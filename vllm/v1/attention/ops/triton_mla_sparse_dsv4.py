# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention kernels for DeepSeek-V4 (head_dim=512).

Adapted from ``rocm_aiter_mla_sparse.py`` (PR #41812), which targets
ROCm but uses architecture-agnostic Triton.  V4 constants:

    NOPE_DIM = 448   (head_dim - qk_rope_head_dim)
    ROPE_DIM = 64    (qk_rope_head_dim)
    COMB_DIM = 512   (head_dim)

These replace the V3.2 ``triton_mla_sparse_kernel.py`` values that
hard-code ``_DIM_QK=576``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# V4 constants
# ---------------------------------------------------------------------------
NOPE_DIM = 448
ROPE_DIM = 64
COMB_DIM = NOPE_DIM + ROPE_DIM  # 512
NOPE_BLOCK = triton.next_power_of_2(NOPE_DIM)  # 512


# ---------------------------------------------------------------------------
# Prefill kernel — standard softmax attention over ragged KV indices
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    kv_lens_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Online-softmax attention over a ragged list of KV tokens per query.

    Each query ``t`` attends to KV slots
    ``kv_indices[kv_indptr[t] : kv_indptr[t+1]]``.
    """
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    # load Q  [BLOCK_H, head_dim]
    q_row = q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h
    q_val = tl.load(q_row + d_offsets[None, :],
                    mask=head_mask[:, None] & d_mask[None, :], other=0.0)

    # online softmax state
    neg_inf = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_inf, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # CUDA-graph safe row-major layout: row t occupies
    # [t * ROW_STRIDE, (t+1) * ROW_STRIDE) in kv_indices; valid prefix
    # length is kv_len (per-row), tail columns hold 0-slot garbage that
    # must NOT be attended. ROW_STRIDE==0 means the legacy ragged layout
    # (indptr-based, kept for non-graph callers).
    if ROW_STRIDE > 0:
        kv_start = query_idx * ROW_STRIDE
        kv_len = tl.load(kv_lens_ptr + query_idx)
    else:
        kv_start = tl.load(kv_indptr_ptr + query_idx)
        kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
        kv_len = kv_end - kv_start
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos,
                       mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        # load K  [BLOCK_K, head_dim]
        k_row = kv_ptr + safe_slot[:, None] * kv_stride_n
        k_val = tl.load(k_row + d_offsets[None, :],
                        mask=valid[:, None] & d_mask[None, :], other=0.0)

        # S = Q @ K^T * scale  [BLOCK_H, BLOCK_K]
        scores = tl.dot(q_val, tl.trans(k_val)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_inf)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_val.dtype), k_val)
        m_i = m_new

    # attn sink (if present)
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets,
                       mask=head_mask, other=neg_inf).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1e-30)
        out_val = tl.where(l_final[:, None] > 0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    else:
        denom = tl.maximum(l_i, 1e-30)
        out_val = tl.where(l_i[:, None] > 0, acc / denom[:, None], 0.0)

    # store O  [BLOCK_H, head_dim]
    out_row = out_ptr + query_idx * out_stride_t + head_offsets[:, None] * out_stride_h
    tl.store(out_row + d_offsets[None, :], out_val,
             mask=head_mask[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# Decode kernel — FP8 NOPE dequant + BF16 ROPE, online softmax
# ---------------------------------------------------------------------------
@triton.jit
def _decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    kv_lens_ptr,
    extra_kv_lens_ptr,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HEAD_BYTES: tl.constexpr,  # per-token cache stride (584 FP8, 512 BF16)
    TOKEN_STRIDE: tl.constexpr,  # data bytes per token (576 FP8, 512 BF16)
    SCALE_DIM: tl.constexpr,  # scale bytes per token (8 FP8, 0 BF16)
    BLOCK_SIZE: tl.constexpr,  # tokens per block (64)
    IS_BF16_CACHE: tl.constexpr,  # True → read NOPE as bf16; False → read FP8
    NOPE_DIM_C: tl.constexpr,  # 448
    NOPE_BLOCK_C: tl.constexpr,  # triton.next_power_of_2(448)
    ROPE_DIM_C: tl.constexpr,  # 64
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    EXTRA_ROW_STRIDE: tl.constexpr,
):
    """Single-launch decode attention over two ragged KV lists.

    FP8 cache layout (IS_BF16_CACHE=False, HEAD_BYTES=584):
        Data region: TOKEN_STRIDE bytes (576) per token
            [0..447]     NOPE as uint8 (FP8 e4m3nv)
            [448..575]   ROPE as bf16
        Scale region: SCALE_DIM bytes (8) per token, stored at offset
            BLOCK_SIZE * TOKEN_STRIDE in the block.

    BF16 cache layout (IS_BF16_CACHE=True, HEAD_BYTES=512):
        Each token: [0..447] NOPE bf16, [448..511] ROPE bf16.
        No scale region (SCALE_DIM=0, TOKEN_STRIDE=512).
    """
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK_C)
    nope_mask = nope_offsets < NOPE_DIM_C
    rope_offsets = tl.arange(0, ROPE_DIM_C)

    # load Q nope + rope  [BLOCK_H, NOPE_BLOCK] + [BLOCK_H, ROPE_DIM]
    q_row = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(q_row + nope_offsets[None, :],
                     mask=head_mask[:, None] & nope_mask[None, :], other=0.0)
    q_rope = tl.load(q_row + NOPE_DIM_C + rope_offsets[None, :],
                     mask=head_mask[:, None], other=0.0)

    neg_inf = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_inf, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK_C), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM_C), dtype=tl.float32)

    k_offsets = tl.arange(0, BLOCK_K)
    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK_C), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM_C), dtype=tl.bfloat16)

    # Helper macro: one ragged-KV attention loop
    # (inlined — Triton JIT does not support nested def)

    # ---- main (SWA) attention ---------------------------------------------
    # ROW_STRIDE > 0: row-major static layout (CUDA-graph safe) — row t
    # occupies [t*ROW_STRIDE, (t+1)*ROW_STRIDE) of the index tensor, valid
    # prefix length is kv_lens[t]. ROW_STRIDE == 0: legacy ragged layout
    # (indptr-based, kept for non-graph callers).
    if ROW_STRIDE > 0:
        main_start = query_idx * ROW_STRIDE
        main_len = tl.load(kv_lens_ptr + query_idx)
    else:
        main_start = tl.load(main_indptr_ptr + query_idx)
        main_end = tl.load(main_indptr_ptr + query_idx + 1)
        main_len = main_end - main_start
    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos,
                       mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)
        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = block_ptr + pos_in_block * TOKEN_STRIDE

        if IS_BF16_CACHE:
            nope_ptr = token_data.to(tl.pointer_type(tl.bfloat16))
            k_nope = tl.load(nope_ptr[:, None] + nope_offsets[None, :],
                             mask=valid[:, None] & nope_mask[None, :], other=0.0)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)
        else:
            x_uint8 = tl.load(token_data[:, None] + nope_offsets[None, :],
                              mask=valid[:, None] & nope_mask[None, :], other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            scale_ptr = block_ptr + BLOCK_SIZE * TOKEN_STRIDE + pos_in_block * SCALE_DIM
            encoded = tl.load(scale_ptr[:, None] + nope_offsets[None, :] // 64,
                              mask=valid[:, None] & nope_mask[None, :], other=127)
            scales = tl.exp2(encoded.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data + NOPE_DIM_C).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(rope_ptr[:, None] + rope_offsets[None, :],
                         mask=valid[:, None], other=0.0)
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_inf)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new

    # ---- extra (compressed) attention ------------------------------------
    if HAS_EXTRA:
        if EXTRA_ROW_STRIDE > 0:
            extra_start = query_idx * EXTRA_ROW_STRIDE
            extra_len = tl.load(extra_kv_lens_ptr + query_idx)
        else:
            extra_start = tl.load(extra_indptr_ptr + query_idx)
            extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
            extra_len = extra_end - extra_start
        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(extra_indices_ptr + extra_start + k_pos,
                           mask=in_range, other=-1)
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)
            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            block_ptr = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            token_data = block_ptr + pos_in_block * TOKEN_STRIDE

            x_uint8 = tl.load(token_data[:, None] + nope_offsets[None, :],
                              mask=valid[:, None] & nope_mask[None, :], other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            # extra cache uses the SAME fp8_ds_mla block layout as the main
            # cache (scales live in a per-block region AFTER the data
            # region) — the old inline-scale formula read garbage scales
            # whenever the extra (DSA global) path activated, corrupting
            # decode attention once a sequence outgrew the SWA window.
            scale_ptr = (
                block_ptr
                + extra_block_size * TOKEN_STRIDE
                + pos_in_block * SCALE_DIM
            )
            encoded = tl.load(scale_ptr[:, None] + nope_offsets[None, :] // 64,
                              mask=valid[:, None] & nope_mask[None, :], other=127)
            scales = tl.exp2(encoded.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data + NOPE_DIM_C).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(rope_ptr[:, None] + rope_offsets[None, :],
                             mask=valid[:, None], other=0.0)
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_inf)
            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new

    # ---- finalize ---------------------------------------------------------
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets,
                       mask=head_mask, other=neg_inf).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1e-30)
        out_nope = tl.where(l_final[:, None] > 0,
                            (acc_nope * alpha[:, None]) / denom[:, None], 0.0)
        out_rope = tl.where(l_final[:, None] > 0,
                            (acc_rope * alpha[:, None]) / denom[:, None], 0.0)
    else:
        denom = tl.maximum(l_i, 1e-30)
        out_nope = tl.where(l_i[:, None] > 0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0, acc_rope / denom[:, None], 0.0)

    # store output  [BLOCK_H, NOPE_DIM] + [BLOCK_H, ROPE_DIM]
    out_row = (out_ptr + query_idx * out_stride0
               + head_offsets[:, None] * out_stride1)
    tl.store(out_row + nope_offsets[None, :], out_nope,
             mask=head_mask[:, None] & nope_mask[None, :])
    tl.store(out_row + NOPE_DIM_C + rope_offsets[None, :], out_rope,
             mask=head_mask[:, None])


# ---------------------------------------------------------------------------
# Host-side launchers
# ---------------------------------------------------------------------------

def _dense_to_ragged(
    indices: torch.Tensor,   # [T, 1, K]
    lengths: torch.Tensor,   # [T]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense indices to ragged (indptr, values) format.

    Returns ``(indptr, ragged_indices)`` where
    ``ragged_indices[indptr[t]:indptr[t+1]]`` are the valid slots for query ``t``.
    """
    T = lengths.shape[0]
    L = lengths.to(torch.int64)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=indices.device)
    indptr[1:] = L.cumsum(0).to(torch.int32)
    # gather valid elements
    K = indices.shape[-1]
    idx = indices.squeeze(1)  # [T, K]
    mask = torch.arange(K, device=indices.device).unsqueeze(0) < L.unsqueeze(1)
    ragged = idx[mask].contiguous()
    return indptr, ragged


def sparse_attn_prefill(
    q: torch.Tensor,          # [T, H, head_dim]  bf16
    kv: torch.Tensor,         # [N, 1, head_dim]  bf16
    indices: torch.Tensor,    # [T, 1, topk]  int32
    sm_scale: float,
    attn_sink: torch.Tensor | None,  # [H]  bf16
    topk_length: torch.Tensor,       # [T]  int32
    out: torch.Tensor,               # [T, H, head_dim]  bf16
) -> None:
    """Prefill sparse MLA attention for DeepSeek-V4 (head_dim=512)."""
    T, H, head_dim = q.shape
    assert head_dim == COMB_DIM, f"expected head_dim={COMB_DIM}, got {head_dim}"
    N = kv.shape[0]
    assert kv.shape[1] == 1 and kv.shape[2] == head_dim

    # CUDA-graph safe: skip the boolean-mask ragged compression
    # (_dense_to_ragged is illegal during capture — dynamic output shape).
    # Use the static row-major layout directly: row t occupies
    # [t*K, (t+1)*K) of the flattened index tensor, valid prefix is
    # topk_length[t], tail columns hold 0-slot garbage that the kernel
    # masks out via kv_len.
    K = indices.shape[-1]
    ragged_idx = indices.squeeze(1).reshape(-1).contiguous()
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=indices.device)

    BLOCK_H = min(16, triton.next_power_of_2(H))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_K = 16 if head_dim < 256 else 32

    grid = (T, triton.cdiv(H, BLOCK_H))
    _prefill_kernel[grid](
        q, kv.squeeze(1), ragged_idx, indptr, topk_length,
        attn_sink if attn_sink is not None else q,  # dummy ptr if no sink
        out,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(2),  # squeezed: [N, head_dim]
        out.stride(0), out.stride(1), out.stride(2),
        H, head_dim, N, sm_scale,
        HAS_ATTN_SINK=attn_sink is not None,
        ROW_STRIDE=K,
        BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )


def sparse_attn_decode(
    q: torch.Tensor,                # [D, 1, head_dim]  bf16
    k_cache: torch.Tensor,          # [num_blocks, block_size, 1, head_bytes]  uint8
    indices: torch.Tensor,          # [D, 1, max_swa]  int32
    topk_length: torch.Tensor,      # [D]  int32
    softmax_scale: float,
    attn_sink: torch.Tensor | None, # [padded_heads]
    extra_k_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    out: torch.Tensor,              # [D, 1, head_dim]  bf16
) -> None:
    """Decode sparse MLA attention for DeepSeek-V4 (head_dim=512).

    Supports both fp8_ds_mla and bf16 KV cache formats.
    """
    D, _, _ = q.shape
    num_heads = q.shape[1]
    assert q.shape[2] == COMB_DIM

    # Detect cache format: uint8 → fp8_ds_mla, bf16 → pure bf16
    is_bf16_cache = k_cache.dtype == torch.bfloat16
    head_bytes = 512 if is_bf16_cache else 584
    token_stride = 512 if is_bf16_cache else NOPE_DIM + ROPE_DIM * 2  # 576
    scale_dim = 0 if is_bf16_cache else NOPE_DIM // 64 + 1  # 8

    # Build index lists. CUDA-graph note: skip the boolean-mask ragged
    # compression (_dense_to_ragged is illegal during capture — dynamic
    # output shape). Use the static row-major layout directly: row t
    # occupies [t*K, (t+1)*K) of the flattened index tensor, valid prefix
    # is topk_length[t]; tail columns hold -1 padding that the kernel
    # masks out via kv_len.
    K = indices.shape[-1]
    main_indptr = torch.zeros(D + 1, dtype=torch.int32, device=indices.device)
    main_ragged = indices.squeeze(1).reshape(-1).contiguous()
    main_lens = topk_length
    has_extra = extra_k_cache is not None and extra_indices is not None
    extra_indptr = extra_ragged = None
    extra_lens = None
    extra_rows = extra_bs = 0
    EXTRA_ROW_STRIDE = 0
    if has_extra:
        assert extra_topk_length is not None
        EK = extra_indices.shape[-1]
        extra_indptr = torch.zeros(D + 1, dtype=torch.int32, device=indices.device)
        extra_ragged = extra_indices.squeeze(1).reshape(-1).contiguous()
        extra_lens = extra_topk_length
        EXTRA_ROW_STRIDE = EK
        extra_rows = extra_k_cache.shape[0] * extra_k_cache.shape[1]
        extra_bs = extra_k_cache.shape[1]

    # cache shape → compute stride parameters
    main_cache = k_cache.squeeze(2)  # [num_blocks, block_size, head_bytes]
    main_stride0 = main_cache.shape[1] * head_bytes  # block_size * head_bytes
    main_rows = main_cache.shape[0] * main_cache.shape[1]
    main_bs = main_cache.shape[1]
    extra_stride0 = 0
    extra_cache_flat = extra_k_cache
    extra_rows = extra_bs = 0
    if has_extra:
        extra_cache_flat = extra_k_cache.squeeze(2)
        extra_stride0 = extra_cache_flat.shape[1] * head_bytes
        extra_rows = extra_cache_flat.shape[0] * extra_cache_flat.shape[1]
        extra_bs = extra_cache_flat.shape[1]

    BLOCK_H = min(16, triton.next_power_of_2(num_heads))
    BLOCK_K = 16

    # Pass a 1D byte view for FP8, or as bf16 for BF16 cache
    if is_bf16_cache:
        main_cache_flat = main_cache.contiguous().view(torch.bfloat16).reshape(-1)
        extra_cache_flat_u8 = None
        if has_extra:
            extra_cache_flat_u8 = extra_cache_flat.contiguous().view(torch.bfloat16).reshape(-1)
    else:
        main_cache_flat = main_cache.contiguous().view(torch.uint8).reshape(-1)
        extra_cache_flat_u8 = None
        if has_extra:
            extra_cache_flat_u8 = extra_cache_flat.contiguous().view(torch.uint8).reshape(-1)
    if has_extra:
        extra_cache_flat_u8 = extra_cache_flat.contiguous().view(torch.uint8).reshape(-1)

    grid = (D, triton.cdiv(num_heads, BLOCK_H))
    _decode_kernel[grid](
        q, main_cache_flat, main_ragged, main_indptr,
        extra_cache_flat_u8 if has_extra else main_cache_flat,
        extra_ragged if has_extra else main_ragged,
        extra_indptr if has_extra else main_indptr,
        attn_sink if attn_sink is not None else q,
        out,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        main_lens,
        extra_lens if has_extra else main_lens,
        main_stride0, extra_stride0,
        main_rows, extra_rows, main_bs, extra_bs,
        softmax_scale, num_heads,
        HAS_ATTN_SINK=attn_sink is not None,
        HAS_EXTRA=has_extra,
        HEAD_BYTES=head_bytes,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        BLOCK_SIZE=main_bs,
        IS_BF16_CACHE=is_bf16_cache,
        NOPE_DIM_C=NOPE_DIM,
        NOPE_BLOCK_C=NOPE_BLOCK,
        ROPE_DIM_C=ROPE_DIM,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        ROW_STRIDE=K,
        EXTRA_ROW_STRIDE=EXTRA_ROW_STRIDE,
    )
