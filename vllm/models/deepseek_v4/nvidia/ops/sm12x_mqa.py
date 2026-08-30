# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels used by the local DeepSeek V4 path."""

import torch
import os

from vllm.triton_utils import tl, triton


def _bucketed_logits_buffer(
    num_rows: int, row_width: int, device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fp32 (num_rows, row_width) output, allocated in power-of-two buckets.

    During a chunked prefill the logits width (the compressed KV length) grows
    monotonically with every chunk, so a fresh exact-size ``torch.empty`` per
    call means no request can ever be served from a cached block: the caching
    allocator maps a new segment each chunk and ``memory_reserved`` ratchets
    toward the SUM of the distinct sizes. On unified-memory devices (GB10)
    that reserve is system RAM -- ~2.6-3 GiB per 32K needle prompt
    (jasl/vllm#31). Rounding the flat allocation up to a power of two makes
    consecutive chunks share a bucket, bounding the transient footprint at
    ~2x the largest live buffer instead of the running sum. The trailing view
    keeps the tensor contiguous, so kernel stride assumptions are unchanged.
    """
    numel = num_rows * row_width
    if numel == 0:
        return torch.empty((num_rows, row_width), device=device, dtype=dtype)
    alloc = 1 << (numel - 1).bit_length()
    flat = torch.empty((alloc,), device=device, dtype=dtype)
    return flat[:numel].view(num_rows, row_width)


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions")
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q,
    seq_len_kv,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (
        offs_n[None, :] < seq_end[:, None]
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + d[None, :] * stride_kd,
                mask=valid_n[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, tl.trans(k), input_precision="tf32")
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(seq_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    logits = _bucketed_logits_buffer(num_q, seq_len_kv, q.device)
    if num_q == 0 or seq_len_kv == 0:
        return logits

    block_m = _fp8_mqa_logits_block_m(num_q, seq_len_kv)
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, 128))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


def _fp8_mqa_logits_block_m(num_q: int, seq_len_kv: int) -> int:
    if seq_len_kv <= 16 * 1024:
        return 16
    return 64


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr
        + block_idx.to(tl.int64) * stride_sb
        + block_offset[None, :] * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[:, :, None].to(tl.int64) * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=context_mask[:, :, None] & (d[None, None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(context_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel optimised for long ``token_count``.

    Each Triton program handles one logical row (``batch * next_n + q_pos``)
    across a ``BLOCK_N``-wide window of token positions. Q is loaded once per
    head tile and reused for every K element in the window, which preserves
    L2 / register locality and avoids the M-axis padding waste of the
    generic 2D-tiled kernel at long contexts (mt-bench c=1 MTP=2 num_rows=3
    with token_count=131072 launches 12k programs of 128 logits each rather
    than 8k programs of 64 logits with 25 % M-axis waste).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    )

    scale = tl.load(
        scale_ptr + block_idx.to(tl.int64) * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_pos * stride_qn
                + heads[:, None] * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[None, :].to(tl.int64) * stride_kvb
                + block_offset[None, :] * stride_kvs
                + d[:, None] * stride_kvd,
                mask=context_mask[None, :] & (d[:, None] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    """Rowwise paged-MQA logits wrapper.

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` inside ``_fp8_paged_mqa_logits_rowwise_kernel`` lands on
    tensor-core friendly tile shapes. DSv4-Flash (head_dim=128,
    num_heads=64) satisfies both and is the only model that exercises this
    path today; the generic 2D kernel below remains the fallback for
    misaligned shapes.
    """
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = _bucketed_logits_buffer(num_rows, token_count, q.device)
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # Aligned head shapes (DSv4-Flash and any future MQA model with
    # ``head_dim % 64 == 0`` and ``num_heads % 4 == 0``) get the rowwise
    # kernel, which keeps long-context decode (>100K tokens) on a per-row
    # grid that re-uses Q across the full token window. The generic 2D
    # kernel below still handles misaligned shapes and remains the canonical
    # reference for the rowwise variant.
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = _bucketed_logits_buffer(num_rows, token_count, q.device)
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    # Adaptive BLOCK_M: the kernel masks off positions >= num_rows, so a fixed
    # BLOCK_M=4 wastes ~75% of M-axis work in the common single-stream decode
    # case (num_rows=1). Pick the smallest power-of-2 tile that still covers
    # num_rows so we keep one grid-program for typical decode while still
    # benefiting from larger tiles when batch / MTP push num_rows higher.
    if num_rows <= 1:
        block_m = 1
    elif num_rows <= 2:
        block_m = 2
    elif num_rows <= 4:
        block_m = 4
    else:
        block_m = 8
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(token_count, 64))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


@triton.jit
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


@triton.jit
def _indexer_score_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_ks_ptr,
    cu_ke_ptr,
    out_ptr,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wm,
    stride_wh,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Indexer dense scoring: logits[m,n] = sum_h W[m,h] * relu(Q[m,h] . (K[n] * scale[n])).

    vs _fp8_mqa_logits_kernel (the original SM12x fallback): K tile is staged
    ONCE per program and reused across a fully unrolled static h-loop; the
    dot runs on fp8 tensor cores (fp8 x fp8 products are exact in fp32
    accumulation, per-n scale applied post-dot) so the result matches the
    fp32 reference semantics.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    valid_m = offs_m < M
    valid_n = offs_n < N

    ks = tl.load(cu_ks_ptr + offs_m, mask=valid_m, other=0)
    ke = tl.load(cu_ke_ptr + offs_m, mask=valid_m, other=0)

    # K tile [D, BLOCK_N] staged once, kept in fp8. The per-n scale is
    # applied AFTER the dot: (q . (k * s))[m,n] == s[n] * (q . k)[m,n], and
    # fp8 x fp8 products are exact in fp32 accumulation, so this matches the
    # fp32 reference semantics without bf16 input rounding.
    k_f8 = tl.load(
        k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
        mask=valid_n[None, :],
        other=0.0,
    )
    k_sc = tl.load(scale_ptr + offs_n, mask=valid_n, other=1.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Single accumulator with BN=64 (see launcher): preserves the sequential
    # h-summation order bit-for-bit while reaching ~460 TFLOPS (vs ~190 at
    # BN=128). A dual-accumulator variant hits ~550 but changes the fp32
    # summation order (~1e-7), which flipped a real tie in validation
    # (ratio 1.000 -> 0.956) and OOM'd at 1M tile shapes — rejected.
    for h in tl.static_range(H):
        q_f8 = tl.load(
            q_ptr
            + offs_m[:, None] * stride_qm
            + h * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=valid_m[:, None],
            other=0.0,
        )
        s = tl.dot(q_f8, k_f8)  # fp8 x fp8 -> fp32 acc, products exact
        s = s * k_sc[None, :]  # per-n K scale
        s = tl.maximum(s, 0.0)  # per-head ReLU
        w = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        acc += s * w[:, None]

    row_valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    store_mask = valid_m[:, None] & valid_n[None, :]
    acc = tl.where(row_valid & store_mask, acc, float("-inf"))
    if OUT_BF16:
        acc = acc.to(tl.bfloat16)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=store_mask,
    )





@triton.jit
def _split_h(f, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """按 h 轴偶奇拆分: [BM, K, BN] -> 两个 [BM, K/2, BN] (h=2a 与 h=2a+1)."""
    r = tl.reshape(f, [BLOCK_M, K // 2, 2, BLOCK_N])
    r = tl.permute(r, (0, 1, 3, 2))       # [BM, K/2, BN, 2]
    lo, hi = tl.split(r)
    return lo, hi


@triton.jit
def _split_h_w(f, K: tl.constexpr, BLOCK_M: tl.constexpr):
    """按 h 轴偶奇拆分 1D 权重: [BM, K] -> 两个 [BM, K/2]."""
    r = tl.reshape(f, [BLOCK_M, K // 2, 2])   # pair 轴已在最后
    lo, hi = tl.split(r)
    return lo, hi


@triton.jit
def _indexer_score_logits_kernel_v2(
    q_ptr, k_ptr, scale_ptr, weights_ptr,
    cu_ks_ptr, cu_ke_ptr, out_ptr,
    M, N,
    H: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """G4 ① GEMM 重排: 64 头摊平成 M 维做单一大 GEMM (消除串行小 dot 延迟界),
    逐头 h 序提取累加 (与参考逐位同)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    valid_m = offs_m < M
    valid_n = offs_n < N

    ks = tl.load(cu_ks_ptr + offs_m, mask=valid_m, other=0)
    ke = tl.load(cu_ke_ptr + offs_m, mask=valid_m, other=0)

    k_f8 = tl.load(
        k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
        mask=valid_n[None, :], other=0.0)
    k_sc = tl.load(scale_ptr + offs_n, mask=valid_n, other=1.0)

    # 摊平 3D 加载 q [BM, H, D] -> reshape [BM*H, D] (行主序视图)
    q3 = tl.load(
        q_ptr + offs_m[:, None, None] * stride_qm
        + offs_h[None, :, None] * stride_qh
        + offs_d[None, None, :] * stride_qd,
        mask=valid_m[:, None, None], other=0.0)
    q_f8 = tl.reshape(q3, [BLOCK_M * H, D])
    w3 = tl.load(
        weights_ptr + offs_m[:, None] * stride_wm + offs_h[None, :] * stride_wh,
        mask=valid_m[:, None], other=0.0)

    s = tl.dot(q_f8, k_f8)              # [BM*H, BN] — K=128 链与参考逐位同
    s = s * k_sc[None, :]
    s = tl.maximum(s, 0.0)
    # 注: w 不在树前乘入 — FMA 链做唯一一次 w 应用 (与参考 acc += s*w 同构)

    s3 = tl.reshape(s, [BLOCK_M, H, BLOCK_N])
    t0_139840926128320_lo, t0_139840926128320_hi = _split_h(s3, 64, BLOCK_M, BLOCK_N)
    t1_139840926872624_lo, t1_139840926872624_hi = _split_h(t0_139840926128320_lo, 32, BLOCK_M, BLOCK_N)
    t1_139840926872688_lo, t1_139840926872688_hi = _split_h(t0_139840926128320_hi, 32, BLOCK_M, BLOCK_N)
    t2_139840927049264_lo, t2_139840927049264_hi = _split_h(t1_139840926872624_lo, 16, BLOCK_M, BLOCK_N)
    t2_139840927048048_lo, t2_139840927048048_hi = _split_h(t1_139840926872624_hi, 16, BLOCK_M, BLOCK_N)
    t2_139840925057072_lo, t2_139840925057072_hi = _split_h(t1_139840926872688_lo, 16, BLOCK_M, BLOCK_N)
    t2_139840925057136_lo, t2_139840925057136_hi = _split_h(t1_139840926872688_hi, 16, BLOCK_M, BLOCK_N)
    t3_139840926872624_lo, t3_139840926872624_hi = _split_h(t2_139840927049264_lo, 8, BLOCK_M, BLOCK_N)
    t3_139840925057392_lo, t3_139840925057392_hi = _split_h(t2_139840927049264_hi, 8, BLOCK_M, BLOCK_N)
    t3_139840925057456_lo, t3_139840925057456_hi = _split_h(t2_139840927048048_lo, 8, BLOCK_M, BLOCK_N)
    t3_139840925057520_lo, t3_139840925057520_hi = _split_h(t2_139840927048048_hi, 8, BLOCK_M, BLOCK_N)
    t3_139840925057840_lo, t3_139840925057840_hi = _split_h(t2_139840925057072_lo, 8, BLOCK_M, BLOCK_N)
    t3_139840925057904_lo, t3_139840925057904_hi = _split_h(t2_139840925057072_hi, 8, BLOCK_M, BLOCK_N)
    t3_139840925058224_lo, t3_139840925058224_hi = _split_h(t2_139840925057136_lo, 8, BLOCK_M, BLOCK_N)
    t3_139840925058288_lo, t3_139840925058288_hi = _split_h(t2_139840925057136_hi, 8, BLOCK_M, BLOCK_N)
    t4_139840927049264_lo, t4_139840927049264_hi = _split_h(t3_139840926872624_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925057136_lo, t4_139840925057136_hi = _split_h(t3_139840926872624_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925057072_lo, t4_139840925057072_hi = _split_h(t3_139840925057392_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925058544_lo, t4_139840925058544_hi = _split_h(t3_139840925057392_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840927048048_lo, t4_139840927048048_hi = _split_h(t3_139840925057456_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925058608_lo, t4_139840925058608_hi = _split_h(t3_139840925057456_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925058672_lo, t4_139840925058672_hi = _split_h(t3_139840925057520_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925058736_lo, t4_139840925058736_hi = _split_h(t3_139840925057520_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925058800_lo, t4_139840925058800_hi = _split_h(t3_139840925057840_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925058864_lo, t4_139840925058864_hi = _split_h(t3_139840925057840_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925059056_lo, t4_139840925059056_hi = _split_h(t3_139840925057904_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925059120_lo, t4_139840925059120_hi = _split_h(t3_139840925057904_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925059312_lo, t4_139840925059312_hi = _split_h(t3_139840925058224_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925059376_lo, t4_139840925059376_hi = _split_h(t3_139840925058224_hi, 4, BLOCK_M, BLOCK_N)
    t4_139840925059568_lo, t4_139840925059568_hi = _split_h(t3_139840925058288_lo, 4, BLOCK_M, BLOCK_N)
    t4_139840925059632_lo, t4_139840925059632_hi = _split_h(t3_139840925058288_hi, 4, BLOCK_M, BLOCK_N)
    t5_139840925058288_lo, t5_139840925058288_hi = _split_h(t4_139840927049264_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925058224_lo, t5_139840925058224_hi = _split_h(t4_139840927049264_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925057904_lo, t5_139840925057904_hi = _split_h(t4_139840925057136_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925057840_lo, t5_139840925057840_hi = _split_h(t4_139840925057136_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925057520_lo, t5_139840925057520_hi = _split_h(t4_139840925057072_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925057456_lo, t5_139840925057456_hi = _split_h(t4_139840925057072_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925057392_lo, t5_139840925057392_hi = _split_h(t4_139840925058544_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925059760_lo, t5_139840925059760_hi = _split_h(t4_139840925058544_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925059824_lo, t5_139840925059824_hi = _split_h(t4_139840927048048_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925059888_lo, t5_139840925059888_hi = _split_h(t4_139840927048048_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925060080_lo, t5_139840925060080_hi = _split_h(t4_139840925058608_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925060144_lo, t5_139840925060144_hi = _split_h(t4_139840925058608_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925060336_lo, t5_139840925060336_hi = _split_h(t4_139840925058672_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925060400_lo, t5_139840925060400_hi = _split_h(t4_139840925058672_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925060592_lo, t5_139840925060592_hi = _split_h(t4_139840925058736_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925060656_lo, t5_139840925060656_hi = _split_h(t4_139840925058736_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925060848_lo, t5_139840925060848_hi = _split_h(t4_139840925058800_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925060912_lo, t5_139840925060912_hi = _split_h(t4_139840925058800_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925061104_lo, t5_139840925061104_hi = _split_h(t4_139840925058864_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925061168_lo, t5_139840925061168_hi = _split_h(t4_139840925058864_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925061360_lo, t5_139840925061360_hi = _split_h(t4_139840925059056_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925061424_lo, t5_139840925061424_hi = _split_h(t4_139840925059056_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925061616_lo, t5_139840925061616_hi = _split_h(t4_139840925059120_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925061680_lo, t5_139840925061680_hi = _split_h(t4_139840925059120_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925061872_lo, t5_139840925061872_hi = _split_h(t4_139840925059312_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925061936_lo, t5_139840925061936_hi = _split_h(t4_139840925059312_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925062128_lo, t5_139840925062128_hi = _split_h(t4_139840925059376_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925062192_lo, t5_139840925062192_hi = _split_h(t4_139840925059376_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925062384_lo, t5_139840925062384_hi = _split_h(t4_139840925059568_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925062448_lo, t5_139840925062448_hi = _split_h(t4_139840925059568_hi, 2, BLOCK_M, BLOCK_N)
    t5_139840925062640_lo, t5_139840925062640_hi = _split_h(t4_139840925059632_lo, 2, BLOCK_M, BLOCK_N)
    t5_139840925062704_lo, t5_139840925062704_hi = _split_h(t4_139840925059632_hi, 2, BLOCK_M, BLOCK_N)
    w0_139840926720544_lo, w0_139840926720544_hi = _split_h_w(w3, 64, BLOCK_M)
    w1_139840925062640_lo, w1_139840925062640_hi = _split_h_w(w0_139840926720544_lo, 32, BLOCK_M)
    w1_139840925062448_lo, w1_139840925062448_hi = _split_h_w(w0_139840926720544_hi, 32, BLOCK_M)
    w2_139840925062384_lo, w2_139840925062384_hi = _split_h_w(w1_139840925062640_lo, 16, BLOCK_M)
    w2_139840925062192_lo, w2_139840925062192_hi = _split_h_w(w1_139840925062640_hi, 16, BLOCK_M)
    w2_139840925062128_lo, w2_139840925062128_hi = _split_h_w(w1_139840925062448_lo, 16, BLOCK_M)
    w2_139840925061936_lo, w2_139840925061936_hi = _split_h_w(w1_139840925062448_hi, 16, BLOCK_M)
    w3_139840925062640_lo, w3_139840925062640_hi = _split_h_w(w2_139840925062384_lo, 8, BLOCK_M)
    w3_139840925062704_lo, w3_139840925062704_hi = _split_h_w(w2_139840925062384_hi, 8, BLOCK_M)
    w3_139840925061872_lo, w3_139840925061872_hi = _split_h_w(w2_139840925062192_lo, 8, BLOCK_M)
    w3_139840925061680_lo, w3_139840925061680_hi = _split_h_w(w2_139840925062192_hi, 8, BLOCK_M)
    w3_139840925061424_lo, w3_139840925061424_hi = _split_h_w(w2_139840925062128_lo, 8, BLOCK_M)
    w3_139840925061360_lo, w3_139840925061360_hi = _split_h_w(w2_139840925062128_hi, 8, BLOCK_M)
    w3_139840925061168_lo, w3_139840925061168_hi = _split_h_w(w2_139840925061936_lo, 8, BLOCK_M)
    w3_139840925061104_lo, w3_139840925061104_hi = _split_h_w(w2_139840925061936_hi, 8, BLOCK_M)
    w4_139840925062128_lo, w4_139840925062128_hi = _split_h_w(w3_139840925062640_lo, 4, BLOCK_M)
    w4_139840925062192_lo, w4_139840925062192_hi = _split_h_w(w3_139840925062640_hi, 4, BLOCK_M)
    w4_139840925062384_lo, w4_139840925062384_hi = _split_h_w(w3_139840925062704_lo, 4, BLOCK_M)
    w4_139840925061616_lo, w4_139840925061616_hi = _split_h_w(w3_139840925062704_hi, 4, BLOCK_M)
    w4_139840925061936_lo, w4_139840925061936_hi = _split_h_w(w3_139840925061872_lo, 4, BLOCK_M)
    w4_139840925060912_lo, w4_139840925060912_hi = _split_h_w(w3_139840925061872_hi, 4, BLOCK_M)
    w4_139840925060848_lo, w4_139840925060848_hi = _split_h_w(w3_139840925061680_lo, 4, BLOCK_M)
    w4_139840925060656_lo, w4_139840925060656_hi = _split_h_w(w3_139840925061680_hi, 4, BLOCK_M)
    w4_139840925060592_lo, w4_139840925060592_hi = _split_h_w(w3_139840925061424_lo, 4, BLOCK_M)
    w4_139840925060400_lo, w4_139840925060400_hi = _split_h_w(w3_139840925061424_hi, 4, BLOCK_M)
    w4_139840925060336_lo, w4_139840925060336_hi = _split_h_w(w3_139840925061360_lo, 4, BLOCK_M)
    w4_139840925060144_lo, w4_139840925060144_hi = _split_h_w(w3_139840925061360_hi, 4, BLOCK_M)
    w4_139840925060080_lo, w4_139840925060080_hi = _split_h_w(w3_139840925061168_lo, 4, BLOCK_M)
    w4_139840925059888_lo, w4_139840925059888_hi = _split_h_w(w3_139840925061168_hi, 4, BLOCK_M)
    w4_139840925059824_lo, w4_139840925059824_hi = _split_h_w(w3_139840925061104_lo, 4, BLOCK_M)
    w4_139840925059760_lo, w4_139840925059760_hi = _split_h_w(w3_139840925061104_hi, 4, BLOCK_M)
    w5_139840925061168_lo, w5_139840925061168_hi = _split_h_w(w4_139840925062128_lo, 2, BLOCK_M)
    w5_139840925061360_lo, w5_139840925061360_hi = _split_h_w(w4_139840925062128_hi, 2, BLOCK_M)
    w5_139840925061424_lo, w5_139840925061424_hi = _split_h_w(w4_139840925062192_lo, 2, BLOCK_M)
    w5_139840925061680_lo, w5_139840925061680_hi = _split_h_w(w4_139840925062192_hi, 2, BLOCK_M)
    w5_139840925061872_lo, w5_139840925061872_hi = _split_h_w(w4_139840925062384_lo, 2, BLOCK_M)
    w5_139840925062704_lo, w5_139840925062704_hi = _split_h_w(w4_139840925062384_hi, 2, BLOCK_M)
    w5_139840925062640_lo, w5_139840925062640_hi = _split_h_w(w4_139840925061616_lo, 2, BLOCK_M)
    w5_139840925062448_lo, w5_139840925062448_hi = _split_h_w(w4_139840925061616_hi, 2, BLOCK_M)
    w5_139840925057392_lo, w5_139840925057392_hi = _split_h_w(w4_139840925061936_lo, 2, BLOCK_M)
    w5_139840925057456_lo, w5_139840925057456_hi = _split_h_w(w4_139840925061936_hi, 2, BLOCK_M)
    w5_139840925057520_lo, w5_139840925057520_hi = _split_h_w(w4_139840925060912_lo, 2, BLOCK_M)
    w5_139840925057840_lo, w5_139840925057840_hi = _split_h_w(w4_139840925060912_hi, 2, BLOCK_M)
    w5_139840925057904_lo, w5_139840925057904_hi = _split_h_w(w4_139840925060848_lo, 2, BLOCK_M)
    w5_139840925058224_lo, w5_139840925058224_hi = _split_h_w(w4_139840925060848_hi, 2, BLOCK_M)
    w5_139840925058288_lo, w5_139840925058288_hi = _split_h_w(w4_139840925060656_lo, 2, BLOCK_M)
    w5_139840925068976_lo, w5_139840925068976_hi = _split_h_w(w4_139840925060656_hi, 2, BLOCK_M)
    w5_139840925069040_lo, w5_139840925069040_hi = _split_h_w(w4_139840925060592_lo, 2, BLOCK_M)
    w5_139840925069104_lo, w5_139840925069104_hi = _split_h_w(w4_139840925060592_hi, 2, BLOCK_M)
    w5_139840925069296_lo, w5_139840925069296_hi = _split_h_w(w4_139840925060400_lo, 2, BLOCK_M)
    w5_139840925069360_lo, w5_139840925069360_hi = _split_h_w(w4_139840925060400_hi, 2, BLOCK_M)
    w5_139840925069552_lo, w5_139840925069552_hi = _split_h_w(w4_139840925060336_lo, 2, BLOCK_M)
    w5_139840925069616_lo, w5_139840925069616_hi = _split_h_w(w4_139840925060336_hi, 2, BLOCK_M)
    w5_139840925069808_lo, w5_139840925069808_hi = _split_h_w(w4_139840925060144_lo, 2, BLOCK_M)
    w5_139840925069872_lo, w5_139840925069872_hi = _split_h_w(w4_139840925060144_hi, 2, BLOCK_M)
    w5_139840925070064_lo, w5_139840925070064_hi = _split_h_w(w4_139840925060080_lo, 2, BLOCK_M)
    w5_139840925070128_lo, w5_139840925070128_hi = _split_h_w(w4_139840925060080_hi, 2, BLOCK_M)
    w5_139840925070320_lo, w5_139840925070320_hi = _split_h_w(w4_139840925059888_lo, 2, BLOCK_M)
    w5_139840925070384_lo, w5_139840925070384_hi = _split_h_w(w4_139840925059888_hi, 2, BLOCK_M)
    w5_139840925070576_lo, w5_139840925070576_hi = _split_h_w(w4_139840925059824_lo, 2, BLOCK_M)
    w5_139840925070640_lo, w5_139840925070640_hi = _split_h_w(w4_139840925059824_hi, 2, BLOCK_M)
    w5_139840925070832_lo, w5_139840925070832_hi = _split_h_w(w4_139840925059760_lo, 2, BLOCK_M)
    w5_139840925070896_lo, w5_139840925070896_hi = _split_h_w(w4_139840925059760_hi, 2, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    t5_139840925058288_lo = tl.reshape(t5_139840925058288_lo, [BLOCK_M, BLOCK_N])
    w5_139840925061168_lo = tl.reshape(w5_139840925061168_lo, [BLOCK_M])
    acc += t5_139840925058288_lo * w5_139840925061168_lo[:, None]
    t5_139840925060848_lo = tl.reshape(t5_139840925060848_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069040_lo = tl.reshape(w5_139840925069040_lo, [BLOCK_M])
    acc += t5_139840925060848_lo * w5_139840925069040_lo[:, None]
    t5_139840925059824_lo = tl.reshape(t5_139840925059824_lo, [BLOCK_M, BLOCK_N])
    w5_139840925057392_lo = tl.reshape(w5_139840925057392_lo, [BLOCK_M])
    acc += t5_139840925059824_lo * w5_139840925057392_lo[:, None]
    t5_139840925061872_lo = tl.reshape(t5_139840925061872_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070064_lo = tl.reshape(w5_139840925070064_lo, [BLOCK_M])
    acc += t5_139840925061872_lo * w5_139840925070064_lo[:, None]
    t5_139840925057520_lo = tl.reshape(t5_139840925057520_lo, [BLOCK_M, BLOCK_N])
    w5_139840925061872_lo = tl.reshape(w5_139840925061872_lo, [BLOCK_M])
    acc += t5_139840925057520_lo * w5_139840925061872_lo[:, None]
    t5_139840925061360_lo = tl.reshape(t5_139840925061360_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069552_lo = tl.reshape(w5_139840925069552_lo, [BLOCK_M])
    acc += t5_139840925061360_lo * w5_139840925069552_lo[:, None]
    t5_139840925060336_lo = tl.reshape(t5_139840925060336_lo, [BLOCK_M, BLOCK_N])
    w5_139840925057904_lo = tl.reshape(w5_139840925057904_lo, [BLOCK_M])
    acc += t5_139840925060336_lo * w5_139840925057904_lo[:, None]
    t5_139840925062384_lo = tl.reshape(t5_139840925062384_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070576_lo = tl.reshape(w5_139840925070576_lo, [BLOCK_M])
    acc += t5_139840925062384_lo * w5_139840925070576_lo[:, None]
    t5_139840925057904_lo = tl.reshape(t5_139840925057904_lo, [BLOCK_M, BLOCK_N])
    w5_139840925061424_lo = tl.reshape(w5_139840925061424_lo, [BLOCK_M])
    acc += t5_139840925057904_lo * w5_139840925061424_lo[:, None]
    t5_139840925061104_lo = tl.reshape(t5_139840925061104_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069296_lo = tl.reshape(w5_139840925069296_lo, [BLOCK_M])
    acc += t5_139840925061104_lo * w5_139840925069296_lo[:, None]
    t5_139840925060080_lo = tl.reshape(t5_139840925060080_lo, [BLOCK_M, BLOCK_N])
    w5_139840925057520_lo = tl.reshape(w5_139840925057520_lo, [BLOCK_M])
    acc += t5_139840925060080_lo * w5_139840925057520_lo[:, None]
    t5_139840925062128_lo = tl.reshape(t5_139840925062128_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070320_lo = tl.reshape(w5_139840925070320_lo, [BLOCK_M])
    acc += t5_139840925062128_lo * w5_139840925070320_lo[:, None]
    t5_139840925057392_lo = tl.reshape(t5_139840925057392_lo, [BLOCK_M, BLOCK_N])
    w5_139840925062640_lo = tl.reshape(w5_139840925062640_lo, [BLOCK_M])
    acc += t5_139840925057392_lo * w5_139840925062640_lo[:, None]
    t5_139840925061616_lo = tl.reshape(t5_139840925061616_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069808_lo = tl.reshape(w5_139840925069808_lo, [BLOCK_M])
    acc += t5_139840925061616_lo * w5_139840925069808_lo[:, None]
    t5_139840925060592_lo = tl.reshape(t5_139840925060592_lo, [BLOCK_M, BLOCK_N])
    w5_139840925058288_lo = tl.reshape(w5_139840925058288_lo, [BLOCK_M])
    acc += t5_139840925060592_lo * w5_139840925058288_lo[:, None]
    t5_139840925062640_lo = tl.reshape(t5_139840925062640_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070832_lo = tl.reshape(w5_139840925070832_lo, [BLOCK_M])
    acc += t5_139840925062640_lo * w5_139840925070832_lo[:, None]
    t5_139840925058224_lo = tl.reshape(t5_139840925058224_lo, [BLOCK_M, BLOCK_N])
    w5_139840925061360_lo = tl.reshape(w5_139840925061360_lo, [BLOCK_M])
    acc += t5_139840925058224_lo * w5_139840925061360_lo[:, None]
    t5_139840925060912_lo = tl.reshape(t5_139840925060912_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069104_lo = tl.reshape(w5_139840925069104_lo, [BLOCK_M])
    acc += t5_139840925060912_lo * w5_139840925069104_lo[:, None]
    t5_139840925059888_lo = tl.reshape(t5_139840925059888_lo, [BLOCK_M, BLOCK_N])
    w5_139840925057456_lo = tl.reshape(w5_139840925057456_lo, [BLOCK_M])
    acc += t5_139840925059888_lo * w5_139840925057456_lo[:, None]
    t5_139840925061936_lo = tl.reshape(t5_139840925061936_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070128_lo = tl.reshape(w5_139840925070128_lo, [BLOCK_M])
    acc += t5_139840925061936_lo * w5_139840925070128_lo[:, None]
    t5_139840925057456_lo = tl.reshape(t5_139840925057456_lo, [BLOCK_M, BLOCK_N])
    w5_139840925062704_lo = tl.reshape(w5_139840925062704_lo, [BLOCK_M])
    acc += t5_139840925057456_lo * w5_139840925062704_lo[:, None]
    t5_139840925061424_lo = tl.reshape(t5_139840925061424_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069616_lo = tl.reshape(w5_139840925069616_lo, [BLOCK_M])
    acc += t5_139840925061424_lo * w5_139840925069616_lo[:, None]
    t5_139840925060400_lo = tl.reshape(t5_139840925060400_lo, [BLOCK_M, BLOCK_N])
    w5_139840925058224_lo = tl.reshape(w5_139840925058224_lo, [BLOCK_M])
    acc += t5_139840925060400_lo * w5_139840925058224_lo[:, None]
    t5_139840925062448_lo = tl.reshape(t5_139840925062448_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070640_lo = tl.reshape(w5_139840925070640_lo, [BLOCK_M])
    acc += t5_139840925062448_lo * w5_139840925070640_lo[:, None]
    t5_139840925057840_lo = tl.reshape(t5_139840925057840_lo, [BLOCK_M, BLOCK_N])
    w5_139840925061680_lo = tl.reshape(w5_139840925061680_lo, [BLOCK_M])
    acc += t5_139840925057840_lo * w5_139840925061680_lo[:, None]
    t5_139840925061168_lo = tl.reshape(t5_139840925061168_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069360_lo = tl.reshape(w5_139840925069360_lo, [BLOCK_M])
    acc += t5_139840925061168_lo * w5_139840925069360_lo[:, None]
    t5_139840925060144_lo = tl.reshape(t5_139840925060144_lo, [BLOCK_M, BLOCK_N])
    w5_139840925057840_lo = tl.reshape(w5_139840925057840_lo, [BLOCK_M])
    acc += t5_139840925060144_lo * w5_139840925057840_lo[:, None]
    t5_139840925062192_lo = tl.reshape(t5_139840925062192_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070384_lo = tl.reshape(w5_139840925070384_lo, [BLOCK_M])
    acc += t5_139840925062192_lo * w5_139840925070384_lo[:, None]
    t5_139840925059760_lo = tl.reshape(t5_139840925059760_lo, [BLOCK_M, BLOCK_N])
    w5_139840925062448_lo = tl.reshape(w5_139840925062448_lo, [BLOCK_M])
    acc += t5_139840925059760_lo * w5_139840925062448_lo[:, None]
    t5_139840925061680_lo = tl.reshape(t5_139840925061680_lo, [BLOCK_M, BLOCK_N])
    w5_139840925069872_lo = tl.reshape(w5_139840925069872_lo, [BLOCK_M])
    acc += t5_139840925061680_lo * w5_139840925069872_lo[:, None]
    t5_139840925060656_lo = tl.reshape(t5_139840925060656_lo, [BLOCK_M, BLOCK_N])
    w5_139840925068976_lo = tl.reshape(w5_139840925068976_lo, [BLOCK_M])
    acc += t5_139840925060656_lo * w5_139840925068976_lo[:, None]
    t5_139840925062704_lo = tl.reshape(t5_139840925062704_lo, [BLOCK_M, BLOCK_N])
    w5_139840925070896_lo = tl.reshape(w5_139840925070896_lo, [BLOCK_M])
    acc += t5_139840925062704_lo * w5_139840925070896_lo[:, None]
    t5_139840925058288_hi = tl.reshape(t5_139840925058288_hi, [BLOCK_M, BLOCK_N])
    w5_139840925061168_hi = tl.reshape(w5_139840925061168_hi, [BLOCK_M])
    acc += t5_139840925058288_hi * w5_139840925061168_hi[:, None]
    t5_139840925060848_hi = tl.reshape(t5_139840925060848_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069040_hi = tl.reshape(w5_139840925069040_hi, [BLOCK_M])
    acc += t5_139840925060848_hi * w5_139840925069040_hi[:, None]
    t5_139840925059824_hi = tl.reshape(t5_139840925059824_hi, [BLOCK_M, BLOCK_N])
    w5_139840925057392_hi = tl.reshape(w5_139840925057392_hi, [BLOCK_M])
    acc += t5_139840925059824_hi * w5_139840925057392_hi[:, None]
    t5_139840925061872_hi = tl.reshape(t5_139840925061872_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070064_hi = tl.reshape(w5_139840925070064_hi, [BLOCK_M])
    acc += t5_139840925061872_hi * w5_139840925070064_hi[:, None]
    t5_139840925057520_hi = tl.reshape(t5_139840925057520_hi, [BLOCK_M, BLOCK_N])
    w5_139840925061872_hi = tl.reshape(w5_139840925061872_hi, [BLOCK_M])
    acc += t5_139840925057520_hi * w5_139840925061872_hi[:, None]
    t5_139840925061360_hi = tl.reshape(t5_139840925061360_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069552_hi = tl.reshape(w5_139840925069552_hi, [BLOCK_M])
    acc += t5_139840925061360_hi * w5_139840925069552_hi[:, None]
    t5_139840925060336_hi = tl.reshape(t5_139840925060336_hi, [BLOCK_M, BLOCK_N])
    w5_139840925057904_hi = tl.reshape(w5_139840925057904_hi, [BLOCK_M])
    acc += t5_139840925060336_hi * w5_139840925057904_hi[:, None]
    t5_139840925062384_hi = tl.reshape(t5_139840925062384_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070576_hi = tl.reshape(w5_139840925070576_hi, [BLOCK_M])
    acc += t5_139840925062384_hi * w5_139840925070576_hi[:, None]
    t5_139840925057904_hi = tl.reshape(t5_139840925057904_hi, [BLOCK_M, BLOCK_N])
    w5_139840925061424_hi = tl.reshape(w5_139840925061424_hi, [BLOCK_M])
    acc += t5_139840925057904_hi * w5_139840925061424_hi[:, None]
    t5_139840925061104_hi = tl.reshape(t5_139840925061104_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069296_hi = tl.reshape(w5_139840925069296_hi, [BLOCK_M])
    acc += t5_139840925061104_hi * w5_139840925069296_hi[:, None]
    t5_139840925060080_hi = tl.reshape(t5_139840925060080_hi, [BLOCK_M, BLOCK_N])
    w5_139840925057520_hi = tl.reshape(w5_139840925057520_hi, [BLOCK_M])
    acc += t5_139840925060080_hi * w5_139840925057520_hi[:, None]
    t5_139840925062128_hi = tl.reshape(t5_139840925062128_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070320_hi = tl.reshape(w5_139840925070320_hi, [BLOCK_M])
    acc += t5_139840925062128_hi * w5_139840925070320_hi[:, None]
    t5_139840925057392_hi = tl.reshape(t5_139840925057392_hi, [BLOCK_M, BLOCK_N])
    w5_139840925062640_hi = tl.reshape(w5_139840925062640_hi, [BLOCK_M])
    acc += t5_139840925057392_hi * w5_139840925062640_hi[:, None]
    t5_139840925061616_hi = tl.reshape(t5_139840925061616_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069808_hi = tl.reshape(w5_139840925069808_hi, [BLOCK_M])
    acc += t5_139840925061616_hi * w5_139840925069808_hi[:, None]
    t5_139840925060592_hi = tl.reshape(t5_139840925060592_hi, [BLOCK_M, BLOCK_N])
    w5_139840925058288_hi = tl.reshape(w5_139840925058288_hi, [BLOCK_M])
    acc += t5_139840925060592_hi * w5_139840925058288_hi[:, None]
    t5_139840925062640_hi = tl.reshape(t5_139840925062640_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070832_hi = tl.reshape(w5_139840925070832_hi, [BLOCK_M])
    acc += t5_139840925062640_hi * w5_139840925070832_hi[:, None]
    t5_139840925058224_hi = tl.reshape(t5_139840925058224_hi, [BLOCK_M, BLOCK_N])
    w5_139840925061360_hi = tl.reshape(w5_139840925061360_hi, [BLOCK_M])
    acc += t5_139840925058224_hi * w5_139840925061360_hi[:, None]
    t5_139840925060912_hi = tl.reshape(t5_139840925060912_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069104_hi = tl.reshape(w5_139840925069104_hi, [BLOCK_M])
    acc += t5_139840925060912_hi * w5_139840925069104_hi[:, None]
    t5_139840925059888_hi = tl.reshape(t5_139840925059888_hi, [BLOCK_M, BLOCK_N])
    w5_139840925057456_hi = tl.reshape(w5_139840925057456_hi, [BLOCK_M])
    acc += t5_139840925059888_hi * w5_139840925057456_hi[:, None]
    t5_139840925061936_hi = tl.reshape(t5_139840925061936_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070128_hi = tl.reshape(w5_139840925070128_hi, [BLOCK_M])
    acc += t5_139840925061936_hi * w5_139840925070128_hi[:, None]
    t5_139840925057456_hi = tl.reshape(t5_139840925057456_hi, [BLOCK_M, BLOCK_N])
    w5_139840925062704_hi = tl.reshape(w5_139840925062704_hi, [BLOCK_M])
    acc += t5_139840925057456_hi * w5_139840925062704_hi[:, None]
    t5_139840925061424_hi = tl.reshape(t5_139840925061424_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069616_hi = tl.reshape(w5_139840925069616_hi, [BLOCK_M])
    acc += t5_139840925061424_hi * w5_139840925069616_hi[:, None]
    t5_139840925060400_hi = tl.reshape(t5_139840925060400_hi, [BLOCK_M, BLOCK_N])
    w5_139840925058224_hi = tl.reshape(w5_139840925058224_hi, [BLOCK_M])
    acc += t5_139840925060400_hi * w5_139840925058224_hi[:, None]
    t5_139840925062448_hi = tl.reshape(t5_139840925062448_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070640_hi = tl.reshape(w5_139840925070640_hi, [BLOCK_M])
    acc += t5_139840925062448_hi * w5_139840925070640_hi[:, None]
    t5_139840925057840_hi = tl.reshape(t5_139840925057840_hi, [BLOCK_M, BLOCK_N])
    w5_139840925061680_hi = tl.reshape(w5_139840925061680_hi, [BLOCK_M])
    acc += t5_139840925057840_hi * w5_139840925061680_hi[:, None]
    t5_139840925061168_hi = tl.reshape(t5_139840925061168_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069360_hi = tl.reshape(w5_139840925069360_hi, [BLOCK_M])
    acc += t5_139840925061168_hi * w5_139840925069360_hi[:, None]
    t5_139840925060144_hi = tl.reshape(t5_139840925060144_hi, [BLOCK_M, BLOCK_N])
    w5_139840925057840_hi = tl.reshape(w5_139840925057840_hi, [BLOCK_M])
    acc += t5_139840925060144_hi * w5_139840925057840_hi[:, None]
    t5_139840925062192_hi = tl.reshape(t5_139840925062192_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070384_hi = tl.reshape(w5_139840925070384_hi, [BLOCK_M])
    acc += t5_139840925062192_hi * w5_139840925070384_hi[:, None]
    t5_139840925059760_hi = tl.reshape(t5_139840925059760_hi, [BLOCK_M, BLOCK_N])
    w5_139840925062448_hi = tl.reshape(w5_139840925062448_hi, [BLOCK_M])
    acc += t5_139840925059760_hi * w5_139840925062448_hi[:, None]
    t5_139840925061680_hi = tl.reshape(t5_139840925061680_hi, [BLOCK_M, BLOCK_N])
    w5_139840925069872_hi = tl.reshape(w5_139840925069872_hi, [BLOCK_M])
    acc += t5_139840925061680_hi * w5_139840925069872_hi[:, None]
    t5_139840925060656_hi = tl.reshape(t5_139840925060656_hi, [BLOCK_M, BLOCK_N])
    w5_139840925068976_hi = tl.reshape(w5_139840925068976_hi, [BLOCK_M])
    acc += t5_139840925060656_hi * w5_139840925068976_hi[:, None]
    t5_139840925062704_hi = tl.reshape(t5_139840925062704_hi, [BLOCK_M, BLOCK_N])
    w5_139840925070896_hi = tl.reshape(w5_139840925070896_hi, [BLOCK_M])
    acc += t5_139840925062704_hi * w5_139840925070896_hi[:, None]

    row_valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    store_mask = valid_m[:, None] & valid_n[None, :]
    acc = tl.where(row_valid & store_mask, acc, float("-inf"))
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc, mask=store_mask)

def indexer_score_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Optimized Triton indexer dense scoring (bf16 tensor cores).

    Same semantics as fp8_mqa_logits_triton (which see); invalid positions
    outside [cu_seqlen_ks, cu_seqlen_ke) are written as -inf, matching
    clean_logits=True so downstream top-k kernels can consume it directly.
    """
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    _out_dtype = (
        torch.bfloat16 if os.environ.get("VLLM_IDX_BF16", "0") == "1"
        else torch.float32
    )
    logits = _bucketed_logits_buffer(num_q, seq_len_kv, q.device,
                                     dtype=_out_dtype)
    if num_q == 0 or seq_len_kv == 0:
        return logits

    block_m = int(os.environ.get("VLLM_IDX_BM", "32"))  # G4: BM 变体
    block_n = int(os.environ.get("VLLM_IDX_BN", "64"))  # G4: BN 变体
    num_warps = int(os.environ.get("VLLM_IDX_WARPS", "4"))  # G4: warps 变体
    if os.environ.get("VLLM_IDX_GEMM_V2", "0") == "1":
        # G4 ① GEMM 重排 v2: 摊平 64 头大 GEMM (位级同, env 门控)
        block_m = int(os.environ.get("VLLM_IDX_BM_V2", "2"))
        grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, block_n))
        _indexer_score_logits_kernel_v2[grid](
            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke, logits,
            num_q, seq_len_kv, num_heads, head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp8.stride(0), k_fp8.stride(1),
            weights.stride(0), weights.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=num_warps)
        return logits
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, block_n))
    _indexer_score_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        OUT_BF16=_out_dtype == torch.bfloat16,
        num_warps=num_warps,
    )
    return logits
