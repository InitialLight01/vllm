"""Triton prefill 稀疏 MLA — FlashInfer TRTLLM cubin 的确定性替代。

接口与 flashinfer_trtllm_batch_decode_sparse_mla_dsv4 对齐:
- query [sum_q, heads, 512] bf16
- swa_kv_cache / compressed_kv_cache: packed uint8 (584B/token:
  448 fp8 + 128 bf16 + 8 UE8M0, 7 有效 + 1 pad)
- sparse_indices [sum_q, 128+topk] (前 128 列 = SWA 全局槽, 其余 = 压缩段槽, -1 无效)
- sparse_topk_lens [sum_q] (每 token 有效候选数, 含固定 128)
- seq_lens [batch] (SWA 有效窗口上界: 槽 < seq_len)
- sinks [heads] fp32
- 输出 [sum_q, heads, 512] bf16

确定性: 每 token 一个 program, 候选按固定序流式累加, 无原子。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_prefill_sparse_mla_kernel(
    q_ptr,
    swa_ptr,
    comp_ptr,
    idx_ptr,
    lens_ptr,
    seq_lens_ptr,
    req_idx_ptr,
    sinks_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_it,
    stride_ot,
    stride_oh,
    stride_od,
    bmm1_scale,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,  # 128 + padded_topk
    BLOCK_N: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,  # 584
    SWA_SEG: tl.constexpr,  # 128
    NOPE_DIM: tl.constexpr,  # 448
    ROPE_DIM: tl.constexpr,  # 64
):
    token_id = tl.program_id(0)
    offs_h = tl.arange(0, num_heads)
    offs_nope = tl.arange(0, NOPE_DIM)
    offs_rope = tl.arange(0, ROPE_DIM)

    # q: [H, 448] + [H, 64] 分段
    q_nope = tl.load(
        q_ptr + token_id * stride_qt + offs_h[:, None] * stride_qh
        + offs_nope[None, :] * stride_qd
    )
    q_rope = tl.load(
        q_ptr + token_id * stride_qt + offs_h[:, None] * stride_qh
        + (NOPE_DIM + offs_rope)[None, :] * stride_qd
    )

    topk_len = tl.load(lens_ptr + token_id)
    req_idx = tl.load(req_idx_ptr + token_id)
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # sink-seeded 在线 softmax
    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)
    e_max = sink
    e_sum = tl.full((num_heads,), 1.0, dtype=tl.float32)
    acc_nope = tl.zeros((num_heads, NOPE_DIM), dtype=tl.float32)
    acc_rope = tl.zeros((num_heads, ROPE_DIM), dtype=tl.float32)

    scale_offs = offs_nope // 64  # [448] 每 64 块一个 scale
    rope_lo = NOPE_DIM + offs_rope * 2
    rope_hi = rope_lo + 1

    neg_large = -1.0e30
    for c_start in range(0, num_candidates, BLOCK_N):
        offs_c = c_start + tl.arange(0, BLOCK_N)
        mask_c = offs_c < topk_len
        is_swa = offs_c < SWA_SEG
        slot = tl.load(
            idx_ptr + token_id * stride_it + offs_c, mask=mask_c, other=-1
        )
        valid = mask_c & tl.where(is_swa, slot < seq_len, slot >= 0)
        base = slot[None, :] * TOKEN_BYTES

        # --- nope 段: fp8 + UE8M0 scale ---
        u8_swa = tl.load(
            swa_ptr + base + offs_nope[:, None],
            mask=valid[None, :] & is_swa[None, :],
            other=0,
        )
        u8_comp = tl.load(
            comp_ptr + base + offs_nope[:, None],
            mask=valid[None, :] & ~is_swa[None, :],
            other=0,
        )
        sc_swa = tl.load(
            swa_ptr + base + (576 + scale_offs)[:, None],
            mask=valid[None, :] & is_swa[None, :],
            other=0,
        )
        sc_comp = tl.load(
            comp_ptr + base + (576 + scale_offs)[:, None],
            mask=valid[None, :] & ~is_swa[None, :],
            other=0,
        )
        k_scale = tl.exp2((sc_swa + sc_comp).to(tl.float32) - 127.0)
        k_nope = ((u8_swa + u8_comp).to(tl.float32) * k_scale).to(tl.bfloat16)

        # --- rope 段: bf16 (小端 2 字节) ---
        lo_swa = tl.load(
            swa_ptr + base + rope_lo[:, None],
            mask=valid[None, :] & is_swa[None, :],
            other=0,
        )
        hi_swa = tl.load(
            swa_ptr + base + rope_hi[:, None],
            mask=valid[None, :] & is_swa[None, :],
            other=0,
        )
        lo_comp = tl.load(
            comp_ptr + base + rope_lo[:, None],
            mask=valid[None, :] & ~is_swa[None, :],
            other=0,
        )
        hi_comp = tl.load(
            comp_ptr + base + rope_hi[:, None],
            mask=valid[None, :] & ~is_swa[None, :],
            other=0,
        )
        u16 = ((hi_swa + hi_comp).to(tl.uint16) << 8) | (
            lo_swa + lo_comp
        ).to(tl.uint16)
        k_rope = u16.to(tl.bfloat16, bitcast=True)

        qk_nope = tl.dot(q_nope, tl.trans(k_nope))  # [H, BLOCK_N]
        qk_rope = tl.dot(q_rope, tl.trans(k_rope))
        qk = (qk_nope + qk_rope).to(tl.float32) * bmm1_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc_nope *= re_scale[:, None]
        acc_rope *= re_scale[:, None]
        acc_nope += tl.dot(p.to(tl.bfloat16), k_nope)
        acc_rope += tl.dot(p.to(tl.bfloat16), k_rope)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    o_nope = (acc_nope / e_sum_safe[:, None]).to(tl.bfloat16)
    o_rope = (acc_rope / e_sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + token_id * stride_ot + offs_h[:, None] * stride_oh
        + offs_nope[None, :] * stride_od,
        o_nope,
    )
    tl.store(
        out_ptr + token_id * stride_ot + offs_h[:, None] * stride_oh
        + (NOPE_DIM + offs_rope)[None, :] * stride_od,
        o_rope,
    )


def triton_prefill_sparse_mla(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_idx: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Triton prefill 稀疏 MLA (确定性替代, 接口对齐 FlashInfer)."""
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    num_candidates = sparse_indices.shape[1]
    assert swa_kv_cache.dtype == torch.uint8, "SWA cache 须 packed uint8 视图"
    assert compressed_kv_cache.dtype == torch.uint8
    swa_flat = swa_kv_cache.view(-1)
    comp_flat = compressed_kv_cache.view(-1)
    BLOCK_N = 16
    _triton_prefill_sparse_mla_kernel[(num_tokens,)](
        query,
        swa_flat,
        comp_flat,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        req_idx,
        sinks,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        sparse_indices.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        bmm1_scale,
        num_heads=num_heads,
        num_candidates=num_candidates,
        BLOCK_N=BLOCK_N,
        TOKEN_BYTES=584,
        SWA_SEG=128,
        NOPE_DIM=448,
        ROPE_DIM=64,
        num_warps=4,
    )
