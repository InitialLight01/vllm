"""Triton kernel: gather KV cache entries (FP8 NOPE + BF16 ROPE) to BF16."""
import torch
import triton
import triton.language as tl

NOPE_DIM = 448
ROPE_DIM = 64
COMB_DIM = 512
TOKEN_STRIDE = NOPE_DIM + ROPE_DIM * 2  # 576


@triton.jit
def _gather_kv_kernel(
    cache_ptr,         # [total_bytes] uint8 — flat byte view of KV cache
    indices_ptr,       # [D, max_len] int32 — which KV slots to gather
    lens_ptr,          # [D] int32 — how many slots per token
    kv_out_ptr,        # [D*max_len, COMB_DIM] bf16 — gathered KV output
    kv_out_stride_0,
    block_size,        # tokens per block (64)
    head_bytes,        # total bytes per token slot (584)
    scale_dim: tl.constexpr,   # 8
    token_stride: tl.constexpr,  # 576
    BLOCK_K: tl.constexpr,  # processing chunk for NOPE
):
    """Gather KV: 1 program per (decode_token, kv_slot_index)."""
    pid = tl.program_id(0)  # flat index: D * max_len
    max_len = indices_ptr.shape[1] if hasattr(indices_ptr, 'shape') else None
    # Compute (d, k) from flat pid
    # We need to know D and max_len, but these are runtime values.
    # Instead, each program processes ONE kv slot for ONE decode token.
    # The grid is launched with grid=(D * max_len,).
    # pid = d * max_len + k, so d = pid // max_len, k = pid % max_len
    # But we can't pass max_len as a tensor arg to the kernel easily.
    # Alternative: each program processes a fixed block of KV.

    # Simpler approach: launch with grid=(D, max_len)
    d = tl.program_id(0)  # decode token index
    k = tl.program_id(1)  # kv slot index within this token

    # Load the slot_id for this decode token and kv position
    slot_ptr = indices_ptr + d * max_len + k
    slot = tl.load(slot_ptr)
    # Load the lens for this token
    L = tl.load(lens_ptr + d)

    # Check if this kv slot is valid
    valid = (k < L) & (slot >= 0)

    # Compute byte offsets in cache
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    data_off = block_idx * block_size * head_bytes + pos_in_block * token_stride
    scale_base = block_idx * block_size * head_bytes + block_size * token_stride + pos_in_block * scale_dim

    # Gather NOPE: 7 blocks of 64 FP8 values → dequant to BF16
    nope_offs = tl.arange(0, BLOCK_K)  # [0..63]
    for g in range(7):
        fp8_off = data_off + g * 64 + nope_offs
        fp8_bytes = tl.load(cache_ptr + fp8_off, mask=valid, other=0)
        fp8_vals = fp8_bytes.to(tl.float8e4nv, bitcast=True)

        scale_byte = tl.load(cache_ptr + scale_base + g, mask=valid, other=127)
        scale_val = tl.exp2(scale_byte.to(tl.float32) - 127.0)
        nope_deq = fp8_vals.to(tl.bfloat16) * scale_val.to(tl.bfloat16)

        out_off = kv_out_ptr + d * kv_out_stride_0 + k * COMB_DIM + g * 64 + nope_offs
        tl.store(out_off.to(tl.pointer_type(tl.bfloat16)), nope_deq, mask=valid)

    # Gather ROPE: 64 BF16 values at offset 448 in the data region
    rope_offs = tl.arange(0, ROPE_DIM)
    rope_fp8_off = data_off + NOPE_DIM + rope_offs * 2  # byte offset for bf16
    rope_bf16_ptr = (cache_ptr + rope_fp8_off).to(tl.pointer_type(tl.bfloat16))
    rope_vals = tl.load(rope_bf16_ptr, mask=valid, other=0.0)
    out_rope_off = kv_out_ptr + d * kv_out_stride_0 + k * COMB_DIM + NOPE_DIM + rope_offs
    tl.store(out_rope_off.to(tl.pointer_type(tl.bfloat16)), rope_vals, mask=valid)


def gather_kv_triton(
    cache: torch.Tensor,        # [num_blocks, block_size, 1, head_bytes] uint8
    indices: torch.Tensor,      # [D, max_len] int32
    topk_length: torch.Tensor,  # [D] int32
) -> torch.Tensor:
    """Gather KV cache entries to BF16 using Triton.

    Returns: [D, max_len, COMB_DIM] bf16
    """
    D, max_len = indices.shape
    device = indices.device

    # Flat uint8 view of cache
    cache_u8 = cache.contiguous().view(torch.uint8).reshape(-1)
    bs = cache.shape[1]
    hb = cache.shape[3]

    # Output tensor: [D * max_len, COMB_DIM]
    out = torch.empty(D, max_len, COMB_DIM, dtype=torch.bfloat16, device=device)

    grid = (D, max_len)
    _gather_kv_kernel[grid](
        cache_u8, indices, topk_length, out, out.stride(0),
        bs, hb,
        scale_dim=8, token_stride=TOKEN_STRIDE,
        BLOCK_K=64,
    )
    torch.cuda.synchronize()
    return out
