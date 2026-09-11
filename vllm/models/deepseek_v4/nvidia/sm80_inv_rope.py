# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 逆 RoPE 融合内核（env 门控默认关，VLLM_SM80_FUSED_INV_ROPE=1）。

替换 _o_proj_bf16_sm80 中的 torch 逆 RoPE 链（~15 ops/调用 × 43 层/步
≈ 645 节点/步: gather/slice/reshape/mul×4/add×2/stack/cat）。语义与
参考链逐位对齐:
- cache 为 f32: 链中所有运算 f32 (bf16 x 提升为精确 f32), 输出 f32,
  最终 bf16 舍入发生在调用方 .to(bf16) — 内核同构
- cache 为 bf16: 链中每次 mul/add 均 bf16 舍入 — 内核逐 op 显式
  .to(bf16) 强制同序舍入 (防 FMA 融合)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _inv_rope_sm80_kernel(
    o_ptr,
    out_ptr,
    pos_ptr,
    cache_ptr,
    nope_dim,
    rope_dim,
    cache_stride,
    o_stride_t,      # token 维步长 = H*head_dim
    o_stride_h,      # H 维步长 = head_dim
    HALF_ROPE: tl.constexpr,
    IS_F32_CACHE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    base = t * o_stride_t + h * o_stride_h
    off_n = tl.arange(0, BLOCK_HEAD)

    # nope 段: [nope_dim] (非 2 幂, mask)
    n_mask = off_n < nope_dim
    nope = tl.load(o_ptr + base + off_n, mask=n_mask, other=0.0)

    # rope 段: 偶/奇分量
    off_p = tl.arange(0, HALF_ROPE)
    x = tl.load(o_ptr + base + nope_dim + off_p * 2)
    y = tl.load(o_ptr + base + nope_dim + off_p * 2 + 1)

    pos = tl.load(pos_ptr + t)
    cos = tl.load(cache_ptr + pos * cache_stride + off_p)
    sin = tl.load(cache_ptr + pos * cache_stride + HALF_ROPE + off_p)

    if IS_F32_CACHE:
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        cf = cos.to(tl.float32)
        sf = sin.to(tl.float32)
        x_inv = xf * cf + yf * sf
        y_inv = -xf * sf + yf * cf
        # 输出 f32: nope 精确提升
        tl.store(out_ptr + base + off_n, nope.to(tl.float32), mask=n_mask)
        tl.store(out_ptr + base + nope_dim + off_p * 2, x_inv)
        tl.store(out_ptr + base + nope_dim + off_p * 2 + 1, y_inv)
    else:
        # bf16 逐 op 舍入对齐 (显式 .to 防 FMA 融合)
        xb = x.to(tl.bfloat16)
        yb = y.to(tl.bfloat16)
        cb = cos.to(tl.bfloat16)
        sb = sin.to(tl.bfloat16)
        m1 = (xb * cb).to(tl.bfloat16)
        m2 = (yb * sb).to(tl.bfloat16)
        x_inv = (m1 + m2).to(tl.bfloat16)
        m3 = (-xb * sb).to(tl.bfloat16)
        m4 = (yb * cb).to(tl.bfloat16)
        y_inv = (m3 + m4).to(tl.bfloat16)
        tl.store(out_ptr + base + off_n, nope, mask=n_mask)
        tl.store(out_ptr + base + nope_dim + off_p * 2, x_inv)
        tl.store(out_ptr + base + nope_dim + off_p * 2 + 1, y_inv)


def inv_rope_sm80(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """逆 RoPE 融合: cat(nope, inv_rope(rope_part)) — 与 torch 链逐位对齐。

    o: [T, H, nope_dim+rope_dim] bf16; 输出 dtype = f32 (cache f32 时,
    与链一致) 或 bf16。
    """
    assert o.dtype == torch.bfloat16
    T, H, head_dim = o.shape
    half_rope = rope_dim // 2
    assert head_dim == nope_dim + rope_dim and rope_dim % 2 == 0

    is_f32 = cos_sin_cache.dtype == torch.float32
    out_dtype = torch.float32 if is_f32 else torch.bfloat16
    out = torch.empty(T, H, head_dim, dtype=out_dtype, device=o.device)

    _inv_rope_sm80_kernel[(T, H)](
        o,
        out,
        positions,
        cos_sin_cache,
        nope_dim,
        rope_dim,
        cos_sin_cache.stride(0),
        o.stride(-3),
        o.stride(-2),
        HALF_ROPE=half_rope,
        IS_F32_CACHE=is_f32,
        BLOCK_HEAD=triton.next_power_of_2(head_dim),
        num_warps=2,
    )
    return out
