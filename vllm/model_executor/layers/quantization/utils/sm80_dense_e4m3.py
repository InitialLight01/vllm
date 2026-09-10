# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 dense block-FP8 GEMM — Triton DECODE_E4M3 (env-gated, default off).

蓝本: lvllm 线 fp8_utils.py 的 w8a8_triton_block_scaled_mm DECODE_E4M3
路径（Ampere 无 FP8 tensor core: 权重 fp8_e4m3 以 uint8 bitcast 传入,
内核内解码为 bf16 后 tl.dot; block scale 在 dot 后乘）。

与本栈的适配差异:
- 激活 A 是 bf16（未量化）——不解码、无 a_s
- 权重 B = float8_e4m3fn 原始值, 真实权重 = B × scale（UE8M0 2 的幂,
  f32 由外部 upcast 并缓存, 见 fp8.py apply 的 VLLM_SM80_DENSE_E4M3 分支）
- 支持 bias（HAS_BIAS）
- 数值: 与 w_bf16 物化路径 (scale 先乘后 round bf16) 的差异 = scale
  在 dot 后乘（f32 精度, 不 round bf16）→ 亚 bf16-ulp 级, 精度闸验收
  (smoke30→50 题→600q), 非逐位
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_uint8_to_f32(u):
    """Decode e4m3fn byte (1-4-3, exp bias 7) -> f32 (Ampere 无 fp8e4nv)."""
    ui = u.to(tl.int32)
    sign = (ui >> 7) & 1
    exp = (ui >> 3) & 0xF
    man = ui & 0x7
    mant = man.to(tl.float32) * 0.125
    val = tl.where(
        exp != 0,
        tl.exp2((exp - 7).to(tl.float32)) * (1.0 + mant),
        0.015625 * mant,
    )
    return tl.where(sign != 0, -val, val)


@triton.jit
def _sm80_dense_e4m3_kernel(
    A,
    B,
    C,
    Bs,
    bias,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Bs_k,
    stride_Bs_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        # A: bf16 原生; B: uint8 位型 e4m3 -> 内核内解码 bf16
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b_u8 = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0
        ).to(tl.uint8)
        b = _e4m3_uint8_to_f32(b_u8).to(tl.bfloat16)
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias_vals = tl.load(bias + offs_bn)
        accumulator += bias_vals[None, :]

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float32:
        c = accumulator
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sm80_dense_e4m3_mm(
    A: torch.Tensor,
    B_fp8: torch.Tensor,
    Bs: torch.Tensor,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """A(bf16)[..., K] @ (B_fp8(e4m3)[N, K] * Bs_block)[N/blk_n, K/blk_k]^T.

    Bs 为 f32 块乘法系数（= 2^scale，外部已 upcast 并缓存）。
    """
    assert A.dtype == torch.bfloat16
    assert B_fp8.dtype == torch.float8_e4m3fn
    assert B_fp8.is_contiguous()
    M = A.numel() // A.shape[-1]
    N, K = B_fp8.shape
    B = B_fp8.view(torch.uint8)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    # M=1 decode 形状专用默认配置（K=4096, N≤16384, block 128）
    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_M = 1
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _sm80_dense_e4m3_kernel[grid](
        A,
        B,
        C,
        Bs,
        bias,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_M,
        num_warps=4,
        num_stages=2,
    )
    return C
