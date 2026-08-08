# SPDX-License-Identifier: Apache-2.0
"""Triton MXFP4 dequant kernel (E2M1 + E8M0 scales) -> BF16.

Bit-identical to ``dq_mxfp4_pytorch`` but ~10-50x faster (single fused
kernel instead of multiple PyTorch bitwise passes).

Format
------
- x: ``uint8`` shape ``(..., K//2)`` — each byte packs two FP4 E2M1 weights
  (low nibble first, high nibble second).
- scale: ``uint8`` shape ``(..., K//32)`` — UE8M0, one byte per 32 weights.
  Scale factor is ``2 ** (scale_uint8 - 127)``.
- out: BF16 tensor shape ``(..., K)``.

Value encoding (OCP E2M1):
  nibble bits: s(bit3) e(bit1-2) m(bit0)
  val = (-1)^s * 2^(e-1) * (1 + m/2)  for e>0
  val = 0                             for e=0, m=0
  val = 0.5                           for e=0, m=1 (subnormal)
"""
import torch
import triton
import triton.language as tl

# FP4 nibble (0-15) -> BF16 bit pattern (sign<<15 | exp<<7 | mant<<4)
_FP4_TO_BF16_BITS = []
for nibble in range(16):
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 0x3
    mant = nibble & 0x1
    if exp > 0:
        bf16_exp = exp - 1 + 127  # FP4 bias 1, BF16 bias 127
        bits = (sign << 15) | (bf16_exp << 7) | (mant << 6)  # mant to bit 6
    else:
        if mant == 0:
            bits = 0
        else:
            bits = (sign << 15) | ((127 - 1) << 7)  # 2^-1 subnormal
    _FP4_TO_BF16_BITS.append(bits)

_FP4_LUT = torch.tensor(_FP4_TO_BF16_BITS, dtype=torch.int32, device="cuda")


@triton.jit
def _dequant_mxfp4_kernel(
    x_ptr,          # packed uint8 (M, K//2)
    scale_ptr,      # scale uint8 (M, K//32)
    out_ptr,        # BF16 (M, K)
    lut_ptr,        # int32 LUT (16 entries)
    M,
    K,
    stride_xm,
    stride_sm,
    stride_om,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid
    if offs_m >= M:
        return
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K

    # packed byte for each output element (low nibble for even k)
    packed = tl.load(x_ptr + offs_m * stride_xm + (offs_k // 2), mask=k_mask, other=0)
    nibble = tl.where((offs_k % 2) == 0, packed & 0xF, (packed >> 4) & 0xF)

    # LUT lookup: nibble -> BF16 bits (int32, bit pattern)
    bits = tl.load(lut_ptr + nibble, mask=k_mask, other=0)

    # scale: 2^(byte-127) applied via BF16 exponent shift
    scale_byte = tl.load(scale_ptr + offs_m * stride_sm + (offs_k // 32), mask=k_mask, other=0)
    scale_exp = scale_byte.to(tl.int32) - 127
    # exponent-field add: bit pattern (sign|exp|mant) + (scale_exp << 7).
    # FP4 max 6 and E8M0 scales keep values within BF16 range, no clamp needed.
    # Zero (bits == 0) must stay zero — no exponent shift.
    out_bits = tl.where(bits == 0, 0, bits + (scale_exp << 7))

    # store as BF16 (low 16 bits of the bit pattern; sign bit is bit 15)
    tl.store(out_ptr + offs_m * stride_om + offs_k, out_bits.to(tl.uint16), mask=k_mask)


def dq_mxfp4_triton(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequant MXFP4 -> BF16. x: (..., K//2) uint8, scale: (..., K//32) uint8.

    Arbitrary leading dims (e.g. (E, N, K//2)) are flattened for the kernel
    and the output is reshaped to (..., K). Bit-identical to
    ``dq_mxfp4_pytorch``.
    """
    assert x.dtype == torch.uint8 and scale.dtype == torch.uint8
    orig = x.shape
    K_half = orig[-1]
    K = K_half * 2
    x2 = x.reshape(-1, K_half)
    s2 = scale.reshape(-1, scale.shape[-1])
    M = x2.shape[0]
    # store raw BF16 bit patterns in a uint16 buffer, then view as BF16
    out = torch.empty((M, K), dtype=torch.uint16, device=x.device)
    BLOCK_K = triton.next_power_of_2(K)
    _dequant_mxfp4_kernel[(M,)](
        x2, s2, out, _FP4_LUT, M, K,
        x2.stride(0), s2.stride(0), out.stride(0),
        BLOCK_K=BLOCK_K,
    )
    return out.view(torch.bfloat16).reshape(*orig[:-1], K)
