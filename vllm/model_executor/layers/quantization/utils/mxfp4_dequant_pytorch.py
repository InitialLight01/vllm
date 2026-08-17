# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch MXFP4 dequant (no ``amd-quark`` dependency).

Drop-in replacement for ``quark.torch.kernel.mx.dq_mxfp4``.

Format
------
- weights ``x``:  ``uint8`` shape ``(..., K//2)`` — each byte packs
  two FP4 E2M1 weights (low nibble first, high nibble second).
- scales ``scale``: ``uint8`` shape ``(..., K//32)`` — UE8M0, one byte
  per 32 weights. Scale factor is ``2 ** (scale_uint8 - 127)``.
- output: ``float_dtype`` tensor of shape ``(..., K)``.

Matches ``tests/quantization/reference_mxfp4.py::dq_mxfp4_torch``.
"""

import torch

BFLOAT16_EXP_BIAS = 127
BFLOAT16_MANTISSA_BITS = 7
FLOAT16_EXP_BIAS = 15
FLOAT16_MANTISSA_BITS = 10
FLOAT4_EXP_BIAS = 1


def _ue8m0_to_float(scale: torch.Tensor, float_dtype: torch.dtype) -> torch.Tensor:
    """UE8M0 uint8 → float: ``2 ** (val - 127)``."""
    assert scale.dtype == torch.uint8
    scale_int = scale.to(torch.int16) - 127
    return (2.0 ** scale_int.float()).to(float_dtype)


def _unpack_fp4_to_float(val: torch.Tensor, float_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed-uint8 FP4 E2M1 → *float_dtype*.

    ``val``: ``(..., K//2)`` → returns ``(..., K)``.
    """
    assert val.dtype == torch.uint8

    if float_dtype == torch.float16:
        half_exp_bias = FLOAT16_EXP_BIAS
        half_mantissa_bits = FLOAT16_MANTISSA_BITS
    elif float_dtype == torch.bfloat16:
        half_exp_bias = BFLOAT16_EXP_BIAS
        half_mantissa_bits = BFLOAT16_MANTISSA_BITS
    else:
        raise ValueError(f"Unsupported float_dtype: {float_dtype}")

    K_packed = val.shape[-1]
    unpacked = torch.empty(*val.shape[:-1], K_packed * 2, dtype=torch.uint8, device=val.device)
    unpacked[..., ::2] = val & 0x0F          # low nibble → even positions
    unpacked[..., 1::2] = (val >> 4) & 0x0F  # high nibble → odd positions

    sign = unpacked >> 3              # 1 bit
    exp = (unpacked >> 1) & 0x3       # 2 bits
    mantissa = unpacked & 0x1         # 1 bit

    # Compute target exponent: unbiased_fp4 = fp4_exp - 1
    #   target_exp = fp4_exp - 1 + half_bias
    # int16 arithmetic: BF16 exp ≤ 255 << 7 = 0x7F80; FP16 exp ≤ 31 << 10 =
    # 0x7C00; both fit in int16 without overflow.  Bit patterns of int16 are
    # identical to uint16, so the final .view(float_dtype) is correct.
    new_exp = (exp.to(torch.int16) - FLOAT4_EXP_BIAS + half_exp_bias)

    # Case 0000 → real zero: zero out exponent
    new_exp = new_exp * torch.logical_or(exp > 0, mantissa > 0).to(torch.int16)

    # Subnormals (001 → 0.5): clear mantissa (value encoded purely by exponent)
    # NB: torch.logical_and promotes to int64 — cast back to int16 to keep
    # 16-bit packing (else .view(bf16) would quadruple the last dim).
    new_mantissa = (
        torch.logical_and(mantissa.to(torch.int16), exp > 0).to(torch.int16)
        << (half_mantissa_bits - 1)
    )

    sign_bits = sign.to(torch.int16) << 15
    exp_bits = new_exp << half_mantissa_bits

    bits = sign_bits | exp_bits | new_mantissa
    return bits.view(float_dtype)


def dq_mxfp4_pytorch(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize MXFP4 weights.

    Args:
        x: Packed FP4 ``uint8``, shape ``(..., K//2)``.
        scale: UE8M0 ``uint8``, shape ``(..., K//32)``.
        float_dtype: ``torch.bfloat16`` or ``torch.float16``.

    Returns:
        Dequantized tensor of shape ``(..., K)``.
    """
    assert x.dtype == torch.uint8
    assert scale.dtype == torch.uint8

    scale_f = _ue8m0_to_float(scale, float_dtype)          # (..., K//32)
    x_f = _unpack_fp4_to_float(x, float_dtype)              # (..., K)
    x_f = x_f.reshape(*x_f.shape[:-1], -1, 32)
    x_f = x_f * scale_f.unsqueeze(-1)
    x_f = x_f.reshape(*x_f.shape[:-2], -1)
    return x_f


def dq_mxfp4_tile(
    x_tile: torch.Tensor,
    scale_tile: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    """Tile-wise dequant (same as dq_mxfp4_pytorch with tile-size contract)."""
    return dq_mxfp4_pytorch(x_tile, scale_tile, float_dtype)
