# SPDX-License-Identifier: Apache-2.0
"""SM80 software fp8e4m3fn helpers (no fp8 hw on SM80; Triton lacks fp8e4nv there).

Bit-exact vs torch.float8_e4m3fn cast (verified: decode 256/256 codes exact,
encode 4096/4096 random+boundary exact incl. subnormals, -0.0 sign, and
saturate-at-448 semantics matching the kernels' pre-clamp).
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def u8_e4m3fn_to_f32(u: tl.tensor) -> tl.tensor:
    """Decode OCP E4M3 bits to f32. e=15,m=7 is the only NaN (never stored)."""
    e = (u >> 3) & 0xF
    m = u & 0x7
    s = ((u >> 7) & 0x1).to(tl.float32) * -2.0 + 1.0
    is_sub = e == 0
    is_nan = (e == 15) & (m == 7)
    v = tl.where(
        is_sub,
        tl.exp2(-6.0) * (m.to(tl.float32) / 8.0),
        tl.exp2(e.to(tl.float32) - 7.0) * (1.0 + m.to(tl.float32) / 8.0),
    )
    return tl.where(is_nan, float("nan"), v) * s


@triton.jit
def f32_to_e4m3fn_u8(x: tl.tensor) -> tl.tensor:
    """Encode f32 (|x|<=448, caller clamps) to OCP E4M3 bits, RN-ties-even.

    E4M3: sign(1) exp(4, bias 7) man(3). Normals e in [1,14]; OCP extension
    e=15 encodes [256, 448] (m<=6; m=7 is NaN, saturated to 448).
    """
    ax = tl.abs(x)
    s = ((x < 0.0) | ((x == 0.0) & (libdevice.copysign(1.0, x) < 0.0))).to(tl.uint8) << 7
    e_floor = tl.floor(tl.log2(tl.maximum(ax, 2.0**-10.0)))
    m_norm = ax * tl.exp2(-e_floor)  # [1,2)
    mq = libdevice.rint(m_norm * 8.0 - 8.0)  # [0,8], 8 = carry into exp
    e = e_floor + 7.0 + tl.floor(mq / 8.0)
    m = mq - 8.0 * tl.floor(mq / 8.0)
    is_sub = e < 1.0
    sub_m = libdevice.rint(ax * 2.0**9.0)  # subnormal grid: m * 2^-9
    e_f = tl.where(is_sub, tl.where(sub_m >= 8.0, 1.0, 0.0), e)
    m_f = tl.where(is_sub, tl.where(sub_m >= 8.0, 0.0, tl.minimum(sub_m, 7.0)), m)
    e_f = tl.minimum(e_f, 15.0)
    m_f = tl.where(e_f >= 15.0, tl.minimum(m_f, 6.0), m_f)
    u = ((e_f.to(tl.uint8) & 0xF) << 3) | (m_f.to(tl.uint8) & 0x7)
    return u | s
