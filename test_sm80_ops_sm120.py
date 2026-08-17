#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""SM120 上的 DECODE_E4M3 合并验证 (切片 1)。

- e4m3 codec round-trip (合并的 _f32_to_e4m3_uint8/_e4m3_uint8_to_f32)
- block-FP8 GEMM 原生路径回归 (DECODE_E4M3=False, 与合并前一致)
- block-FP8 GEMM 强制 decode 路径 (uint8 bitcast + DECODE_E4M3=True 直呼 kernel)
"""
import torch
import triton

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _e4m3_uint8_to_f32,
    _f32_to_e4m3_uint8,
    _get_default_w8a8_block_fp8_config,
    _w8a8_triton_block_scaled_mm,
    w8a8_triton_block_scaled_mm,
)


def check_e4m3_codec() -> None:
    import triton.language as tl

    @triton.jit
    def _roundtrip_kernel(x_ptr, enc_ptr, dec_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        bits = _f32_to_e4m3_uint8(x)
        tl.store(enc_ptr + off, bits, mask=mask)
        tl.store(dec_ptr + off, _e4m3_uint8_to_f32(bits), mask=mask)

    dev = "cuda"
    x = torch.cat([
        torch.linspace(-460, 460, 4096, device=dev),
        torch.logspace(-9, 2.65, 2048, base=2, device=dev),
        -torch.logspace(-9, 2.65, 2048, base=2, device=dev),
    ]).contiguous()
    n = x.numel()
    enc = torch.empty(n, dtype=torch.uint8, device=dev)
    dec = torch.empty(n, dtype=torch.float32, device=dev)
    _roundtrip_kernel[(triton.cdiv(n, 256),)](x, enc, dec, n, BLOCK=256)

    ref_fp8 = x.to(torch.float8_e4m3fn)
    ref_bytes = ref_fp8.view(torch.uint8)
    ref_val = ref_fp8.to(torch.float32)
    byte_match = (enc == ref_bytes).float().mean().item()
    rel = ((dec - ref_val).abs() / (ref_val.abs() + 1e-3)).max().item()
    print(f"e4m3 codec     byte_match={byte_match*100:.1f}%  decode_max_rel={rel:.4f}")
    assert byte_match > 0.99 and rel < 0.02
    print("  [OK] e4m3 codec\n")


def _reference(M, N, K, block_n, block_k, A, B, As, Bs):
    a_deq = A.to(torch.float32) * As.repeat_interleave(block_k, dim=1)
    b_deq = B.to(torch.float32) * Bs.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )
    return (a_deq @ b_deq.t()).to(torch.bfloat16)


def check_native_path() -> None:
    torch.manual_seed(0)
    M, N, K = 128, 256, 256
    block_n = block_k = 128
    dev = "cuda"
    a_f = (torch.randn(M, K, device=dev) * 0.25).clamp(-3, 3)
    b_f = (torch.randn(N, K, device=dev) * 0.25).clamp(-3, 3)
    A, B = a_f.to(torch.float8_e4m3fn), b_f.to(torch.float8_e4m3fn)
    As = torch.rand(M, K // block_k, device=dev) * 0.5 + 0.5
    Bs = torch.rand(N // block_n, K // block_k, device=dev) * 0.5 + 0.5

    out = w8a8_triton_block_scaled_mm(A, B, As, Bs, [block_n, block_k],
                                      output_dtype=torch.bfloat16)
    ref = _reference(M, N, K, block_n, block_k, A, B, As, Bs)
    rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()
    print(f"native GEMM    max_rel={rel:.4f}")
    assert rel < 0.06
    print("  [OK] native path (regression)\n")


def check_decode_forced_path() -> None:
    torch.manual_seed(1)
    M, N, K = 128, 256, 256
    block_n = block_k = 128
    dev = "cuda"
    a_f = (torch.randn(M, K, device=dev) * 0.25).clamp(-3, 3)
    b_f = (torch.randn(N, K, device=dev) * 0.25).clamp(-3, 3)
    A, B = a_f.to(torch.float8_e4m3fn), b_f.to(torch.float8_e4m3fn)
    As = torch.rand(M, K // block_k, device=dev) * 0.5 + 0.5
    Bs = torch.rand(N // block_n, K // block_k, device=dev) * 0.5 + 0.5

    config = _get_default_w8a8_block_fp8_config(M, block_n, block_k)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    Au, Bu = A.view(torch.uint8).contiguous(), B.view(torch.uint8).contiguous()
    _w8a8_triton_block_scaled_mm[grid](
        Au, Bu, C, As, Bs, M, N, K, block_n, block_k,
        Au.stride(-2), Au.stride(-1), Bu.stride(1), Bu.stride(0),
        C.stride(-2), C.stride(-1), As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        DECODE_E4M3=True, **config,
    )
    torch.cuda.synchronize()

    ref = _reference(M, N, K, block_n, block_k, A, B, As, Bs)
    diff = (C.float() - ref.float()).abs()
    rel = (diff / (ref.float().abs() + 1e-3)).max().item()
    print(f"decode GEMM    max_abs={diff.max().item():.4f}  max_rel={rel:.4f}")
    assert rel < 0.06
    print("  [OK] DECODE_E4M3 forced path\n")


def main() -> None:
    check_e4m3_codec()
    check_native_path()
    check_decode_forced_path()
    print("ALL SLICE-1 OP CHECKS PASSED")


if __name__ == "__main__":
    main()
