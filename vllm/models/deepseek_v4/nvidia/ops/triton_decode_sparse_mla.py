# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton decode 稀疏 MLA 替代 (附3 更新42/43) — 替换 cubin decode 注意力。

cubin decode kernel (flashinfer_trtllm_batch_decode_sparse_mla_dsv4) 为
闭源二进制且逐次不确定 (残余翻转噪声源 — 更新42 定论: marlin MoE 位级
确定, 噪声 = draft 层 decode 注意力传导)。本模块:

1. gather+decode kernel: 按 (token, candidate) 从分页池 (576B/token:
   448 fp8 e4m3 nope + 128B bf16 rope + 块尾 UE8M0 scale) 物化 kv 行
   [T, K, 512] bf16 + valid 掩码
2. split-kv + sink 注意力 (复用 v1 portable 内核
   splitkv_sparse_mla_attention_with_sink)
"""
import os

import torch
import triton
import triton.language as tl

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    choose_sparse_mla_splitkv_splits,
    splitkv_sparse_mla_attention_with_sink,
)

_DATA_BYTES = 576
_NOPE = 448
_ROPE_DIM = 64
_HEAD_DIM = _NOPE + _ROPE_DIM  # 512


@triton.jit
def _gather_decode_kv_kernel(
    kv_ptr,             # [T, K, 512] bf16
    valid_ptr,          # [T, K] int8
    idx_ptr,            # [T, K] int64 flat 槽
    idx_stride_t,
    lens_ptr,           # [T] int32
    cache_ptr,          # uint8 池
    cache_stride0,      # int (block stride bytes)
    cache_sbase,        # int (块内 scale 区起点 = cb*576)
    cb,                 # 块槽数
    kv_off,             # 本段在 K 维的起始
    DATA_BYTES: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    c_start = tl.program_id(1) * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < K_TOTAL
    slot = tl.load(idx_ptr + token * idx_stride_t + offs_c, mask=mask_c, other=-1)
    lens = tl.load(lens_ptr + token)
    valid = mask_c & (slot >= 0) & (offs_c < lens)
    # int64 地址运算 (block*stride 可达 3.6e9)
    block = (slot // cb).to(tl.int64)
    pos = (slot % cb).to(tl.int64)
    base = block[None, :] * cache_stride0 + pos[None, :] * DATA_BYTES

    # nope: 448 字节 fp8 e4m3 (u8 bitcast) × UE8M0 scale
    offs_n = tl.arange(0, 512)
    nope_mask = offs_n < NOPE
    u8 = tl.load(
        cache_ptr + base + offs_n[:, None],
        mask=valid[None, :] & nope_mask[:, None],
        other=0,
    )
    scale_offs = offs_n // 64  # [512] 每 64 元素一个 scale (块 7 = pad)
    sc = tl.load(
        cache_ptr + block[None, :] * cache_stride0 + cache_sbase
        + pos[None, :] * 8 + scale_offs[:, None],
        mask=valid[None, :] & nope_mask[:, None],
        other=0,
    )
    k_scale = tl.exp2(sc.to(tl.float32) - 127.0)
    k_nope = (
        u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale
    ).to(tl.bfloat16)

    # rope: 64 bf16 位于行内字节 [448, 576)
    offs_r = tl.arange(0, 64)
    rlo = tl.load(cache_ptr + base + NOPE + offs_r[:, None] * 2, mask=valid[None, :], other=0)
    rhi = tl.load(cache_ptr + base + NOPE + 1 + offs_r[:, None] * 2, mask=valid[None, :], other=0)
    u16 = (rhi.to(tl.uint16) << 8) | rlo.to(tl.uint16)
    k_rope = u16.to(tl.bfloat16, bitcast=True)

    # 输出 [K, 512]: nope [0:448), rope [448:512)
    out_base = kv_ptr + token * K_TOTAL * (_NOPE + _ROPE_DIM) + (offs_c[None, :] + kv_off) * (_NOPE + _ROPE_DIM)
    tl.store(
        out_base + offs_n[:, None],
        k_nope,
        mask=valid[None, :] & nope_mask[:, None],
    )
    tl.store(
        out_base + NOPE + offs_r[:, None],
        k_rope,
        mask=valid[None, :],
    )
    tl.store(valid_ptr + token * K_TOTAL + kv_off + offs_c, valid.to(tl.int8), mask=mask_c)


def triton_decode_sparse_mla_sm120(
    query: torch.Tensor,              # [T, H, 512] bf16
    swa_kv_cache: torch.Tensor,       # uint8 池视图
    swa_indices: torch.Tensor,        # [T, 1, 128] int64 flat 槽
    swa_lens: torch.Tensor,           # [T] int32
    compressed_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,   # [T, 1, 512]
    extra_lens: torch.Tensor | None,      # [T] int32
    sinks: torch.Tensor,              # [H] fp32
    out: torch.Tensor,                # [T, H_pad, 512] bf16
    bmm1_scale: float,
) -> None:
    T, num_heads, head_dim = query.shape
    assert head_dim == _HEAD_DIM
    device = query.device
    K_SWA = swa_indices.shape[-1]
    K_EXTRA = extra_indices.shape[-1] if extra_indices is not None else 0
    K_TOTAL = K_SWA + K_EXTRA

    kv = torch.empty((T, K_TOTAL, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    valid = torch.empty((T, K_TOTAL), dtype=torch.int8, device=device)
    BLOCK_C = 32

    def _launch_part(cache, idx, lens, kv_off, k_width):
        if cache is None or idx is None:
            return
        _idx = idx.contiguous().view(T, k_width)
        _gather_decode_kv_kernel[
            (T, triton.cdiv(k_width, BLOCK_C))
        ](
            kv, valid,
            _idx, _idx.stride(0),
            lens,
            cache, cache.stride(0), cache.shape[1] * _DATA_BYTES, cache.shape[1],
            kv_off,
            DATA_BYTES=_DATA_BYTES,
            NOPE=_NOPE,
            ROPE_DIM=_ROPE_DIM,
            K_TOTAL=K_TOTAL,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

    _launch_part(swa_kv_cache, swa_indices, swa_lens, 0, K_SWA)
    _launch_part(compressed_kv_cache, extra_indices, extra_lens, K_SWA, K_EXTRA)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_splits = choose_sparse_mla_splitkv_splits(T, num_heads, K_TOTAL, sm_count)
    mid = torch.empty(
        (T, num_heads, num_splits, _HEAD_DIM + 1), dtype=torch.float32, device=device
    )
    splitkv_sparse_mla_attention_with_sink(
        query,
        kv,
        valid,
        bmm1_scale,
        sinks,
        out,
        mid,
        num_splits,
        num_heads=num_heads,
    )
