"""Triton prefill 稀疏 MLA — FlashInfer TRTLLM cubin 的确定性替代 (SM120 路径)。

布局 (与生产者 kernel 一致, 2026-08-24 定论, 见附3 更新21):
- SWA / 压缩缓存: 每 token 数据 576B (448 fp8 e4m3 + 128 bf16 rope),
  UE8M0 scale 在块尾 [64*576, 64*576 + 64*8)。张量 [B, 64, 584] 的
  行距 584B 是分配粒度 (行尾 8B pad 从不写入), 池块间距 = stride(0)
  (共享池, 非连续)。
- 槽 = flat paged 坐标: block = slot // 64, pos = slot % 64。

接口 (与 SM120 类 _forward_prefill 的分支索引约定对齐):
- query [n, heads, 512] bf16 (已 pad)
- swa_indices [n, window] flat 槽 (-1 无效), swa_lens [n]
- extra_indices [n, topk] flat 压缩槽, extra_lens [n]
- seq_lens [num_reqs], req_idx [n]
- sinks [heads] fp32
- 输出 [n, heads, 512] bf16

确定性: 每 (token, 4 头组) 一个 program, 候选按固定序流式累加, 无原子。
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_prefill_sparse_mla_kernel(
    q_ptr,
    swa_ptr,
    comp_ptr,
    swa_idx_ptr,
    swa_len_ptr,
    extra_idx_ptr,
    extra_len_ptr,
    seq_lens_ptr,
    req_idx_ptr,
    sinks_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_sw0,
    stride_cp0,
    stride_swt,
    stride_ext,
    stride_ot,
    stride_oh,
    stride_od,
    bmm1_scale,
    swa_block,      # SWA 池块 token 数 (64)
    swa_sbase,      # SWA 块内 scale 区起点 (block*576)
    comp_block,     # 压缩池块 token 数 (64 C4A / 2 C128A)
    comp_sbase,     # 压缩块内 scale 区起点
    BLOCK_H: tl.constexpr,      # 每 program 头数 (4)
    NUM_HEADS: tl.constexpr,    # 总头数
    SWA_W: tl.constexpr,        # window (128)
    EXTRA_W: tl.constexpr,      # topk 宽度 (0 = 无压缩段)
    BLOCK_N: tl.constexpr,      # 候选分块 (16)
    DATA_BYTES: tl.constexpr,   # 每 token 数据字节 (576)
    NOPE_DIM: tl.constexpr,     # 448
    ROPE_DIM: tl.constexpr,     # 64
):
    pid = tl.program_id(0)
    n_groups = NUM_HEADS // BLOCK_H
    token_id = pid // n_groups
    h_base = (pid % n_groups) * BLOCK_H

    offs_h = h_base + tl.arange(0, BLOCK_H)
    # 448 非 2 的幂 → 512 宽 tile + mask (零填充对 dot 无贡献)
    offs_nope = tl.arange(0, 512)
    nope_mask = offs_nope < NOPE_DIM
    offs_rope = tl.arange(0, ROPE_DIM)

    # q: [BLOCK_H, 512(448 有效)] + [BLOCK_H, 64]
    q_nope = tl.load(
        q_ptr + token_id * stride_qt + offs_h[:, None] * stride_qh
        + offs_nope[None, :] * stride_qd,
        mask=nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_ptr + token_id * stride_qt + offs_h[:, None] * stride_qh
        + (NOPE_DIM + offs_rope)[None, :] * stride_qd
    )

    req_idx = tl.load(req_idx_ptr + token_id)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    swa_len = tl.load(swa_len_ptr + token_id)
    if EXTRA_W > 0:
        extra_len = tl.load(extra_len_ptr + token_id)
    else:
        extra_len = 0

    # sink-seeded 在线 softmax
    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)
    e_max = sink
    e_sum = tl.full((BLOCK_H,), 1.0, dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, 512), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)

    scale_offs = offs_nope // 64  # [512] 每 64 块一个 scale (块 7 = pad)
    rope_lo = 448 + offs_rope * 2  # rope bf16 字节位置 (数据行内 [448,576))
    rope_hi = rope_lo + 1
    neg_large = -1.0e30

    # ── SWA 段 ──
    for c_start in range(0, SWA_W, BLOCK_N):
        offs_c = c_start + tl.arange(0, BLOCK_N)
        mask_c = offs_c < SWA_W
        slot = tl.load(
            swa_idx_ptr + token_id * stride_swt + offs_c, mask=mask_c, other=-1
        )
        # swa_lens 已编码窗口/长度截断; 槽是物理坐标 (block×64+pos),
        # 与 seq_len 不同域, 不能比较 (会误杀全部高位槽)
        valid = mask_c & (slot >= 0) & (offs_c < swa_len)
        # int64 地址运算: block×stride 可达 3.6e9, int32 会溢出 (非法地址)
        block = (slot // swa_block).to(tl.int64)
        pos = (slot % swa_block).to(tl.int64)
        base = block[None, :] * stride_sw0 + pos[None, :] * DATA_BYTES

        u8_nope = tl.load(
            swa_ptr + base + offs_nope[:, None],
            mask=valid[None, :] & nope_mask[:, None],
            other=0,
        )
        sc = tl.load(
            swa_ptr + block[None, :] * stride_sw0 + swa_sbase
            + pos[None, :] * 8 + scale_offs[:, None],
            mask=valid[None, :] & nope_mask[:, None],
            other=0,
        )
        k_scale = tl.exp2(sc.to(tl.float32) - 127.0)
        k_nope = (
            u8_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale
        ).to(tl.bfloat16)

        lo = tl.load(
            swa_ptr + base + rope_lo[:, None], mask=valid[None, :], other=0
        )
        hi = tl.load(
            swa_ptr + base + rope_hi[:, None], mask=valid[None, :], other=0
        )
        u16 = (hi.to(tl.uint16) << 8) | lo.to(tl.uint16)
        k_rope = u16.to(tl.bfloat16, bitcast=True)

        qk = (
            tl.dot(q_nope, k_nope)
            + tl.dot(q_rope, k_rope)
        ).to(tl.float32) * bmm1_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc_nope *= re_scale[:, None]
        acc_rope *= re_scale[:, None]
        acc_nope += tl.dot(p.to(tl.bfloat16), tl.trans(k_nope))
        acc_rope += tl.dot(p.to(tl.bfloat16), tl.trans(k_rope))
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # ── 压缩段 (EXTRA_W=0 时循环为空) ──
    for c_start in range(0, EXTRA_W, BLOCK_N):
        offs_c = c_start + tl.arange(0, BLOCK_N)
        mask_c = offs_c < EXTRA_W
        slot = tl.load(
            extra_idx_ptr + token_id * stride_ext + offs_c, mask=mask_c, other=-1
        )
        valid = mask_c & (slot >= 0) & (offs_c < extra_len)
        block = (slot // comp_block).to(tl.int64)
        pos = (slot % comp_block).to(tl.int64)
        base = block[None, :] * stride_cp0 + pos[None, :] * DATA_BYTES

        u8_nope = tl.load(
            comp_ptr + base + offs_nope[:, None],
            mask=valid[None, :] & nope_mask[:, None],
            other=0,
        )
        sc = tl.load(
            comp_ptr + block[None, :] * stride_cp0 + comp_sbase
            + pos[None, :] * 8 + scale_offs[:, None],
            mask=valid[None, :] & nope_mask[:, None],
            other=0,
        )
        k_scale = tl.exp2(sc.to(tl.float32) - 127.0)
        k_nope = (
            u8_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale
        ).to(tl.bfloat16)

        lo = tl.load(
            comp_ptr + base + rope_lo[:, None], mask=valid[None, :], other=0
        )
        hi = tl.load(
            comp_ptr + base + rope_hi[:, None], mask=valid[None, :], other=0
        )
        u16 = (hi.to(tl.uint16) << 8) | lo.to(tl.uint16)
        k_rope = u16.to(tl.bfloat16, bitcast=True)

        qk = (
            tl.dot(q_nope, k_nope)
            + tl.dot(q_rope, k_rope)
        ).to(tl.float32) * bmm1_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc_nope *= re_scale[:, None]
        acc_rope *= re_scale[:, None]
        acc_nope += tl.dot(p.to(tl.bfloat16), tl.trans(k_nope))
        acc_rope += tl.dot(p.to(tl.bfloat16), tl.trans(k_rope))
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    o_nope = (acc_nope / e_sum_safe[:, None]).to(tl.bfloat16)
    o_rope = (acc_rope / e_sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + token_id * stride_ot + offs_h[:, None] * stride_oh
        + offs_nope[None, :] * stride_od,
        o_nope,
        mask=nope_mask[None, :],
    )
    tl.store(
        out_ptr + token_id * stride_ot + offs_h[:, None] * stride_oh
        + (NOPE_DIM + offs_rope)[None, :] * stride_od,
        o_rope,
    )


def _flat_pool_view(cache: torch.Tensor) -> torch.Tensor:
    """[B, block, 584] (池块间距 stride(0), 非连续) → [B, block*584]
    as_strided 视图。kernel 通过 block*stride(0) + off 寻址。

    block = 64 (C4A/SWA) 或 2 (C128A 压缩缓存)。
    """
    assert cache.dim() == 3, f"cache 须 3-D [B, block, 584], got {cache.shape}"
    assert cache.shape[2] >= 584, cache.shape
    return cache.as_strided(
        (cache.shape[0], cache.shape[1] * 584), (cache.stride(0), 1)
    )


def triton_prefill_sparse_mla_sm120(
    query: torch.Tensor,            # [n, heads, 512] bf16 (已 pad)
    swa_kv_cache: torch.Tensor,     # [B, 64, 584] uint8 池视图
    swa_indices: torch.Tensor,      # [n, window] int64 flat 槽
    swa_lens: torch.Tensor,         # [n] int
    compressed_kv_cache: torch.Tensor | None,  # [B', 64, 584] uint8
    extra_indices: torch.Tensor | None,        # [n, topk] int64 flat 槽
    extra_lens: torch.Tensor | None,           # [n] int
    seq_lens: torch.Tensor,         # [num_reqs] int
    req_idx: torch.Tensor,          # [n] int
    sinks: torch.Tensor,            # [heads] fp32
    out: torch.Tensor,              # [n, heads, 512] bf16
    bmm1_scale: float,
) -> None:
    n, num_heads, head_dim = query.shape
    assert head_dim == 512 and num_heads % 4 == 0

    # 逐调用覆写小张量捕获 (VLLM_TRITON_CAPLAST=<path>): 保存索引/lens/
    # seq/req 等寻址相关的小张量 + 缓存的 shape/stride — 崩溃后可用于
    # 独立复现寻址越界 (缓存内容与寻址无关)。
    if os.environ.get("VLLM_TRITON_CAPLAST"):
        try:
            torch.save(
                {
                    "n": int(n),
                    "H": int(num_heads),
                    "q_slice": query[:64].detach().cpu(),
                    "swa_shape": list(swa_kv_cache.shape),
                    "swa_stride": list(swa_kv_cache.stride()),
                    "comp_shape": None if compressed_kv_cache is None
                    else list(compressed_kv_cache.shape),
                    "comp_stride": None if compressed_kv_cache is None
                    else list(compressed_kv_cache.stride()),
                    "swa_indices": swa_indices.detach().cpu(),
                    "swa_lens": swa_lens.detach().cpu(),
                    "extra_indices": None if extra_indices is None
                    else extra_indices.detach().cpu(),
                    "extra_lens": None if extra_lens is None
                    else extra_lens.detach().cpu(),
                    "seq_lens": seq_lens.detach().cpu(),
                    "req_idx": req_idx.detach().cpu(),
                    "sinks": sinks.detach().cpu(),
                    "out_shape": list(out.shape),
                    "out_stride": list(out.stride()),
                    "scale": bmm1_scale,
                },
                os.environ["VLLM_TRITON_CAPLAST"],
            )
        except Exception:
            pass

    # 逐调用轻量统计 (VLLM_TRITON_CAPLOG=<path>, JSONL) — 定位崩溃调用
    if os.environ.get("VLLM_TRITON_CAPLOG"):
        try:
            import json as _jcl
            _swa_slots = swa_indices.view(-1)[swa_indices.view(-1) >= 0]
            _rec = {
                "n": int(n),
                "H": int(num_heads),
                "swa_shape": list(swa_indices.shape),
                "swa_slot_min": int(_swa_slots.min().item()) if _swa_slots.numel() else -1,
                "swa_slot_max": int(_swa_slots.max().item()) if _swa_slots.numel() else -1,
                "swa_len_max": int(swa_lens.max().item()),
                "comp": None if compressed_kv_cache is None
                else list(compressed_kv_cache.shape),
                "extra_shape": None if extra_indices is None
                else list(extra_indices.shape),
                "extra_len_max": None if extra_lens is None
                else int(extra_lens.max().item()),
                "req_max": int(req_idx.max().item()),
                "seq_len": int(seq_lens.max().item()),
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
            }
            with open(os.environ["VLLM_TRITON_CAPLOG"], "a") as _f:
                _f.write(_jcl.dumps(_rec) + "\n")
        except Exception:
            pass

    # 一次性输入捕获 (VLLM_TRITON_CAP=1, 首个 n>1000 的 prefill chunk,
    # 按层类型各存一份) — 独立复现 kernel 崩溃用, 默认关。
    if os.environ.get("VLLM_TRITON_CAP") == "1" and n > 1000 and bool(
        swa_lens.max() > 0
    ):
        _caps = getattr(triton_prefill_sparse_mla_sm120, "_caps", None)
        if _caps is None:
            _caps = {}
            triton_prefill_sparse_mla_sm120._caps = _caps
        _key = (
            "swa_only"
            if compressed_kv_cache is None
            else f"comp{compressed_kv_cache.shape[1]}"
        )
        if _key not in _caps:
            _caps[_key] = True
            try:
                torch.save(
                    {
                        "query": query,
                        "swa_kv_cache": swa_kv_cache,
                        "swa_indices": swa_indices,
                        "swa_lens": swa_lens,
                        "compressed_kv_cache": compressed_kv_cache,
                        "extra_indices": extra_indices,
                        "extra_lens": extra_lens,
                        "seq_lens": seq_lens,
                        "req_idx": req_idx,
                        "sinks": sinks,
                        "out": out,
                        "scale": bmm1_scale,
                    },
                    f"/root/autodl-tmp/tmp/triton_cap_{_key}.pt",
                )
                print(
                    f"[TRITON-CAP] saved {_key} n={n} H={num_heads} "
                    f"swa={swa_kv_cache.shape}/{swa_kv_cache.stride()} "
                    f"comp={compressed_kv_cache.shape if compressed_kv_cache is not None else None} "
                    f"swa_idx={swa_indices.shape} extra={extra_indices.shape if extra_indices is not None else None} "
                    f"seq_lens={seq_lens}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[TRITON-CAP] save failed: {_e}", flush=True)

    swa_flat = _flat_pool_view(swa_kv_cache)
    swa_block = swa_kv_cache.shape[1]
    if extra_indices is not None and compressed_kv_cache is not None:
        comp_flat = _flat_pool_view(compressed_kv_cache)
        extra_w = extra_indices.shape[1]
        comp_block = compressed_kv_cache.shape[1]
    else:
        comp_flat = swa_flat  # 占位 (循环为空, 不访问)
        extra_indices = swa_indices[:0]
        extra_lens = swa_lens[:0]
        extra_w = 0
        comp_block = swa_block

    BLOCK_H = 4
    grid = (n * (num_heads // BLOCK_H),)
    _triton_prefill_sparse_mla_kernel[grid](
        query,
        swa_flat,
        comp_flat,
        swa_indices,
        swa_lens,
        extra_indices,
        extra_lens,
        seq_lens,
        req_idx,
        sinks,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        swa_flat.stride(0),
        comp_flat.stride(0),
        swa_indices.stride(0),
        extra_indices.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        bmm1_scale,
        swa_block,
        swa_block * 576,
        comp_block,
        comp_block * 576,
        BLOCK_H=BLOCK_H,
        NUM_HEADS=num_heads,
        SWA_W=swa_indices.shape[-1],
        EXTRA_W=extra_w,
        BLOCK_N=16,
        DATA_BYTES=576,
        NOPE_DIM=448,
        ROPE_DIM=64,
        num_warps=4,
    )


def triton_prefill_sparse_mla(*args, **kwargs) -> None:
    """旧入口 (基础类组合索引约定) — 组合索引的 span 重映射适配未实现。

    保留占位以便旧调用点 (非 SM120 类的 _forward) 显式失败而非静默错误。
    """
    raise NotImplementedError(
        "triton_prefill_sparse_mla (组合索引约定) 未适配新布局; "
        "SM120 路径请用 triton_prefill_sparse_mla_sm120"
    )
