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
    dense_ptr,      # [num_keys, 512] bf16 稠密工作区 (DENSE=1 时)
    stride_dd,
    BLOCK_H: tl.constexpr,      # 每 program 头数 (4)
    NUM_HEADS: tl.constexpr,    # 总头数
    SWA_W: tl.constexpr,        # window (128)
    EXTRA_W: tl.constexpr,      # topk 宽度 (0 = 无压缩段)
    SWA_BN: tl.constexpr,       # SWA 段分块 (u8 staging SMEM 敏感, 上限 32)
    BLOCK_N: tl.constexpr,      # EXTRA 段候选分块 (16; 稠密路径可 64)
    DATA_BYTES: tl.constexpr,   # 每 token 数据字节 (576)
    NOPE_DIM: tl.constexpr,     # 448
    ROPE_DIM: tl.constexpr,     # 64
    DENSE: tl.constexpr,        # 1 = 压缩段走稠密工作区
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
    # #50: scale 读放大修复 — [8,BLOCK_N] 加载后广播, 取代 [512,BLOCK_N]
    # (8KB 读 112B 有效 ≈ 70× 放大)。逐位无损 (同值广播)。
    offs8 = tl.arange(0, 8)
    rope_lo = 448 + offs_rope * 2  # rope bf16 字节位置 (数据行内 [448,576))
    rope_hi = rope_lo + 1
    neg_large = -1.0e30

    # ── SWA 段 ──
    for c_start in range(0, SWA_W, SWA_BN):
        offs_c = c_start + tl.arange(0, SWA_BN)
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
        sc8 = tl.load(
            swa_ptr + block[None, :] * stride_sw0 + swa_sbase
            + pos[None, :] * 8 + offs8[:, None],
            mask=valid[None, :] & (offs8[:, None] < 7),
            other=0,
        )
        k_scale = tl.exp2(sc8.to(tl.float32) - 127.0)  # [8, BLOCK_N]
        k_scale_full = tl.reshape(
            tl.broadcast_to(k_scale[:, None, :], (8, 64, SWA_BN)),
            (512, BLOCK_N),
        )
        k_nope = (
            u8_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale_full
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
        if DENSE:
            # #50: 稠密工作区读取 — k 已整池反量化 (逐位同值),
            # 免 scale 加载/反量化/rope 字节拼装。
            k_nope = tl.load(
                dense_ptr + slot[None, :] * stride_dd + offs_nope[:, None],
                mask=valid[None, :] & nope_mask[:, None],
                other=0.0,
            )
            k_rope = tl.load(
                dense_ptr + slot[None, :] * stride_dd
                + (NOPE_DIM + offs_rope)[:, None],
                mask=valid[None, :],
                other=0.0,
            )
        else:
            block = (slot // comp_block).to(tl.int64)
            pos = (slot % comp_block).to(tl.int64)
            base = block[None, :] * stride_cp0 + pos[None, :] * DATA_BYTES

            u8_nope = tl.load(
                comp_ptr + base + offs_nope[:, None],
                mask=valid[None, :] & nope_mask[:, None],
                other=0,
            )
            sc8 = tl.load(
                comp_ptr + block[None, :] * stride_cp0 + comp_sbase
                + pos[None, :] * 8 + offs8[:, None],
                mask=valid[None, :] & (offs8[:, None] < 7),
                other=0,
            )
            k_scale = tl.exp2(sc8.to(tl.float32) - 127.0)  # [8, SWA_BN]
            k_scale_full = tl.reshape(
                tl.broadcast_to(k_scale[:, None, :], (8, 64, BLOCK_N)),
                (512, SWA_BN),
            )
            k_nope = (
                u8_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale_full
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


@triton.jit
def _splitkv_partial_kernel(
    q_ptr, swa_ptr, dense_ptr,
    swa_idx_ptr, swa_len_ptr, extra_idx_ptr, extra_len_ptr,
    seq_lens_ptr, req_idx_ptr,
    mid_o_ptr,       # [n, heads, SPLITS, 576] bf16 (o_partial)
    mid_lse_ptr,     # [n, heads, SPLITS] fp32 (ln-lse)
    stride_qt, stride_qh, stride_qd,
    stride_sw0, stride_swt, stride_ext,
    stride_mot, stride_moh, stride_mos, stride_mod,
    stride_mlt, stride_mlh, stride_mls,
    bmm1_scale,
    swa_block, swa_sbase,
    BLOCK_H: tl.constexpr, NUM_HEADS: tl.constexpr,
    SWA_W: tl.constexpr, EXTRA_W: tl.constexpr,
    BLOCK_N: tl.constexpr, DATA_BYTES: tl.constexpr,
    NOPE_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    SPLITS: tl.constexpr, SPLIT_SIZE: tl.constexpr,
):
    """#50 split-KV partial: 每 (token, head 组, split) 独立计算分段注意力
    (split0 = SWA + EXTRA 前半; split1 = EXTRA 后半), 存归一化 o + ln-lse。
    与参考数学精确等价 (merge 端权重 exp(lse - M)), 与串行版非位级同。"""
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    n_groups = NUM_HEADS // BLOCK_H
    token_id = pid // n_groups
    h_base = (pid % n_groups) * BLOCK_H

    offs_h = h_base + tl.arange(0, BLOCK_H)
    offs_nope = tl.arange(0, 512)
    nope_mask = offs_nope < NOPE_DIM
    offs_rope = tl.arange(0, ROPE_DIM)

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

    neg_large = -1.0e30
    e_max = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, 512), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)

    offs8 = tl.arange(0, 8)
    rope_lo = 448 + offs_rope * 2
    rope_hi = rope_lo + 1

    if split_id == 0:
        # ── SWA 段 (串行, 与主内核同构) ──
        for c_start in range(0, SWA_W, BLOCK_N):
            offs_c = c_start + tl.arange(0, BLOCK_N)
            mask_c = offs_c < SWA_W
            slot = tl.load(
                swa_idx_ptr + token_id * stride_swt + offs_c, mask=mask_c, other=-1
            )
            valid = mask_c & (slot >= 0) & (offs_c < swa_len)
            block = (slot // swa_block).to(tl.int64)
            pos = (slot % swa_block).to(tl.int64)
            base = block[None, :] * stride_sw0 + pos[None, :] * DATA_BYTES

            u8_nope = tl.load(
                swa_ptr + base + offs_nope[:, None],
                mask=valid[None, :] & nope_mask[:, None],
                other=0,
            )
            sc8 = tl.load(
                swa_ptr + block[None, :] * stride_sw0 + swa_sbase
                + pos[None, :] * 8 + offs8[:, None],
                mask=valid[None, :] & (offs8[:, None] < 7),
                other=0,
            )
            k_scale = tl.exp2(sc8.to(tl.float32) - 127.0)
            k_scale_full = tl.reshape(
                tl.broadcast_to(k_scale[:, None, :], (8, 64, BLOCK_N)),
                (512, BLOCK_N),
            )
            k_nope = (
                u8_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale_full
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
                tl.dot(q_nope, k_nope) + tl.dot(q_rope, k_rope)
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

    # ── EXTRA 段 (每 split 各半; 稠密工作区读取) ──
    if EXTRA_W > 0:
        c_lo = split_id * SPLIT_SIZE
        c_hi = tl.minimum((split_id + 1) * SPLIT_SIZE, EXTRA_W)
        for c_start in range(c_lo, c_hi, BLOCK_N):
            offs_c = c_start + tl.arange(0, BLOCK_N)
            mask_c = offs_c < c_hi
            slot = tl.load(
                extra_idx_ptr + token_id * stride_ext + offs_c, mask=mask_c, other=-1
            )
            valid = mask_c & (slot >= 0) & (offs_c < extra_len)
            k_nope = tl.load(
                dense_ptr + slot[None, :] * 512 + offs_nope[:, None],
                mask=valid[None, :] & nope_mask[:, None],
                other=0.0,
            )
            k_rope = tl.load(
                dense_ptr + slot[None, :] * 512 + (NOPE_DIM + offs_rope)[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            qk = (
                tl.dot(q_nope, k_nope) + tl.dot(q_rope, k_rope)
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
    lse = tl.where(e_sum > 0, e_max + tl.log(e_sum), neg_large)
    o_nope = (acc_nope / e_sum_safe[:, None]).to(tl.bfloat16)
    o_rope = (acc_rope / e_sum_safe[:, None]).to(tl.bfloat16)

    tl.store(
        mid_o_ptr + token_id * stride_mot + offs_h[:, None] * stride_moh
        + split_id * stride_mos + offs_nope[None, :] * stride_mod,
        o_nope,
        mask=nope_mask[None, :],
    )
    tl.store(
        mid_o_ptr + token_id * stride_mot + offs_h[:, None] * stride_moh
        + split_id * stride_mos + (NOPE_DIM + offs_rope)[None, :] * stride_mod,
        o_rope,
    )
    tl.store(
        mid_lse_ptr + token_id * stride_mlt + offs_h * stride_mlh
        + split_id * stride_mls,
        lse,
    )


@triton.jit
def _splitkv_merge_kernel(
    mid_o_ptr, mid_lse_ptr, sinks_ptr, out_ptr,
    stride_mot, stride_moh, stride_mos, stride_mod,
    stride_mlt, stride_mlh, stride_mls,
    stride_ot, stride_oh, stride_od,
    BLOCK_H: tl.constexpr, NUM_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """#50 split-KV merge: 分段 o/lse 加权合并 + sink (torch 参考同构)。"""
    pid = tl.program_id(0)
    n_groups = NUM_HEADS // BLOCK_H
    token_id = pid // n_groups
    h_base = (pid % n_groups) * BLOCK_H

    offs_h = h_base + tl.arange(0, BLOCK_H)
    offs_nope = tl.arange(0, 512)
    nope_mask = offs_nope < NOPE_DIM
    offs_rope = tl.arange(0, ROPE_DIM)

    m = tl.full((BLOCK_H,), -1.0e30, dtype=tl.float32)
    o_nope = tl.zeros((BLOCK_H, 512), dtype=tl.float32)
    o_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)

    for s in tl.static_range(SPLITS):
        lse = tl.load(
            mid_lse_ptr + token_id * stride_mlt + offs_h * stride_mlh
            + s * stride_mls
        )
        n_m = tl.maximum(m, lse)
        w = tl.exp(lse - n_m)
        res = tl.exp(m - n_m)
        o_nope = o_nope * res[:, None]
        o_rope = o_rope * res[:, None]
        o_nope += w[:, None] * tl.load(
            mid_o_ptr + token_id * stride_mot + offs_h[:, None] * stride_moh
            + s * stride_mos + offs_nope[None, :] * stride_mod,
            mask=nope_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        o_rope += w[:, None] * tl.load(
            mid_o_ptr + token_id * stride_mot + offs_h[:, None] * stride_moh
            + s * stride_mos + (NOPE_DIM + offs_rope)[None, :] * stride_mod
        ).to(tl.float32)
        m = n_m

    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)
    n_m = tl.maximum(m, sink)
    res = tl.exp(m - n_m)
    o_nope = o_nope * res[:, None]
    o_rope = o_rope * res[:, None]
    wsum = tl.exp(sink - n_m)
    # 分段权重之和 = Σ exp(lse_s - n_m) (lse 已含段内 e_sum)
    for s in tl.static_range(SPLITS):
        lse = tl.load(
            mid_lse_ptr + token_id * stride_mlt + offs_h * stride_mlh
            + s * stride_mls
        )
        wsum += tl.exp(lse - n_m)
    o_nope = (o_nope / wsum[:, None]).to(tl.bfloat16)
    o_rope = (o_rope / wsum[:, None]).to(tl.bfloat16)

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


@triton.jit
def _dequant_compressed_dense_kernel(
    comp_ptr,       # [B, block, 584] uint8 池
    dense_ptr,      # [num_keys, 512] bf16
    stride_c0,
    stride_d0,
    num_keys,
    BLOCK_KEYS: tl.constexpr,
    NOPE_DIM: tl.constexpr,      # 448
    ROPE_DIM: tl.constexpr,      # 64
    comp_block: tl.constexpr,    # 2
    comp_sbase,                  # block*576
):
    """#50: 整池反量化 → 稠密 bf16 [num_keys, 512] (逐位同于注意力内核内联反量化)。

    key-id = 槽 = block*comp_block + pos; 每 key 576B (448 fp8 + 64 bf16 rope
    + 64 pad), UE8M0 scale 在块尾。"""
    pid = tl.program_id(0)
    k_start = pid * BLOCK_KEYS
    offs_k = k_start + tl.arange(0, BLOCK_KEYS)
    mask_k = offs_k < num_keys
    block = (offs_k // comp_block).to(tl.int64)
    pos = (offs_k % comp_block).to(tl.int64)
    base = block[None, :] * stride_c0 + pos[None, :] * 576

    offs_n = tl.arange(0, 512)
    nope_mask = offs_n < NOPE_DIM
    u8 = tl.load(
        comp_ptr + base + offs_n[:, None],
        mask=mask_k[None, :] & nope_mask[:, None],
        other=0,
    )
    offs8 = tl.arange(0, 8)
    sc8 = tl.load(
        comp_ptr + block[None, :] * stride_c0 + comp_sbase
        + pos[None, :] * 8 + offs8[:, None],
        mask=mask_k[None, :] & (offs8[:, None] < 7),
        other=0,
    )
    k_scale = tl.exp2(sc8.to(tl.float32) - 127.0)
    k_scale_full = tl.reshape(
        tl.broadcast_to(k_scale[:, None, :], (8, 64, BLOCK_KEYS)),
        (512, BLOCK_KEYS),
    )
    k_nope = (
        u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale_full
    ).to(tl.bfloat16)

    offs_r = tl.arange(0, ROPE_DIM)
    lo = tl.load(
        comp_ptr + base + 448 + offs_r[:, None] * 2,
        mask=mask_k[None, :],
        other=0,
    )
    hi = tl.load(
        comp_ptr + base + 448 + 1 + offs_r[:, None] * 2,
        mask=mask_k[None, :],
        other=0,
    )
    u16 = (hi.to(tl.uint16) << 8) | lo.to(tl.uint16)
    k_rope = u16.to(tl.bfloat16, bitcast=True)

    offs_k2 = offs_k[:, None]
    tl.store(
        dense_ptr + offs_k2 * stride_d0 + offs_n[None, :],
        tl.trans(k_nope),
        mask=mask_k[:, None] & nope_mask[None, :],
    )
    tl.store(
        dense_ptr + offs_k2 * stride_d0 + (448 + offs_r)[None, :],
        tl.trans(k_rope),
        mask=mask_k[:, None],
    )


def _dequant_compressed_dense(
    compressed_kv_cache: torch.Tensor, num_keys: int
) -> torch.Tensor:
    """整池反量化 → [num_keys, 512] bf16 (调用方管理生命周期)。"""
    dense = torch.empty(
        (num_keys, 512), dtype=torch.bfloat16, device=compressed_kv_cache.device
    )
    comp_flat = _flat_pool_view(compressed_kv_cache)
    _cb = compressed_kv_cache.shape[1]
    BLOCK_KEYS = 64
    _dequant_compressed_dense_kernel[(triton.cdiv(num_keys, BLOCK_KEYS),)](
        comp_flat,
        dense,
        comp_flat.stride(0),
        dense.stride(0),
        num_keys,
        BLOCK_KEYS=BLOCK_KEYS,
        NOPE_DIM=448,
        ROPE_DIM=64,
        comp_block=_cb,
        comp_sbase=_cb * 576,
        num_warps=4,
    )
    return dense


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

    # #50: BLOCK_H 旋钮 (env, 默认 4) — 每 token 的 key 加载冗余 = 64/BLOCK_H 倍。
    # 纯 tiling 变更 (每 head 数学不变) = 逐位无损。
    BLOCK_H = int(os.environ.get("VLLM_PREFILL_BLOCK_H", "4"))
    assert num_heads % BLOCK_H == 0
    grid = (n * (num_heads // BLOCK_H),)
    _bn = int(os.environ.get("VLLM_PREFILL_BLOCK_N", "16"))
    _swa_bn = int(os.environ.get("VLLM_PREFILL_SWA_BN", "16"))
    _nw = int(os.environ.get("VLLM_PREFILL_WARPS", "4"))
    _ns = int(os.environ.get("VLLM_PREFILL_STAGES", "0"))  # 0 = Triton 默认
    # #50: 稠密工作区 (VLLM_PREFILL_DENSE=1, 默认关) — 整池反量化一次,
    # 注意力压缩段读稠密 bf16 (逐位同值, 免 scale/反量化/rope 拼装)。
    _dense_on = int(os.environ.get("VLLM_PREFILL_DENSE", "0")) == 1
    # #50: split-KV (VLLM_PREFILL_SPLITKV=1, 默认关) — EXTRA 段拆 2 路并行
    # partial + merge (断串行软最大值链; 与参考数学精确等价, 非位级同)。
    _splitkv_on = int(os.environ.get("VLLM_PREFILL_SPLITKV", "0")) == 1
    _dense = None
    if (_dense_on or _splitkv_on) and extra_w > 0 and compressed_kv_cache is not None:
        _dense = _dequant_compressed_dense(
            compressed_kv_cache,
            compressed_kv_cache.shape[0] * compressed_kv_cache.shape[1],
        )
    if _splitkv_on and _dense is not None:
        _splits = 2
        _split_size = extra_w // _splits
        _mid_o = torch.empty(
            (n, num_heads, _splits, 576),
            dtype=torch.bfloat16,
            device=query.device,
        )
        _mid_lse = torch.empty(
            (n, num_heads, _splits), dtype=torch.float32, device=query.device
        )
        _hg = num_heads // BLOCK_H
        _splitkv_partial_kernel[(n * _hg, _splits)](
            query,
            swa_flat,
            _dense,
            swa_indices,
            swa_lens,
            extra_indices,
            extra_lens,
            seq_lens,
            req_idx,
            _mid_o,
            _mid_lse,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            swa_flat.stride(0),
            swa_indices.stride(0),
            extra_indices.stride(0),
            _mid_o.stride(0),
            _mid_o.stride(1),
            _mid_o.stride(2),
            _mid_o.stride(3),
            _mid_lse.stride(0),
            _mid_lse.stride(1),
            _mid_lse.stride(2),
            bmm1_scale,
            swa_block,
            swa_block * 576,
            BLOCK_H=BLOCK_H,
            NUM_HEADS=num_heads,
            SWA_W=swa_indices.shape[-1],
            EXTRA_W=extra_w,
            BLOCK_N=_bn,
            DATA_BYTES=576,
            NOPE_DIM=448,
            ROPE_DIM=64,
            SPLITS=_splits,
            SPLIT_SIZE=_split_size,
            num_warps=_nw,
            num_stages=_ns if _ns > 0 else 3,
        )
        _splitkv_merge_kernel[(n * _hg,)](
            _mid_o,
            _mid_lse,
            sinks,
            out,
            _mid_o.stride(0),
            _mid_o.stride(1),
            _mid_o.stride(2),
            _mid_o.stride(3),
            _mid_lse.stride(0),
            _mid_lse.stride(1),
            _mid_lse.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_H=BLOCK_H,
            NUM_HEADS=num_heads,
            NOPE_DIM=448,
            ROPE_DIM=64,
            SPLITS=_splits,
            num_warps=4,
        )
        return
    if _dense is None:
        _dense = torch.empty((1, 1), dtype=torch.bfloat16, device=query.device)
        _dense_s = 1
        _dense_flag = 0
    else:
        _dense_s = _dense.stride(0)
        _dense_flag = 1
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
        _dense,
        _dense_s,
        BLOCK_H=BLOCK_H,
        NUM_HEADS=num_heads,
        SWA_W=swa_indices.shape[-1],
        EXTRA_W=extra_w,
        SWA_BN=_swa_bn,
        BLOCK_N=_bn,
        DATA_BYTES=576,
        NOPE_DIM=448,
        ROPE_DIM=64,
        DENSE=_dense_flag,
        num_warps=_nw,
        num_stages=_ns if _ns > 0 else 3,
    )


def triton_prefill_sparse_mla(*args, **kwargs) -> None:
    """旧入口 (基础类组合索引约定) — 组合索引的 span 重映射适配未实现。

    保留占位以便旧调用点 (非 SM120 类的 _forward) 显式失败而非静默错误。
    """
    raise NotImplementedError(
        "triton_prefill_sparse_mla (组合索引约定) 未适配新布局; "
        "SM120 路径请用 triton_prefill_sparse_mla_sm120"
    )
