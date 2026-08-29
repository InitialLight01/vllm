"""融合免物化 indexer 打分+topk (G4 ②, 更新53y 设计定案).

参考语义 = 默认路径 (indexer_score_logits_triton + C++ top_k_per_row_prefill):
  logits[m,n] = Σ_h w[m,h] * relu(dot(q[m,h] fp8, k[n] fp8) * scale[n])
  topk = 前 topk_tokens 个, canonical 序 (score desc, index asc), 无效区 -1
数值硬约束 (见 sm12x_mqa.py 注释): h 序单累加器; 每输出元素 K 累加链与
参考 BN=64 dot 逐位同 (N 宽只影响 mma 网格铺排 — TDD 验证).

融合策略: 每 program 一行组 (BLOCK_M), 流式 BLOCK_N=TOPK=512 片; run 与
tile → tl.join (交错) → split-pair bitonic 全排序 (canonical 比较器,
numpy 模拟全验证: canonical/全并列/与 hypercube 参照逐位一致) → 取前半.
无 logits 物化, 无原子, 位级确定.

⚠️ triton 3.6 实证约束 (历次 GPU 迭代): 全 2 轴 hypercube reshape 内容
损坏; 嵌套 JIT 表达式 constexpr 语义注入断裂; 通用 reduce 表达式 axis
断裂; 无张量切片. 本实现: 全部内联顶层内核 + 安全 reshape 形状族
([BM,X,2,step]) + join/permute/split 配对.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _split_pair_v(fv, X: tl.constexpr, STEP: tl.constexpr,
                  BLOCK_M: tl.constexpr, PAD: tl.constexpr):
    """值通道配对拆分 (单张量 helper — 3.6 多张量 reshape 布局污染规避)."""
    rv = tl.reshape(fv, [BLOCK_M, X, 2, STEP])
    rv = tl.permute(rv, (0, 1, 3, 2))       # [BM, X, STEP, 2]
    lo_v, hi_v = tl.split(rv)
    return lo_v, hi_v


@triton.jit
def _split_pair_i(fi, X: tl.constexpr, STEP: tl.constexpr,
                  BLOCK_M: tl.constexpr, PAD: tl.constexpr):
    """索引通道配对拆分."""
    ri = tl.reshape(fi, [BLOCK_M, X, 2, STEP])
    ri = tl.permute(ri, (0, 1, 3, 2))
    lo_i, hi_i = tl.split(ri)
    return lo_i, hi_i


@triton.jit
def _join_back_v(nlo_v, nhi_v, X: tl.constexpr, STEP: tl.constexpr,
                 BLOCK_M: tl.constexpr, PAD: tl.constexpr):
    """值通道写回."""
    return tl.reshape(tl.permute(tl.join(nlo_v, nhi_v), (0, 1, 3, 2)),
                      [BLOCK_M, PAD])


@triton.jit
def _join_back_i(nlo_i, nhi_i, X: tl.constexpr, STEP: tl.constexpr,
                 BLOCK_M: tl.constexpr, PAD: tl.constexpr):
    """索引通道写回."""
    return tl.reshape(tl.permute(tl.join(nlo_i, nhi_i), (0, 1, 3, 2)),
                      [BLOCK_M, PAD])


@triton.jit
def _fused_indexer_topk_kernel(
    q_ptr, k_ptr, scale_ptr, weights_ptr,
    cu_ks_ptr, cu_ke_ptr, out_ptr,
    M, N,
    H: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LOG2_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    valid_m = offs_m < M

    ks = tl.load(cu_ks_ptr + offs_m, mask=valid_m, other=0)
    ke = tl.load(cu_ke_ptr + offs_m, mask=valid_m, other=0)

    PAD: tl.constexpr = 1 << LOG2_PAD
    PAD_IDX: tl.constexpr = 1 << 30
    NEG_INF = float("-inf")

    # 运行 topk: 恒保持 canonical 序 (score desc, idx asc), 宽 = BLOCK_N
    run_v = tl.full((BLOCK_M, BLOCK_N), NEG_INF, dtype=tl.float32)
    run_i = tl.full((BLOCK_M, BLOCK_N), PAD_IDX, dtype=tl.int32)

    for n0 in range(0, N, BLOCK_N):
        valid_n = offs_n < (N - n0)
        k_f8 = tl.load(
            k_ptr + offs_d[:, None] * stride_kd
            + (n0 + offs_n)[None, :] * stride_kn,
            mask=valid_n[None, :], other=0.0)
        k_sc = tl.load(scale_ptr + n0 + offs_n, mask=valid_n, other=1.0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h in tl.static_range(H):
            q_f8 = tl.load(
                q_ptr + offs_m[:, None] * stride_qm + h * stride_qh
                + offs_d[None, :] * stride_qd,
                mask=valid_m[:, None], other=0.0)
            s = tl.dot(q_f8, k_f8)          # fp8 x fp8 → fp32 acc (与参考同链)
            s = s * k_sc[None, :]
            s = tl.maximum(s, 0.0)          # per-head ReLU
            w = tl.load(weights_ptr + offs_m * stride_wm + h * stride_wh,
                        mask=valid_m, other=0.0)
            acc += s * w[:, None]           # 单累加器 h 序 (硬约束)

        # 全局列偏移 (n0 + offs_n) — 多块 + 截断 ke 时块内偏移会误掩
        row_valid = ((n0 + offs_n)[None, :] >= ks[:, None]) & (
            (n0 + offs_n)[None, :] < ke[:, None])
        acc = tl.where(row_valid & valid_m[:, None] & valid_n[None, :],
                       acc, NEG_INF)
        tile_i = tl.broadcast_to((n0 + offs_n)[None, :], (BLOCK_M, BLOCK_N))

        # ── merge: join (交错) → split-pair bitonic 全排序 → 取前半 ──
        jv = tl.join(run_v, acc)            # [BM, BLOCK_N, 2]
        ji = tl.join(run_i, tile_i)
        fv = tl.reshape(jv, [BLOCK_M, PAD])
        fi = tl.reshape(ji, [BLOCK_M, PAD])
        # bitonic 55 轮 (字面量展开, 单张量 helper 规避布局污染)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 1)) != 0) | (1 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 2)) != 0) | (2 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 2)) != 0) | (2 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 3)) != 0) | (3 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 3)) != 0) | (3 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 3)) != 0) | (3 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 4)) != 0) | (4 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 4)) != 0) | (4 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 4)) != 0) | (4 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 4)) != 0) | (4 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 5)) != 0) | (5 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 5)) != 0) | (5 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 5)) != 0) | (5 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 5)) != 0) | (5 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 5)) != 0) | (5 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 16, 32, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 16, 32, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 16)[None, :, None] * 64
                     + tl.arange(0, 32)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 16, 32, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 16, 32, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 6)) != 0) | (6 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 8, 64, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 8, 64, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 8)[None, :, None] * 128
                     + tl.arange(0, 64)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 8, 64, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 8, 64, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 16, 32, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 16, 32, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 16)[None, :, None] * 64
                     + tl.arange(0, 32)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 16, 32, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 16, 32, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 7)) != 0) | (7 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 4, 128, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 4, 128, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 4)[None, :, None] * 256
                     + tl.arange(0, 128)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 4, 128, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 4, 128, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 8, 64, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 8, 64, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 8)[None, :, None] * 128
                     + tl.arange(0, 64)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 8, 64, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 8, 64, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 16, 32, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 16, 32, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 16)[None, :, None] * 64
                     + tl.arange(0, 32)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 16, 32, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 16, 32, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 8)) != 0) | (8 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 2, 256, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 2, 256, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 2)[None, :, None] * 512
                     + tl.arange(0, 256)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 2, 256, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 2, 256, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 4, 128, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 4, 128, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 4)[None, :, None] * 256
                     + tl.arange(0, 128)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 4, 128, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 4, 128, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 8, 64, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 8, 64, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 8)[None, :, None] * 128
                     + tl.arange(0, 64)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 8, 64, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 8, 64, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 16, 32, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 16, 32, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 16)[None, :, None] * 64
                     + tl.arange(0, 32)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 16, 32, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 16, 32, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 9)) != 0) | (9 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 1, 512, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 1, 512, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 1)[None, :, None] * 1024
                     + tl.arange(0, 512)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 1, 512, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 1, 512, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 2, 256, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 2, 256, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 2)[None, :, None] * 512
                     + tl.arange(0, 256)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 2, 256, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 2, 256, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 4, 128, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 4, 128, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 4)[None, :, None] * 256
                     + tl.arange(0, 128)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 4, 128, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 4, 128, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 8, 64, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 8, 64, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 8)[None, :, None] * 128
                     + tl.arange(0, 64)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 8, 64, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 8, 64, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 16, 32, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 16, 32, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 16)[None, :, None] * 64
                     + tl.arange(0, 32)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 16, 32, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 16, 32, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 32, 16, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 32, 16, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 32)[None, :, None] * 32
                     + tl.arange(0, 16)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 32, 16, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 32, 16, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 64, 8, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 64, 8, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 64)[None, :, None] * 16
                     + tl.arange(0, 8)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 64, 8, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 64, 8, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 128, 4, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 128, 4, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 128)[None, :, None] * 8
                     + tl.arange(0, 4)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 128, 4, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 128, 4, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 256, 2, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 256, 2, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 256)[None, :, None] * 4
                     + tl.arange(0, 2)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 256, 2, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 256, 2, BLOCK_M, PAD)
        lo_v, hi_v = _split_pair_v(fv, 512, 1, BLOCK_M, PAD)
        lo_i, hi_i = _split_pair_i(fi, 512, 1, BLOCK_M, PAD)
        lower_idx = (tl.arange(0, BLOCK_M)[:, None, None] * PAD
                     + tl.arange(0, 512)[None, :, None] * 2
                     + tl.arange(0, 1)[None, None, :])
        desc = ((lower_idx & (1 << 10)) != 0) | (10 == LOG2_PAD)
        lo_better = (lo_v > hi_v) | ((lo_v == hi_v) & (lo_i < hi_i))
        swap = tl.where(desc, ~lo_better,
                        (hi_v < lo_v) | ((hi_v == lo_v) & (lo_i < hi_i)))
        nlo_v = tl.where(swap, hi_v, lo_v)
        nhi_v = tl.where(swap, lo_v, hi_v)
        nlo_i = tl.where(swap, hi_i, lo_i)
        nhi_i = tl.where(swap, lo_i, hi_i)
        fv = _join_back_v(nlo_v, nhi_v, 512, 1, BLOCK_M, PAD)
        fi = _join_back_i(nlo_i, nhi_i, 512, 1, BLOCK_M, PAD)
        # 取前半 (前 BLOCK_N = 新 run): [BM, 2, BLOCK_N] → permute → split
        wv = tl.permute(tl.reshape(fv, [BLOCK_M, 2, BLOCK_N]), (0, 2, 1))
        wi = tl.permute(tl.reshape(fi, [BLOCK_M, 2, BLOCK_N]), (0, 2, 1))
        run_v, _ = tl.split(wv)
        run_i, _ = tl.split(wi)

    out_i = tl.where(run_v <= NEG_INF, -1, run_i)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om
        + tl.arange(0, BLOCK_N)[None, :] * stride_on,
        out_i, mask=valid_m[:, None])


def fused_indexer_topk_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """融合免物化打分+topk (G4 ②). 接口与 fp8_fp4_mqa_topk_indices 对齐.

    返回 True 表示已填充 topk_indices (canonical 序, 无效区 -1).
    """
    if not (q.dim() == 3 and q.dtype == torch.float8_e4m3fn):
        return False
    k_fp8, scale = kv
    M, H, D = q.shape
    N = k_fp8.shape[0]
    topk_tokens = topk_indices.shape[1]
    if M == 0 or N == 0 or topk_tokens == 0:
        topk_indices.fill_(-1)
        return True
    assert topk_tokens & (topk_tokens - 1) == 0, "TOPK 须为 2 的幂"

    BLOCK_M = int(os.environ.get("VLLM_IDX_FUSED_BM", "8"))
    BLOCK_N = topk_tokens
    LOG2_PAD = (2 * topk_tokens).bit_length() - 1
    assert 2 * topk_tokens == (1 << LOG2_PAD), (topk_tokens, LOG2_PAD)

    grid = (triton.cdiv(M, BLOCK_M),)
    _fused_indexer_topk_kernel[grid](
        q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices,
        M, N, H, D,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        weights.stride(0), weights.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        LOG2_PAD=LOG2_PAD,
        num_warps=8, num_stages=1,
    )
    return True
