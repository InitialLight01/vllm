"""TDD 单测: GEMM 化打分与当前 paged kernel 的数值一致性。

已确认的 kernel 语义 (逐位验证):
  logits[m,n] = sum_h W_eff[m,h] * relu( Q_fp8[m,h,:] . K_fp8[n,:] * k_scale[n] )
  (per-head ReLU, 负 logits 截 0; W_eff 已含 q 的 per-token scale)

GEMM 路径 (bf16 dequant + 分块流式 top-k) 与 kernel 的判据:
  1. logits 相对误差 < 5% (fp8 量化 + bf16 GEMM 的精度量级)
  2. top-512 索引一致率 >= 99.9% (仅允许 tie 附近翻转)
"""
import torch
import vllm._custom_ops as ops
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

torch.manual_seed(42)
dev = "cuda"
M, H, D, N, TOPK = 512, 64, 128, 4096, 512

# --- 构造与真实 prefill 相同布局的输入 ---
q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = torch.randn(N, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
# per-element K scale
k_scale = torch.rand(N, device=dev, dtype=torch.float32) * 0.1 + 0.5
# W_eff: 已含 q per-token scale + softmax + head scale (模拟真实 weights_out)
weights = torch.randn(M, H, device=dev, dtype=torch.float32).abs() + 0.5
weights = weights * (torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.1 + 0.5)
# 每行有效 K 范围: 模拟 chunk 内位置 (历史长度递增)
cu_ke = torch.linspace(1024, N, M, device=dev, dtype=torch.int32)
cu_ks = torch.zeros(M, device=dev, dtype=torch.int32)

# --- 参考: 当前 paged kernel ---
logits_ref = fp8_fp4_mqa_logits(
    (q, None), (k, k_scale), weights, cu_ks, cu_ke, clean_logits=True
)
topk_ref = torch.full((M, TOPK), -1, device=dev, dtype=torch.int32)
ops.top_k_per_row_prefill(
    logits_ref, cu_ks, cu_ke, topk_ref, M,
    logits_ref.stride(0), logits_ref.stride(1), TOPK,
)

# --- 新实现 (GEMM 路径参考版): bf16 dequant + per-head GEMM + relu + 加权 + 分块 topk ---
q_bf = q.float()  # 注意: 真实 GEMM 路径用 bf16 RoPE 原值 Q (不做 fp8 往返)
k_bf = k.float() * k_scale.unsqueeze(1)  # K dequant [N, D]

# 分块流式 top-k (物化峰值受限): K 分块 1024 列
BLK = 1024
topk_new = torch.full((M, TOPK), -1, device=dev, dtype=torch.int32)
vals_new = torch.full((M, TOPK), float("-inf"), device=dev, dtype=torch.float32)
for c0 in range(0, N, BLK):
    c1 = min(c0 + BLK, N)
    k_blk = k_bf[c0:c1]  # [BLK, D]
    logits_blk = q_bf @ k_blk.T  # [M, H, BLK] — per-head GEMM
    logits_blk = logits_blk.clamp(min=0)  # per-head ReLU
    logits_blk = (logits_blk * weights.unsqueeze(-1)).sum(1)  # [M, BLK]
    # 行内有效范围 mask (cu_ks..cu_ke)
    col_idx = torch.arange(c0, c1, device=dev).unsqueeze(0)
    logits_blk = logits_blk.masked_fill(
        (col_idx >= cu_ke.unsqueeze(1)) | (col_idx < cu_ks.unsqueeze(1)),
        float("-inf"),
    )
    # 与当前 top-k 合并
    merged_v = torch.cat([vals_new, logits_blk], dim=1)
    merged_i = torch.cat([topk_new, col_idx.expand(M, -1)], dim=1)
    vals_new, idx_in_merged = merged_v.topk(TOPK, dim=1)
    topk_new = merged_i.gather(1, idx_in_merged)

# --- 断言 1: logits 一致性 (全量对比一次, 只测公式与数值误差) ---
logits_full = (q_bf @ k_bf.T).clamp(min=0)
logits_full = (logits_full * weights.unsqueeze(-1)).sum(1)
valid = (torch.arange(N, device=dev).unsqueeze(0) < cu_ke.unsqueeze(1)) & (
    torch.arange(N, device=dev).unsqueeze(0) >= cu_ks.unsqueeze(1)
)
rel_err = ((logits_ref[valid] - logits_full[valid]).abs() /
           (logits_ref[valid].abs() + 1e-3)).mean().item()
print(f"logits 相对误差 (valid region): {rel_err:.5f}")
assert rel_err < 0.05, f"logits 误差过大: {rel_err}"

# --- 断言 2: topk 一致率 ---
ref_sorted, _ = topk_ref.sort(dim=1)
new_sorted, _ = topk_new.sort(dim=1)
match = (ref_sorted == new_sorted).all(dim=1).float().mean().item()
print(f"top-{TOPK} 索引完全一致的行比例: {match*100:.2f}%")
assert match >= 0.999, f"topk 一致率过低: {match}"

print("PARITY TEST PASSED ✅")
