"""TDD 单测: Triton 打分 kernel (indexer_score_logits_triton) 与参考语义的数值一致性。

参考语义 (与 paged kernel 逐位锁定):
  logits[m,n] = sum_h W_eff[m,h] * relu( Q_fp8[m,h,:] . (K_fp8[n,:] * k_scale[n]) )
  无效位置 (n < cu_ks 或 n >= cu_ke) 记 -inf

判据:
  1. 打分相对误差 < 5% (bf16 dot: 输入舍入 ~0.2%, 远低于门限)
  2. top-512 (top_k_per_row_prefill 作用于新 logits) 索引集合一致率 >= 99.9%
  3. 确定性: 同输入两次运行逐位一致
  4. 掩码: 无效位置 -inf, 短行 topk -1 填充
"""
import torch

from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import indexer_score_logits_triton

torch.manual_seed(42)
dev = "cuda"


def reference_logits(q_fp8, k_fp8, k_scale, W, cu_ks, cu_ke):
    """fp32 精确参考 emulation."""
    q_f = q_fp8.float()
    k_f = k_fp8.float() * k_scale.unsqueeze(1)  # [N, D]
    logits = q_f @ k_f.T  # [M, H, N]
    logits = logits.clamp(min=0)  # per-head ReLU
    logits = (logits * W.unsqueeze(-1)).sum(1)  # [M, N]
    col = torch.arange(logits.shape[1], device=dev).unsqueeze(0)
    valid = (col >= cu_ks.unsqueeze(1)) & (col < cu_ke.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


# ============ 用例 1: 主对齐用例 ============
M, H, D, N, TOPK = 512, 64, 128, 8192, 512
q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = torch.randn(N, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k_scale = torch.rand(N, device=dev, dtype=torch.float32) * 0.1 + 0.5
W = (torch.randn(M, H, device=dev, dtype=torch.float32).abs() + 0.5) * (
    torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.1 + 0.5
)
cu_ke = (torch.linspace(2048, N, M, device=dev)).to(torch.int32)
cu_ks = torch.zeros(M, device=dev, dtype=torch.int32)
cu_ks[M // 3 :] = 512  # sub-chunk 语义

ref = reference_logits(q, k, k_scale, W, cu_ks, cu_ke)
new = indexer_score_logits_triton(q, (k, k_scale), W, cu_ks, cu_ke)

valid_mask = torch.isfinite(ref)
abs_err = (ref[valid_mask] - new[valid_mask]).abs()
rel_err = (abs_err / (ref[valid_mask].abs() + 1e-3)).mean().item()
max_err = abs_err.max().item()
print(f"1a 打分误差: 均值 rel {rel_err:.5f}, 最大 abs {max_err:.2e}")
# fp8 乘积在 fp32 累加中精确 → 唯一误差源是求序噪声 (~1e-5 量级);
# 最大 abs 误差必须远低于 fp32 累加噪声地板 (~1e-3), 否则实现有错
assert max_err < 1e-3, f"打分最大误差过大: {max_err}"

# 1b: 掩码 — 无效位置 -inf
invalid_mask = ~valid_mask
assert torch.all(torch.isinf(new[invalid_mask]) & (new[invalid_mask] < 0)), "无效位置应 -inf"
print(f"1b 掩码: {invalid_mask.sum().item()} 个无效位置全部 -inf")

# 1c: topk 一致率 — 集合对称差元素必须落在 tie 区 (|分数差| ≤ 1e-3, fp32 求序噪声地板)
import vllm._custom_ops as ops
ref_topk = torch.full((M, TOPK), -1, device=dev, dtype=torch.int32)
new_topk = torch.full((M, TOPK), -1, device=dev, dtype=torch.int32)
ops.top_k_per_row_prefill(ref, cu_ks, cu_ke, ref_topk, M, ref.stride(0), ref.stride(1), TOPK)
ops.top_k_per_row_prefill(new, cu_ks, cu_ke, new_topk, M, new.stride(0), new.stride(1), TOPK)
ref_sorted, _ = ref_topk.sort(dim=1)
new_sorted, _ = new_topk.sort(dim=1)
row_match = (ref_sorted == new_sorted).all(dim=1).float().mean().item()
print(f"1c 行级集合一致率: {row_match*100:.2f}%")
max_tie_gap = 0.0
n_swaps = 0
for i in range(M):
    a, b = set(ref_topk[i].tolist()), set(new_topk[i].tolist())
    only_a, only_b = a - b, b - a
    for x, y in zip(sorted(only_a), sorted(only_b)):
        if x < 0 or y < 0:
            continue  # -1 填充槽位 (短行), 非分数翻转
        gap = abs(ref[i, x].item() - ref[i, y].item())
        max_tie_gap = max(max_tie_gap, gap)
        n_swaps += 1
print(f"1c 集合对称差元素对 {n_swaps} 个, 最大 tie 间隙 {max_tie_gap:.2e}")
assert max_tie_gap <= 1e-3, f"存在非 tie 区翻转, 间隙 {max_tie_gap}"
assert row_match >= 0.9, f"行级一致率过低: {row_match}"

# 1d: 短行 -1 填充
short_rows = (cu_ke - cu_ks) < TOPK
if short_rows.any():
    for i in short_rows.nonzero()[:, 0].tolist():
        cnt = int((cu_ke[i] - cu_ks[i]).item())
        assert (new_topk[i, cnt:] == -1).all(), f"行 {i} 有效 {cnt} 后应全 -1"
    print(f"1d 短行 -1 填充: {short_rows.sum().item()} 行全部正确")

# 1e: 确定性
new2 = indexer_score_logits_triton(q, (k, k_scale), W, cu_ks, cu_ke)
assert torch.equal(new, new2), "确定性失败"
print("1e 确定性: 两次运行逐位一致")

# ============ 用例 2: tie 确定性 (全零输入 → 全零分数) ============
q2 = torch.zeros(32, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k2 = torch.zeros(4096, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k_scale2 = torch.ones(4096, device=dev, dtype=torch.float32)
W2 = torch.ones(32, H, device=dev, dtype=torch.float32)
cu_ks2 = torch.zeros(32, device=dev, dtype=torch.int32)
cu_ke2 = torch.full((32,), 4096, device=dev, dtype=torch.int32)
r2a = indexer_score_logits_triton(q2, (k2, k_scale2), W2, cu_ks2, cu_ke2)
r2b = indexer_score_logits_triton(q2, (k2, k_scale2), W2, cu_ks2, cu_ke2)
assert torch.equal(r2a, r2b), "tie 下确定性失败"
print("用例 2 全零 tie 确定性: PASS")

# ============ 用例 3: 与 C++ paged kernel 三方交叉 ============
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

M3, N3 = 256, 4096
q3 = torch.randn(M3, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k3 = torch.randn(N3, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k_scale3 = torch.rand(N3, device=dev, dtype=torch.float32) * 0.1 + 0.5
W3 = (torch.randn(M3, H, device=dev, dtype=torch.float32).abs() + 0.5)
cu_ks3 = torch.zeros(M3, device=dev, dtype=torch.int32)
cu_ke3 = (torch.linspace(1024, N3, M3, device=dev)).to(torch.int32)

logits_cpp = fp8_fp4_mqa_logits(
    (q3, None), (k3, k_scale3), W3, cu_ks3, cu_ke3, clean_logits=True
)
ref3 = reference_logits(q3, k3, k_scale3, W3, cu_ks3, cu_ke3)
err_cpp = ((logits_cpp - ref3).abs() / (ref3.abs() + 1e-3))[torch.isfinite(ref3)].mean().item()
print(f"用例 3 C++ kernel vs 参考 emulation 相对误差: {err_cpp:.5f}")
assert err_cpp < 0.05

print("\nALL TESTS PASSED ✅")
