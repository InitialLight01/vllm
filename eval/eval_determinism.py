#!/usr/bin/env python
"""确定性评测对比脚本 — logprobs 级逐位对齐验证。

用法:
    # 30 题 smoke 确定性对比
    python eval/eval_determinism.py \
        --baseline eval/baseline_runA_smoke.jsonl \
        --compare eval/baseline_runB_smoke.jsonl

    # EM 差异模式（不需要 logprobs）
    python eval/eval_determinism.py \
        --baseline eval/baseline_smoke.jsonl \
        --compare eval/m1_smoke.jsonl \
        --mode em_diff
"""
import argparse, json, sys
from collections import Counter


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="基线结果文件 (run A)")
    ap.add_argument("--compare", required=True, help="待比较结果文件 (run B)")
    ap.add_argument("--mode", default="full",
                    choices=["full", "em_diff", "logprobs"],
                    help="full=EM+logprobs, em_diff=仅EM, logprobs=仅logprobs")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="logprobs 差异容忍阈值")
    args = ap.parse_args()

    base = load_jsonl(args.baseline)
    comp = load_jsonl(args.compare)

    if len(base) != len(comp):
        print(f"WARNING: 结果数不一致 ({len(base)} vs {len(comp)})")

    # Index by question_id
    base_by_id = {r["question_id"]: r for r in base}
    comp_by_id = {r["question_id"]: r for r in comp}

    common = set(base_by_id) & set(comp_by_id)
    only_base = set(base_by_id) - set(comp_by_id)
    only_comp = set(comp_by_id) - set(base_by_id)

    print(f"===== 确定性对比报告 =====")
    print(f"基线: {args.baseline} ({len(base)} 题)")
    print(f"对比: {args.compare} ({len(comp)} 题)")
    print(f"共同题目: {len(common)}")
    if only_base:
        print(f"仅在基线: {only_base}")
    if only_comp:
        print(f"仅在对比: {only_comp}")
    print()

    # EM comparison
    em_same = 0
    em_diff = 0
    em_ids = []

    for qid in sorted(common):
        b_pred = base_by_id[qid].get("pred")
        c_pred = comp_by_id[qid].get("pred")
        if b_pred == c_pred:
            em_same += 1
        else:
            em_diff += 1
            em_ids.append(qid)

    print(f"--- EM 一致性 ---")
    print(f"一致: {em_same}/{len(common)} ({em_same/len(common)*100:.1f}%)")
    print(f"不一致: {em_diff}/{len(common)}")
    if em_ids:
        print(f"不一致题目 ID: {em_ids}")
    print()

    # Logprobs comparison (if mode is full or logprobs)
    if args.mode in ("full", "logprobs"):
        logprobs_available = all(
            "logprobs" in base_by_id[qid] for qid in common
        ) and all(
            "logprobs" in comp_by_id[qid] for qid in common
        )

        if not logprobs_available:
            print("--- Logprobs 对比 ---")
            print("WARNING: 结果文件中无 logprobs 字段，跳过逐 token 对比。")
            print("如需 logprobs 对比，请在评测时请求 logprobs 输出。")
        else:
            total_tokens = 0
            diff_tokens = 0
            max_diff = 0.0
            mse_sum = 0.0
            per_q_diff = {}

            for qid in sorted(common):
                b_lp = base_by_id[qid]["logprobs"]
                c_lp = comp_by_id[qid]["logprobs"]
                if len(b_lp) != len(c_lp):
                    print(f"  WARNING: qid={qid} token 数不一致 ({len(b_lp)} vs {len(c_lp)})")
                    continue
                q_diff = 0
                for i, (a, b) in enumerate(zip(b_lp, c_lp)):
                    d = abs(a - b)
                    mse_sum += d * d
                    total_tokens += 1
                    if d > args.tol:
                        diff_tokens += 1
                        q_diff += 1
                    if d > max_diff:
                        max_diff = d
                per_q_diff[qid] = q_diff

            print(f"--- Logprobs 对比 (tol={args.tol}) ---")
            print(f"总 token 数: {total_tokens}")
            print(f"差异 > tol 的 token: {diff_tokens} ({diff_tokens/max(1,total_tokens)*100:.4f}%)")
            print(f"最大差异: {max_diff:.8f}")
            print(f"MSE: {mse_sum/max(1,total_tokens):.8f}")
            if per_q_diff:
                q_with_diff = {k: v for k, v in per_q_diff.items() if v > 0}
                if q_with_diff:
                    print(f"有差异的题目: {len(q_with_diff)}/{len(common)}")
                    for qid, cnt in sorted(q_with_diff.items(), key=lambda x: -x[1])[:10]:
                        print(f"  qid={qid}: {cnt} diff tokens")

    # Summary verdict
    print()
    print("===== 结论 =====")
    if em_diff == 0:
        print("EM 完全一致 ✓")
    else:
        print(f"EM 差异: {em_diff} 题 ✗")

    if args.mode in ("full", "logprobs") and logprobs_available:
        if diff_tokens == 0:
            print("Logprobs 逐位对齐 ✓")
        elif max_diff < 1e-5 and diff_tokens / max(1, total_tokens) < 0.001:
            print("Logprobs 差异可忽略（仅浮点舍入误差）✓")
        else:
            print(f"Logprobs 存在显著差异 ✗ (max_diff={max_diff:.2e})")

    return 0 if em_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
