#!/usr/bin/env python
"""MMLU-Pro 600 题分层抽样构建脚本。

从 datasets/mmlu_pro/test.jsonl (12032 题, 14 类) 按类别比例分层抽样，
生成 600 题回归测试集和 30 题 smoke 快速验证子集。
"""
import argparse, json, random, collections

ROOT = "/root/paddlejob/gpfsspace/huxiaoguang/llm_deploy"
LETTERS = "ABCDEFGHIJKLMNOP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=f"{ROOT}/datasets/mmlu_pro/test.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--smoke", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"{ROOT}/datasets/mmlu_pro/test_600.jsonl")
    ap.add_argument("--out-smoke", default=f"{ROOT}/datasets/mmlu_pro/test_smoke_30.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    test = [json.loads(l) for l in open(args.test)]
    by_cat = collections.defaultdict(list)
    for r in test:
        by_cat[r["category"]].append(r)

    total = len(test)
    sample = []
    for cat, rows in sorted(by_cat.items()):
        n = max(1, round(len(rows) / total * args.n))
        n = min(n, len(rows))
        sampled = random.sample(rows, n)
        sample.extend(sampled)
        print(f"  {cat:20s} total={len(rows):5d}  sampled={n:3d}")

    # Trim to exact count
    if len(sample) > args.n:
        sample = random.sample(sample, args.n)
    elif len(sample) < args.n:
        sample_ids = {r["question_id"] for r in sample}
        extra = [r for r in test if r["question_id"] not in sample_ids]
        sample.extend(random.sample(extra, args.n - len(sample)))

    print(f"\nTotal: {len(sample)} questions, {len(by_cat)} categories")

    with open(args.out, "w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Smoke subset: up to 3 per category from the 600 set
    smoke = []
    for cat in sorted(by_cat.keys()):
        cat_rows = [r for r in sample if r["category"] == cat][:3]
        smoke.extend(cat_rows)
    smoke = smoke[:args.smoke]

    with open(args.out_smoke, "w") as f:
        for r in smoke:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Smoke subset: {len(smoke)} questions -> {args.out_smoke}")


if __name__ == "__main__":
    main()
