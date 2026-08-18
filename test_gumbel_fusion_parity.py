#!/usr/bin/env python
"""Bit-parity test: fused dspark_gumbel_argmax_sample vs eager gumbel_sample."""
import torch

from vllm.models.deepseek_v4.nvidia.dspark_triton import (
    dspark_gumbel_argmax_sample,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

torch.manual_seed(0)
R, S, V = 8, 5, 129280
dev = "cuda"

for trial in range(3):
    logits = (torch.randn(R, V, device=dev) * 2.0).contiguous()
    temps = torch.tensor([0.0, 1.0, 0.7, 0.0, 1.3, 0.0, 0.5, 0.0], device=dev)
    seeds = torch.randint(0, 2**31 - 1, (R,), device=dev, dtype=torch.int64)
    pos = torch.randint(0, 4096, (R,), device=dev, dtype=torch.int32)
    idx_map = torch.arange(R, dtype=torch.int64, device=dev)
    col = torch.tensor(0, dtype=torch.int32, device=dev)

    # eager reference (mirrors the speculator's gumbel_sample call)
    eager_cache = torch.zeros(R, S, V, dtype=torch.float32, device=dev)
    tokens_eager = gumbel_sample(
        logits, idx_map, temps, seeds, pos,
        apply_temperature=True,
        output_processed_logits=eager_cache,
        output_processed_logits_col=col,
        use_fp64=False,
    )

    # fused path
    fused_cache = torch.zeros(R, S, V, dtype=torch.float32, device=dev)
    tokens_fused = torch.empty(R, dtype=torch.int64, device=dev)
    scratch = {
        "block_gval": torch.zeros(R, (V + 1023) // 1024, dtype=torch.float32, device=dev),
        "block_gid": torch.zeros(R, (V + 1023) // 1024, dtype=torch.int64, device=dev),
    }
    dspark_gumbel_argmax_sample(
        logits, temps, tokens_fused, fused_cache[:, 0, :],
        seeds, pos, scratch, use_fp64=False,
    )

    tok_eq = torch.equal(tokens_eager, tokens_fused)
    logit_eq = torch.equal(eager_cache[:, 0, :], fused_cache[:, 0, :])
    max_logit_diff = (eager_cache[:, 0, :] - fused_cache[:, 0, :]).abs().max().item()
    print(f"trial {trial}: tokens_equal={tok_eq} logits_equal={logit_eq} "
          f"max_logit_diff={max_logit_diff:.3e}")
    if not tok_eq:
        print("  eager:", tokens_eager.tolist())
        print("  fused:", tokens_fused.tolist())
    assert tok_eq and logit_eq, "bit-parity broken"

print("ALL GUMBEL FUSION PARITY CHECKS PASSED")
