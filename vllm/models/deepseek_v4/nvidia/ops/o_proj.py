# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    _use_deepseek_v4_sm12x_triton_fp8_einsum,
    deepseek_v4_fp8_einsum,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    # MarlinFP8.process_weights_after_loading renames block-FP8 scales to
    # weight_scale_inv. Non-Marlin kernels keep the on-disk weight_scale name.
    wo_a_scale = getattr(wo_a, "weight_scale_inv", None)
    if wo_a_scale is None:
        wo_a_scale = getattr(wo_a, "weight_scale", None)
    if _use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", list(einsum_recipe), wo_a_scale
    ):
        # Triton einsum: SM 8.x (DECODE_E4M3 on Ampere / UPCAST on Ada) and
        # SM12x with the legacy (1,128,128) FP32 block-scale layout.
        deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a.weight,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(einsum_recipe),
        )
    else:
        # DeepGEMM C++ fp8_einsum (SM90/SM100, SM12x TMA recipe) — the
        # original direct-call semantics with the 2D weight.
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wo_a.weight, wo_a_scale),
            z,
            recipe=einsum_recipe,
        )
    return wo_b(z.flatten(1))
