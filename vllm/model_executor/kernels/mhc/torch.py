# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_flat.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    if norm_weight is not None:
        norm_w = norm_weight.float()
        rms = torch.rsqrt(layer_input.float().pow(2).mean(-1, keepdim=True) + norm_eps)
        layer_input = (layer_input.float() * rms * norm_w).to(torch.bfloat16)

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input,
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_fused_post_pre_torch(
    x: torch.Tensor, residual: torch.Tensor,
    post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
    fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float,
    hc_post_mult_value: float, sinkhorn_repeat: int,
    n_splits: int = 1, tile_n: int = 1,
    norm_weight: torch.Tensor | None = None, norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_cur = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_torch(
        residual_cur, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, n_splits,
        norm_weight=norm_weight, norm_eps=norm_eps,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def hc_head_fused_torch(
    hs_flat: torch.Tensor, fn: torch.Tensor,
    hc_scale: torch.Tensor, hc_base: torch.Tensor,
    rms_eps: float, hc_eps: float,
) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    if num_tokens == 0:
        return torch.empty(0, hidden_size, dtype=torch.bfloat16, device=hs_flat.device)
    hc_dim = hc_mult * hidden_size
    hs_f32 = hs_flat.float()
    sqrsum = hs_f32.pow(2).sum(dim=[1, 2], keepdim=True)
    rsqrt = torch.rsqrt(sqrsum.reshape(num_tokens, 1) / hc_dim + rms_eps)
    hs_flat_f32 = hs_f32.reshape(num_tokens, hc_dim)
    fn_f32 = fn[:hc_mult].to(torch.float32)
    mixes = torch.matmul(hs_flat_f32, fn_f32.T)
    hc_scale_val = hc_scale.to(torch.float32).view(1)
    hc_base_val = hc_base.to(torch.float32).view(1, hc_mult)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale_val + hc_base_val) + hc_eps
    out = torch.einsum("tch,tc->th", hs_flat.to(torch.bfloat16), pre_mix.to(torch.bfloat16))
    return out


