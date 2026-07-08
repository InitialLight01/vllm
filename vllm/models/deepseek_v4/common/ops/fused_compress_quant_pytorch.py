# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch fp8e4nv requantize for SM80 where Triton lacks fp8e4nv support.

The Triton compress+quant kernel uses tl.float8e4b15 (the only fp8 dtype
available on SM80), which encodes values DIFFERENTLY from fp8e4nv.
This module provides a post-processing step that converts the stored
fp8e4b15 bytes to fp8e4nv bytes via float roundtrip.
"""

import torch

# fp8e4nv max (E4M3: exp=14, bias=7, man=7 -> 2^7 * 1.875 = 240)
_FP8_MAX = 448.0  # clamping value used by the kernel (safe upper bound)


def _decode_fp8e4b15(uint8_bytes: torch.Tensor) -> torch.Tensor:
    """Decode fp8e4b15 uint8 bytes to float32 values.

    fp8e4b15 is E4M3 format where exp=15 (1111) encodes large normal
    values (unlike fp8e4nv where exp=15 is NaN). Exponent bias = 7.
    """
    sign = (uint8_bytes >> 7) & 1
    exp = (uint8_bytes >> 3) & 0xF
    man = uint8_bytes & 0x7

    is_normal = exp > 0
    # Normal: (-1)^s * 2^(e-7) * (1 + m/8)
    normal_val = (
        ((-1.0) ** sign.float())
        * (2.0 ** (exp.float() - 7.0))
        * (1.0 + man.float() / 8.0)
    )
    # Subnormal: (-1)^s * 2^(-6) * (m/8)
    subnormal_val = ((-1.0) ** sign.float()) * (2.0 ** (-6.0)) * (man.float() / 8.0)

    return torch.where(is_normal, normal_val, subnormal_val)


def requantize_kv_cache_fp8e4nv(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    token_stride: int,
    scale_dim: int,
    fp8_dim: int,
    n_blocks: int,
    quant_block: int,
    scale_is_fp32: bool = False,
) -> None:
    """Post-process: convert fp8e4b15 bytes in KV cache to fp8e4nv.

    Reads fp8 data bytes + scale bytes written by the Triton kernel
    (in fp8e4b15 encoding), decodes them to float, and re-encodes them as
    fp8e4nv via torch.float8_e4m3fn.

    Args:
        kv_cache: [num_blocks, block_size, head_bytes] the target KV cache.
        slot_mapping: [num_tokens] int32, -1 = padding.
        token_stride: bytes per token in kv_cache (576 for sparse).
        scale_dim: bytes of scale data per token (8 for sparse).
        fp8_dim: number of fp8 values per token (448 for sparse).
        n_blocks: number of quant blocks (7 for sparse).
        quant_block: fp8 values per quant block (64 for sparse).
        scale_is_fp32: True for indexer float32 scales, False for ue8m0.
    """
    cache_u8 = kv_cache.reshape(-1)  # flatten to 1D byte view
    block_size = kv_cache.shape[1]
    block_stride = kv_cache.stride(0)  # in uint8 (block_size * head_bytes)

    if n_blocks <= 0:
        # indexer variant (head_dim=128) uses single-block, no per-block scales
        return

    valid_mask = slot_mapping >= 0
    if not valid_mask.any():
        return

    valid_slots = slot_mapping[valid_mask]

    for b in range(n_blocks):
        # Convert slot → flat byte offset for this quant block.
        block_indices = valid_slots // block_size
        pos_in_block = valid_slots % block_size
        flat_offsets = (
            block_indices * block_stride
            + pos_in_block * token_stride
            + b * quant_block
        ).long()

        # Read fp8e4b15 bytes [num_valid, quant_block]
        fp8_bytes = torch.stack(
            [cache_u8[off : off + quant_block] for off in flat_offsets]
        )

        # Scale bytes are stored AFTER all token data in each block:
        #   block_base + block_size * token_stride + pos_in_block * scale_dim + b
        scale_offsets = (
            block_indices * block_stride
            + block_size * token_stride
            + pos_in_block * scale_dim
            + b
        ).long()
        # Read scale bytes (format depends on scale_is_fp32)
        if scale_is_fp32:
            # Indexer format: float32 scales, 4 bytes per block
            scale_bytes = torch.stack(
                [cache_u8[off : off + 4] for off in scale_offsets]
            )
            scale = scale_bytes.view(torch.float32).view(-1, 1)  # [num_valid, 1]
        else:
            # SWA format: ue8m0, 1 byte per block
            scale_u8 = torch.stack(
                [cache_u8[off : off + 1] for off in scale_offsets]
            ).squeeze(-1)
            scale = 2.0 ** (scale_u8.float() - 127.0)  # [num_valid]
            scale = scale.unsqueeze(1)  # [num_valid, 1]

        # Decode fp8e4b15 → float, then apply scale to recover original values
        fp8_float = _decode_fp8e4b15(fp8_bytes)  # [num_valid, quant_block]
        # scale is already [num_valid, 1]
        values = fp8_float * scale  # [num_valid, quant_block]

        # Re-quantize to fp8e4nv
        inv_scale = 1.0 / scale.clamp(min=1e-10)
        x_scaled = torch.clamp(values * inv_scale, -_FP8_MAX, _FP8_MAX)
        x_fp8nv = x_scaled.to(torch.float8_e4m3fn)
        new_bytes = x_fp8nv.view(torch.uint8)  # [num_valid, quant_block]

        # Write back
        for i, off in enumerate(flat_offsets):
            cache_u8[off : off + quant_block] = new_bytes[i]
