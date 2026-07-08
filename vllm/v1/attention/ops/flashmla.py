# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


# ---------------------------------------------------------------------------
# SM80 PyTorch fallbacks for fp8_ds_mla sparse attention
# ---------------------------------------------------------------------------

# fp8_ds_mla KV cache layout helpers
_FP8_DIM_SWA = 448  # number of fp8_e4m3 values in SWA NoPE
_ROPE_BF16_DIM = 64  # number of bf16 RoPE values
_SCALE_U8_DIM = 8  # number of ue8m0 scale bytes (one per 64-element block)
_N_BLOCKS_SWA = _FP8_DIM_SWA // 64  # 7 blocks
_TOKEN_BYTES_SWA = _FP8_DIM_SWA + _ROPE_BF16_DIM * 2 + _SCALE_U8_DIM  # 584

# Indexer fp8_ds_mla: 512 fp8 NoPE + 16 fp32 scale + 128 bf16 RoPE = 656
_FP8_DIM_IDX = 512
_N_BLOCKS_IDX = _FP8_DIM_IDX // 128  # 4 blocks
_TOKEN_BYTES_IDX = _FP8_DIM_IDX + 16 + _ROPE_BF16_DIM * 2  # 656


# __SM80_FALLBACK_MARKER__


def _fallback_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table=None,
    cache_seqlens=None,
    head_dim_v: int = 512,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    out=None,
):
    """PyTorch fallback for flash_mla_with_kvcache sparse decode.

    Handles fp8_ds_mla dequantization and mixed-dim attention
    (SWA K=512 dim vs Indexer K=576 dim) in pure PyTorch.
    """
    device = q.device
    num_tokens, sq, num_heads_q, dim_qk = q.shape  # sq == 1 for decode
    assert sq == 1, f"fallback supports sq=1 decode only, got sq={sq}"

    if softmax_scale is None:
        softmax_scale = dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(
            num_tokens, sq, num_heads_q, head_dim_v, dtype=q.dtype, device=device
        )

    def _dequant_cache(raw_cache: torch.Tensor, idx_tensor: torch.Tensor):
        """Dequantize tokens at given indices from raw cache.

        Uses SPLIT BLOCK LAYOUT: token data at block_base + pos * token_stride,
        scales at block_base + block_size * token_stride + pos * scale_dim.

        Returns [num_tokens, num_idx, kv_dim] bf16.
        """
        token_bytes = raw_cache.shape[-1]
        cache_block_size = raw_cache.shape[1]  # tokens per block
        block_stride = raw_cache.stride(0)  # bytes (block_size * token_bytes)

        # Determine format from token_bytes
        if token_bytes == _TOKEN_BYTES_SWA:
            fp8_dim = _FP8_DIM_SWA
            n_blocks = _N_BLOCKS_SWA
            quant_block = 64
            scale_dim = _SCALE_U8_DIM
            scale_is_fp32 = False
            token_stride = _TOKEN_BYTES_SWA - _SCALE_U8_DIM  # 584 - 8 = 576
        else:
            fp8_dim = _FP8_DIM_IDX
            n_blocks = _N_BLOCKS_IDX
            quant_block = 128
            scale_dim = 16  # 4 float32 scales * 4 bytes each = 16
            scale_is_fp32 = True
            token_stride = _TOKEN_BYTES_IDX - 16  # 656 - 16 = 640

        kv_dim = fp8_dim + _ROPE_BF16_DIM
        num_tokens_, num_idx, _ = idx_tensor.shape  # [N, 1, topk]

        cache_flat = raw_cache.reshape(-1)  # 1D byte view
        idx_flat = idx_tensor.reshape(num_tokens_, -1).long()  # [N, topk]

        # Convert global token indices to byte offsets
        block_indices = idx_flat // cache_block_size  # [N, topk]
        pos_in_block = idx_flat % cache_block_size

        out_k = torch.empty(
            num_tokens_, num_idx, kv_dim, dtype=torch.bfloat16, device=device
        )

        for t in range(num_tokens_):
            for kk in range(num_idx):
                bi = block_indices[t, kk].item()
                pos = pos_in_block[t, kk].item()
                data_base = bi * block_stride + pos * token_stride
                scale_base = (
                    bi * block_stride
                    + cache_block_size * token_stride
                    + pos * scale_dim
                )

                # Dequantize fp8 blocks
                for b in range(n_blocks):
                    fp8_start = data_base + b * quant_block
                    fp8_end = fp8_start + quant_block
                    fp8_bytes = cache_flat[fp8_end - quant_block : fp8_end]
                    if fp8_bytes.numel() == 0:
                        continue
                    fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()

                    if scale_is_fp32:
                        s_bytes = cache_flat[
                            scale_base + b * 4 : scale_base + b * 4 + 4
                        ]
                        s = s_bytes.view(torch.float32).item()
                    else:
                        s_byte = cache_flat[scale_base + b].item()
                        s = 2.0 ** (s_byte - 127.0)

                    out_k[t, kk, b * quant_block : b * quant_block + quant_block] = (
                        fp8_vals * s
                    ).to(torch.bfloat16)

                # Copy RoPE (bf16, at end of token data: fp8_dim)
                rope_start = data_base + fp8_dim
                rope_bytes = cache_flat[rope_start : rope_start + _ROPE_BF16_DIM * 2]
                out_k[t, kk, fp8_dim:] = rope_bytes.view(torch.bfloat16)

        return out_k

    # Compute per-source scores + V, handling mixed K dimensions
    all_scores = []
    all_v = []

    # 1) SWA cache
    if indices is not None:
        k_swa = _dequant_cache(k_cache, indices)  # [num_tokens, swa_topk, 512]
        k_dim = k_swa.shape[-1]
        dim_k = min(dim_qk, k_dim)
        q_slice = q[:, 0, :, :dim_k]  # [num_tokens, heads, dim_k]
        s = (
            torch.bmm(q_slice.float(), k_swa[..., :dim_k].float().transpose(1, 2))
            * softmax_scale
        )
        all_scores.append(s)
        all_v.append(k_swa[..., :head_dim_v])

    # 2) Extra (compressed) cache
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        k_extra = _dequant_cache(extra_k_cache, extra_indices_in_kvcache)
        k_dim_extra = k_extra.shape[-1]
        dim_k_e = min(dim_qk, k_dim_extra)
        q_slice_e = q[:, 0, :, :dim_k_e]  # [num_tokens, heads, dim_k_e]
        s_e = (
            torch.bmm(
                q_slice_e.float(), k_extra[..., :dim_k_e].float().transpose(1, 2)
            )
            * softmax_scale
        )
        all_scores.append(s_e)
        all_v.append(k_extra[..., :head_dim_v])

    # Concatenate
    scores = torch.cat(all_scores, dim=-1)  # [num_tokens, heads, total_kv]
    v_all = torch.cat(all_v, dim=1)  # [num_tokens, total_kv, head_dim_v]

    # Apply topk_length masking
    if topk_length is not None or extra_topk_length is not None:
        n_kv = scores.shape[-1]
        mask = torch.zeros(num_tokens, n_kv, dtype=torch.bool, device=device)
        offset = 0
        if indices is not None:
            swa_n = indices.shape[-1]
            if topk_length is not None:
                for i in range(num_tokens):
                    tl = topk_length[i].item()
                    if tl < swa_n:
                        mask[i, offset + tl : offset + swa_n] = True
            offset += swa_n
        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            extra_n = extra_indices_in_kvcache.shape[-1]
            if extra_topk_length is not None:
                for i in range(num_tokens):
                    etl = extra_topk_length[i].item()
                    if etl < extra_n:
                        mask[i, offset + etl : offset + extra_n] = True
            offset += extra_n
        if mask.any():
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

    # NOTE: attn_sink support omitted in SM80 fallback for simplicity
    lse = torch.logsumexp(scores, dim=-1)  # [num_tokens, heads]
    # [num_tokens, heads, 1] to match FlashMLA [batch, heads, sq]
    lse = lse.unsqueeze(-1)
    attn_w = torch.softmax(scores, dim=-1).to(v_all.dtype)
    out_squeezed = torch.bmm(attn_w, v_all)  # [num_tokens, heads, head_dim_v]
    out[:, 0, :, :] = out_squeezed

    return out, lse


def _fallback_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink=None,
    topk_length=None,
    out=None,
):
    """PyTorch fallback for flash_mla_sparse_fwd (prefill sparse attention).

    q:      [s_q, h_q, d_qk] bf16
    kv:     [s_kv, h_kv=1, d_qk] bf16 (already dequantized)
    indices: [s_q, h_kv=1, topk] int32
    Returns: (output [s_q, h_q, d_v], max_logits [s_q, h_q], lse [s_q, h_q])
    """
    device = q.device
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    # Clamp indices
    idx = indices.clamp(0, s_kv - 1).reshape(s_q, -1)  # [s_q, topk]

    # Gather K [s_q, topk, d_qk]
    k_flat = kv.reshape(-1, d_qk)  # [s_kv, d_qk]
    k = k_flat[idx]  # [s_q, topk, d_qk]

    # V is first d_v dims of K
    v = k[..., :d_v]  # [s_q, topk, d_v]

    # Each query token attends independently: scores [s_q, h_q, topk]
    scores = (
        torch.bmm(
            q.float(),  # [s_q, h_q, d_qk]
            k.float().transpose(1, 2),  # [s_q, d_qk, topk]
        )
        * sm_scale
    )

    # NOTE: attn_sink support omitted in SM80 fallback for simplicity

    # Mask invalid topk_length
    if topk_length is not None:
        tl = topk_length.view(-1)  # [s_q]
        topk = scores.shape[-1]
        mask = torch.arange(topk, device=device).unsqueeze(0).expand(
            s_q, -1
        ) >= tl.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

    attn_w = torch.softmax(scores, dim=-1).to(v.dtype)
    attn_out = torch.bmm(attn_w, v)  # [s_q, h_q, d_v]

    if out is None:
        out = attn_out.to(torch.bfloat16)
    else:
        out.copy_(attn_out)

    max_logits = scores.max(dim=-1).values.to(torch.float32)  # [s_q, h_q]
    lse = torch.logsumexp(scores, dim=-1).to(torch.float32)  # [s_q, h_q]
    return out, max_logits, lse


if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        have_initialized: bool = False
        tile_scheduler_metadata: torch.Tensor | None = None
        num_splits: torch.Tensor | None = None

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _fallback_flash_mla_sparse_fwd  # type: ignore[assignment]
    flash_mla_with_kvcache = _fallback_flash_mla_with_kvcache  # type: ignore[assignment]
    get_mla_metadata = lambda *a, **kw: (FlashMLASchedMeta(), None)  # noqa: E731


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
