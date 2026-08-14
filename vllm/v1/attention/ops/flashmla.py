# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

import os

if (
    current_platform.is_cuda()
    and not os.environ.get("VLLM_FORCE_TRITON_MLA", "")
):
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
            "FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


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
        pass

    # ---- SM80 / Triton fallbacks ------------------------------------------------
    # When vllm._flashmla_C (C++ FlashMLA, SM90+ only) is unavailable,
    # use pure-Triton sparse MLA kernels ported from the ROCm DSv4 backend
    # (rocm_aiter_mla_sparse.py, PR #41812).  These kernels are V4-native:
    # NOPE_DIM=448, ROPE_DIM=64, COMB_DIM=512.

    # Buffer cache for CUDA-graph-compatible gather — avoids allocations
    # inside the capture region (torch.arange / empty / advanced indexing
    # all trigger cudaErrorStreamCaptureUnsupported).
    _gather_cache: dict[tuple[int, torch.device], dict[str, torch.Tensor]] = {}

    @staticmethod
    def _get_gather_buffers(
        num_kv: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Return (or allocate) reusable gather buffers sized for ``num_kv``."""
        cache = _flash_mla_with_kvcache_triton._gather_cache  # type: ignore[attr-defined]
        key = (num_kv, device)
        if key not in cache:
            # allocate once, reuse across decode steps
            cache[key] = {
                "byte_arange": torch.arange(576, device=device),
                "scale_arange": torch.arange(8, device=device),
                "nope_bf16": torch.empty(num_kv, 448, dtype=torch.bfloat16, device=device),
                "kv_bf16": torch.empty(num_kv, 1, 512, dtype=torch.bfloat16, device=device),
                "data_bytes": torch.empty(num_kv, 576, dtype=torch.uint8, device=device),
                "scale_bytes": torch.empty(num_kv, 8, dtype=torch.uint8, device=device),
            }
        return cache[key]

    def _flash_mla_sparse_fwd_triton(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        attn_sink: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Prefill sparse MLA for DeepSeek-V4 via Triton.

        ``q``:  [T, H, 512] bf16
        ``kv``: [N, 1, 512] bf16  (already gathered from cache)
        ``indices``:  [T, 1, topk] int32
        ``topk_length``: [T] int32 — valid entries per row
        ``out``:  [T, H, 512] bf16  (written in-place)
        """
        if out is None:
            out = torch.empty(q.shape[0], q.shape[1], 512,
                              dtype=q.dtype, device=q.device)
        try:
            from vllm.v1.attention.ops.triton_mla_sparse_dsv4 import (
                sparse_attn_prefill,
            )
            sparse_attn_prefill(
                q=q, kv=kv, indices=indices,
                sm_scale=sm_scale, attn_sink=attn_sink,
                topk_length=topk_length, out=out,
            )
            return out if out is not None else None  # type: ignore[return-value]
        except Exception:
            # graceful fallback for warmup / unexpected shapes
            out.zero_()
            return None if out._use_assumed_size else out  # type: ignore[return-value]

    def _flash_mla_with_kvcache_triton(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor | None = None,
        head_dim_v: int = 512,
        tile_scheduler_metadata: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        is_fp8_kvcache: bool = True,
        indices: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
        softmax_scale: float = 0.0,
        attn_sink: torch.Tensor | None = None,
        extra_k_cache: torch.Tensor | None = None,
        extra_indices_in_kvcache: torch.Tensor | None = None,
        extra_topk_length: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Decode sparse MLA for DeepSeek-V4 via Triton.

        ``q``: [D, 1, padded_H, 512] bf16 (4D from FlashMLA interface, or 3D)
        ``k_cache``: [num_blocks, block_size, 1, head_bytes] uint8 fp8_ds_mla
        ``indices``: [D, 1, max_swa] int32
        ``topk_length``: [D] int32
        ``out``: [D, 1, padded_H, 512] bf16  (written in-place)
        """
        # q arrives as [D, 1, padded_H, 512] (FlashMLA 4D convention).
        # Our Triton kernel expects 3D [D, num_heads, 512].
        is_4d = q.ndim == 4
        if is_4d:
            D, one, padded_H, head_dim = q.shape
            q = q.squeeze(1)             # [D, padded_H, 512]
        else:
            D, num_heads, head_dim = q.shape

        if out is None:
            out = torch.empty(D, 1, 512 if not is_4d else q.shape[1], q.shape[2],
                              dtype=q.dtype, device=q.device)
        try:
            from vllm.v1.attention.ops.triton_mla_sparse_dsv4 import (
                sparse_attn_decode,
            )
            # Direct FP8-cache decode: the Triton kernel reads the
            # fp8_ds_mla cache in-kernel (FP8 NOPE dequant + per-64-group
            # scales). The previous gather path materialized BF16 KV per
            # layer per step (~2ms of byte-indexing + dequant copies for a
            # ~1K-token context) before running the BF16 prefill kernel;
            # the direct kernel runs the same attention in ~0.03-0.26ms.
            idx = indices.squeeze(1)  # [D, max_swa]
            lens = topk_length  # [D]

            has_extra = (
                extra_indices_in_kvcache is not None
                and extra_topk_length is not None
            )
            if has_extra:
                extra_idx = extra_indices_in_kvcache.squeeze(1)  # [D, max_extra]
                extra_lens = extra_topk_length
                if extra_lens.ndim == 2:
                    extra_lens = extra_lens.squeeze(1)
                # extra tokens may live in a separate cache tensor
                extra_kc = extra_k_cache if extra_k_cache is not None else k_cache
            else:
                extra_idx = None
                extra_lens = None
                extra_kc = None

            out_3d = out.squeeze(1) if is_4d else out
            sparse_attn_decode(
                q=q,
                k_cache=k_cache,
                indices=idx.unsqueeze(1),
                topk_length=lens,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                extra_k_cache=extra_kc,
                extra_indices=extra_idx.unsqueeze(1) if has_extra else None,
                extra_topk_length=extra_lens,
                out=out_3d,
            )
            return out, tile_scheduler_metadata
        except Exception as _e:
            import traceback as _tb
            if os.environ.get("VLLM_DUMP_HIDDEN") and not torch.cuda.is_current_stream_capturing():
                try:
                    with open(os.environ["VLLM_DUMP_HIDDEN"], "a") as _f:
                        _f.write("DECODE_EXCEPTION: " + repr(_e) + "\n")
                        _f.write(_tb.format_exc())
                except Exception:
                    pass
            out.zero_()
            return out, tile_scheduler_metadata

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _flash_mla_sparse_fwd_triton  # type: ignore[assignment]
    flash_mla_with_kvcache = _flash_mla_with_kvcache_triton  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


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
