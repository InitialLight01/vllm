# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch
import os

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    build_flashinfer_mixed_sparse_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import MultipleOf

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


def _packed_block_span(pool: torch.Tensor) -> int:
    """Per-block stride of ``pool`` in tokens (``stride(0)//stride(-2)``): ==
    block_size for unpacked KV, larger when packed (#44577). Raises if not
    token-aligned."""
    block_stride = pool.stride(0)
    token_stride = pool.stride(-2)
    if block_stride % token_stride != 0:
        raise NotImplementedError(
            "FLASHINFER_MLA_SPARSE_DSV4 packed KV requires the per-block stride "
            f"({block_stride}) to be a multiple of the per-token stride "
            f"({token_stride}); this layout is not supported yet."
        )
    return block_stride // token_stride


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4FlashMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout.

    Inheriting from the FlashMLA V4 backend reuses its
    ``DeepseekV4FlashMLAMetadata`` builder.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV4"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [10, 12]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major == 10:
            if kv_cache_dtype == "fp8_ds_mla":
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM10x uses the plain "
                    "per-tensor FP8 KV layout, not fp8_ds_mla"
                )
            if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
                return "kv_cache_dtype not supported"
            return None
        if device_capability.major == 12:
            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
                return "kv_cache_dtype not supported"
            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

            if not has_flashinfer_sparse_mla_sm120():
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                    "sparse MLA decode API"
                )
            return None
        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        device_capability = current_platform.get_device_capability()
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV4FlashMLABackend.get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str,
            )
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):
    """FlashInfer TRTLLM-gen sparse MLA attention layer for SM100 DeepSeek V4."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        # Per-tensor FP8 scale buffers + precomputed scalar BMM scales. Only the
        # per-tensor FP8 cache path consumes these; bf16 reads ``self.scale``.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # TRTLLM-gen takes scalar scale args on a distinct C++ path vs
        # one-element tensors, so these are Python floats.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The TRTLLM-gen kernel requires h_q in {64, 128}, so the output buffer
        # is allocated at the padded head count while q arrives at the local
        # head count; _forward pads q to match before the launcher.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: FlashInfer reads the cache directly and lazily
            # allocates its workspace, so nothing to reserve here.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward(
            q=q,
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
            output=output,
        )

    def _build_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the combined sparse-index tensors for the mixed batch.

        Returns ``(compressed_kv_cache, seq_lens, sparse_indices,
        sparse_topk_lens)``.
        """
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, self.window_size
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                if num_prefill_tokens > 0:
                    prefill_topk_indices = self.topk_indices_buffer[
                        num_decode_tokens:num_tokens
                    ]
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                    top_k = 0

                decode_compressed_indices_are_local = True
                assert swa_metadata.is_valid_token is not None
                decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
                if num_decode_tokens > 0:
                    decode_compressed_indices = self.topk_indices_buffer[
                        :num_decode_tokens
                    ]
                else:
                    # Keep the logical width aligned with the mixed-batch case so
                    # pure-prefill steps reuse the same Triton specialization.
                    decode_compressed_indices = prefill_topk_indices[:0]
            else:
                if num_prefill_tokens > 0:
                    assert attn_metadata.c128a_prefill_topk_indices is not None
                    prefill_topk_indices = attn_metadata.c128a_prefill_topk_indices
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = decode_swa_indices[:0, :0]
                    top_k = 0

                if num_decode_tokens > 0:
                    assert attn_metadata.c128a_global_decode_topk_indices is not None
                    assert attn_metadata.c128a_decode_topk_lens is not None
                    decode_compressed_indices = (
                        attn_metadata.c128a_global_decode_topk_indices.view(
                            num_decode_tokens, -1
                        )
                    )
                    decode_compressed_topk_lens = attn_metadata.c128a_decode_topk_lens
                    if num_prefill_tokens == 0:
                        prefill_topk_indices = decode_compressed_indices[:0, :0]
                else:
                    decode_compressed_indices = prefill_topk_indices[:0]
                    decode_compressed_topk_lens = swa_metadata.seq_lens[:0]

        # 确定性修复: 请求首 chunk 时压缩缓存尚无有效条目 (topk 全 -1),
        # 退化路径可能读取 KV 池的陈旧区块 (请求间内容不同 → 输出不确定)。
        # 防御性清零本请求的压缩缓存区块 — 有效区域读前必写, 语义等价;
        # 消除陈旧读取的确定性影响。VLLM_C4A_ZERO_STALE=0 可关闭。
        if (
            num_prefill_tokens > 0
            and os.environ.get("VLLM_C4A_ZERO_STALE", "1") != "0"
            and compressed_block_table is not None
            and compressed_kv_cache is not None
            and bool((prefill_topk_indices[:num_prefill_tokens] < 0).all().item())
        ):
            _rows = torch.unique(compressed_block_table.flatten())
            compressed_kv_cache[_rows] = 0

        # 退化短路: prefill 选择全空 (首 chunk, 压缩缓存无有效块) 时,
        # 改为 0 宽度选择 (与 dense 层同构) — 数学等价 (空选择零贡献),
        # 且避开 FlashInfer sparse kernel 的全 -1 退化路径 (该路径输出
        # 逐次不确定, 见附3 更新9)。VLLM_C4A_EMPTY_SHORT=0 可关闭。
        if (
            num_prefill_tokens > 0
            and os.environ.get("VLLM_C4A_EMPTY_SHORT", "1") != "0"
            and bool((prefill_topk_indices[:num_prefill_tokens] < 0).all().item())
        ):
            prefill_topk_indices = prefill_topk_indices[:, :0]
            top_k = 0

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        # cache for SWA-only and C128A that build the same mixed sparse indices
        # C4A stays uncached.
        cache_key = (
            "swa_only"
            if swa_only
            else ("c128a" if self.compress_ratio == 128 else "c4a")
        )
        cached_sparse = swa_metadata.flashinfer_sparse_index_cache.get(cache_key, None)
        if cached_sparse is None:
            swa_block_span = _packed_block_span(swa_k_cache)
            compressed_block_span = _packed_block_span(compressed_kv_cache)
            sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
                decode_swa_indices,
                decode_compressed_indices,
                decode_compressed_topk_lens,
                prefill_topk_indices[:num_prefill_tokens],
                query_start_loc,
                seq_lens,
                swa_metadata.token_to_req_indices[:num_tokens],
                swa_metadata.block_table[:num_reqs],
                swa_metadata.block_size,
                compressed_block_table,
                compressed_block_size,
                self.window_size,
                self.compress_ratio,
                top_k,
                decode_compressed_indices_are_local=decode_compressed_indices_are_local,
                decode_is_valid_token=decode_is_valid_token,
                swa_block_span=swa_block_span,
                compressed_block_span=compressed_block_span,
            )
            if cache_key != "c4a":
                swa_metadata.flashinfer_sparse_index_cache[cache_key] = (
                    sparse_indices,
                    sparse_topk_lens,
                )
        else:
            sparse_indices, sparse_topk_lens = cached_sparse
        return compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens

    def _forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        import os as _os
        _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
        if _prof:
            _pb0 = torch.cuda.Event(enable_timing=True)
            _pb1 = torch.cuda.Event(enable_timing=True)
            _pb0.record()
        (
            compressed_kv_cache,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )
        if _prof:
            _pb1.record()
            torch.cuda.synchronize()
            with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                _f.write(f"idxbuild r{self.compress_ratio} {_pb0.elapsed_time(_pb1):.3f}ms\n")

        # CUDA graph execution can pad q/output past the scheduled token count;
        # restrict to the real tokens (the launcher validates sparse indices).
        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        # The TRTLLM-gen sparse-MLA kernel requires h_q in {64, 128}; zero-pad
        # the query heads to the allocated output head count. Padded heads attend
        # to the shared KV and are sliced off downstream (output is padded too).
        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        workspace = _get_flashinfer_dsv4_workspace(q.device)
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc is not None and query_start_loc_cpu is not None

        # Keep the TRTLLM-gen decode/prefill split: the launcher is tuned for
        # uniform-q batches, and this avoids flattening mixed batches into one call.
        if num_decode_tokens > 0:
            decode_cu = query_start_loc[: num_decodes + 1]
            decode_cu_cpu = query_start_loc_cpu[: num_decodes + 1]
            decode_lens_cpu = decode_cu_cpu[1:] - decode_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[:num_decode_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                seq_lens=seq_lens[:num_decodes],
                out=output[:num_decode_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=int(decode_lens_cpu.max().item()),
            )

        if num_prefill_tokens > 0:
            # The prefill query view re-anchors at offset 0, so rebase the
            # cumulative query offsets to start at 0.
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
            import os as _os
            _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
            if _prof:
                _pk0 = torch.cuda.Event(enable_timing=True)
                _pk1 = torch.cuda.Event(enable_timing=True)
                _pk0.record()
            if os.environ.get("VLLM_TRITON_SPARSE_PREFILL", "0") == "1":
                # 确定性 Triton 替代路径 (FlashInfer cubin 的逐次不确定规避)。
                # 接口与下方 FlashInfer 调用对齐; 默认关, 原路径不变。
                from vllm.models.deepseek_v4.nvidia.ops.triton_prefill_sparse_mla import (
                    triton_prefill_sparse_mla,
                )

                triton_prefill_sparse_mla(
                    query=query[num_decode_tokens:num_tokens],
                    swa_kv_cache=swa_k_cache,
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                    sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                    seq_lens=seq_lens[num_decodes:num_reqs],
                    req_idx=swa_metadata.token_to_req_indices[
                        num_decode_tokens:num_tokens
                    ],
                    sinks=self.attn_sink,
                    out=output[num_decode_tokens:num_tokens],
                    bmm1_scale=bmm1_scale,
                )
            else:
                flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                    query=query[num_decode_tokens:num_tokens],
                    swa_kv_cache=swa_k_cache,
                    workspace_buffer=workspace,
                    sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                    seq_lens=seq_lens[num_decodes:num_reqs],
                    out=output[num_decode_tokens:num_tokens],
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    sinks=self.attn_sink,
                    cum_seq_lens_q=prefill_cu,
                    max_q_len=int(prefill_lens_cpu.max().item()),
                )
            if _prof:
                _pk1.record()
                torch.cuda.synchronize()
                with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                    _f.write(f"sparsekernel r{self.compress_ratio} {_pk0.elapsed_time(_pk1):.3f}ms")


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = True

    @staticmethod
    def _get_workspace(device: torch.device) -> torch.Tensor:
        return _get_flashinfer_dsv4_workspace(device)

    @staticmethod
    def _as_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if kv_cache.dim() == 4:
            return kv_cache
        return kv_cache.unsqueeze(-2)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
            "(SM120 kernel requires h_q in {16, 32, 64, 128})."
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        # Per-tensor FP8 cache path scales.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # FlashInfer expects scalar scale arguments for this path.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        # 确定性 Triton decode 替代 (VLLM_TRITON_SPARSE_DECODE=1, 附3
        # 更新43): cubin decode kernel 为闭源二进制且逐次不确定 (残余
        # 翻转噪声源 — 更新42 定论), SM80 实机亦必经此路径。布局与
        # 生产者一致 (576B/token + 块尾 UE8M0 scale)。
        # T 上限: 首请求 draft 全窗口 (8192 tokens) 物化 kv 需 5GB+
        # → 大 batch 回退 cubin (首请求为已知劣化, 评测协议排除)。
        if os.environ.get("VLLM_TRITON_SPARSE_DECODE", "0") == "1" and q.shape[0] <= 512:
            from vllm.models.deepseek_v4.nvidia.ops.triton_decode_sparse_mla import (
                triton_decode_sparse_mla_sm120,
            )

            triton_decode_sparse_mla_sm120(
                query=q,
                swa_kv_cache=self.swa_cache_layer.kv_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                compressed_kv_cache=kv_cache,
                extra_indices=extra_sparse_indices,
                extra_lens=extra_sparse_lengths,
                sinks=self.attn_sink,
                out=output,
                bmm1_scale=self.scale,
            )
            return
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            import os as _os
            _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
            if _prof:
                _pm0 = torch.cuda.Event(enable_timing=True)
                _pm1 = torch.cuda.Event(enable_timing=True)
                _pm0.record()
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )
            if _prof:
                _pm1.record()
                torch.cuda.synchronize()
                with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                    _f.write(f"topkmap r{self.compress_ratio} {_pm0.elapsed_time(_pm1):.3f}ms\n")

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_query(q, output)
        swa_kv_paged = self._as_sparse_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = self._as_sparse_cache(compressed_k_cache)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            # 读侧指纹 (L2 + chunk0 限定): 注意力读取的槽行内容 + 索引/lens
            # — 与写入侧 (swa_write/comp_write) 配对判定读侧分歧
            if (
                os.environ.get("VLLM_DUMP_HIDDEN_FP")
                and not torch.cuda.is_current_stream_capturing()
                and self.prefix == "model.layers.2.attn"
                and chunk_idx == 0
                and swa_metadata.block_table is not None
            ):
                try:
                    import json as _jrs
                    torch.cuda.synchronize()
                    _sidx = swa_indices_chunk.contiguous().view(-1).to(torch.int64)
                    _vl = _sidx[_sidx >= 0]
                    if _vl.numel():
                        _sflat = swa_k_cache.as_strided(
                            (swa_k_cache.shape[0], swa_k_cache.shape[1] * swa_k_cache.shape[2]),
                            (swa_k_cache.stride(0), 1),
                        )
                        # 只 gather 有效槽引用的行 (去重) — 整块 gather 会
                        # 混入未写的残留行 (伪像)
                        _uslots = _vl.clamp(max=_sflat.shape[0] - 1).unique()
                        _rows = _sflat[_uslots]
                        _bitsum = int(_rows.sum(dtype=torch.int64).item())
                        # 压缩读侧: extra 槽引用的行
                        _cbitsum = None
                        _cextra = None
                        if extra_sparse_indices_chunk is not None and compressed_k_cache is not None:
                            _eidx = extra_sparse_indices_chunk.contiguous().view(-1).to(torch.int64)
                            _ev = _eidx[_eidx >= 0]
                            if _ev.numel():
                                _cflat = compressed_k_cache.as_strided(
                                    (compressed_k_cache.shape[0], compressed_k_cache.shape[1] * compressed_k_cache.shape[2]),
                                    (compressed_k_cache.stride(0), 1),
                                )
                                _cbitsum = int(_cflat[_ev.clamp(max=_cflat.shape[0] - 1).unique()].sum(dtype=torch.int64).item())
                                _cextra = [int(_ev.min().item()), int(_ev.max().item())]
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jrs.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "attn_read_swa",
                                "prefix": self.prefix,
                                "pos0": int(swa_metadata.query_start_loc_cpu[
                                    num_decodes].item()) if swa_metadata.query_start_loc_cpu is not None else -1,
                                "nrows": int(_uslots.numel()),
                                "bitsum": _bitsum,
                                "cbitsum": _cbitsum,
                                "cextra": _cextra,
                                "idx_sum": int(_sidx.sum().item()),
                                "idx_min": int(_vl.min().item()) if _vl.numel() else -1,
                                "idx_max": int(_vl.max().item()) if _vl.numel() else -1,
                                "lens_sum": int(swa_lens_chunk.sum().item()),
                                "is_valid_sum": int(swa_metadata.is_valid_token.sum().item()) if swa_metadata.is_valid_token is not None else -1,
                                "bt16": [int(x) for x in swa_metadata.block_table[
                                    0, :16].tolist()],
                                "bt_valid": int((swa_metadata.block_table[0] >= 0).sum().item()),
                                "idx0_16": [int(x) for x in swa_indices_chunk[
                                    0, 0, :16].tolist()],
                            }) + "\n")
                except Exception as _ers:
                    try:
                        import json as _jers
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jers.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "attn_read_swa_err",
                                "prefix": self.prefix,
                                "err": str(_ers)[:200],
                                "bt_none": swa_metadata.block_table is None,
                                "bt_shape": list(swa_metadata.block_table.shape) if swa_metadata.block_table is not None else None,
                            }) + "\n")
                    except Exception:
                        pass
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            import os as _os
            _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
            if _prof:
                _pk0 = torch.cuda.Event(enable_timing=True)
                _pk1 = torch.cuda.Event(enable_timing=True)
                _pk0.record()
            # 双请求 kernel 边界全量捕获 (VLLM_TRITON_DIFFCAP=<path>):
            # L2 chunk0 前 2 请求 — 保存 q/索引/lens/seq/req/sinks + 按
            # 真布局 (576 数据 + 块尾 scale) 提取的读行, 供离线位级比对。
            # 附3 更新39: 同时捕获最后 chunk (chunk_idx == num_chunks-1,
            # 短 chunk 1961 行 — run 35 指纹显示 L3 注意力在此分歧),
            # 独立计数与文件后缀 .last{N}.pt。
            _capn = None
            _cap_suffix = ""
            if (
                os.environ.get("VLLM_TRITON_DIFFCAP")
                and self.prefix == "model.layers.2.attn"
                and q_chunk.shape[0] > 1000
                and bool(swa_lens_chunk.max() > 0)
                and int(swa_metadata.seq_lens[0].item()) == 8188
                and (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                if chunk_idx == 0:
                    _capn = getattr(self, "_diffcap_n", 0)
                elif chunk_idx == num_chunks - 1:
                    _capn = getattr(self, "_diffcap_last_n", 0)
                    _cap_suffix = "last"
            if _capn is not None:
                try:
                    _cap_attr = "_diffcap_n" if not _cap_suffix else "_diffcap_last_n"
                    if _capn < 2:
                        setattr(self, _cap_attr, _capn + 1)
                        torch.cuda.synchronize()
                        _sidx2 = swa_indices_chunk.contiguous().view(-1).to(torch.int64)
                        _v2 = _sidx2[_sidx2 >= 0].unique()
                        _f576 = swa_k_cache.as_strided(
                            (swa_k_cache.shape[0], swa_k_cache.shape[1] * 576),
                            (swa_k_cache.stride(0), 1),
                        )
                        _fs = swa_k_cache.as_strided(
                            (swa_k_cache.shape[0], 512), (swa_k_cache.stride(0), 1)
                        )
                        _blk = _v2 // 64
                        _pos = _v2 % 64
                        _bd = _f576[_blk].view(-1, 64, 576)
                        _rowdata = _bd[torch.arange(_v2.numel(), device=_v2.device), _pos]  # [n, 576]
                        _sc = _fs[_blk].view(-1, 64, 8)
                        _rowsc = _sc[torch.arange(_v2.numel(), device=_v2.device), _pos]   # [n, 8]
                        # 写域行: 按 token 顺序 (slot_mapping[t] → 576B 行)
                        # — 槽值排序会导致跨请求行序错位 (伪像)
                        _sm = swa_metadata.slot_mapping[: q_chunk.shape[0]]
                        _sm_b = _sm // 64
                        _sm_p = _sm % 64
                        _smd = _f576[_sm_b].view(-1, 64, 576)
                        _wrowdata = _smd[torch.arange(_sm.numel(), device=_sm.device), _sm_p]
                        _sms = _fs[_sm_b].view(-1, 64, 8)
                        _wrowsc = _sms[torch.arange(_sm.numel(), device=_sm.device), _sm_p]
                        # 压缩读行指纹: unique 槽 sum (顺序无关, 无伪像)
                        # + token 2040-2059 的完整读行序列 (2051 分歧起点采样)
                        _csum = None
                        _csample = None
                        if extra_sparse_indices_chunk is not None and compressed_k_cache is not None:
                            _cflat576 = compressed_k_cache.as_strided(
                                (compressed_k_cache.shape[0], compressed_k_cache.shape[1] * 576),
                                (compressed_k_cache.stride(0), 1),
                            )
                            _cflatsc = compressed_k_cache.as_strided(
                                (compressed_k_cache.shape[0], 512),
                                (compressed_k_cache.stride(0), 1),
                            )
                            _cb = compressed_k_cache.shape[1]
                            _eidx = extra_sparse_indices_chunk
                            _ev = _eidx.view(-1).to(torch.int64)
                            _ev = _ev[_ev >= 0]
                            if _ev.numel():
                                _cu = _ev.unique()
                                _cblk = _cu // _cb
                                _cpos = _cu % _cb
                                _cd = _cflat576[_cblk].view(-1, _cb, 576)
                                _crows = _cd[torch.arange(_cu.numel(), device=_cu.device), _cpos]
                                _csum = int(_crows.sum(dtype=torch.int64).item())
                                _csc = _cflatsc[_cblk].view(-1, _cb, 8)
                                _csum_s = int(_csc[torch.arange(_cu.numel(), device=_cu.device), _cpos].sum(dtype=torch.int64).item())
                            # 采样: token 2040-2059
                            _s0, _s1 = min(2040, _eidx.shape[0]), min(2060, _eidx.shape[0])
                            if _s1 > _s0:
                                _es = _eidx[_s0:_s1]
                                _els = extra_sparse_lengths_chunk[_s0:_s1]
                                _sample_rows = []
                                for _t in range(_es.shape[0]):
                                    for _c in range(int(_els[_t].item())):
                                        _sv = int(_es[_t, _c].item())
                                        if _sv >= 0:
                                            _b2, _p2 = _sv // _cb, _sv % _cb
                                            _sample_rows.append(_cflat576[_b2, _p2 * 576:(_p2 + 1) * 576].cpu())
                                            _sample_rows.append(_cflatsc[_b2, _p2 * 8:(_p2 + 1) * 8].cpu())
                                _csample = torch.cat(_sample_rows) if _sample_rows else torch.zeros(0)
                        torch.save(
                            {
                                "n": int(_capn),
                                "csum": _csum,
                                "csum_s": _csum_s,
                                "csample": _csample,
                                "q": q_chunk.detach().contiguous().cpu(),
                                "swa_indices": swa_indices_chunk.detach().cpu(),
                                "swa_lens": swa_lens_chunk.detach().cpu(),
                                "extra_indices": None if extra_sparse_indices_chunk is None
                                else extra_sparse_indices_chunk.detach().cpu(),
                                "extra_lens": None if extra_sparse_lengths_chunk is None
                                else extra_sparse_lengths_chunk.detach().cpu(),
                                "seq_lens": swa_metadata.seq_lens.detach().cpu(),
                                "req_idx": swa_metadata.token_to_req_indices[
                                    num_decode_tokens + query_start :
                                    num_decode_tokens + query_end].detach().cpu(),
                                "sinks": self.attn_sink.detach().cpu(),
                                "slot_mapping": swa_metadata.slot_mapping.detach().cpu(),
                                "topk_buffer": None if local_topk_indices is None
                                else local_topk_indices[:query_end].detach().cpu(),
                                "block_table": None if attn_metadata is None
                                else attn_metadata.block_table.detach().cpu(),
                                "comp_block_size": None if attn_metadata is None
                                else attn_metadata.block_size // self.compress_ratio,
                                "rowdata": _rowdata.detach().cpu(),
                                "rowsc": _rowsc.detach().cpu(),
                                "wrowdata": _wrowdata.detach().cpu(),
                                "wrowsc": _wrowsc.detach().cpu(),
                                "scale": self.scale,
                            },
                            os.environ["VLLM_TRITON_DIFFCAP"]
                            + f".{_cap_suffix + '.' if _cap_suffix else ''}n{q_chunk.shape[0]}.{_capn}.pt",
                        )
                except Exception as _edc:
                    print(f"[DIFFCAP] err: {_edc}", flush=True)
            # 确定性 Triton 替代 (VLLM_TRITON_SPARSE_PREFILL=1): 固定序
            # 流式注意力, 布局与生产者一致 (576B/token + 块尾 UE8M0
            # scales)。默认关, 原 cubin 路径不变。
            if os.environ.get("VLLM_TRITON_SPARSE_PREFILL", "0") == "1":
                from vllm.models.deepseek_v4.nvidia.ops.triton_prefill_sparse_mla import (
                    triton_prefill_sparse_mla_sm120,
                )

                triton_prefill_sparse_mla_sm120(
                    query=q_chunk,
                    swa_kv_cache=swa_k_cache,
                    swa_indices=swa_indices_chunk,
                    swa_lens=swa_lens_chunk,
                    compressed_kv_cache=compressed_k_cache,
                    extra_indices=extra_sparse_indices_chunk,
                    extra_lens=extra_sparse_lengths_chunk,
                    # 完整批序数: req_idx (全批 token→req) 直接索引
                    seq_lens=swa_metadata.seq_lens,
                    req_idx=swa_metadata.token_to_req_indices[
                        num_decode_tokens + query_start :
                        num_decode_tokens + query_end
                    ],
                    sinks=self.attn_sink,
                    out=output[query_start:query_end],
                    bmm1_scale=self.scale,
                )
            else:
                flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                    query=q_chunk,
                    swa_kv_cache=swa_kv_paged,
                    workspace_buffer=self._get_workspace(q.device),
                    sparse_indices=swa_indices_chunk,
                    compressed_kv_cache=extra_kv_paged,
                    out=output[query_start:query_end],
                    bmm1_scale=self.scale,
                    sinks=self.attn_sink,
                    kv_layout="NHD",
                    swa_topk_lens=swa_lens_chunk,
                    extra_sparse_indices=extra_sparse_indices_chunk,
                    extra_sparse_topk_lens=extra_sparse_lengths_chunk,
                )
            if _prof:
                _pk1.record()
                torch.cuda.synchronize()
                with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                    _f.write(f"sparsekernel r{self.compress_ratio} {_pk0.elapsed_time(_pk1):.3f}ms\n")
