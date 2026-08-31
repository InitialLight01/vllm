# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
def compute_dsv4_index_cache_skip_flags(
    compress_ratios,
    num_hidden_layers,
    *,
    index_topk_freq: int = 1,
    index_topk_pattern=None,
    index_skip_topk_offset: int = 2,
    local_start_layer: int = 0,
    local_end_layer=None,
) -> tuple:
    """DSv4 C4 层 IndexCache 跳过决策 (上游 PR #49085 移植, #71).

    仅 C4A (compress_ratio==4) 层可跳过; 跳过的层不复算 topk, 复用前一个
    F 层的共享 topk 缓冲 (同步序内 F 层先执行)。freq=4/offset=2 默认
    F,F,S,S,S 循环 (43→11 PFLOP)。首层本地 C4 强制 F。
    """
    if local_end_layer is None:
        local_end_layer = num_hidden_layers
    ratios = list(compress_ratios[:num_hidden_layers])
    skip_flags = [False] * num_hidden_layers
    c4_ids = [i for i, r in enumerate(ratios) if r == 4]
    if not c4_ids:
        return tuple(skip_flags)
    if index_topk_pattern is not None:
        pattern = list(index_topk_pattern)
        assert set(pattern) <= {"F", "S"}, pattern
        assert len(pattern) == len(c4_ids), (len(pattern), len(c4_ids))
        assert pattern[0] == "F"
        for rank, lid in enumerate(c4_ids):
            skip_flags[lid] = pattern[rank] == "S"
    else:
        assert index_topk_freq > 0
        assert index_skip_topk_offset >= 1
        for rank, lid in enumerate(c4_ids):
            skip_flags[lid] = (
                max(rank - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
    local_c4 = [lid for lid in c4_ids if local_start_layer <= lid < local_end_layer]
    if local_c4:
        skip_flags[local_c4[0]] = False
    return tuple(skip_flags)

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.skip_topk = skip_topk
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if os.environ.get("VLLM_DUMP_HIDDEN") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _json
                with open(os.environ["VLLM_DUMP_HIDDEN"], "a") as _f:
                    _f.write(_json.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "attn_fwd_enter",
                        "cls": type(self).__name__,
                        "hs_shape": list(hidden_states.shape),
                    }) + "\n")
            except Exception:
                pass

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Metadata-independent input GEMMs + RMSNorm stay in the captured
        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
        # compressor, MLA attention) runs in the eager break.
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _tqkv0 = torch.cuda.Event(enable_timing=True)
            _tqkv1 = torch.cuda.Event(enable_timing=True)
            _tqkv0.record()
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _tqkv1.record()
            torch.cuda.synchronize()
            with open(os.environ["VLLM_PROFILE"], "a") as _f:
                _f.write(f"attn_qkv {_tqkv0.elapsed_time(_tqkv1):.3f}ms\n")

        # attention_impl is wrapped with @eager_break_during_capture: this is
        # where the breakable cudagraph capture breaks (the attention op runs
        # eagerly between captured graph segments).
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _tatt0 = torch.cuda.Event(enable_timing=True)
            _tatt1 = torch.cuda.Event(enable_timing=True)
            _tatt0.record()
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]
        # 注意力输出 o 位级捕获 (VLLM_TRITON_DIFFCAP2) — L2/L3 chunk0 前 2
        # 请求 + 最后 chunk (附3 更新39: run 35 指纹显示 L3 注意力在最后
        # chunk 首个分歧, 需捕获最后 chunk 的 L2/L3 o 与 chunk0 对照)。
        # L0 (draft) 捕获: draft 注意力 = cubin decode kernel — 判别
        # decode 注意力确定性 (更新41 假说 b)。
        # 附3 更新48b: rank 0 门控 — TP=2 下两 rank 写同一文件名互覆盖,
        # run 52/53 o3last/olast 对 = 跨 rank 混拼伪像 (99.63% "分歧" 实为
        # rank0-vs-rank1 的不同头分片)
        _rank0o = (
            (torch.distributed.get_rank() == 0)
            if torch.distributed.is_initialized()
            else True
        )
        if os.environ.get("VLLM_TRITON_DIFFCAP2") and _rank0o and self.prefix in (
            "model.layers.0.attn",
            "model.layers.2.attn",
            "model.layers.3.attn",
            "model.layers.36.attn",
        ):
            try:
                _p0 = int(positions.view(-1)[0].item())
                _l0 = self.prefix.endswith("layers.0.attn")
                _l3 = self.prefix.endswith("layers.3.attn")
                _l36 = self.prefix.endswith("layers.36.attn")
                if _l0:
                    # draft (decode 注意力): 每步局部 positions 从 0 起 —
                    # 捕获首步 (cap 2) 判别 cubin decode 确定性
                    _cd = getattr(self, "_diffcap2_odn", 0)
                    if _p0 == 0 and _cd < 2:
                        setattr(self, "_diffcap2_odn", _cd + 1)
                        torch.cuda.synchronize()
                        _slotd = (_cd + 1) % 2
                        torch.save(
                            {"n": _cd, "o": o.detach().contiguous().cpu()},
                            os.environ["VLLM_TRITON_DIFFCAP2"] + f".odraft{_slotd}.pt",
                        )
                elif _p0 == 0:
                    _fctx2 = get_forward_context()
                    _smd2 = _fctx2.attn_metadata.get(self.swa_cache_layer.prefix) if isinstance(_fctx2.attn_metadata, dict) else None
                    _sl2 = int(_smd2.seq_lens[0].item()) if _smd2 is not None and _smd2.seq_lens is not None else None
                    if _sl2 == 8188:
                        _cattr = (
                            "_diffcap2_o3n" if _l3
                            else ("_diffcap2_o36n" if _l36 else "_diffcap2_on")
                        )
                        _cn2 = getattr(self, _cattr, 0)
                        if _cn2 < 2:
                            setattr(self, _cattr, _cn2 + 1)
                            torch.cuda.synchronize()
                            _slot2 = (_cn2 + 1) % 2
                            _fname = (
                                f".o3_{_slot2}.pt" if _l3
                                else (f".o36_{_slot2}.pt" if _l36 else f".o{_slot2}.pt")
                            )
                            torch.save(
                                {"n": _cn2, "o": o.detach().contiguous().cpu()},
                                os.environ["VLLM_TRITON_DIFFCAP2"] + _fname,
                            )
                elif _p0 > 100000 and 1000 < o.shape[0] < 8188:
                    # 最后短 chunk (130838 ctx: 1961 行; chunk 15 为 8188 行,
                    # decode 步为 <100 行, 均须排除) — 环形 cap 4 保留
                    # 最近 2 = 暖对 (请求 3/4) (更新46: 分歧源 = 短 chunk)
                    _l3 = self.prefix.endswith("layers.3.attn")
                    _l36 = self.prefix.endswith("layers.36.attn")
                    _cattr = (
                        "_diffcap2_ol3n" if _l3
                        else ("_diffcap2_ol36n" if _l36 else "_diffcap2_oln")
                    )
                    _cnl = getattr(self, _cattr, 0)
                    if _cnl < 4:
                        setattr(self, _cattr, _cnl + 1)
                        torch.cuda.synchronize()
                        _slotl = (_cnl + 1) % 2
                        _fname = (
                            f".o3last{_slotl}.pt" if _l3
                            else (f".o36last{_slotl}.pt" if _l36 else f".olast{_slotl}.pt")
                        )
                        torch.save(
                            {"n": _cnl, "o": o.detach().contiguous().cpu()},
                            os.environ["VLLM_TRITON_DIFFCAP2"] + _fname,
                        )
            except Exception as _e3:
                print(f"[DIFFCAP2-o] err: {_e3}", flush=True)
        # 更新53d (G1): L2 verify 首步全量捕获 (VLLM_VFY_CAP) — 判别 L2
        # 注意力分歧源 (run 72 指纹: verify 首步 pos0=130665 出生层 = L2
        # attn_out; L0/L1 全同, 输入侧位级同已证)。捕获 hidden/qr/o +
        # SWA 索引/长度/有效性 + block_table + topk buffer — 判别
        # 输入张量跨请求不同 vs kernel 同输入异输出。rank0 + cap 4。
        if (
            os.environ.get("VLLM_VFY_CAP")
            and _rank0o
            and self.prefix == "model.layers.2.attn"
        ):
            try:
                _p0v = int(positions.view(-1)[0].item())
                if _p0v == 130665:
                    _cnv = getattr(self, "_vfycap_n", 0)
                    if _cnv < 12:
                        setattr(self, "_vfycap_n", _cnv + 1)
                        torch.cuda.synchronize()
                        _fctxv = get_forward_context()
                        _smdv = (
                            _fctxv.attn_metadata.get(self.swa_cache_layer.prefix)
                            if isinstance(_fctxv.attn_metadata, dict)
                            else None
                        )
                        _smd2v = (
                            _fctxv.attn_metadata.get(self.prefix)
                            if isinstance(_fctxv.attn_metadata, dict)
                            else None
                        )
                        _g = lambda t: t.detach().cpu() if t is not None else None
                        torch.save(
                            {
                                "n": _cnv,
                                "hidden": _g(hidden_states),
                                "qr": _g(qr),
                                "o": _g(o),
                                "positions": _g(positions),
                                "swa_indices": _g(
                                    getattr(_smdv, "decode_swa_indices", None)
                                ),
                                "swa_lens": _g(
                                    getattr(_smdv, "decode_swa_lens", None)
                                ),
                                "is_valid": _g(
                                    getattr(_smdv, "is_valid_token", None)
                                ),
                                "block_table": _g(
                                    getattr(_smd2v, "block_table", None)
                                ),
                                "topk_buffer": _g(
                                    getattr(self, "topk_indices_buffer", None)
                                ),
                            },
                            os.environ["VLLM_VFY_CAP"] + f".v{_cnv}.pt",
                        )
            except Exception as _e4:
                print(f"[VFY-CAP] err: {_e4}", flush=True)
        # 指纹: 注意力输出 (o_proj 前) — 与 attn_r 配对判别 o_proj 分歧
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jao
                torch.cuda.synchronize()
                _ov = o.detach().contiguous().view(torch.int16)
                _os = int(_ov.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                # 行级 bucket (行%8) + 首 8 行 h8 — 定位分歧位置
                _bk = [int(_ov[_i::8].sum(dtype=torch.int32).sum(dtype=torch.int64).item())
                       for _i in range(8)]
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jao.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "attn_o",
                        "prefix": self.prefix,
                        "pos0": int(positions.view(-1)[0].item()),
                        "bitsum": _os,
                        "bk": _bk,
                        "h8": [_ov.view(-1)[i].item() for i in range(8)],
                    }) + "\n")
            except Exception:
                pass
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _tatt1.record()
            torch.cuda.synchronize()
            with open(os.environ["VLLM_PROFILE"], "a") as _f:
                _f.write(f"attn_impl {_tatt0.elapsed_time(_tatt1):.3f}ms\n")
        if os.environ.get("VLLM_DUMP_HIDDEN") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _json
                torch.cuda.synchronize()
                _o_stats = o.detach().float().cpu()
                _o_pad_stats = o_padded.detach().float().cpu()
                with open(os.environ["VLLM_DUMP_HIDDEN"], "a") as _f:
                    _f.write(_json.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "layer": getattr(self, "_active_idx", -1),
                        "comp": "attn_o_pre_proj",
                        "shape": list(o.shape),
                        "nz": int((_o_stats != 0).sum().item()),
                        "mean": round(_o_stats.mean().item(), 6),
                        "std": round(_o_stats.std().item(), 6),
                        "max_abs": round(_o_stats.abs().max().item(), 6),
                        "o_padded_nz": int((_o_pad_stats != 0).sum().item()),
                        "o_padded_mean": round(_o_pad_stats.mean().item(), 6),
                        "impl": type(self.attention_impl).__name__,
                        "num_tokens": int(hidden_states.shape[0]),
                    }) + "\n")
            except Exception:
                pass
        if os.environ.get("VLLM_DUMP_HIDDEN_FULL") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _json
                torch.cuda.synchronize()
                _o_stats = o.detach().float().cpu()
                with open(os.environ["VLLM_DUMP_HIDDEN_FULL"], "a") as _f:
                    _f.write(_json.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "attn_o_pre_proj_full",
                        "shape": list(o.shape),
                        "data": _o_stats[:4].reshape(-1).tolist(),  # 4 tokens x heads*head_dim
                    }) + "\n")
            except Exception:
                pass

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _top0 = torch.cuda.Event(enable_timing=True)
            _top1 = torch.cuda.Event(enable_timing=True)
            _top0.record()
        _r = self._o_proj(o, positions)
        # o_proj 输出 r 位级捕获 (VLLM_TRITON_DIFFCAP2) — L2 step1 环形
        if os.environ.get("VLLM_TRITON_DIFFCAP2") and self.prefix == "model.layers.2.attn":
            try:
                _cn3 = getattr(self, "_diffcap2_rn", 0)
                if int(positions.view(-1)[0].item()) == 0:
                    _fctx3 = get_forward_context()
                    _smd3 = _fctx3.attn_metadata.get(self.swa_cache_layer.prefix) if isinstance(_fctx3.attn_metadata, dict) else None
                    _sl3 = int(_smd3.seq_lens[0].item()) if _smd3 is not None and _smd3.seq_lens is not None else None
                    if _sl3 == 8188 and _cn3 < 2:
                        self._diffcap2_rn = _cn3 + 1
                        torch.cuda.synchronize()
                        _slot3 = (_cn3 + 1) % 2
                        torch.save(
                            {"n": _cn3, "r": _r.detach().contiguous().cpu()},
                            os.environ["VLLM_TRITON_DIFFCAP2"] + f".r{_slot3}.pt",
                        )
            except Exception as _e4:
                print(f"[DIFFCAP2-r] err: {_e4}", flush=True)
        # 指纹: o_proj 输出 — 与 attn_o 配对判别 o_proj (wo_a/wo_b) 分歧
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jar
                torch.cuda.synchronize()
                _rv = _r.detach().contiguous().view(torch.int16)
                _rs = int(_rv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jar.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "attn_r",
                        "prefix": self.prefix,
                        "pos0": int(positions.view(-1)[0].item()),
                        "bitsum": _rs,
                        "h8": [_rv.view(-1)[i].item() for i in range(8)],
                    }) + "\n")
            except Exception:
                pass
        if os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
            _top1.record()
            torch.cuda.synchronize()
            with open(os.environ["VLLM_PROFILE"], "a") as _f:
                _f.write(f"attn_oproj {_top0.elapsed_time(_top1):.3f}ms\n")
        return _r

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jp3
                torch.cuda.synchronize()
                _rec = {
                    "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                    "comp": "p3way",
                    "prefix": self.prefix,
                    "ntok": int(hidden_states.shape[0]),
                }
                for _name, _t in (("qr_kv", qr_kv), ("kv_score", kv_score),
                                  ("indexer_kv_score", indexer_kv_score),
                                  ("indexer_weights", indexer_weights)):
                    _v = _t.detach().contiguous().view(torch.int16)
                    _rec[_name] = int(_v.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                    _rec[_name + "_h8"] = [_v.view(-1)[i].item() for i in range(4)]
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jp3.dumps(_rec) + "\n")
            except Exception:
                pass

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # 位级捕获 (VLLM_TRITON_DIFFCAP2=<path>): L2 + step1 (seq_len 8188)
        # 前 2 请求 — hidden (层输入) / qr (wq_a 输出) / kv (fused_wqa_wkv
        # kv 列) — 定位写入分歧的源头 (附3 更新28: q 净 kv 脏)。
        if (
            os.environ.get("VLLM_TRITON_DIFFCAP2")
            and (
                (torch.distributed.get_rank() == 0)
                if torch.distributed.is_initialized()
                else True
            )
            and self.prefix in (
                "model.layers.2.attn",
                "model.layers.3.attn",
            )
        ):
            try:
                _l3 = self.prefix.endswith("layers.3.attn")
                _cattr = "_diffcap2_r3n" if _l3 else "_diffcap2_n"
                _cn = getattr(self, _cattr, 0)
                if int(positions.view(-1)[0].item()) == 0:
                    _sm_len = None
                    if isinstance(attn_metadata, dict):
                        _smd = attn_metadata.get(self.swa_cache_layer.prefix)
                        _sm_len = int(_smd.seq_lens[0].item()) if _smd is not None and _smd.seq_lens is not None else None
                    if _sm_len == 8188 and _cn < 2:
                        setattr(self, _cattr, _cn + 1)
                        torch.cuda.synchronize()
                        _slot = (_cn + 1) % 2  # 环形: 保留最近 2 个
                        _fname = f".ring3_{_slot}.pt" if _l3 else f".ring{_slot}.pt"
                        torch.save(
                            {
                                "n": _cn,
                                "hidden": hidden_states.detach().contiguous().cpu(),
                                "qr": qr.detach().contiguous().cpu(),
                                "kv": kv.detach().contiguous().cpu(),
                            },
                            os.environ["VLLM_TRITON_DIFFCAP2"] + _fname,
                        )
            except Exception as _e2:
                print(f"[DIFFCAP2] err: {_e2}", flush=True)

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                    try:
                        import json as _jqr
                        torch.cuda.synchronize()
                        _rv = qr.detach().contiguous().view(torch.int16)
                        _rs = int(_rv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jqr.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "wq_b_qr",
                                "prefix": self.prefix,
                                "pos0": int(positions.view(-1)[0].item()),
                                "qr_shape": list(qr.shape),
                                "qrsum": _rs,
                                "h8": [_rv.view(-1)[i].item() for i in range(8)],
                            }) + "\n")
                    except Exception:
                        pass
                import os as _os2
                _prof2 = _os2.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
                if _prof2:
                    _w0 = torch.cuda.Event(enable_timing=True)
                    _w1 = torch.cuda.Event(enable_timing=True)
                    _w0.record()
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                    try:
                        import json as _jqw
                        torch.cuda.synchronize()
                        _qv = q.detach().contiguous().view(torch.int16)
                        _qs = int(_qv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jqw.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "attn_q_wqb",
                                "prefix": self.prefix,
                                "pos0": int(positions.view(-1)[0].item()),
                                "qsum": _qs,
                                "h8": [_qv.view(-1)[i].item() for i in range(8)],
                            }) + "\n")
                    except Exception:
                        pass
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                if _prof2:
                    _w1.record()
                    torch.cuda.synchronize()
                    with open(_os2.environ["VLLM_PROFILE"], "a") as _f2:
                        _f2.write(f"wqbkv r{self.compress_ratio} {_w0.elapsed_time(_w1):.3f}ms\n")
                return q

            # 3-way overlap (matches TRT-LLM PR #14142 Level 1): default runs
            # wq_b+kv_insert; slot [0] runs the full indexer; slot [1] runs the
            # MLA compressor. Slot [2] is reserved for the indexer's inner
            # overlap. ROCm (aux_streams is None) falls back to sequential.
            # Measured 2026-08-21: with the fast Triton indexer the branches are
            # wqbkv 0.9ms / indexer 7.5ms / compressor 0.1ms but the parallel
            # block's wall time is ~30ms (join/stream-queue overhead) — the
            # overlap saves <1ms and costs ~22ms. VLLM_SKIP_ATTN_OVERLAP=1
            # runs the branches sequentially on the default stream.
            # NOTE(2026-08-20): VLLM_INDEXER_STRIDE>1 (skip rescoring on some
            # prefill chunks) is semantically broken for long context — stale
            # topk selections leave newly-written compressed KV blocks unseen
            # (2needle 1M n=10 all-fail at stride=4). The correct fix is an
            # incremental scoring kernel (score only new blocks + merge).
            import os as _os
            _overlap_enabled = (
                aux_streams is not None
                and _os.environ.get("VLLM_SKIP_ATTN_OVERLAP", "0") != "1"
            )
            _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
            if _prof:
                _pp0 = torch.cuda.Event(enable_timing=True)
                _pp1 = torch.cuda.Event(enable_timing=True)
                _pp0.record()
            if self.skip_topk:
                # #71 IndexCache: 跳过本层 topk 重算, 复用前 F 层的共享缓冲
                _indexer_fn = lambda: None
            else:
                _indexer_fn = lambda: indexer(
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    self.indexer_rotary_emb,
                )
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    _indexer_fn,
                    lambda: _timed_compressor(
                        compressor, kv_score, positions, self.rotary_emb,
                        self.compress_ratio,
                    ),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=_overlap_enabled,
            )
            if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                try:
                    import json as _jq
                    torch.cuda.synchronize()
                    _qv = q.detach().contiguous().view(torch.int16)
                    _qs = int(_qv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jq.dumps({
                            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                            "comp": "attn_q",
                            "prefix": self.prefix,
                            "pos0": int(positions.view(-1)[0].item()),
                            "qsum": _qs,
                            "h8": [_qv.view(-1)[i].item() for i in range(8)],
                        }) + "\n")
                except Exception:
                    pass
            if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                try:
                    import json as _jtopk
                    torch.cuda.synchronize()
                    _nt = hidden_states.shape[0]
                    _tk = self.topk_indices_buffer[:min(_nt, 16)].detach().to(torch.int64)
                    _tk8 = [_tk.view(-1)[i].item() for i in range(min(8, _tk.numel()))]
                    _tkall = self.topk_indices_buffer[:min(_nt, self.topk_indices_buffer.shape[0])].detach().to(torch.int64)
                    _tksum = int(_tkall.sum().item())
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jtopk.dumps({
                            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                            "comp": "idx_topk",
                            "prefix": self.prefix,
                            "pos0": int(positions.view(-1)[0].item()),
                            "tksum": _tksum,
                            "tk8": _tk8,
                        }) + "\n")
                except Exception:
                    pass
            if _prof:
                _pp1.record()
                torch.cuda.synchronize()
                with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                    _f.write(f"parallel3way r{self.compress_ratio} {_pp0.elapsed_time(_pp1):.3f}ms\n")
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                import os as _os2
                _prof2 = _os2.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
                if _prof2:
                    _w0 = torch.cuda.Event(enable_timing=True)
                    _w1 = torch.cuda.Event(enable_timing=True)
                    _w0.record()
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                    try:
                        import json as _jqc128
                        torch.cuda.synchronize()
                        _qv = q.detach().contiguous().view(torch.int16)
                        _qs = int(_qv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jqc128.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "attn_q_c128",
                                "prefix": self.prefix,
                                "pos0": int(positions.view(-1)[0].item()),
                                "qsum": _qs,
                                "h8": [_qv.view(-1)[i].item() for i in range(8)],
                            }) + "\n")
                    except Exception:
                        pass
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                if _prof2:
                    _w1.record()
                    torch.cuda.synchronize()
                    with open(_os2.environ["VLLM_PROFILE"], "a") as _f2:
                        _f2.write(f"wqbkv r{self.compress_ratio} {_w0.elapsed_time(_w1):.3f}ms\n")
                return q

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            # 指纹: kv 输入 (fused_wqa_wkv GEMM 输出) — 写入侧因果链的源头
            if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                try:
                    import json as _jkvin
                    torch.cuda.synchronize()
                    _vv = kv.detach().contiguous().view(torch.int16)
                    _vs = int(_vv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jkvin.dumps({
                            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                            "comp": "kv_in",
                            "prefix": self.prefix,
                            "pos0": int(positions.view(-1)[0].item()),
                            "bitsum": _vs,
                            "h8": [_vv.view(-1)[i].item() for i in range(8)],
                        }) + "\n")
                except Exception:
                    pass
            q = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )
            # 指纹: 刚写入的 SWA cache 行 (flat slot, 584B/token packed uint8)
            if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
                try:
                    import json as _jsww
                    torch.cuda.synchronize()
                    _sm = swa_metadata.slot_mapping
                    _ok = _sm >= 0
                    if _ok.any():
                        # flat token 行: 原始 [num_blocks, block_size, 584] → [-1, 584]
                        _swa_flat = swa_kv_cache.reshape(-1, swa_kv_cache.shape[-1])
                        _nrows_tot = _swa_flat.shape[0]
                        _oob = int((_sm[_ok] >= _nrows_tot).sum().item())
                        _idx = _sm[_ok].to(torch.int64).clamp(0, _nrows_tot - 1)
                        _rows = _swa_flat[_idx]
                        _bitsum = int(_rows.sum(dtype=torch.int64).item())
                        _h8 = [_rows.view(-1)[i].item() for i in range(8)]
                        # 区域指纹: [0:448) fp8 数据 / [448:576) rope bf16 /
                        # [576:584) UE8M0 scale 字节
                        _w = _rows.shape[-1]
                        _bsd = int(_rows[..., :448].sum(dtype=torch.int64).item())
                        _bsr = int(_rows[..., 448:576].sum(dtype=torch.int64).item())
                        _bsc = int(_rows[..., 576:_w].sum(dtype=torch.int64).item())
                        # 行分桶指纹: 行号 %8 → 8 桶和; 尾部行残留 vs 均匀翻转判别
                        _nb = _rows.shape[0]
                        _bk = [int(_rows[_i::8].sum(dtype=torch.int64).item())
                               for _i in range(8)]
                        # 逐行指纹: bucket 3 (已知脏桶) 的每行和 — 锁定具体行
                        _r3 = _rows[3::8]
                        _s3 = _r3.sum(dim=1).to(torch.int64).tolist() if _r3.shape[0] else []
                        # 最后一行全字节 + 槽号 + 倒数第二行 (干净参照)
                        _slotl = int(_sm[_ok][-1].item())
                        _rowl = _rows[-1].to(torch.int64).tolist()
                        _rowp = _rows[-2].to(torch.int64).tolist() if _rows.shape[0] > 1 else []
                        # 张量布局 + kernel append-scale 位置的 8 字节
                        _tshape = list(swa_kv_cache.shape)
                        _tstride = list(swa_kv_cache.stride())
                        try:
                            _b0 = (_slotl // 64) * _tstride[0] if len(_tstride) > 0 else 0
                            _scbl = swa_kv_cache.reshape(-1)[
                                _b0 + 64 * 576 + (_slotl % 64) * 8:
                                _b0 + 64 * 576 + (_slotl % 64) * 8 + 8
                            ].to(torch.int64).tolist()
                        except Exception:
                            _scbl = []
                        _h8l = [_rows.view(-1)[_i].item()
                                for _i in range((_nb - 1) * _w, min(_nb * _w, (_nb - 1) * _w + 8))]
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jsww.dumps({
                                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                "comp": "swa_write",
                                "prefix": self.prefix,
                                "pos0": int(positions.view(-1)[0].item()),
                                "nrows": int(_ok.sum().item()),
                                "oob": _oob,
                                "bitsum": _bitsum,
                                "bd": _bsd, "br": _bsr, "bc": _bsc,
                                "bk": _bk,
                                "s3": _s3,
                                "slotl": _slotl,
                                "rowl": _rowl,
                                "rowp": _rowp,
                                "tshape": _tshape,
                                "tstride": _tstride,
                                "scbl": _scbl,
                                "h8": _h8,
                                "h8l": _h8l,
                            }) + "\n")
                except Exception as _esw:
                    try:
                        import json as _jesw
                        with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                            _f.write(_jesw.dumps({"comp": "swa_write_err", "err": str(_esw),
                                                  "prefix": self.prefix}) + "\n")
                    except Exception:
                        pass
            return q

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        compressor = self.compressor
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jiw
                torch.cuda.synchronize()
                _w = indexer_weights.detach().contiguous().view(torch.int16)
                _ws = int(_w.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jiw.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "idx_weights",
                        "prefix": self.prefix,
                        "pos0": int(positions.view(-1)[0].item()),
                        "wsum": _ws,
                        "h8": [_w.view(-1)[i].item() for i in range(8)],
                    }) + "\n")
            except Exception:
                pass

        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        import os as _os
        _prof = _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing()
        if _prof:
            _pc0 = torch.cuda.Event(enable_timing=True)
            _pc1 = torch.cuda.Event(enable_timing=True)
            _pc0.record()
        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            None
            if _os.environ.get("VLLM_SKIP_ATTN_OVERLAP", "0") == "1"
            else self.aux_stream,
        )
        if _prof:
            _pc1.record()
            torch.cuda.synchronize()
            with open(_os.environ["VLLM_PROFILE"], "a") as _f:
                _f.write(f"compressor+wq r{self.compress_ratio} {_pc0.elapsed_time(_pc1):.3f}ms\n")

        # GEMM-based prefill scoring (VLLM_INDEXER_GEMM=1): additionally
        # prepare the bf16 RoPE'd Q and the unfolded per-head weights
        # (q per-token scale x softmax x head scale). The paged kernel still
        # serves decode steps; indexer_gemm_prefill replaces the prefill
        # scoring inside SparseAttnIndexer.forward_cuda.
        q_rope_bf16: torch.Tensor | None = None
        weights_eff: torch.Tensor | None = None
        if os.environ.get("VLLM_INDEXER_GEMM") == "1" and not self.use_fp4_kv:
            q_bf, _ = self.wq_b(qr)
            q_bf = q_bf.view(-1, self.n_head, self.head_dim)
            q_rope_bf16, _ = rotary_emb(positions, q_bf)
            q_scale = q_rope_bf16.abs().amax(dim=-1, keepdim=True).div(448.0)
            weights_eff = (indexer_weights * q_scale.squeeze(-1)) * (
                self.softmax_scale * self.n_head**-0.5
            )
        return self.indexer_op(
            hidden_states, q_quant, k, weights, q_rope_bf16, weights_eff
        )


def _timed_compressor(compressor, kv_score, positions, rotary_emb, ratio) -> None:
    import os as _os
    if _os.environ.get("VLLM_PROFILE") and not torch.cuda.is_current_stream_capturing():
        _c0 = torch.cuda.Event(enable_timing=True)
        _c1 = torch.cuda.Event(enable_timing=True)
        _c0.record()
        compressor(kv_score, positions, rotary_emb)
        _c1.record()
        torch.cuda.synchronize()
        with open(_os.environ["VLLM_PROFILE"], "a") as _f:
            _f.write(f"mlacompress r{ratio} {_c0.elapsed_time(_c1):.3f}ms\n")
    else:
        compressor(kv_score, positions, rotary_emb)
