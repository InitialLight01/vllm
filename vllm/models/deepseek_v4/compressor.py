# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_pytorch import (
    requantize_kv_cache_fp8e4nv,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jprobe
                torch.cuda.synchronize()
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jprobe.dumps({"rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                                            "comp": "comp_fwd_probe", "pos0": int(positions.view(-1)[0].item()),
                                            "shape": list(kv_score.shape)}) + "\n")
            except Exception:
                pass
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jmt
                torch.cuda.synchronize()
                _rec = {
                    "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                    "comp": "comp_meta_probe",
                    "prefix": self.prefix,
                    "pos0": int(positions.view(-1)[0].item()),
                    "meta_type": type(attn_metadata).__name__,
                    "cache_bitsum": None,
                    "cache_h8": None,
                }
                try:
                    _cv = self.state_cache.kv_cache.detach().view(torch.uint8)
                    _rec["cache_bitsum"] = int(_cv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                    _rec["cache_h8"] = [_cv.view(-1)[i].item() for i in range(8)]
                except Exception as _e2:
                    _rec["cache_err"] = str(_e2)
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jmt.dumps(_rec) + "\n")
            except Exception as _e3:
                try:
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f2:
                        _f2.write(_jmt.dumps({"comp": "comp_probe_err", "err": str(_e3)}) + "\n")
                except Exception:
                    pass
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )
        # 指纹: save_partial_states 刚写入的 state cache 行 (compress kernel 的输入)
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jstw
                torch.cuda.synchronize()
                _ss = slot_mapping
                _sc = state_cache.detach()
                _sflat = _sc.reshape(-1, _sc.shape[-1])
                _ok2 = _ss >= 0
                if _ok2.any():
                    _idx2 = _ss[_ok2].to(torch.int64).clamp(0, _sflat.shape[0] - 1)
                    _rows2 = _sflat[_idx2].view(torch.int32)
                    _bitsum2 = int(_rows2.sum(dtype=torch.int64).item())
                    _h82 = [_rows2.view(-1)[i].item() for i in range(8)]
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jstw.dumps({
                            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                            "comp": "state_write",
                            "prefix": self.prefix,
                            "pos0": int(positions.view(-1)[0].item()),
                            "nrows": int(_ok2.sum().item()),
                            "bitsum": _bitsum2,
                            "h8": _h82,
                        }) + "\n")
            except Exception:
                pass

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        # SM80 / FORCE_SM80: use Triton compressor — CuTeDSL is compiled for
        # the real hardware arch and its output format may differ from what the
        # pure-Triton decode kernel expects.
        compress_norm_rope_store_fn: Any
        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and not current_platform.is_sm80_context()
        ):
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA (SM90+) uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jpre
                torch.cuda.synchronize()
                _cv = kv_cache.detach().contiguous().view(torch.uint8)
                _bitsum = int(_cv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jpre.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "comp_pre_call",
                        "prefix": self.prefix,
                        "pos0": int(positions.view(-1)[0].item()),
                        "pre_bitsum": _bitsum,
                    }) + "\n")
            except Exception:
                pass
        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
        # comp 写行位级捕获 (VLLM_TRITON_DIFFCAP=<path>): token 序提取, 全请求跨 chunk 累积
        # (slot_mapping[t] → 576B 行 + 块尾 8B scale) — L2 compressor, rank0 专用 (防 TP 双写)
        # 环形: 请求 N 的完整捕获在请求 N+1 的 chunk0 到达时落盘到 .cw{(N+1)%2}.pt
        if os.environ.get("VLLM_TRITON_DIFFCAP") and self.prefix == "model.layers.2.attn.compressor":
            try:
                _rank0 = (torch.distributed.get_rank() == 0) if torch.distributed.is_initialized() else True
                if _rank0:
                    _cwn = getattr(self, "_diffcap_w_n", 0)
                    _p0 = int(positions.view(-1)[0].item())
                    _nn = k_cache_metadata.slot_mapping.shape[0]
                    _okm = k_cache_metadata.slot_mapping >= 0
                    if _nn > 1000 and bool(_okm.any()):
                        torch.cuda.synchronize()
                        _cb = kv_cache.shape[1]
                        _k576 = kv_cache.detach().as_strided(
                            (kv_cache.shape[0], _cb * 576),
                            (kv_cache.stride(0), 1),
                        )
                        _ksc = kv_cache.detach().as_strided(
                            (kv_cache.shape[0], _cb * 584), (kv_cache.stride(0), 1)
                        )[:, _cb * 576 : _cb * 576 + _cb * 8]
                        _ksmv = k_cache_metadata.slot_mapping[_okm]
                        _b = _ksmv // _cb
                        _p = _ksmv % _cb
                        _d = _k576[_b].view(-1, _cb, 576)
                        _rows = _d[torch.arange(_ksmv.numel(), device=_ksmv.device), _p]
                        _sc = _ksc[_b].view(-1, _cb, 8)
                        _rowsc = _sc[torch.arange(_ksmv.numel(), device=_ksmv.device), _p]
                        _pend = getattr(self, "_diffcap_pend", None)
                        if _p0 == 0:
                            if _pend is not None and _pend["wrows"].shape[0] > 10000:
                                # 前一请求已累积完整 → 落盘到其槽位, 开启新请求
                                _slot = (_pend["n"] + 1) % 2
                                torch.save(_pend, os.environ["VLLM_TRITON_DIFFCAP"] + f".cw{_slot}.pt")
                                print(
                                    f"[DIFFCAP-cw] flush n={_pend['n']} rows={_pend['wrows'].shape[0]} "
                                    f"slot={_slot}", flush=True)
                                self._diffcap_w_n = _cwn + 1
                            elif _pend is not None:
                                # 同请求 chunk0 重放 (verify 等) → 丢弃残片重启
                                print(f"[DIFFCAP-cw] restart n={_cwn} (prev rows={_pend['wrows'].shape[0]})",
                                      flush=True)
                            else:
                                self._diffcap_w_n = _cwn + 1
                            self._diffcap_pend = {
                                "n": _cwn,
                                "wrows": _rows.detach().cpu(),
                                "wrowsc": _rowsc.detach().cpu(),
                                "slots": _ksmv.detach().cpu(),
                            }
                            print(f"[DIFFCAP-cw] new req n={_cwn} rows={_rows.shape[0]} cb={_cb}", flush=True)
                        elif _pend is not None:
                            _pend["wrows"] = torch.cat([_pend["wrows"], _rows.detach().cpu()], 0)
                            _pend["wrowsc"] = torch.cat([_pend["wrowsc"], _rowsc.detach().cpu()], 0)
                            _pend["slots"] = torch.cat([_pend["slots"], _ksmv.detach().cpu()], 0)
            except Exception as _ecw2:
                print(f"[DIFFCAP-cw] err: {_ecw2}", flush=True)
        # 指纹: 刚写入的压缩 KV cache 行 (k_cache_metadata.slot_mapping 为
        # flat slot: block = slot // block_size, 行 = block*stride0 + off*width)
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jcw
                torch.cuda.synchronize()
                _ks = k_cache_metadata.slot_mapping
                _kc = kv_cache.detach()
                # 压缩 cache 与 KV blocks 共享物理张量, 可能非连续 → reshape
                _kflat = _kc.reshape(-1, _kc.shape[-1])
                _ok = _ks >= 0
                if _ok.any():
                    _idx = _ks[_ok].to(torch.int64).clamp(0, _kflat.shape[0] - 1)
                    _rows = _kflat[_idx]
                    _bitsum = int(_rows.sum(dtype=torch.int64).item())
                    _h8 = [_rows.view(-1)[i].item() for i in range(8)]
                    _w = _rows.shape[-1]
                    _bsd = int(_rows[..., :448].sum(dtype=torch.int64).item())
                    _bsr = int(_rows[..., 448:576].sum(dtype=torch.int64).item())
                    _bsc = int(_rows[..., 576:_w].sum(dtype=torch.int64).item())
                    _nb = _rows.shape[0]
                    _bk = [int(_rows[_i::8].sum(dtype=torch.int64).item())
                           for _i in range(8)]
                    _r6 = _rows[6::8]
                    _s6 = _r6.sum(dim=1).to(torch.int64).tolist() if _r6.shape[0] else []
                    _slotl = int(_ks[_ok][-1].item())
                    _rowl = _rows[-1].to(torch.int64).tolist()
                    _rowp = _rows[-2].to(torch.int64).tolist() if _rows.shape[0] > 1 else []
                    _h8l = [_rows.view(-1)[_i].item()
                            for _i in range((_nb - 1) * _w, min(_nb * _w, (_nb - 1) * _w + 8))]
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jcw.dumps({
                            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                            "comp": "comp_write",
                            "prefix": self.prefix,
                            "pos0": int(positions.view(-1)[0].item()),
                            "nrows": int(_ok.sum().item()),
                            "bitsum": _bitsum,
                            "bd": _bsd, "br": _bsr, "bc": _bsc,
                            "bk": _bk,
                            "s6": _s6,
                            "slotl": _slotl,
                            "rowl": _rowl,
                            "rowp": _rowp,
                            "h8": _h8,
                            "h8l": _h8l,
                        }) + "\n")
            except Exception as _ecw:
                try:
                    import json as _jecw
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                        _f.write(_jecw.dumps({"comp": "comp_write_err", "err": str(_ecw),
                                              "prefix": self.prefix}) + "\n")
                except Exception:
                    pass
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jcomp
                torch.cuda.synchronize()
                _cv = kv_cache.detach().contiguous().view(torch.uint8)
                _bitsum = int(_cv.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                _h8 = [_cv.view(-1)[i].item() for i in range(8)]
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jcomp.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "comp_kv_cache",
                        "prefix": self.prefix,
                        "pos0": int(positions.view(-1)[0].item()),
                        "shape": list(kv_cache.shape),
                        "bitsum": _bitsum,
                        "h8": _h8,
                    }) + "\n")
            except Exception as _ekv:
                try:
                    import json as _jekv
                    with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f3:
                        _f3.write(_jekv.dumps({"comp": "comp_tail_err", "err": str(_ekv),
                                               "prefix": self.prefix}) + "\n")
                except Exception:
                    pass
