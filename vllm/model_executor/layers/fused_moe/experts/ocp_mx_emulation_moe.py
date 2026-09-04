# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OCP MX quantization emulation for MoE.

This file implements OCP MX (MXFP4/MXFP6) emulation for MoE in case the
hardware used does not natively support OCP MX MoE.

Weights are dequantized on the fly during each forward, we fall back to calling
`TritonExperts` using BF16, and fake OCP MX quantize-dequantize
is applied on activations via `moe_kernel_quantize_input`.
"""

import json as _json_mod
import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

# debug counter for VLLM_DUMP_SILU (per-process)
_EMU_DUMP_N = [0]
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp4_dequant_pytorch import (
    dq_mxfp4_pytorch,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_dequant_triton import (
    dq_mxfp4_triton,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_Scheme,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class OCP_MXQuantizationEmulationTritonExperts(TritonExperts):
    """
    Extension of TritonExperts to support emulated OCP MX MoE experts.

    It may be used for OCP MX (MXFP4/MXFP6) models when the device does not
    have native support for these dtypes.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using OCP_MXQuantizationEmulationTritonExperts MOE backend. This"
            " will dequantize weights on the fly and may be slower than native"
            " quantized MOE. Consider using a device with native OCP MX"
            " quantization support for better performance."
        )

        self.ocp_mx_scheme = quant_config.ocp_mx_scheme
        assert self.ocp_mx_scheme is not None, (
            "ocp_mx_scheme must be set in quant_config for"
            " OCP_MXQuantizationEmulationTritonExperts"
        )

        # `TritonExperts.apply` expects pre-dequantized weights,
        # which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.quantization_emulation = True

        # ---- 夜4 Phase2: 冷专家卸载状态 (VLLM_MOE_COLD_OFFLOAD=1 开启) ----------
        # 设计见 docs/A800_MEMORY_OFFLOAD_SURVEY_PLAN_2026-09-04.md 附录。
        # 权重只读 → LRU 无写回; w1/w2 收缩为 [hot_K + M] (热表重排 0..K-1,
        # 冷槽 K..K+M-1); scales 不收缩 (按逻辑 id 索引, 仅 ~3GB 总量)。
        # H2D 为精确复制 → 与全 GPU 位级一致。
        self._off_enabled = os.environ.get("VLLM_MOE_COLD_OFFLOAD", "0") == "1"
        self._off_ready = False
        self._off_hot_k = int(os.environ.get("VLLM_MOE_HOT_K", "128"))
        self._off_slots = int(os.environ.get("VLLM_MOE_LRU_SLOTS", "32"))
        # host 侧冷专家权重 (pinned), key = 逻辑本地 id
        self._off_host_w1: dict[int, torch.Tensor] | None = None
        self._off_host_w2: dict[int, torch.Tensor] | None = None
        # GPU 槽位状态
        self._off_slot_of: dict[int, int] | None = None  # 逻辑 id -> 槽
        self._off_slot_free: list[int] | None = None
        self._off_slot_lru: list[int] | None = None      # 槽 (最近使用在尾)
        # 逻辑 id -> 物理行 (hot_pos 或 K+slot)
        self._off_phys: dict[int, int] | None = None
        self._off_k_hot_actual: int = 0

        if self.ocp_mx_scheme in {
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
        }:
            # Weight has to be dequantized for mxfp4 emulation.
            if os.environ.get("VLLM_EMU_FP8_ACT"):
                # Experiment: emulate DeepGEMM's FP8 activation QDQ instead
                # of MXFP4 (which is ~2x coarser: FP4 1-bit mantissa vs FP8
                # E4M3 3-bit).  Tests the "training-distribution alignment"
                # hypothesis — if FP8-act matches DeepGEMM results, the
                # accuracy gap is activation-quantization format, not BF16.
                logger.warning_once(
                    "VLLM_EMU_FP8_ACT=1: activation QDQ switched to FP8 "
                    "(DeepGEMM-compatible) instead of MXFP4."
                )
                self._quant_dtype = current_platform.fp8_dtype()
            else:
                self._quant_dtype = "mxfp4"
        elif self.ocp_mx_scheme in [
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3,
            OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2,
            OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3,
        ]:
            self._quant_dtype = "mxfp6"
        elif self.ocp_mx_scheme in [
            OCP_MX_Scheme.w_mxfp4_a_fp8,
            OCP_MX_Scheme.w_mxfp6_e3m2_a_fp8,
        ]:
            self._quant_dtype = current_platform.fp8_dtype()

    def _offload_setup(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        """首次 apply 时执行: 热/冷分区 + 收缩 GPU 张量 + host pinned 拷贝。

        要求调用环境无 CUDA graph 捕获 (本类在 cudagraph NONE 配置下运行)。
        """
        if self._off_ready:
            return
        self._off_ready = True
        n_local = int(w1.shape[0])
        # 热表: 默认 = 本地 id 前缀 K (占位); 画像后经 VLLM_MOE_HOT_TABLE
        # (<path> JSON: {"local_hot": [ids]}) 覆盖
        hot_ids = list(range(min(self._off_hot_k, n_local)))
        hot_table = os.environ.get("VLLM_MOE_HOT_TABLE")
        if hot_table:
            with open(hot_table) as _f:
                hot_ids = [int(e) for e in _json_mod.load(_f).get("local_hot", [])]
        hot_ids = [e for e in hot_ids if 0 <= e < n_local]
        hot_set = set(hot_ids)
        cold_ids = [e for e in range(n_local) if e not in hot_set]
        k_hot = len(hot_ids)
        m_slots = min(self._off_slots, len(cold_ids))
        if self._off_hot_k != k_hot:
            logger.warning(
                "COLD_OFFLOAD: hot_K=%d (table=%d cold=%d slots=%d)",
                self._off_hot_k, k_hot, len(cold_ids), m_slots,
            )

        # host: pinned 冷专家权重拷贝 (逐专家; 必须同步 D2H 后再 pin —
        # non_blocking D2H 与 pin_memory 并发会读到未落地数据, 2026-09-05 实测竞态)
        self._off_host_w1 = {
            e: w1[e].detach().to("cpu").pin_memory() for e in cold_ids
        }
        self._off_host_w2 = {
            e: w2[e].detach().to("cpu").pin_memory() for e in cold_ids
        }

        # GPU: 收缩张量 [k_hot + m_slots, ...]
        dev = w1.device
        new_w1 = torch.empty(
            (k_hot + m_slots, *w1.shape[1:]), dtype=w1.dtype, device=dev
        )
        new_w2 = torch.empty(
            (k_hot + m_slots, *w2.shape[1:]), dtype=w2.dtype, device=dev
        )
        for pos, e in enumerate(hot_ids):
            new_w1[pos].copy_(w1[e], non_blocking=True)
            new_w2[pos].copy_(w2[e], non_blocking=True)
        # 槽位行初始置零 (未加载状态由 LRU dict 判定, 内容无意义)
        # 重绑定 Parameter.data — 调用方 (fused_moe_modular_method) 每步传
        # layer.w13_weight 本对象, data 重绑定后自动生效
        w1.data = new_w1
        w2.data = new_w2

        self._off_phys = {e: pos for pos, e in enumerate(hot_ids)}
        self._off_k_hot_actual = k_hot
        self._off_slot_of = {}
        self._off_slot_free = list(range(m_slots))
        self._off_slot_lru = []

    def _off_ensure(self, eid: int, w1: torch.Tensor, w2: torch.Tensor) -> int:
        """确保冷专家 eid 常驻, 返回其物理槽行; 热专家返回 hot_pos。"""
        phys = self._off_phys.get(eid)
        if phys is not None:
            return phys
        slot = self._off_slot_of.get(eid)
        if slot is not None:
            # LRU touch
            self._off_slot_lru.remove(slot)
            self._off_slot_lru.append(slot)
            return self._off_k_hot_actual + slot
        # 分配槽 (空闲或淘汰 LRU)
        if self._off_slot_free:
            slot = self._off_slot_free.pop()
        else:
            slot = self._off_slot_lru.pop(0)
            self._off_slot_of = {
                e: s for e, s in self._off_slot_of.items() if s != slot
            }
        row = slot  # 物理行 = K + slot
        # 同步 H2D (正确性先行; 预取优化后置)
        w1[row].copy_(self._off_host_w1[eid], non_blocking=True)
        w2[row].copy_(self._off_host_w2[eid], non_blocking=True)
        self._off_slot_of[eid] = slot
        self._off_slot_lru.append(slot)
        return self._off_k_hot_actual + slot

    def _off_phys_rows(self, chunk_ids, w1: torch.Tensor, w2: torch.Tensor):
        """chunk 逻辑 id 列表 → 物理行索引列表 (含 ensure-resident)。"""
        return [
            self._off_ensure(int(eid), w1, w2)
            for eid in chunk_ids
        ]

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self._quant_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key,
        activation_key,
    ) -> bool:
        # This class is used for emulation only - the oracle selects it
        # directly rather than via quant scheme matching.
        return True

    def _dequantize_weights(
        self,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize weights based on the OCP MX scheme.

        Uses pure-PyTorch FP4 unpack (no ``amd-quark`` dependency) for
        ``w_mxfp4`` schemes.
        """
        if self.ocp_mx_scheme.startswith("w_mxfp4"):  # type: ignore[union-attr]
            return dq_mxfp4_pytorch(w, w_scale, dtype)
        elif self.ocp_mx_scheme.startswith("w_mxfp6_e3m2"):  # type: ignore[union-attr]
            return dequant_mxfp6(w, w_scale, quant_dtype="fp6_e3m2", float_dtype=dtype)
        elif self.ocp_mx_scheme.startswith("w_mxfp6_e2m3"):  # type: ignore[union-attr]
            return dequant_mxfp6(w, w_scale, quant_dtype="fp6_e2m3", float_dtype=dtype)
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={self.ocp_mx_scheme}")

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply emulated quantized MoE computation.

        Dequantizes only the routed-to experts in **chunks** and calls the
        parent forward once per chunk, accumulating into ``output``.  This
        bounds peak temporary memory: prefill activating all 256 experts
        would otherwise need a ~15 GB BF16 tensor; chunking (≤16 experts)
        keeps it under ~1 GB.
        """
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        # 夜4 Phase0: 专家激活分布画像 — 本地 id 逐 token dump (VLLM_DUMP_EXPERT_IDS)
        if os.environ.get("VLLM_DUMP_EXPERT_IDS") and not torch.cuda.is_current_stream_capturing():
            try:
                torch.cuda.synchronize()
                _ids = topk_ids.detach().reshape(-1).tolist()
                with open(os.environ["VLLM_DUMP_EXPERT_IDS"], "a") as _f:
                    _f.write(_json_mod.dumps({
                        "rank": torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else -1,
                        "layer": getattr(self, "_dbg_layer_idx", -1),
                        "T": int(topk_ids.shape[0]),
                        "K": int(topk_ids.shape[1]),
                        "ids": _ids,
                    }) + "\n")
            except Exception:
                pass

        # 夜4 Phase2: 冷专家卸载 (仅 w_mxfp4 方案; setup 在首次 apply)
        if self._off_enabled and not self.ocp_mx_scheme.startswith("w_mxfp4"):
            logger.warning(
                "COLD_OFFLOAD: scheme=%s 不支持, 关闭卸载",
                self.ocp_mx_scheme,
            )
            self._off_enabled = False
        if self._off_enabled and not self._off_ready:
            self._offload_setup(w1, w2)

        if os.environ.get("VLLM_PROFILE"):
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()

        if os.environ.get("VLLM_DUMP_FC1"):
            try:
                import json as _json
                with open(os.environ["VLLM_DUMP_FC1"], "a") as _f:
                    _f.write(_json.dumps({
                        "rank": torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else -1,
                        "backend": "emu_shapes",
                        "w1_shape": list(w1.shape),
                        "w2_shape": list(w2.shape),
                        "hs_shape": list(hidden_states.shape),
                        "expert_map": None if expert_map is None else
                        [int(v) for v in expert_map[:16].tolist()],
                        "layer_name": getattr(self, "layer_name", "?"),
                    }) + "\n")
            except Exception:
                pass

        if os.environ.get("VLLM_DUMP_TOPK"):
            try:
                import json as _json
                _tk = topk_ids.detach().float().cpu()
                with open(os.environ["VLLM_DUMP_TOPK"], "a") as _f:
                    _f.write(_json.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "layer": getattr(self, "_active_idx", -1),
                        "n_valid": int((topk_ids >= 0).sum().item()),
                        "topk0": [int(v) for v in topk_ids[0].tolist()],
                        "uniq": [int(v) for v in topk_ids.unique().tolist()[:20]],
                    }) + "\n")
            except Exception:
                pass

        target_dtype = hidden_states.dtype

        # ---- capture-safe static expert partition -----------------------------
        # CUDA-graph note: torch.unique() and .tolist() are illegal during
        # graph capture, and Python loop trip counts must not depend on GPU
        # values. We therefore chunk over a FIXED partition of the expert-id
        # space: chunk c covers experts [c*MAX_CHUNK, (c+1)*MAX_CHUNK).
        # Slot (t, k) belongs to chunk topk_ids[t, k] // MAX_CHUNK with local
        # id topk_ids[t, k] % MAX_CHUNK — pure arithmetic, no dedup needed,
        # and each slot is counted exactly once (padding -1 matches no chunk).
        # Each expert is dequantized exactly once per forward, matching the
        # old unique() dedup's cost. Chunking by slot POSITION instead would
        # re-dequantize the same expert ~once per token that routes to it
        # (a 2048-token prefill would run ~768 dequant-chunks per layer vs
        # 16 here — >5 min per prefill step, exceeding the 300 s execute-model
        # RPC timeout).
        # Chunk size bounds the dequant peak: dq_mxfp4_pytorch allocates
        # several int32 intermediates ≈ 6× the BF16 output.
        MAX_CHUNK = 16
        num_experts = w1.shape[0]  # static Python int (from shape)
        n_chunks = (num_experts + MAX_CHUNK - 1) // MAX_CHUNK

        # Non-FP4/FP6 path: full dequant (rare)
        if not (
            self.ocp_mx_scheme.startswith("w_mxfp4")  # type: ignore[union-attr]
            or self.ocp_mx_scheme.startswith("w_mxfp6")  # type: ignore[union-attr]
        ):
            w1_full = self._dequantize_weights(w1, self.w1_scale_val, target_dtype)
            w2_full = self._dequantize_weights(w2, self.w2_scale_val, target_dtype)
            super().apply(
                output=output, hidden_states=hidden_states,
                w1=w1_full, w2=w2_full,
                topk_weights=topk_weights, topk_ids=topk_ids,
                activation=activation, global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=None, a2_scale=None,
                workspace13=workspace13, workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            return

        is_mxfp6 = self.ocp_mx_scheme.startswith("w_mxfp6")  # type: ignore[union-attr]
        qtype = None
        if is_mxfp6:
            qtype = "fp6_e3m2" if "e3m2" in self.ocp_mx_scheme else "fp6_e2m3"  # type: ignore[union-attr]

        # Accumulate per-chunk parent-forward results.  ``moe_sum`` in the
        # parent sums over top_k slots; slots whose weight is zero contribute
        # nothing, so summing chunk outputs is exact.
        acc = torch.zeros_like(output)

        # ---- decode / tiny-batch fast path -----------------------------------
        # When the whole routing fits one chunk (e.g. single-stream decode,
        # M=1 → 6 slots), dequantize only the ROUTED experts via one
        # positional chunk. The fixed expert-id partition would dequantize
        # all 256 experts per step (16 chunks), which is correct but wastes
        # ~40× the dequant traffic for single-stream decode. Both branches
        # are static-shape (the branch is on a Python int from the tensor
        # shape), so each is individually CUDA-graph capture-safe.
        num_slots = topk_ids.numel()  # static Python int (from shape)
        if num_slots <= MAX_CHUNK:
            chunk_ids = topk_ids.reshape(-1)  # <= MAX_CHUNK routed slots
            chunk_clamped = torch.clamp(chunk_ids, min=0)  # -1 pad → expert 0
            if is_mxfp6:
                w1_chunk = torch.stack([
                    dequant_mxfp6(w1[eid], self.w1_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk_clamped.tolist()
                ], dim=0)
                w2_chunk = torch.stack([
                    dequant_mxfp6(w2[eid], self.w2_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk_clamped.tolist()
                ], dim=0)
            else:
                if self._off_enabled:
                    # 物理行重映射 (热=hot_pos, 冷=槽; miss 触发 H2D)
                    _phys = self._off_phys_rows(chunk_clamped.tolist(), w1, w2)
                    w1_chunk = dq_mxfp4_triton(
                        w1[_phys], self.w1_scale_val[chunk_clamped],
                    )
                    w2_chunk = dq_mxfp4_triton(
                        w2[_phys], self.w2_scale_val[chunk_clamped],
                    )
                else:
                    w1_chunk = dq_mxfp4_triton(
                        w1[chunk_clamped], self.w1_scale_val[chunk_clamped],
                    )
                    w2_chunk = dq_mxfp4_triton(
                        w2[chunk_clamped], self.w2_scale_val[chunk_clamped],
                    )
            # positional local ids: slot j → row j of the chunk
            local_ids = (
                torch.arange(num_slots, device=topk_ids.device, dtype=torch.int64)
                .view(topk_ids.shape)
                .to(torch.int32)
            )
            temp_out = torch.empty_like(output)
            super().apply(
                output=temp_out,
                hidden_states=hidden_states,
                w1=w1_chunk,
                w2=w2_chunk,
                topk_weights=topk_weights,
                topk_ids=local_ids,
                activation=activation,
                global_num_experts=num_slots,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            acc.add_(temp_out)
            output.copy_(acc)
            if os.environ.get("VLLM_PROFILE"):
                try:
                    _t1.record()
                    torch.cuda.synchronize()
                    with open(os.environ["VLLM_PROFILE"], "a") as _f:
                        _f.write(
                            f"emu_apply_fastpath {_t0.elapsed_time(_t1):.3f}ms "
                            f"M={hidden_states.shape[0]} slots={num_slots}\n"
                        )
                except Exception:
                    pass
            return

        for c in range(n_chunks):
            c0 = c * MAX_CHUNK
            c1 = min(c0 + MAX_CHUNK, num_experts)
            chunk_ids = list(range(c0, c1))  # static Python ints

            # ---- dequantize only this chunk's experts -------------------------
            if is_mxfp6:
                w1_chunk = torch.stack([
                    dequant_mxfp6(w1[eid], self.w1_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk_ids
                ], dim=0)
                w2_chunk = torch.stack([
                    dequant_mxfp6(w2[eid], self.w2_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk_ids
                ], dim=0)
            else:
                if os.environ.get("VLLM_PROFILE"):
                    _td0 = torch.cuda.Event(enable_timing=True)
                    _td1 = torch.cuda.Event(enable_timing=True)
                    _td0.record()
                if self._off_enabled:
                    # 物理行重映射 (热=hot_pos, 冷=槽; miss 触发 H2D)
                    _phys = self._off_phys_rows(list(chunk_ids), w1, w2)
                    w1_chunk = dq_mxfp4_triton(
                        w1[_phys], self.w1_scale_val[c0:c1],
                    )
                    w2_chunk = dq_mxfp4_triton(
                        w2[_phys], self.w2_scale_val[c0:c1],
                    )
                else:
                    w1_chunk = dq_mxfp4_triton(
                        w1[c0:c1], self.w1_scale_val[c0:c1],
                    )
                    w2_chunk = dq_mxfp4_triton(
                        w2[c0:c1], self.w2_scale_val[c0:c1],
                    )
                if os.environ.get("VLLM_PROFILE"):
                    _td1.record()
                    torch.cuda.synchronize()
                    with open(os.environ["VLLM_PROFILE"], "a") as _f:
                        _f.write(
                            f"  dequant {_td0.elapsed_time(_td1):.3f}ms "
                            f"chunk={len(chunk_ids)}\n"
                        )
            if os.environ.get("VLLM_DUMP_FC1"):
                try:
                    import json as _j
                    _dq = [float(v) for v in w1_chunk[0][0][:16].tolist()]
                    with open(os.environ["VLLM_DUMP_FC1"], "a") as _f:
                        _f.write(_j.dumps({
                            "rank": torch.distributed.get_rank()
                            if torch.distributed.is_initialized() else -1,
                            "backend": "emu_wchunk",
                            "chunk": chunk_ids,
                            "w1c0_dq16": _dq,
                        }) + "\n")
                except Exception:
                    pass
            if os.environ.get("VLLM_DUMP_FC1"):
                try:
                    import json as _json
                    with open(os.environ["VLLM_DUMP_FC1"], "a") as _f:
                        _f.write(_json.dumps({
                            "rank": torch.distributed.get_rank()
                            if torch.distributed.is_initialized() else -1,
                            "backend": "emu_wbytes",
                            "w1_bytes": [int(v) for v in w1[c0][0][:8].tolist()],
                            "w2_bytes": [int(v) for v in w2[c0][0][:8].tolist()],
                            "w1s_bytes": [int(v) for v in self.w1_scale_val[c0][0][:8].tolist()],
                            "w2s_bytes": [int(v) for v in self.w2_scale_val[c0][0][:8].tolist()],
                            "w2s_shape": list(self.w2_scale_val.shape),
                            "w1_dq": w1_chunk[0][0][:16].tolist(),
                            "w2_dq": w2_chunk[0][0][:16].tolist(),
                        }) + "\n")
                except Exception:
                    pass

            # ---- remap this chunk's experts  global → local -------------------
            # Fixed expert-id partition: slot (t, k) belongs to chunk
            # topk_ids[t, k] // MAX_CHUNK with local id % MAX_CHUNK — pure
            # arithmetic, each slot is counted exactly once. Padding slots
            # (-1) fail the range check and get zero weight. Out-of-chunk
            # local ids are clamped to [0, MAX_CHUNK) — ids >= len(chunk)
            # would make moe_align_block_size index out of bounds
            # (per-expert counters / w1_chunk rows).
            in_chunk = (topk_ids >= c0) & (topk_ids < c1)
            local_ids = (topk_ids - c0).clamp(min=0, max=MAX_CHUNK - 1).to(torch.int32)
            local_weights = torch.where(
                in_chunk, topk_weights, torch.zeros_like(topk_weights)
            )
            if os.environ.get("VLLM_DUMP_FC1"):
                try:
                    import json as _j
                    with open(os.environ["VLLM_DUMP_FC1"], "a") as _f:
                        _f.write(_j.dumps({
                            "rank": torch.distributed.get_rank()
                            if torch.distributed.is_initialized() else -1,
                            "backend": "emu_localids",
                            "topk_ids_shape": list(topk_ids.shape),
                            "topk_ids0": [int(v) for v in topk_ids[0].tolist()],
                            "local_ids0": [int(v) for v in local_ids[0].tolist()],
                            "dtype": str(topk_ids.dtype),
                        }) + "\n")
                except Exception:
                    pass

            # ---- forward this chunk into a temp output, then accumulate ------
            temp_out = torch.empty_like(output)
            if os.environ.get("VLLM_DUMP_SILU") or os.environ.get("VLLM_DUMP_FC1"):
                # publish chunk→global expert map for the inner dump hook
                try:
                    import json as _j
                    os.environ["VLLM_DUMP_CHUNK_EXPERTS"] = _j.dumps(chunk_ids)
                    os.environ["VLLM_DUMP_CHUNK_TOPK0"] = _j.dumps(
                        [int(v) for v in topk_ids[0].tolist()]
                    )
                    if os.environ.get("VLLM_DUMP_FC1"):
                        with open(os.environ["VLLM_DUMP_FC1"], "a") as _f:
                            _f.write(_j.dumps({
                                "rank": torch.distributed.get_rank()
                                if torch.distributed.is_initialized() else -1,
                                "backend": "emu_chunk",
                                "chunk": chunk_ids,
                                "topk0": [int(v) for v in topk_ids[0].tolist()],
                            }) + "\n")
                except Exception:
                    pass
            super().apply(
                output=temp_out,
                hidden_states=hidden_states,
                w1=w1_chunk,
                w2=w2_chunk,
                topk_weights=local_weights,
                topk_ids=local_ids,
                activation=activation,
                global_num_experts=len(chunk_ids),
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            if os.environ.get("VLLM_DUMP_SILU") and os.path.exists(
                os.environ.get(
                    "VLLM_DUMP_SILU_TRIGGER", "/root/autodl-tmp/tmp/.silu_trigger"
                )
            ):
                try:
                    import json as _json
                    _EMU_DUMP_N[0] += 1
                    _t = temp_out.detach().float().cpu()
                    _lid = local_ids.detach().cpu()
                    _tk = topk_ids.detach().cpu()
                    with open(os.environ["VLLM_DUMP_SILU"], "a") as _f:
                        _f.write(_json.dumps({
                            "n": _EMU_DUMP_N[0],
                            "rank": torch.distributed.get_rank()
                            if torch.distributed.is_initialized() else -1,
                            "backend": "emulation",
                            "chunk_experts": chunk_ids,
                            "local_ids0": [int(v) for v in _lid[0].tolist()],
                            "topk0": [int(v) for v in _tk[0].tolist()],
                            "fc2_out": _t[:1].reshape(-1)[:64].tolist(),
                        }) + "\n")
                except Exception:
                    pass
            acc.add_(temp_out)

        output.copy_(acc)

        if os.environ.get("VLLM_PROFILE"):
            try:
                _t1.record()
                torch.cuda.synchronize()
                with open(os.environ["VLLM_PROFILE"], "a") as _f:
                    _f.write(
                        f"emu_apply {_t0.elapsed_time(_t1):.3f}ms "
                        f"M={hidden_states.shape[0]} experts={num_experts}\n"
                    )
            except Exception:
                pass
