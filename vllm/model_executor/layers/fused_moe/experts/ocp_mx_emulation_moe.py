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

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
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

        if self.ocp_mx_scheme in {
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
        }:
            # Weight has to be dequantized for mxfp4 emulation.
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

        target_dtype = hidden_states.dtype

        # ---- find which experts are actually active ---------------------------
        active_experts = topk_ids.unique()
        num_active = active_experts.numel()
        MAX_CHUNK = 16

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

        for c in range(0, num_active, MAX_CHUNK):
            chunk = active_experts[c : c + MAX_CHUNK]

            # ---- dequantize only this chunk's experts -------------------------
            if is_mxfp6:
                w1_chunk = torch.stack([
                    dequant_mxfp6(w1[eid], self.w1_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk.tolist()
                ], dim=0)
                w2_chunk = torch.stack([
                    dequant_mxfp6(w2[eid], self.w2_scale_val[eid],
                                  quant_dtype=qtype, float_dtype=target_dtype)
                    for eid in chunk.tolist()
                ], dim=0)
            else:
                w1_chunk = dq_mxfp4_pytorch(
                    w1[chunk], self.w1_scale_val[chunk], target_dtype,
                )
                w2_chunk = dq_mxfp4_pytorch(
                    w2[chunk], self.w2_scale_val[chunk], target_dtype,
                )

            # ---- remap this chunk's experts  global → local -------------------
            # Slots not routed to this chunk get id 0 with zero weight, so
            # they are computed (redundantly) but contribute nothing.
            local_ids = torch.zeros_like(topk_ids)
            for i, eid in enumerate(chunk.tolist()):
                local_ids[topk_ids == eid] = i
            in_chunk = torch.isin(topk_ids, chunk)
            local_weights = torch.where(
                in_chunk, topk_weights, torch.zeros_like(topk_weights)
            )

            # ---- forward this chunk into a temp output, then accumulate ------
            temp_out = torch.empty_like(output)
            super().apply(
                output=temp_out,
                hidden_states=hidden_states,
                w1=w1_chunk,
                w2=w2_chunk,
                topk_weights=local_weights,
                topk_ids=local_ids,
                activation=activation,
                global_num_experts=len(chunk),
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
