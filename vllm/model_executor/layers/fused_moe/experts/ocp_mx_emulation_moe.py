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

        Dequantizes **only the routed-to experts** (not all experts) to avoid
        the 4× memory spike of full FP4→BF16 expansion.  Uses pure-PyTorch
        FP4 unpack (no ``amd-quark`` dependency).
        """
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        target_dtype = hidden_states.dtype

        # ---- find which experts are actually active ---------------------------
        # topk_ids: [num_tokens, top_k]  — expert indices each token routes to
        active_experts = topk_ids.unique()
        num_active = active_experts.numel()

        # ---- dequantize only the active experts --------------------------------
        if self.ocp_mx_scheme.startswith("w_mxfp4"):  # type: ignore[union-attr]
            # w1 shape: [E_total, intermediate*2, hidden//2]
            # w1_scale shape: [E_total, intermediate*2, hidden//32]
            w1_dequant = torch.empty(
                num_active, w1.shape[1], w1.shape[2] * 2,
                dtype=target_dtype, device=w1.device,
            )
            w2_dequant = torch.empty(
                num_active, w2.shape[1], w2.shape[2] * 2,
                dtype=target_dtype, device=w2.device,
            )
            for i, eid in enumerate(active_experts.tolist()):
                w1_dequant[i] = dq_mxfp4_pytorch(
                    w1[eid].unsqueeze(0),
                    self.w1_scale_val[eid].unsqueeze(0),
                    target_dtype,
                ).squeeze(0)
                w2_dequant[i] = dq_mxfp4_pytorch(
                    w2[eid].unsqueeze(0),
                    self.w2_scale_val[eid].unsqueeze(0),
                    target_dtype,
                ).squeeze(0)
            a1q_scale_arg = None
            a2_scale_arg = None
        elif self.ocp_mx_scheme.startswith("w_mxfp6"):  # type: ignore[union-attr]
            w1_dequant = torch.empty(
                num_active, w1.shape[1], w1.shape[2] * 2,
                dtype=target_dtype, device=w1.device,
            )
            w2_dequant = torch.empty(
                num_active, w2.shape[1], w2.shape[2] * 2,
                dtype=target_dtype, device=w2.device,
            )
            for i, eid in enumerate(active_experts.tolist()):
                w1_dequant[i] = dequant_mxfp6(
                    w1[eid], self.w1_scale_val[eid],
                    quant_dtype="fp6_e3m2"
                    if "e3m2" in self.ocp_mx_scheme  # type: ignore[union-attr]
                    else "fp6_e2m3",
                    float_dtype=target_dtype,
                )
                w2_dequant[i] = dequant_mxfp6(
                    w2[eid], self.w2_scale_val[eid],
                    quant_dtype="fp6_e3m2"
                    if "e3m2" in self.ocp_mx_scheme  # type: ignore[union-attr]
                    else "fp6_e2m3",
                    float_dtype=target_dtype,
                )
            a1q_scale_arg = None
            a2_scale_arg = None
        else:
            w1_dequant = self._dequantize_weights(
                w1, self.w1_scale_val, target_dtype
            )
            w2_dequant = self._dequantize_weights(
                w2, self.w2_scale_val, target_dtype
            )
            a1q_scale_arg = None
            a2_scale_arg = None

        # ---- remap expert IDs  global → local ----------------------------------
        global_to_local = {eid: i for i, eid in enumerate(active_experts.tolist())}
        local_ids = topk_ids.clone()
        for gid, lid in global_to_local.items():
            local_ids[topk_ids == gid] = lid

        # ---- forward to parent with reduced weight set -------------------------
        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=local_ids,
            activation=activation,
            global_num_experts=num_active,
            expert_map=None,   # no remapping needed with local IDs
            a1q_scale=a1q_scale_arg,
            a2_scale=a2_scale_arg,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
