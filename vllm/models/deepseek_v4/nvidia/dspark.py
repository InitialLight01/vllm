# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the DSV4 DSpark,
which reuses the target model's architecture similarly to MTP.

The draft layers run a DSpark-specific attention over a circular window of
target-model hidden-state KVs plus the draft block's own KVs (head-less KV
shared across all query heads). It is computed with dedicated Triton kernels /
an eager fallback and NEVER dispatches to the FlashInfer sparse-MLA backend:
the draft block is only ``dspark_block_size`` (5) tokens wide, far below the
minimum batch the sparse-MLA kernels accept.
"""

import os
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (
    fused_q_kv_rmsnorm,
)

from .dspark_triton import (
    dspark_context_kv_store,
    dspark_qkv_postprocess,
    dspark_triton_attention,
)
from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    return x_float.mul(torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps))


def _apply_rope_gptj_last(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Apply DS4 RoPE to the last rope_dim features.

    Mirrors the DS4 fused qnorm/kv-insert kernels' PyTorch reference:
    GPT-J/interleaved pairs, cos||sin cache layout, and RoPE on the tail of the
    head dimension.
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache.index_select(0, positions.long()).to(torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]

    while cos.dim() < even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone()
    out[..., nope_dim:] = rope_rotated.to(out.dtype)
    return out


class DSparkDecoderLayer(DeepseekV4DecoderLayer):
    """Draft decoder layer: hc pre -> DSpark attention -> hc post -> ffn.

    Identical to ``DeepseekV4DecoderLayer`` except the attention call is
    replaced by the DSpark fused attention over the materialized circular
    main-KV window plus the draft block KV (see module docstring).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        window_size: int,
        max_batch: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(vllm_config, prefix)
        self.register_buffer(
            "_main_kv_cache",
            torch.zeros(
                max_batch,
                window_size,
                self.attn.head_dim,
                dtype=dtype,
            ),
            persistent=False,
        )
        # Triton fused kernels on by default; eager fallbacks for debugging
        # (VLLM_DSPARK_EAGER_ATTN / _QKV / _STORE).
        self._triton_attn = os.environ.get("VLLM_DSPARK_EAGER_ATTN") is None
        self._triton_qkv = os.environ.get("VLLM_DSPARK_EAGER_QKV") is None
        self._triton_store = os.environ.get("VLLM_DSPARK_EAGER_STORE") is None

    @property
    def window_size(self) -> int:
        return int(self.attn.window_size)

    def store_main_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        batch_size: int,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> None:
        """Scatter context KV into the circular window buffer.

        ``main_x`` is the (already combined) target hidden state per context
        token. Rejected-tail tokens are excluded so the buffer only ever holds
        KV for accepted positions.
        """
        if main_x.numel() == 0 or query_start_loc is None:
            return
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(main_x))
        kv_raw = qr_kv[..., self.attn.q_lora_rank :]
        if self._triton_store:
            dspark_context_kv_store(
                kv_raw,
                self._main_kv_cache[:batch_size],
                context_positions,
                query_start_loc,
                batch_size,
                num_rejected_tokens,
                self.attn.kv_norm.weight.data,
                self.attn.rotary_emb.cos_sin_cache,
                self.attn.eps,
            )
            return
        kv = self.attn.kv_norm(kv_raw)
        kv = _apply_rope_gptj_last(
            kv, context_positions, self.attn.rotary_emb.cos_sin_cache
        )

        starts = query_start_loc[:-1].long()
        ends = query_start_loc[1:].long()
        if num_rejected_tokens is not None:
            ends = ends - num_rejected_tokens[:batch_size].long()
        window = self.window_size
        cache = self._main_kv_cache
        for req_idx in range(batch_size):
            start = int(starts[req_idx].item())
            end = int(ends[req_idx].item())
            if end <= start:
                continue
            slots = torch.remainder(context_positions[start:end].long(), window)
            cache[req_idx, slots] = kv[start:end].to(cache.dtype)

    def _project_draft_q_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(x))
        qr, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        if self._triton_qkv:
            # Fused RMSNorm (q/kv) + fused no-weight q-RMSNorm + RoPE tail.
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.attn.q_norm.weight.data,
                self.attn.kv_norm.weight.data,
                self.attn.eps,
            )
            q = _linear_output(self.attn.wq_b(qr)).view(
                -1, self.attn.n_local_heads, self.attn.head_dim
            )
            q, kv = dspark_qkv_postprocess(
                q.contiguous(),
                kv.contiguous(),
                positions,
                self.attn.rotary_emb.cos_sin_cache,
                self.attn.eps,
            )
            return q, kv
        qr = self.attn.q_norm(qr)
        kv = self.attn.kv_norm(kv)
        q = _linear_output(self.attn.wq_b(qr)).view(
            -1, self.attn.n_local_heads, self.attn.head_dim
        )
        # No-weight RMSNorm on q, rounded back to BF16, then RoPE on both.
        q = _rmsnorm_no_weight(q, self.attn.eps).to(x.dtype)
        q = _apply_rope_gptj_last(
            q, positions, self.attn.rotary_emb.cos_sin_cache
        )
        kv = _apply_rope_gptj_last(
            kv, positions, self.attn.rotary_emb.cos_sin_cache
        )
        return q, kv

    def _dspark_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        main_positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Eager DSpark attention over [circular main KV + draft KV].

        Slots 0..window-1 of the circular cache are valid up to the request's
        last context position; the whole draft block is always valid.
        """
        block_size = positions.shape[0] // batch_size
        q, draft_kv = self._project_draft_q_kv(x, positions)
        q = q.view(
            batch_size, block_size, self.attn.n_local_heads, self.attn.head_dim
        )
        draft_kv = draft_kv.view(batch_size, block_size, self.attn.head_dim)

        cache_kv = self._main_kv_cache[:batch_size].to(draft_kv.dtype)
        # 解码首步指纹 (VLLM_DUMP_HIDDEN_FP, 更新51): 判别草稿注意力
        # 输入的奇偶周期状态 — x/cache_kv/draft_kv/main_positions
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jds
                torch.cuda.synchronize()
                _bs = lambda t: int(t.detach().contiguous().view(torch.int16).sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item()) if t.numel() else 0
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jds.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "comp": "dspark_attn_in",
                        "prefix": self.attn.prefix if hasattr(self.attn, "prefix") else "dspark",
                        "pos0": int(positions.view(-1)[0].item()),
                        "main_pos0": int(main_positions.view(-1)[0].item()),
                        "xsum": _bs(x), "cksum": _bs(cache_kv), "dksum": _bs(draft_kv),
                    }) + "\n")
            except Exception:
                pass
        if self._triton_attn:
            # KV-shared tensor-core attention over [circular main KV + draft
            # KV] with the per-head sink folded into the online softmax.
            o = dspark_triton_attention(
                q,
                cache_kv,
                draft_kv,
                main_positions.long(),
                self.attn.attn_sink[: self.attn.n_local_heads],
                float(self.attn.scale),
            ).reshape(
                batch_size * block_size,
                self.attn.n_local_heads,
                self.attn.head_dim,
            )
            return self.attn._o_proj(o, positions)
        kv = torch.cat([cache_kv, draft_kv], dim=1)

        window = self.window_size
        cache_arange = torch.arange(
            window, device=positions.device, dtype=main_positions.dtype
        )
        valid_cache = cache_arange.unsqueeze(0) <= torch.minimum(
            main_positions.long().unsqueeze(1),
            torch.full_like(main_positions.long().unsqueeze(1), window - 1),
        )
        valid = torch.cat(
            [
                valid_cache,
                torch.ones(
                    batch_size,
                    block_size,
                    dtype=torch.bool,
                    device=positions.device,
                ),
            ],
            dim=1,
        )

        scores = torch.einsum("bqhd,bnd->bqhn", q.float(), kv.float())
        scores *= float(self.attn.scale)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))

        sink = self.attn.attn_sink[: self.attn.n_local_heads].view(1, 1, -1, 1)
        max_score = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - max_score).masked_fill(
            ~valid[:, None, None, :], 0.0
        )
        denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - max_score)
        o = torch.einsum("bqhn,bnd->bqhd", exp_scores / denom, kv.float())
        o = o.to(x.dtype).reshape(
            batch_size * block_size, self.attn.n_local_heads, self.attn.head_dim
        )
        return self.attn._o_proj(o, positions)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor | None,
        main_positions: torch.Tensor,
        batch_size: int,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del main_x  # interface parity; the Triton attention reads the cache buffer
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            # Run standalone mhc_pre on first layer
            residual = x
            post_mix, res_mix, x = mhc_pre_tilelang(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        # attn_norm is fused into mhc_pre_tilelang / mhc_fused_post_pre above.
        x = self._dspark_attention(x, positions, main_positions, batch_size)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )

        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)

        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or len(
            self.target_layer_ids
        )
        # The draft block is num_speculative_tokens wide; the checkpoint's
        # dspark_block_size (5) is the trained default, but the DSpark
        # attention handles any block size (non-causal within the block), so
        # honor an explicit num_speculative_tokens override.
        self.block_size = int(
            vllm_config.speculative_config.num_speculative_tokens
            or getattr(config, "dspark_block_size", 0)
        )
        if self.block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size > 0")

        # Shared with the target (aliased by the speculator's loading utility).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        current_vllm_config = get_current_vllm_config()
        max_batch = max(
            1, int(current_vllm_config.scheduler_config.max_num_seqs)
        )
        dtype = current_vllm_config.model_config.dtype
        self.layers = nn.ModuleList(
            [
                DSparkDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    window_size=config.sliding_window,
                    max_batch=max_batch,
                    dtype=dtype,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Heads: final norm + hc_head, and the Markov head
        # Loaded from the "final" MTP layer weights (mtp.*) in the target checkpoint
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> None:
        """Insert the sliding-window context KV into each draft layer's
        materialized circular buffer (used by the DSpark attention).

        ``main_x`` is the combined target hidden state per context token.
        Runs eagerly outside the captured CUDA graph. When
        ``query_start_loc`` is None (legacy memory-profiling path) nothing is
        stored.
        """
        del context_slot_mappings  # materialized buffer; paged draft caches unused
        if query_start_loc is None or main_x.numel() == 0:
            return
        if batch_size is None:
            batch_size = int(query_start_loc.shape[0] - 1)
        for layer in self.layers:
            layer.store_main_kv(
                main_x,
                context_positions,
                query_start_loc,
                batch_size,
                num_rejected_tokens,
            )
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jbuf
                torch.cuda.synchronize()
                _b = self.layers[0]._main_kv_cache.detach().view(torch.int16)
                _bitsum = int(_b.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                _h8 = [_b.view(-1)[i].item() for i in range(8)]
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jbuf.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "layer": -11,
                        "comp": "draft_buf",
                        "pos0": int(context_positions.view(-1)[0].item()),
                        "shape": list(self.layers[0]._main_kv_cache.shape),
                        "bitsum": _bitsum,
                        "zc": int(((_b == 0) | (_b == -32768)).sum().item()),
                        "h8": _h8,
                    }) + "\n")
            except Exception:
                pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Expand to hc_mult copies for hyper-connections ([T, H] -> [T, hc, H]).
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        batch_size = input_ids.shape[0] // self.block_size
        # Last context position per request = the position right before the
        # first draft-block query.
        main_positions = torch.clamp(
            positions.view(batch_size, self.block_size)[:, 0] - 1, min=0
        )

        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                None,
                main_positions,
                batch_size,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # hc_head reduces the hc copies; return the PRE-norm head hidden
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        if os.environ.get("VLLM_DUMP_HIDDEN_FP") and not torch.cuda.is_current_stream_capturing():
            try:
                import json as _jdraft
                torch.cuda.synchronize()
                _b = hidden_states.detach().view(torch.int16)
                _bitsum = int(_b.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())
                _h8 = [_b.view(-1)[i].item() for i in range(8)]
                with open(os.environ["VLLM_DUMP_HIDDEN_FP"], "a") as _f:
                    _f.write(_jdraft.dumps({
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                        "layer": -10,
                        "comp": "draft_fwd_out",
                        "pos0": int(positions.view(-1)[0].item()),
                        "shape": list(hidden_states.shape),
                        "bitsum": _bitsum,
                        "zc": int(((_b == 0) | (_b == -32768)).sum().item()),
                        "h8": _h8,
                    }) + "\n")
            except Exception:
                pass
        return hidden_states


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None
    # The speculator passes query_start_loc / batch_size / num_rejected_tokens
    # to precompute_and_store_context_kv (materialized-buffer DSpark attention).
    uses_query_start_loc_context_kv = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared with the target (aliased by the speculator's load utility).
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix. The paged caches stay allocated but are
        # unused by the materialized DSpark attention.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mappings,
            query_start_loc=query_start_loc,
            batch_size=batch_size,
            num_rejected_tokens=num_rejected_tokens,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm hc_head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        first_layer = self.model.layers[0]
        use_mega_moe = first_layer.ffn.use_mega_moe
        if use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/hc_head/markov_head) load directly — otherwise e.g.
            # "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self._finalize_moe()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _finalize_moe(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        # The confidence head is not wired into inference yet; drop its weights.
        if rest.startswith("confidence_head."):
            return None
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
