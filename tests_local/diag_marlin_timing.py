"""诊断: fused_marlin_moe 输出张量在调用后的时序稳定性 (双 clone + 跨调用 bitsum)."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
import vllm.model_executor.layers.quantization.utils.marlin_utils as _mu


def _fake_quant(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10).float() / 448.0
    q = (x / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s


_mu._quant_fp8_method = _fake_quant

dev = "cuda"
E, M, K, N, TOPK = 16, 8188, 7168, 2048, 6
BLOCK = 64
torch.manual_seed(42)
hidden = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w13_hf = torch.randint(0, 256, (E, 2 * N, K // 2), device=dev, dtype=torch.uint8) & 0x77
w2_hf = torch.randint(0, 256, (E, K, N // 2), device=dev, dtype=torch.uint8) & 0x77
w13_scale_hf = torch.randint(100, 160, (E, 2 * N, K // 32), device=dev, dtype=torch.uint8)
w2_scale_hf = torch.randint(100, 160, (E, K, N // 32), device=dev, dtype=torch.uint8)


class L:
    params_dtype = torch.bfloat16


w1, w2, w1_scale, w2_scale, _, _ = prepare_moe_mxfp4_layer_for_marlin(
    L(), w13_hf, w2_hf, w13_scale_hf, w2_scale_hf, None, None
)
topk_ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32)
topk_weights = torch.rand(M, TOPK, device=dev, dtype=torch.bfloat16)
max_buf = M * TOPK + E * BLOCK
sorted_token_ids = torch.full((max_buf,), -1, device=dev, dtype=torch.int32)
expert_ids = torch.empty((E + 1) * (max_buf // BLOCK), device=dev, dtype=torch.int32)
ntpp = torch.empty((1,), device=dev, dtype=torch.int32)
ops.moe_align_block_size(topk_ids, E, BLOCK, sorted_token_ids, expert_ids, ntpp)
torch.cuda.synchronize()


def run():
    return _fused_marlin_moe(
        hidden, w1, w2, None, None, w1_scale, w2_scale, topk_weights,
        num_topk=TOPK, quant_type=scalar_types.float4_e2m1f,
        apply_router_weight_on_input=True, expert_map=None, block_size_m=BLOCK,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=ntpp, activation=MoEActivation.SILU,
        activation_func=apply_moe_activation, input_dtype=torch.float8_e4m3fn,
        is_k_full=True,
    )


def bitsum(t):
    b = t.detach().view(torch.int16)
    return int(b.sum(dim=1, dtype=torch.int32).sum(dtype=torch.int64).item())


for trial in range(3):
    o1 = run()
    torch.cuda.synchronize()
    b1 = bitsum(o1)
    c1a = o1.clone()
    torch.cuda.synchronize()
    c1b = o1.clone()
    torch.cuda.synchronize()
    o2 = run()
    torch.cuda.synchronize()
    b1_after = bitsum(o1)
    c2 = o2.clone()
    torch.cuda.synchronize()
    print(
        f"trial{trial}: cloneA==cloneB={torch.equal(c1a, c1b)} "
        f"o1 bitsum stable after run2={b1 == b1_after} "
        f"run1_clone vs run2_clone diffs={int((c1a != c2).sum().item())}"
    )
