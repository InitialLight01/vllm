"""TDD 复现: marlin MoE GEMM1 逐次不确定。

目标: 在进程内用相同输入连跑 N 次 fused_marlin_moe, 验证输出是否位级一致。
权重走真实转换路径 (prepare_moe_mxfp4_layer_for_marlin), 排除合成格式问题。
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)

dev = "cuda"

# 绕过 QuantFP8 CustomOp 的 vllm-config 依赖: torch 版 per-token fp8 量化
# (必须在任何 get_marlin_input_dtype 调用之前生效)
import vllm.model_executor.layers.quantization.utils.marlin_utils as _mu


def _fake_quant(x):
    s = x.abs().amax(dim=-1, keepdim=True) / 448.0
    s = s.clamp(min=1e-10).float()
    q = (x / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s


_mu._quant_fp8_method = _fake_quant

# 生产形状 (DSV4-Flash gemm1): M=8188 tokens, K=7168, N=2048, topk=6
E, M, K, N, TOPK = 16, 8188, 7168, 2048, 6
BLOCK = 64  # moe_block_size (对齐粒度)

torch.manual_seed(42)

hidden = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
# HF 格式随机权重 ([E, 2N, K/2] uint8, 2 fp4/byte)
# fp4 位模式须避免 NaN/inf (E2M1 exp=3): 每 nibble 掩到 <= 0b110
w13_hf = torch.randint(0, 256, (E, 2 * N, K // 2), device=dev, dtype=torch.uint8) & 0x77
w2_hf = torch.randint(0, 256, (E, K, N // 2), device=dev, dtype=torch.uint8) & 0x77
# UE8M0 scale 合法范围 (避开 0/denormal): 100..160
w13_scale_hf = torch.randint(100, 160, (E, 2 * N, K // 32), device=dev, dtype=torch.uint8)
w2_scale_hf = torch.randint(100, 160, (E, K, N // 32), device=dev, dtype=torch.uint8)

# 真实转换路径 (与生产相同)
assert get_marlin_input_dtype() == torch.float8_e4m3fn, "需要 VLLM_MARLIN_INPUT_DTYPE=fp8"


class _DummyLayer:
    params_dtype = torch.bfloat16


w1, w2, w1_scale, w2_scale, _b1, _b2 = prepare_moe_mxfp4_layer_for_marlin(
    _DummyLayer(), w13_hf, w2_hf, w13_scale_hf, w2_scale_hf, None, None
)
print("converted: w1", tuple(w1.shape), "w2", tuple(w2.shape),
      "w1s", tuple(w1_scale.shape), "w2s", tuple(w2_scale.shape))

topk_ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32)
topk_weights = torch.rand(M, TOPK, device=dev, dtype=torch.bfloat16)

# align: buffer 需容纳 padding (M*TOPK + E*BLOCK 上界)
max_tokens_padded = M * TOPK
max_tokens_padded_buf = M * TOPK + E * BLOCK
sorted_token_ids = torch.full(
    (max_tokens_padded_buf,), -1, device=dev, dtype=torch.int32
)
expert_ids = torch.empty(
    (E + 1) * max(1, max_tokens_padded_buf // BLOCK), device=dev, dtype=torch.int32
)
num_tokens_post_pad = torch.empty((1,), device=dev, dtype=torch.int32)
ops.moe_align_block_size(
    topk_ids, E, BLOCK, sorted_token_ids, expert_ids, num_tokens_post_pad
)
torch.cuda.synchronize()
print("align ok, post_pad =", num_tokens_post_pad.item())

quant_type = scalar_types.float4_e2m1f


def run_once() -> torch.Tensor:
    return _fused_marlin_moe(
        hidden,
        w1,
        w2,
        None,  # bias1
        None,  # bias2
        w1_scale,
        w2_scale,
        topk_weights,
        num_topk=TOPK,
        quant_type=quant_type,
        apply_router_weight_on_input=True,
        expert_map=None,
        block_size_m=BLOCK,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_pad,
        activation=MoEActivation.SILU,
        activation_func=apply_moe_activation,
        input_dtype=torch.float8_e4m3fn,
        is_k_full=True,
    )


# 每次调用后立即快照, 排除后写覆盖
outs, snaps = [], []
for _ in range(6):
    o = run_once()
    torch.cuda.synchronize()
    outs.append(o)
    snaps.append(o.clone())
print("data_ptr 序列:", [o.data_ptr() for o in outs])
print("同一 tensor 是否被后写覆盖:", [bool(torch.equal(o, s)) for o, s in zip(outs, snaps)])
ref = snaps[0]
all_eq = True
for i, o in enumerate(snaps[1:], 1):
    eq = torch.equal(ref, o)
    n_diff = int((ref != o).sum().item())
    all_eq &= eq
    print(f"run{i} (snapshot): bitwise_equal={eq}  differing_elements={n_diff}/{ref.numel()}")
    if not eq and n_diff < 64:
        idx = (ref != o).nonzero()[:8]
        for t in idx:
            r = ref[tuple(t)].item()
            v = o[tuple(t)].item()
            print(f"    pos {t.tolist()}: {r!r} vs {v!r}")
print("VERDICT:", "DETERMINISTIC" if all_eq else "NONDETERMINISTIC")
