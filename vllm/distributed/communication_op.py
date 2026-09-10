# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import os

import torch
import torch.distributed

from .parallel_state import get_tp_group

# [DIAG] VLLM_DBG_AR_TIMING=1: TP allreduce 逐调用计时探针 (env 门控,
# 默认关; capture 安全 — 捕获期跳过). 用途: 每步 AR 真实耗时占比
# (逐调用事件对求和, 不含夹层计算).
_AR_PROBE: dict[str, Any] = {"count": 0, "evs": [], "bytes": 0}


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if os.environ.get("VLLM_DBG_AR_TIMING") == "1":
        if not torch.cuda.is_current_stream_capturing():
            _ev0 = torch.cuda.Event(enable_timing=True)
            _ev1 = torch.cuda.Event(enable_timing=True)
            _ev0.record()
            _out = get_tp_group().all_reduce(input_)
            _ev1.record()
            _AR_PROBE["evs"].append((_ev0, _ev1))
            _AR_PROBE["count"] += 1
            _AR_PROBE["bytes"] += input_.numel() * input_.element_size()
            return _out
    return get_tp_group().all_reduce(input_)


def ar_timing_snapshot_and_reset() -> tuple[float, int, int] | None:
    """Return (elapsed_ms, call_count, total_bytes) since last snapshot."""
    if not _AR_PROBE["evs"]:
        return None
    torch.cuda.synchronize()
    ms = sum(e0.elapsed_time(e1) for e0, e1 in _AR_PROBE["evs"])
    n = _AR_PROBE["count"]
    b = _AR_PROBE["bytes"]
    _AR_PROBE["evs"] = []
    _AR_PROBE["count"] = 0
    _AR_PROBE["bytes"] = 0
    return ms, n, b


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
