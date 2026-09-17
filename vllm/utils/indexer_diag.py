# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-078 indexer diag dumps (env-gated, evidence-first diagnosis).

VLLM_INDEXER_DIAG=1 时: 在 prefill 问题 token (末 token) 抓取打分输入/输出
落盘, 供离线对比 fp8 打分 vs bf16-Q 打分的 needle 排名。诊断代码绝不干扰
主流程 (save 全 try/except)。
"""
import os

import torch

TAG_FILE = "/root/autodl-tmp/tmp/indexer_diag_tag.txt"
_FUSED_COUNTER = {"n": 0}
_PREFILL_CAP = {}
_DECODE_CAP = {}
_LAST_TAG = None


def _reset_if_tag_changed() -> None:
    global _LAST_TAG
    t = tag()
    if _LAST_TAG is not None and _LAST_TAG != t:
        _FUSED_COUNTER["n"] = 0
        _PREFILL_CAP.clear()
        _DECODE_CAP.clear()
    _LAST_TAG = t


def enabled() -> bool:
    return os.getenv("VLLM_INDEXER_DIAG") == "1"


def dump_dir() -> str:
    return os.getenv("VLLM_INDEXER_DIAG_DIR", "/root/autodl-tmp/tmp/indexer_diag")


def tag() -> str:
    try:
        with open(TAG_FILE) as fh:
            t = fh.read().strip()
            return t or "run"
    except OSError:
        return "run"


def fused_counter() -> int:
    return _FUSED_COUNTER["n"]


def decode_ok(layer: str, cap: int = 30) -> bool:
    _reset_if_tag_changed()
    n = _DECODE_CAP.get(layer, 0)
    if n >= cap:
        return False
    _DECODE_CAP[layer] = n + 1
    return True


def save(layer: str, kind: str, payload: dict, logger=None) -> None:
    try:
        _reset_if_tag_changed()
        d = os.path.join(dump_dir(), tag())
        os.makedirs(d, exist_ok=True)
        if kind == "fused":
            _FUSED_COUNTER["n"] += 1
            name = f"fused_{_FUSED_COUNTER['n']:06d}.pt"
        elif kind == "decode":
            # decode_ok 已先行递增, 当前序号 = counter-1
            n = _DECODE_CAP.get(layer, 1) - 1
            name = f"{layer}_{kind}_{n:02d}.pt"
        else:
            # prefill 环形覆盖: 每个 prefill 步都会命中 (ke[-1] == 该步前缀
            # 长度), 只保留最近 4 份 — 最后一份必含问题 token 行。
            n = _PREFILL_CAP.get(layer, 0)
            _PREFILL_CAP[layer] = n + 1
            name = f"{layer}_{kind}_{n % 4:02d}.pt"
        torch.save(payload, os.path.join(d, name))
    except Exception as e:  # 诊断代码绝不干扰主流程
        if logger is not None:
            logger.warning("indexer diag save failed: %s", e)
