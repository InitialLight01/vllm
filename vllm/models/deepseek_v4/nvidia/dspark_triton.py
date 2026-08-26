# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

from vllm.triton_utils import tldevice
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64


@triton.jit
def _dspark_qkv_postprocess_kernel(
    q_ptr,
    q_out_ptr,
    kv_ptr,
    kv_out_ptr,
    positions_ptr,
    cos_sin_ptr,
    eps: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    token_pid = tl.program_id(0)
    head_pid = tl.program_id(1)

    offs = tl.arange(0, block_d)
    mask = offs < head_dim
    rope_half: tl.constexpr = rope_dim // 2
    nope_dim: tl.constexpr = head_dim - rope_dim

    q_base = (token_pid * n_heads + head_pid) * head_dim
    q = tl.load(q_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(q * q, axis=0) / head_dim
    rrms = tl.rsqrt(variance + eps)
    q_norm = (q * rrms).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(positions_ptr + token_pid).to(tl.int64)
    rope_offsets = offs - nope_dim
    pair_idx = rope_offsets // 2
    pair_base = nope_dim + pair_idx * 2
    even = tl.load(q_ptr + q_base + pair_base, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(q_ptr + q_base + pair_base + 1, mask=mask, other=0.0).to(tl.float32)
    even = (even * rrms).to(tl.bfloat16).to(tl.float32)
    odd = (odd * rrms).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(
        cos_sin_ptr + pos * rope_dim + pair_idx,
        mask=(offs >= nope_dim) & (pair_idx < rope_half),
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * rope_dim + rope_half + pair_idx,
        mask=(offs >= nope_dim) & (pair_idx < rope_half),
        other=0.0,
    ).to(tl.float32)
    q_rope = tl.where(
        rope_offsets % 2 == 0,
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    q_out = tl.where(offs < nope_dim, q_norm, q_rope)
    tl.store(q_out_ptr + q_base + offs, q_out, mask=mask)

    if head_pid == 0:
        kv = tl.load(kv_ptr + token_pid * head_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        kv_even = tl.load(
            kv_ptr + token_pid * head_dim + pair_base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        kv_odd = tl.load(
            kv_ptr + token_pid * head_dim + pair_base + 1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        kv_rope = tl.where(
            rope_offsets % 2 == 0,
            kv_even * cos - kv_odd * sin,
            kv_odd * cos + kv_even * sin,
        )
        kv_out = tl.where(offs < nope_dim, kv, kv_rope)
        tl.store(kv_out_ptr + token_pid * head_dim + offs, kv_out, mask=mask)


def dspark_qkv_postprocess(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse DSpark q no-weight RMSNorm+RoPE and KV RoPE.

    This matches the existing PyTorch reference order: q is RMS-normalized,
    rounded back to BF16, then RoPE is applied on the tail dimensions.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    if kv.dim() != 2:
        raise ValueError(f"kv must be [tokens, dim], got {kv.shape}")
    if q.shape[0] != kv.shape[0] or q.shape[2] != kv.shape[1]:
        raise ValueError(f"q/kv shape mismatch: q={q.shape}, kv={kv.shape}")
    if not q.is_contiguous() or not kv.is_contiguous():
        raise ValueError("q and kv must be contiguous")
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise ValueError("DSpark Triton q/kv postprocess currently requires BF16")

    num_tokens, n_heads, head_dim = q.shape
    if num_tokens == 0:
        return torch.empty_like(q), torch.empty_like(kv)
    rope_dim = cos_sin_cache.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    _dspark_qkv_postprocess_kernel[(num_tokens, n_heads)](
        q,
        q_out,
        kv,
        kv_out,
        positions.contiguous(),
        cos_sin_cache,
        eps=eps,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        block_d=block_d,
        num_warps=8,
    )
    return q_out, kv_out


@triton.jit
def _dspark_context_kv_store_kernel(
    kv_ptr,
    cache_ptr,
    positions_ptr,
    query_start_loc_ptr,
    rejected_ptr,
    kv_weight_ptr,
    cos_sin_ptr,
    eps: tl.constexpr,
    kv_stride: tl.constexpr,
    cache_batch_stride: tl.constexpr,
    cache_window_stride: tl.constexpr,
    batch_size: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    window_size: tl.constexpr,
    block_d: tl.constexpr,
    has_rejected: tl.constexpr,
):
    token_pid = tl.program_id(0)

    req_idx = tl.full((), 0, dtype=tl.int64)
    should_store = tl.full((), False, dtype=tl.int1)
    store_end = tl.full((), 0, dtype=tl.int64)
    for batch_idx in tl.static_range(0, batch_size):
        start = tl.load(query_start_loc_ptr + batch_idx).to(tl.int64)
        end = tl.load(query_start_loc_ptr + batch_idx + 1).to(tl.int64)
        if has_rejected:
            end -= tl.load(rejected_ptr + batch_idx).to(tl.int64)
        in_request = (token_pid >= start) & (token_pid < end)
        req_idx = tl.where(in_request, batch_idx, req_idx)
        should_store = should_store | in_request
        store_end = tl.where(in_request, end, store_end)

    # 更新52k: 只写各槽的最后写入者 — 圆形缓存 slot = pos % window,
    # 同槽多 token 并发写 = 写写竞态 (run 66 捕获 34/128 槽跨请求分歧
    # 的机制)。仅 end-window 起的尾窗 token 写 (每槽唯一写入者) →
    # 确定 + 与语义一致 (尾窗 = 最终状态)。
    should_store = should_store & (token_pid >= store_end - window_size)

    offs = tl.arange(0, block_d)
    mask = offs < head_dim
    rope_half: tl.constexpr = rope_dim // 2
    nope_dim: tl.constexpr = head_dim - rope_dim

    row = kv_ptr + token_pid * kv_stride
    x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    rrms = tl.rsqrt(variance + eps)
    weight = tl.load(kv_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    norm = (x * rrms * weight).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(positions_ptr + token_pid).to(tl.int64)
    slot = pos % window_size
    rope_offsets = offs - nope_dim
    pair_idx = rope_offsets // 2
    pair_base = nope_dim + pair_idx * 2
    rope_mask = (offs >= nope_dim) & (pair_idx < rope_half)

    even_x = tl.load(row + pair_base, mask=rope_mask, other=0.0).to(tl.float32)
    odd_x = tl.load(row + pair_base + 1, mask=rope_mask, other=0.0).to(tl.float32)
    even_w = tl.load(kv_weight_ptr + pair_base, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    odd_w = tl.load(kv_weight_ptr + pair_base + 1, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    even = (even_x * rrms * even_w).to(tl.bfloat16).to(tl.float32)
    odd = (odd_x * rrms * odd_w).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(
        cos_sin_ptr + pos * rope_dim + pair_idx,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * rope_dim + rope_half + pair_idx,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    rope = tl.where(
        rope_offsets % 2 == 0,
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    out = tl.where(offs < nope_dim, norm, rope)

    cache_row = cache_ptr + req_idx * cache_batch_stride + slot * cache_window_stride
    tl.store(cache_row + offs, out, mask=mask & should_store)


def dspark_context_kv_store(
    kv: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    batch_size: int,
    num_rejected_tokens: torch.Tensor | None,
    kv_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    """Fuse DSpark context KV RMSNorm+RoPE and circular cache scatter."""
    if kv.dim() != 2:
        raise ValueError(f"kv must be [tokens, dim], got {kv.shape}")
    if cache.dim() != 3:
        raise ValueError(f"cache must be [batch, window, dim], got {cache.shape}")
    if kv.shape[1] != cache.shape[2]:
        raise ValueError(
            f"kv/cache head-dim mismatch: kv={kv.shape}, cache={cache.shape}"
        )
    if kv.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("kv and cache must have contiguous last dimensions")
    if kv.dtype is not torch.bfloat16 or cache.dtype is not torch.bfloat16:
        raise ValueError("DSpark context KV store currently requires BF16 kv/cache")

    num_tokens, head_dim = kv.shape
    if num_tokens == 0:
        return
    rope_dim = cos_sin_cache.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    rejected = (
        num_rejected_tokens if num_rejected_tokens is not None else query_start_loc
    )
    _dspark_context_kv_store_kernel[(num_tokens,)](
        kv,
        cache,
        positions.contiguous(),
        query_start_loc.contiguous(),
        rejected.contiguous(),
        kv_weight,
        cos_sin_cache,
        eps=eps,
        kv_stride=kv.stride(0),
        cache_batch_stride=cache.stride(0),
        cache_window_stride=cache.stride(1),
        batch_size=batch_size,
        head_dim=head_dim,
        rope_dim=rope_dim,
        window_size=cache.shape[1],
        block_d=block_d,
        has_rejected=num_rejected_tokens is not None,
        num_warps=8,
    )


@triton.jit
def _dspark_attention_kernel(
    q_ptr,
    main_kv_ptr,
    draft_kv_ptr,
    main_pos_ptr,
    sink_ptr,
    out_ptr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """KV-shared tensor-core DSpark draft attention.

    DSpark draft attention is head-less on the KV side: the main-window KV and
    the draft-block KV are shared across all heads, so every (draft-token, head)
    query for a batch element attends the SAME [window + block, head_dim] KV
    under the SAME validity mask. A one-program-per-(token, head) launch would
    therefore re-read that KV block_size*n_heads times per batch element. This
    kernel instead tiles the block_size*n_heads query rows (BLOCK_M at a time),
    streams the shared KV once per tile in BLOCK_N chunks, and runs a flash
    online-softmax with the per-head attention sink folded in as the running-max
    initializer (a keyless logit that contributes to the denominator only).
    QK^T and P@V use tl.dot; KV is bf16 in storage, so P@V runs in bf16.
    """
    batch_idx = tl.program_id(0)
    m_tile = tl.program_id(1)

    rows_per_batch: tl.constexpr = block_size * n_heads
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    m_valid = offs_m < rows_per_batch
    head_of_row = offs_m % n_heads

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # q_flat[batch] is [block*heads, head_dim] contiguous; row = draft*heads + head
    q_base = batch_idx * rows_per_batch * head_dim
    q_ptrs = q_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None] & d_mask[None, :], other=0.0)

    sink = tl.load(sink_ptr + head_of_row, mask=m_valid, other=0.0).to(tl.float32)

    valid_main_end = tl.load(main_pos_ptr + batch_idx)
    valid_main_end = tl.minimum(valid_main_end, window_size - 1)

    # sink folded in as a keyless logit: init running max=sink, denom=1, acc=0
    m_i = sink
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    total_kv: tl.constexpr = window_size + block_size
    for start in range(0, total_kv, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        main_mask = offs_n < window_size
        draft_off = offs_n - window_size
        valid_n = tl.where(main_mask, offs_n <= valid_main_end, draft_off < block_size)

        main_ptrs = (
            main_kv_ptr
            + (batch_idx * window_size + offs_n[:, None]) * head_dim
            + offs_d[None, :]
        )
        draft_ptrs = (
            draft_kv_ptr
            + (batch_idx * block_size + draft_off[:, None]) * head_dim
            + offs_d[None, :]
        )
        kv_ptrs = tl.where(main_mask[:, None], main_ptrs, draft_ptrs)
        kv = tl.load(kv_ptrs, mask=valid_n[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
        scores = tl.where(valid_n[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid_n[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptrs = out_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out, mask=m_valid[:, None] & d_mask[None, :])


def dspark_triton_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    draft_kv: torch.Tensor,
    main_positions: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Small-shape fused DSpark attention over circular main KV + draft KV."""
    if q.dim() != 4:
        raise ValueError(f"q must be [batch, block, heads, dim], got {q.shape}")
    batch_size, block_size, n_heads, head_dim = q.shape
    window_size = main_kv.shape[1]
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)

    # One program per (batch, query-row tile). BLOCK_M x BLOCK_N x BLOCK_D bf16
    # tiles fit the shared-memory budget with num_stages=1 (the KV loop is only
    # a few iterations, so software pipelining buys nothing here).
    block_m, block_n = 32, 32
    rows_per_batch = block_size * n_heads
    grid = (batch_size, triton.cdiv(rows_per_batch, block_m))
    _dspark_attention_kernel[grid](
        q,
        main_kv,
        draft_kv,
        main_positions,
        attn_sink,
        out,
        scale=scale,
        block_size=block_size,
        n_heads=n_heads,
        head_dim=head_dim,
        window_size=window_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return out



@triton.jit
def _dspark_gumbel_argmax_blocks_kernel(
    logits_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    out_logits_ptr,
    block_gval_ptr,
    block_gid_ptr,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    out_logits_row_stride: tl.constexpr,
    scratch_stride: tl.constexpr,
    block_v: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Per-vocab-block gumbel-max for the fused DSpark draft sampler.

    Bit-exact replica of ``gumbel_block_argmax`` semantics (the eager
    ``gumbel_sample`` path this replaces): temperature scaling (div_rn),
    the processed-logits store (temp-applied logits the rejection sampler
    consumes), the (seeds, pos)-keyed Philox draw, and the block gumbel-max.
    Greedy rows (temperature == 0) take the plain argmax with the raw-logits
    store — identical to the reference.
    """
    batch_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    offs_v = block_pid * block_v + tl.arange(0, block_v)
    v_mask = offs_v < vocab_size

    temp = tl.load(temp_ptr + batch_pid).to(tl.float32)
    z = tl.load(
        logits_ptr + batch_pid * logits_row_stride + offs_v,
        mask=v_mask,
        other=-float("inf"),
    ).to(tl.float32)
    if temp != 0.0:
        # Match _temperature_kernel / gumbel_block_argmax: div_rn.
        z = z / temp

    # Processed-logits store: temp-applied logits (raw for greedy rows),
    # exactly what the eager gumbel_sample writes to the draft_logits cache.
    tl.store(
        out_logits_ptr + batch_pid * out_logits_row_stride + offs_v,
        z,
        mask=v_mask,
    )

    if temp != 0.0:
        seed = tl.load(seeds_ptr + batch_pid)
        pos = tl.load(pos_ptr + batch_pid)
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, offs_v, includes_zero=False)
            g = z + (-tl.log(-tl.log(u)))
        else:
            u = tl_rand32(gumbel_seed, offs_v, includes_zero=False)
            # Same tail-preserving form as the reference: the winning tail
            # lives at u -> 0 where fp32 has fine resolution.
            g = z + (-tl.log(-tldevice.log1p(-u)))
    else:
        g = z
    g = tl.where(v_mask, g, -float("inf"))

    bgval = tl.max(g, axis=0)
    bgid = tl.min(
        tl.where((g == bgval) & v_mask, offs_v, vocab_size), axis=0
    )
    tl.store(block_gval_ptr + batch_pid * scratch_stride + block_pid, bgval)
    tl.store(block_gid_ptr + batch_pid * scratch_stride + block_pid, bgid)


@triton.jit
def _dspark_gumbel_pick_kernel(
    block_gval_ptr,
    block_gid_ptr,
    out_tokens_ptr,
    out_tokens_stride,
    num_blocks: tl.constexpr,
    block_nb: tl.constexpr,
    scratch_stride: tl.constexpr,
):
    """Combine per-block gumbel-max results into the sampled token.

    Tie-break matches the reference two-stage reduction: the lowest vocab id
    wins (first occurrence in block-major order).
    """
    batch_pid = tl.program_id(0)
    offs = tl.arange(0, block_nb)
    mask = offs < num_blocks
    base = batch_pid * scratch_stride + offs

    bgval = tl.load(block_gval_ptr + base, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    bgid = tl.load(block_gid_ptr + base, mask=mask, other=2147483647).to(tl.int64)
    gmax = tl.max(bgval, axis=0)
    token = tl.min(tl.where((bgval == gmax) & mask, bgid, 2147483647), axis=0)
    tl.store(out_tokens_ptr + batch_pid * out_tokens_stride, token)


def dspark_gumbel_argmax_sample(
    step_logits: torch.Tensor,
    temperature: torch.Tensor,
    out_tokens: torch.Tensor,
    out_logits: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    scratch: dict[str, torch.Tensor],
    *,
    use_fp64: bool = False,
    block_v: int = 1024,
) -> None:
    """Fused DSpark draft sampler for one sequential block step.

    Drop-in replacement for the eager ``gumbel_sample`` call in the DSpark
    speculator: per-request Gumbel-max sampling over ``step_logits`` (base
    logits + Markov bias, pre-temperature), writing the sampled ``out_tokens``
    and the temp-applied ``out_logits`` (the draft_logits cache the rejection
    sampler consumes). Two launches instead of the eager path's three plus
    intermediates, with bit-identical semantics (verified by op test).
    """
    if step_logits.dim() != 2:
        raise ValueError(
            f"step_logits must be [batch, vocab], got {step_logits.shape}"
        )
    batch_size, vocab_size = step_logits.shape
    if out_logits.shape != step_logits.shape:
        raise ValueError(
            f"out_logits shape {out_logits.shape} must match step_logits "
            f"{step_logits.shape}"
        )
    if temperature.shape[0] < batch_size or seeds.shape[0] < batch_size:
        raise ValueError("temperature and seeds must cover batch_size")
    if pos.shape[0] < batch_size or out_tokens.shape[0] < batch_size:
        raise ValueError("pos and out_tokens must cover batch_size")

    num_blocks = triton.cdiv(vocab_size, block_v)
    block_gval = scratch.get("block_gval")
    block_gid = scratch.get("block_gid")
    if (
        block_gval is None
        or block_gid is None
        or block_gval.shape[0] < batch_size
        or block_gval.shape[1] < num_blocks
        or block_gid.shape[0] < batch_size
        or block_gid.shape[1] < num_blocks
    ):
        raise ValueError(
            "scratch['block_gval'/'block_gid'] too small: "
            f"{None if block_gval is None else block_gval.shape}, "
            f"{None if block_gid is None else block_gid.shape}"
        )

    grid = (batch_size, num_blocks)
    _dspark_gumbel_argmax_blocks_kernel[grid](
        step_logits,
        temperature,
        seeds,
        pos,
        out_logits,
        block_gval,
        block_gid,
        vocab_size=vocab_size,
        logits_row_stride=step_logits.stride(0),
        out_logits_row_stride=out_logits.stride(0),
        scratch_stride=block_gval.stride(0),
        block_v=block_v,
        USE_FP64=use_fp64,
        num_warps=8,
    )
    _dspark_gumbel_pick_kernel[(batch_size,)](
        block_gval,
        block_gid,
        out_tokens,
        out_tokens.stride(0),
        num_blocks=num_blocks,
        block_nb=triton.next_power_of_2(num_blocks),
        scratch_stride=block_gval.stride(0),
        num_warps=4,
    )
