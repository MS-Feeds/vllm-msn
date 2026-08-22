"""Speculative Prefill scoring math — Algorithm 1, lines 12, 14, 16.

Engine- and architecture-agnostic: pure tensor-in/tensor-out functions, no
vLLM engine dependency. Copied verbatim from `../../spec_prefill_llama/
vllm_patch/scoring.py` — see the approved plan's "Files to create / change"
(copy-unchanged list): this module has no notion of a single vs. multi-turn
caller baked in, it just scores whatever Q/K/context_len it's handed. In the
multi-turn pipeline it's called once per turn by `pruner.py`, and a second
time (against the *target* model's own attention instead of the speculator's)
by the oracle-upper-bound path — see EXPERIMENT_PLAN.md's "Oracle upper
bound" section.

    12: A <- compute_attention_score(Q, K)
    14: A <- aggregate_attention_score(A)
    16: T <- chunk_select_from_smoothed_attention(A)   # T = kept token indices

Lines 1-11 (batch splitting, the lookahead loop, Q/K retrieval) live in
prefill_split.py / proposer.py / kv_cache_utils.py. Lines 18-20 (position-id
restoration, request merging, target forward) are out of scope for this pass
— see EXPERIMENT_PLAN.md's "Implementation status".
"""

import math
from typing import List, Optional

import torch

from .config import SpecConfig


def compute_attention_score(
    query_buffer: List[torch.Tensor],
    key_buffer: List[List[torch.Tensor]],
    actual_look_ahead_cnts: List[int],
) -> List[torch.Tensor]:
    """Algorithm line 12: A <- compute_attention_score(Q, K).

    Q @ K^T / sqrt(d) per layer, per sample.

    Args:
        query_buffer: per-layer list of buffered query tensors, each
            [num_prefill_samples, look_ahead_cnt, num_heads * head_dim]
            (the shape produced by stacking one entry per lookahead step —
            see proposer.py's query-capture hook).
        key_buffer: per-layer list of per-sample key tensors, each
            [context_len, num_kv_heads, head_dim] (from kv_cache_utils.py).
        actual_look_ahead_cnts: per-sample count of lookahead steps actually
            used (may be less than the configured look_ahead_cnt if a
            sample hit EOS early).

    Returns:
        Per-sample list of [num_layer, num_head, look_ahead_cnt, context_len]
        attention-score tensors.
    """
    attn_weights: List[List[torch.Tensor]] = []

    for layer_idx in range(len(query_buffer)):
        attn_weights.append([])

        keys = key_buffer[layer_idx]
        # Unbind along the sample dim: query_buffer[layer_idx] is
        # [num_samples, look_ahead_cnt, num_heads*head_dim] -> per-sample
        # [look_ahead_cnt, num_heads*head_dim].
        queries = torch.unbind(query_buffer[layer_idx], dim=0)

        for q, k, c in zip(queries, keys, actual_look_ahead_cnts):
            look_ahead_cnt, num_heads_times_head_dim = q.shape
            num_kv_heads, head_dim = k.shape[-2], k.shape[-1]
            num_heads = num_heads_times_head_dim // head_dim

            # [look_ahead_cnt, num_heads*head_dim] -> [num_heads, look_ahead_cnt, head_dim]
            query = q.view(look_ahead_cnt, num_heads, head_dim).transpose(0, 1)
            key = k.transpose(0, 1)  # [num_kv_heads, context_len, head_dim]

            if num_heads != num_kv_heads:
                # GQA: repeat KV heads to match query heads (mirrors the
                # reference's repeat_interleave-based reshape).
                assert num_heads % num_kv_heads == 0
                key = key.repeat_interleave(num_heads // num_kv_heads, dim=0)

            query = query[:, :c, :]

            attn = torch.matmul(
                query, key.transpose(-1, -2)
            ) / math.sqrt(head_dim)

            attn_weights[-1].append(attn)

    num_samples = len(attn_weights[0])
    return [
        torch.stack(
            [attn_weights[layer_idx][sample_idx] for layer_idx in range(len(attn_weights))],
            dim=0,
        )
        for sample_idx in range(num_samples)
    ]


def scoring_layer_indices(num_layers: int, score_layers: Optional[str]) -> List[int]:
    """Which layer indices get a vote in `aggregate_attention_score`.

    Pure integer arithmetic, no tensors, so the layer-restriction policy is
    unit-testable without a model -- and so the "does this selection do what
    its name says" question is answered in one place rather than inside a
    tensor pipeline.

    `None` means every layer (the reference implementation's behavior and the
    default). Every named selection drops EARLY layers and keeps late ones:
    layers 0-1 are near-universally positional/sink-dominated, and under the
    default `max` aggregation a single peaked early head can set the entire
    importance vector by itself. See ACCURACY_IMPROVEMENTS.md §1.2.

    Always returns at least one layer -- a selection that would empty the
    list on a very shallow model falls back to the last layer rather than
    producing a NaN score vector further down.
    """
    if score_layers is None:
        return list(range(num_layers))
    if score_layers == "skip_first2":
        start = 2
    elif score_layers == "second_half":
        start = num_layers // 2
    elif score_layers == "last_quarter":
        start = (3 * num_layers) // 4
    else:
        raise ValueError(f"unknown score_layers: {score_layers!r}")
    start = min(start, max(num_layers - 1, 0))
    return list(range(start, num_layers))


def _collapse_layer_head(attn: torch.Tensor, score_aggregation: str) -> torch.Tensor:
    """Collapse a [layer*head, look_ahead_cnt, context_len] tensor down the
    (layer, head) axis, per `score_aggregation`. See `SpecConfig`'s field
    docstring for what each mode is for.

    `zmean` normalizes each (layer, head)'s distribution over the CONTEXT
    axis before averaging, so heads contribute their *shape* rather than
    their magnitude. The epsilon guards a head whose pooled distribution is
    genuinely flat (zero variance across the context), which is a real case
    for a sink-dominated head after average pooling, not a hypothetical --
    without it that head alone turns the whole score vector into NaN.
    """
    if score_aggregation == "max":
        return attn.max(0)[0]
    if score_aggregation == "mean":
        return attn.mean(0)
    if score_aggregation == "zmean":
        mean = attn.mean(dim=-1, keepdim=True)
        std = attn.std(dim=-1, keepdim=True)
        return ((attn - mean) / (std + 1e-6)).mean(0)
    raise ValueError(f"unknown score_aggregation: {score_aggregation!r}")


def aggregate_attention_score(
    attn_scores: List[torch.Tensor],
    spec_config: SpecConfig,
) -> List[torch.Tensor]:
    """Algorithm line 14: A <- aggregate_attention_score(A).

    softmax -> optional smoothing pool -> collapse (layer, head) -> mean over
    lookahead steps, producing one importance score per prompt token.

    The collapse step is `max` by default -- the reference implementation's
    behavior, so an unconfigured run reproduces every already-published row
    exactly -- and `spec_config.score_aggregation` /
    `spec_config.score_layers` select the alternatives (`mean`, `zmean`, and
    late-layer-only voting). Those exist because ORACLE-k20 attributed 17.0
    of `scbench_kv`'s 25.0-point degradation to the speculator's estimation
    error, against 8.0 for the sparse-decode mechanism, while this whole
    function costs ~0.007% of a turn's FLOPs. See ACCURACY_IMPROVEMENTS.md.

    Args:
        attn_scores: per-sample [num_layer, num_head, look_ahead_cnt, context_len]
            tensors, as returned by compute_attention_score.
        spec_config: supplies `pool_kernel_size`, `score_aggregation`,
            `score_layers`.

    Returns:
        Per-sample 1D [context_len] token-importance tensors.
    """
    token_importance: List[torch.Tensor] = []

    for attn in attn_scores:
        original_dtype = attn.dtype
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
            original_dtype
        )

        # Layer restriction happens BEFORE the flatten below -- dim 0 is the
        # layer axis only until (layer, head) are folded together.
        layer_indices = scoring_layer_indices(
            attn.shape[0], spec_config.score_layers
        )
        if len(layer_indices) != attn.shape[0]:
            attn = attn[layer_indices]

        # Flatten (layer, head) into one axis so pooling/collapse apply
        # uniformly.
        attn = attn.flatten(0, 1)

        kernel_size = spec_config.pool_kernel_size
        if kernel_size:
            attn = torch.nn.functional.avg_pool1d(
                attn,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )

        attn = _collapse_layer_head(attn, spec_config.score_aggregation)
        attn = attn.mean(0)  # mean over lookahead steps

        token_importance.append(attn)

    return token_importance


def _chunked_topk_indices(
    sample_ti: torch.Tensor, seq_len: int, chunk_size: int, percentage: float
) -> torch.Tensor:
    """Vectorized replacement for a per-chunk Python loop that called
    `.mean()` once per TOTAL chunk and `.item()` once per KEPT chunk --
    O(context_len / chunk_size) separate op dispatches (e.g. ~2,750 chunk
    ops for an 88k-token SCBench context at chunk_size=32), each `.item()`
    additionally a GPU/device sync point under `enforce_eager=True` (no
    CUDA-graph batching to amortize it) on real hardware.

    Pads to a `chunk_size` multiple (zero-padding is safe here: dividing
    each chunk's sum by its own REAL element count, not `chunk_size`,
    means the zero padding never perturbs the mean) so all chunk means can
    be computed in one reshape+sum+divide, then builds the kept token-index
    set for the selected chunks via broadcasting instead of a per-chunk
    `.item()` + `torch.split`-indexing loop. Verified bit-identical output
    against the original per-chunk-loop implementation across context
    lengths both divisible and non-divisible by `chunk_size` -- see
    `test_vllm_patch.py`.
    """
    pad = (-seq_len) % chunk_size
    padded = torch.nn.functional.pad(sample_ti, (0, pad))  # zero-padded
    chunk_cnt = padded.shape[0] // chunk_size
    chunk_ti = padded.view(chunk_cnt, chunk_size)

    real_counts = torch.full(
        (chunk_cnt,), chunk_size, dtype=chunk_ti.dtype, device=chunk_ti.device
    )
    if pad:
        real_counts[-1] = chunk_size - pad
    chunk_means = chunk_ti.sum(-1) / real_counts

    keep_chunk_cnt = math.ceil(chunk_cnt * percentage)
    _, chunk_indices = torch.topk(chunk_means, k=keep_chunk_cnt, dim=-1)

    starts = chunk_indices * chunk_size
    offsets = torch.arange(chunk_size, device=sample_ti.device)
    token_indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    return token_indices[token_indices < seq_len]


def chunk_select_from_smoothed_attention(
    token_importance: List[torch.Tensor],
    spec_config: SpecConfig,
) -> List[torch.LongTensor]:
    """Algorithm line 16: T <- chunk_select_from_smoothed_attention(A).

    Groups tokens into chunks of `keep_kwargs.chunk_size`, averages
    importance per chunk, and keeps the top-k% chunks by
    `keep_kwargs.percentage`. Falls back to flat (non-chunked) top-k% if
    `keep_kwargs.chunk` is false.

    Returns:
        Per-sample sorted 1D LongTensor of kept token indices (T).

    Sorting by index (not by score) is load-bearing for the multi-turn
    DISCARD history mode's "monotonic extension" property — see
    `pruner.py`'s `conversation_state.py` docstring and
    EXPERIMENT_PLAN.md's "KEEP vs. DISCARD candidate pools" section: because
    this always returns indices in original-position order, DISCARD's
    survivors are never silently reordered turn to turn, so turn N's final
    pruned prompt is guaranteed to be turn N-1's plus a suffix, with no
    extra bookkeeping needed here to enforce that.
    """
    kept_indices = []

    for sample_ti in token_importance:
        seq_len = len(sample_ti)
        percentage = spec_config.keep_kwargs.get("percentage", 1.0)

        if spec_config.keep_kwargs.get("chunk", False):
            chunk_size = spec_config.keep_kwargs.get("chunk_size", 32)
            indices = _chunked_topk_indices(sample_ti, seq_len, chunk_size, percentage)
        else:
            topk = math.ceil(seq_len * percentage)
            _, indices = torch.topk(sample_ti, k=topk, dim=-1)

        kept_indices.append(torch.sort(indices)[0])

    return kept_indices


def score_and_select_indices(
    query_buffer: List[torch.Tensor],
    key_buffer_per_layer: List[torch.Tensor],
    actual_look_ahead_cnt: int,
    spec_config: SpecConfig,
) -> List[int]:
    """One-sample convenience wrapper chaining lines 12/14/16 above
    (`compute_attention_score` -> `aggregate_attention_score` ->
    `chunk_select_from_smoothed_attention`) for the single-conversation-
    at-a-time case this pipeline always calls with -- factored out so
    every caller that has a `(query_buffer, key_buffer_per_layer)` pair in
    hand and just wants "which local indices to keep" doesn't duplicate
    the 3-call sequence. Two real callers, in two different PROCESSES:
    `pruner.py`'s `_score_and_select` (oracle path, driver-side, scores
    the TARGET's own Q/K) and `speculator_worker.py`'s
    `SpeculatorGPUModelRunner.end_capture_and_score` (speculator path,
    runs IN-PROCESS inside the speculator's own worker so only the
    resulting small index list -- not the Q/K tensors themselves -- ever
    needs to cross `collective_rpc`'s process boundary; see that method's
    own docstring for the real, measured cost this avoids).

    Caller must not call this with `actual_look_ahead_cnt == 0` (aggregating
    over zero lookahead steps produces silent NaN, not an error -- both
    real callers already guard this themselves before calling in)."""
    key_buffer = [[k] for k in key_buffer_per_layer]  # one sample
    attn_scores = compute_attention_score(query_buffer, key_buffer, [actual_look_ahead_cnt])
    token_importance = aggregate_attention_score(attn_scores, spec_config)
    kept_local_indices = chunk_select_from_smoothed_attention(token_importance, spec_config)[0]
    return kept_local_indices.tolist()
