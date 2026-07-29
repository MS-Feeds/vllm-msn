"""Speculative Prefill scoring math — Algorithm 1, lines 12, 14, 16.

Engine- and architecture-agnostic: pure tensor-in/tensor-out functions, no
vLLM engine dependency. Ported from the paper's reference implementation
(../speculative_prefill/speculative_prefill/vllm_patch/worker/
look_ahead_spec_worker.py's `_get_attention_scores`/
`_token_importance_from_attn_scores`/`_get_kept_indices_from_token_importance`),
renamed to match Algorithm 1's own step names:

    12: A <- compute_attention_score(Q, K)
    14: A <- aggregate_attention_score(A)
    16: T <- chunk_select_from_smoothed_attention(A)   # T = kept token indices

Lines 1-11 (batch splitting, the lookahead loop, Q/K retrieval) live in
prefill_split.py / proposer.py / kv_cache_utils.py. Lines 18-20 (position-id
restoration, request merging, target forward) are out of scope for this pass
— see EXPERIMENT_PLAN.md's "Implementation status".
"""

import math
from typing import List

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


def aggregate_attention_score(
    attn_scores: List[torch.Tensor],
    spec_config: SpecConfig,
) -> List[torch.Tensor]:
    """Algorithm line 14: A <- aggregate_attention_score(A).

    softmax -> optional smoothing pool -> max over (layer, head) -> mean over
    lookahead steps, producing one importance score per prompt token.

    Args:
        attn_scores: per-sample [num_layer, num_head, look_ahead_cnt, context_len]
            tensors, as returned by compute_attention_score.
        spec_config: supplies `pool_kernel_size`.

    Returns:
        Per-sample 1D [context_len] token-importance tensors.
    """
    token_importance: List[torch.Tensor] = []

    for attn in attn_scores:
        original_dtype = attn.dtype
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
            original_dtype
        )

        # Flatten (layer, head) into one axis so pooling/max apply uniformly.
        attn = attn.flatten(0, 1)

        kernel_size = spec_config.pool_kernel_size
        if kernel_size:
            attn = torch.nn.functional.avg_pool1d(
                attn,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )

        attn = attn.max(0)[0]  # max over (layer, head)
        attn = attn.mean(0)  # mean over lookahead steps

        token_importance.append(attn)

    return token_importance


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
    """
    kept_indices = []

    for sample_ti in token_importance:
        seq_len = len(sample_ti)
        percentage = spec_config.keep_kwargs.get("percentage", 1.0)

        if spec_config.keep_kwargs.get("chunk", False):
            chunk_size = spec_config.keep_kwargs.get("chunk_size", 32)
            chunk_ti = torch.split(sample_ti, chunk_size, dim=-1)
            chunk_ti = [cti.mean() for cti in chunk_ti]
            chunk_cnt = len(chunk_ti)

            keep_chunk_cnt = math.ceil(chunk_cnt * percentage)
            _, chunk_indices = torch.topk(
                torch.stack(chunk_ti), k=keep_chunk_cnt, dim=-1
            )
            indices = torch.split(torch.arange(seq_len), chunk_size, dim=-1)
            indices = torch.concat([indices[ci.item()] for ci in chunk_indices])
        else:
            topk = math.ceil(seq_len * percentage)
            _, indices = torch.topk(sample_ti, k=topk, dim=-1)

        kept_indices.append(torch.sort(indices)[0])

    return kept_indices
