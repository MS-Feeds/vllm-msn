#!/usr/bin/env python3
"""Model FLOP Utilization (MFU) and Model Bandwidth Utilization (MBU) for
the target model (google/gemma-4-26B-A4B-it).

ACTIVE PARAMETER COUNT -- verified against the real implementation in
vllm/model_executor/models/gemma4.py, not guessed from config.json field
names. Two facts that aren't obvious from the config alone and would
silently produce a wrong formula if assumed rather than checked:

1. Each of the 30 layers runs a dense MLP block AND a routed MoE block
   "in parallel" (Gemma4DecoderLayer, comment at gemma4.py:644 -- "MoE
   ... block parallel to MLP"), not MoE-only. Only the router-selected
   top_k=8 (of num_experts=128) experts are actually computed per token;
   the other 120 experts' weights are loaded into memory but not touched
   by this token's forward pass, so they're excluded from both the FLOPs
   and bytes-moved accounting here.

2. Attention head dims are heterogeneous by layer type
   (gemma4.py:573-594): sliding_attention layers use head_dim,
   num_key_value_heads; full_attention layers use global_head_dim
   (typically 2x head_dim) AND, when attention_k_eq_v is set, V is
   derived directly from K -- there is no v_proj weight at all for those
   layers ("the checkpoint has no v_proj", gemma4.py:410) -- and the KV
   head count drops to num_global_key_value_heads.

Computed this way for google/gemma-4-26B-A4B-it, active params come out
to ~3.82B, matching the model's own "A4B" (Active 4B) name -- a useful
sanity check that this is structurally correct, not just plausible.

KNOWN SIMPLIFICATIONS (documented, not hidden):

- Only the target model's FLOPs/bytes are counted. When spec_decode is
  on, the MTP assistant/draft model (google/gemma-4-26B-A4B-it-assistant)
  also does a forward pass every round, but it's a much smaller model
  with additional components (a "centroid" mechanism, a
  backbone-hidden-state projection) not reverse-engineered here. MFU/MBU
  for spec-decode-on experiments therefore under-count total GPU work --
  treat them as a lower bound, not an exact figure, when spec_decode=True.
- The attention FLOPs term (quadratic in context length) uses the
  average context length across the batch, not the true per-request
  distribution. Sliding-attention layers cap this at sliding_window
  tokens regardless of true context length (Gemma4Attention only
  attends within the window), so for the context lengths this pipeline
  uses this term is a minor correction on top of the parameter-
  proportional term, not the dominant one.

Published hardware peak specs (BF16 tensor core, dense/no-sparsity,
NVIDIA's own datasheets) -- add entries here for other GPUs as needed.
"""

from __future__ import annotations

BYTES_PER_PARAM_BF16 = 2

GPU_SPECS: dict[str, dict[str, float]] = {
    "NVIDIA A100-SXM4-80GB": dict(peak_tflops_bf16=312.0, peak_bandwidth_gbps=2039.0),
    "NVIDIA A100-SXM4-40GB": dict(peak_tflops_bf16=312.0, peak_bandwidth_gbps=1555.0),
    "NVIDIA A100 80GB PCIe": dict(peak_tflops_bf16=312.0, peak_bandwidth_gbps=1935.0),
    "NVIDIA H100 80GB HBM3": dict(peak_tflops_bf16=989.0, peak_bandwidth_gbps=3350.0),
}


def detect_gpu_specs(device_name: str | None = None) -> dict[str, float] | None:
    """Looks up peak FLOPs/bandwidth for the current (or given) GPU.
    Returns None if unrecognized -- callers must handle that by skipping
    MFU/MBU rather than reporting a wrong number against a guessed spec."""
    if device_name is None:
        import torch

        if not torch.cuda.is_available():
            return None
        device_name = torch.cuda.get_device_name(0)
    return GPU_SPECS.get(device_name)


def _get_text_config(config):
    """Same helper as gemma4.py's _get_text_config: Gemma4Config nests
    the text-generation config under .text_config; dereference it if
    present, otherwise assume config already is the text config."""
    return getattr(config, "text_config", config)


def compute_active_params(config) -> dict[str, int]:
    """Computes active (not total) parameter count for one forward pass
    of one token, broken down by component for transparency/debugging.
    `config` is the value returned by
    transformers.AutoConfig.from_pretrained(MODEL_BASE) -- may or may not
    be nested under .text_config, matching gemma4.py's own handling."""
    cfg = _get_text_config(config)

    hidden_size = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    head_dim = cfg.head_dim
    kv_heads = cfg.num_key_value_heads
    global_head_dim = getattr(cfg, "global_head_dim", head_dim)
    global_kv_heads = getattr(cfg, "num_global_key_value_heads", kv_heads)
    use_k_eq_v_flag = getattr(cfg, "attention_k_eq_v", False)
    intermediate_size = cfg.intermediate_size
    enable_moe = getattr(cfg, "enable_moe_block", False)
    moe_intermediate_size = getattr(
        cfg, "moe_intermediate_size", getattr(cfg, "expert_intermediate_size", 0)
    )
    num_experts = getattr(cfg, "num_experts", 0) or 0
    top_k_experts = getattr(cfg, "top_k_experts", 0) or 0
    vocab_size = cfg.vocab_size

    def attn_params(this_head_dim: int, this_kv_heads: int, has_v: bool) -> int:
        q = hidden_size * (n_heads * this_head_dim)
        k = hidden_size * (this_kv_heads * this_head_dim)
        v = k if has_v else 0
        o = (n_heads * this_head_dim) * hidden_size
        return q + k + v + o

    dense_mlp_params = (
        hidden_size * (2 * intermediate_size) + intermediate_size * hidden_size
    )

    if enable_moe and num_experts and top_k_experts:
        router_params = hidden_size * num_experts
        moe_active_params = top_k_experts * (
            hidden_size * (2 * moe_intermediate_size)
            + moe_intermediate_size * hidden_size
        )
    else:
        router_params = 0
        moe_active_params = 0

    attn_total = 0
    for layer_idx in range(n_layers):
        is_full = layer_types[layer_idx] == "full_attention"
        if is_full:
            use_k_eq_v = use_k_eq_v_flag
            this_head_dim = global_head_dim
            this_kv_heads = global_kv_heads if use_k_eq_v else kv_heads
            attn_total += attn_params(
                this_head_dim, this_kv_heads, has_v=not use_k_eq_v
            )
        else:
            attn_total += attn_params(head_dim, kv_heads, has_v=True)

    mlp_total = n_layers * dense_mlp_params
    router_total = n_layers * router_params
    moe_total = n_layers * moe_active_params
    embedding_params = vocab_size * hidden_size  # tied -- counted once

    return {
        "attn_params": attn_total,
        "dense_mlp_params": mlp_total,
        "router_params": router_total,
        "moe_active_params": moe_total,
        "embedding_params": embedding_params,
        "total_active_params": (
            attn_total + mlp_total + router_total + moe_total + embedding_params
        ),
    }


def flops_per_token(
    active_params: dict[str, int],
    avg_context_len: float,
    config,
) -> float:
    """2 * active_params (one multiply-add per parameter, standard
    convention) plus the attention QK^T + AV quadratic term, averaged
    over layer types and capped at sliding_window for sliding-attention
    layers (see module docstring)."""
    cfg = _get_text_config(config)
    n_heads = cfg.num_attention_heads
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    head_dim = cfg.head_dim
    global_head_dim = getattr(cfg, "global_head_dim", head_dim)
    sliding_window = getattr(cfg, "sliding_window", None)

    param_flops = 2 * active_params["total_active_params"]

    attn_flops = 0.0
    for layer_idx in range(n_layers):
        is_full = layer_types[layer_idx] == "full_attention"
        this_head_dim = global_head_dim if is_full else head_dim
        ctx = avg_context_len
        if not is_full and sliding_window:
            ctx = min(ctx, sliding_window)
        # QK^T + AV, both O(ctx * n_heads * head_dim), 2 FLOPs per MAC.
        attn_flops += 2 * 2 * ctx * n_heads * this_head_dim

    return param_flops + attn_flops


def bytes_per_token(active_params: dict[str, int], bytes_per_param: int = BYTES_PER_PARAM_BF16) -> int:
    """Bytes read from HBM for the active weights (attention + dense MLP
    + router + selected experts + embedding/lm_head) for ONE decode
    STEP -- not one output token. A step reads the weights once and
    produces one token for every concurrently-running request, so
    callers computing MBU under batching must divide by the concurrent
    batch size before multiplying by this value (see run_pipeline.py's
    decode_steps_per_second) -- multiplying by raw aggregate tokens/sec
    instead counts one full weight-read per request per token rather
    than one per step, and reports MBU > 100%, which is physically
    impossible (you cannot exceed the hardware's rated peak bandwidth).
    An earlier version of this docstring claimed >100% was an expected
    batching effect -- that was wrong; fixed after real GPU runs
    reported MBU in the 500-1500% range, which was the tell."""
    return active_params["total_active_params"] * bytes_per_param


def compute_mfu(tokens_per_second: float, flops_per_tok: float, peak_tflops_bf16: float) -> float:
    achieved_flops = tokens_per_second * flops_per_tok
    peak_flops = peak_tflops_bf16 * 1e12
    return achieved_flops / peak_flops if peak_flops > 0 else 0.0


def compute_mbu(tokens_per_second: float, bytes_per_tok: int, peak_bandwidth_gbps: float) -> float:
    achieved_bytes_per_sec = tokens_per_second * bytes_per_tok
    peak_bytes_per_sec = peak_bandwidth_gbps * 1e9
    return achieved_bytes_per_sec / peak_bytes_per_sec if peak_bytes_per_sec > 0 else 0.0
