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

CLOSED GAPS (previously documented simplifications, now fixed -- kept here
so the history of what was wrong and why is not lost):

- MFU no longer blends prefill and decode tokens into one rate. Decode's
  per-token FLOPs figure is divided by mean_accept_length and (under
  spec decode) includes the draft model's FLOPs -- both are decode-only
  corrections. Multiplying that single figure by (prompt+output)/elapsed
  (run_pipeline.py's old approach) silently attributed both corrections
  to prefill tokens too, which never go through a verification round or
  a draft model at all. Fixed by computing prefill and decode achieved-
  FLOPs as two separate compute_mfu() calls, summed (see
  run_pipeline.py's evaluate()) -- prefill tokens are charged one full,
  undivided target forward pass each, at a prefill-shaped context length
  (average L/2 for a length-L prompt processed in one causal pass, not
  the decode side's prompt_len+output_len/2, which assumes the whole
  prompt is already cached). Confirmed by a hand-worked example
  (test_mfu_split_prefill_decode_matches_true_flops_not_blended) that the
  old blended approach OVERSTATED true MFU under spec decode:
  decode_flops_per_tok's numerator scales by (mtp_k+1) (see the next
  bullet) but is only divided by mean_accept_length, which is always
  <= mtp_k+1 since not every candidate position gets accepted -- so
  decode_flops_per_tok is systematically >= a single prefill token's
  true cost, and blending it onto prefill tokens (which should cost
  exactly one undivided target pass each) inflated the total.
- The target model's own per-round FLOPs (flops_per_token_spec_decode)
  now scales by (mtp_k + 1) when mtp_k >= 1: the target's single
  verification pass actually processes mtp_k+1 candidate positions at
  once, not the 1 position flops_per_token() alone accounts for. Was
  previously left at 1x, undercounting the target's own contribution
  under spec decode. At mtp_k <= 0 this is a no-op (flops_per_token()'s
  1-position convention is already exact there -- no verification round
  exists).
- bytes_per_decode_step() now includes a KV WRITE term (one new token's
  K/V written per step), not just the read term -- this applies to
  EVERY decode step, spec decode or not (an ordinary autoregressive step
  also writes exactly one new token's K/V). Previously uncounted
  entirely. bytes_per_decode_step_spec_decode()'s own write term is
  mtp_k additional writes on top of the 1 already included in the base,
  reaching the correct mtp_k+1 total.

KNOWN SIMPLIFICATIONS (documented, not hidden):

- The attention FLOPs term (quadratic in context length) and the KV
  cache bytes term both take a single avg_context_len figure, not the
  true per-step, per-request distribution. The caller (run_pipeline.py)
  passes the batch's MIDPOINT context length (prompt_len +
  output_len/2, not the final prompt_len+output_len) for the decode
  side (a separate, smaller prefill-shaped context length is used for
  MFU's prefill term -- see CLOSED GAPS above) since FLOPs/bytes are
  step-level quantities averaged over every decode step, and context
  grows from prompt_len up to prompt_len+output_len over the course of
  generation -- using the final length instead would overestimate both
  terms by close to 2x for long generations. This midpoint approximation
  assumes roughly linear, stationary growth across requests and (under
  speculative decoding) stationary acceptance length across the
  sequence; neither is tracked precisely here. Sliding-attention layers
  cap this at sliding_window tokens regardless of true context length
  (Gemma4Attention only attends within the window).
- batch_size (used throughout for weight-read sharing and KV-term
  scaling) is num_prompts, a flat constant for the whole decode window --
  it assumes every request submitted in one generate() call stays
  concurrently active for the WHOLE window. Real vLLM continuous
  batching lets faster-finishing requests drop out mid-call, so this is
  an upper bound on true concurrency whenever output-length variance
  within a batch is high (unaffected by any fix in this file; noted here
  since it's the other major "batching" approximation alongside
  avg_context_len).
- Draft-model (speculative decoding) accounting is REAL, not a lower
  bound, PROVIDED callers use the *_spec_decode variants
  (bytes_per_decode_step_spec_decode, flops_per_token_spec_decode) with a
  real draft_config/draft_active_params -- see the DRAFT MODEL section
  below compute_active_params for the full, source-verified architecture.
  compute_active_params/flops_per_token/bytes_per_token/
  kv_cache_bytes_per_token themselves remain target-only; callers that
  don't pass draft_config silently get target-only numbers back, which
  IS still a lower bound in that case.

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
    """Bytes read from HBM for the active WEIGHTS ONLY (attention + dense
    MLP + router + selected experts + embedding/lm_head) for ONE decode
    STEP -- not one output token. A step reads the weights once and
    produces one token for every concurrently-running request, so
    callers computing MBU under batching must divide by the concurrent
    batch size before multiplying by this value (see run_pipeline.py's
    decode_steps_per_second) -- multiplying by raw aggregate tokens/sec
    instead counts one full weight-read per request per token rather
    than one per step, and reports MBU > 100%, which is physically
    impossible (you cannot exceed the hardware's rated peak bandwidth).

    This does NOT include KV cache bytes -- see kv_cache_bytes_per_token()
    for that term and bytes_per_decode_step() for the combined total that
    MBU should actually be computed against. Kept as a standalone
    weights-only function since it's also used to sanity-check active
    param counts against checkpoint size independent of any KV/context
    assumption."""
    return active_params["total_active_params"] * bytes_per_param


def _kv_shape_bytes_for_layer_type(
    is_full: bool,
    config,
    bytes_per_param: int,
) -> float:
    """Per-cached-token KV bytes for ONE attention layer of the given
    type (full vs sliding), sized off `config`'s head_dim/kv_heads/
    attention_k_eq_v fields (matching compute_active_params's
    this_kv_heads selection exactly -- see kv_cache_bytes_per_token's
    docstring for why the head count, not just head_dim, matters here).

    Deliberately does NOT apply the sliding_window cap -- that's a
    property of whichever layer/config actually issues the read at
    runtime, which is not always the same config as the one whose KV
    tensor is physically being read (see
    draft_kv_read_bytes_per_forward_pass, where a draft layer's OWN
    sliding_window governs the cap even though the tensor it reads is
    TARGET-shaped). Callers apply the cap themselves."""
    cfg = _get_text_config(config)
    head_dim = cfg.head_dim
    kv_heads = cfg.num_key_value_heads
    global_head_dim = getattr(cfg, "global_head_dim", head_dim)
    global_kv_heads = getattr(cfg, "num_global_key_value_heads", kv_heads)
    use_k_eq_v_flag = getattr(cfg, "attention_k_eq_v", False)

    if is_full:
        this_head_dim = global_head_dim
        this_kv_heads = global_kv_heads if use_k_eq_v_flag else kv_heads
        kv_multiplier = 1 if use_k_eq_v_flag else 2
    else:
        this_head_dim = head_dim
        this_kv_heads = kv_heads
        kv_multiplier = 2

    return kv_multiplier * this_kv_heads * this_head_dim * bytes_per_param


def kv_cache_bytes_per_token(
    avg_context_len: float,
    config,
    bytes_per_param: int = BYTES_PER_PARAM_BF16,
) -> float:
    """Bytes read from the KV cache for ONE request's attention at one
    decode step, at avg_context_len context length. Companion to
    bytes_per_token() (weights only) -- flops_per_token() already
    accounts for an equivalent attention term on top of param_flops, but
    bytes_per_token() had no such term, silently undercounting HBM
    traffic for every layer's K/V reads.

    Mirrors flops_per_token's per-layer-type branching (is_full,
    sliding_window cap) plus compute_active_params's attention_k_eq_v
    handling (same getattr defaults, same field names) since KV byte
    accounting -- unlike the FLOPs term -- depends on the actual KV head
    count, not just head_dim:

    - sliding_attention layers: always 2x (K and V both read), kv_heads/
      head_dim, context capped at sliding_window (Gemma4Attention only
      attends within the window regardless of true context length).
    - full_attention layers: global_head_dim; if attention_k_eq_v is set,
      V is derived from K at attention time rather than read from a
      separate cache -- only K is read (1x, num_global_key_value_heads).
      Otherwise both K and V are read (2x, num_global_key_value_heads if
      attention_k_eq_v is set else num_key_value_heads) -- matching
      compute_active_params's this_kv_heads selection exactly, since a
      wrong head count here would silently produce a plausible-looking
      but incorrect byte count.

    This is a PER-REQUEST, per-step figure -- unlike weight bytes (read
    once per step and shared across the whole batch), every
    concurrently-running request has its own KV cache and must be read
    separately, so callers must multiply this by the batch size before
    adding it to weight bytes (see bytes_per_decode_step() /
    run_pipeline.py's evaluate()).

    Also reused, at avg_context_len=1.0, as the per-position KV WRITE
    byte footprint (same per-layer-type shape/multiplier logic applies
    whether the position is being read or written) -- see
    bytes_per_decode_step_spec_decode()."""
    cfg = _get_text_config(config)
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    sliding_window = getattr(cfg, "sliding_window", None)

    kv_bytes = 0.0
    for layer_idx in range(n_layers):
        is_full = layer_types[layer_idx] == "full_attention"
        unit_bytes = _kv_shape_bytes_for_layer_type(is_full, config, bytes_per_param)

        ctx = avg_context_len
        if not is_full and sliding_window:
            ctx = min(ctx, sliding_window)

        kv_bytes += unit_bytes * ctx

    return kv_bytes


def bytes_per_decode_step(
    active_params: dict[str, int],
    avg_context_len: float,
    config,
    batch_size: int,
    bytes_per_param: int = BYTES_PER_PARAM_BF16,
) -> float:
    """Total HBM bytes moved for ONE decode step across the whole batch --
    the correct "bytes moved" figure for compute_mbu(), combining
    bytes_per_token() (weights) and kv_cache_bytes_per_token() for both
    the READ (existing cached context) and WRITE (this step's own new
    token) sides of the KV cache, with the right scaling for each:
    weights are read ONCE per step and shared by every concurrently-
    running request, but each request has its own KV cache that must be
    read AND written separately, so both KV terms -- unlike the weight
    term -- scale with batch_size.

    The WRITE term applies here too, not just under speculative decoding:
    every ordinary decode step writes exactly one new token's K/V into
    the cache, same as the speculative-decoding case reduces to at
    mtp_k=0 (see bytes_per_decode_step_spec_decode). It was uncounted
    entirely before -- a real, if usually small, omission (see the
    module docstring) that grows in relative significance at large batch
    sizes / long context where KV traffic dominates weight bytes, same
    dynamic as the read term.

    Flat-summing weight+KV (no batch_size factor on the KV terms)
    undercounts KV traffic by exactly batch_size and, for long-context/
    large-batch runs where KV traffic dominates, was undercounting
    *total* MBU bytes for such runs, not just missing a small term."""
    weight_bytes = bytes_per_token(active_params, bytes_per_param)
    kv_read_bytes = kv_cache_bytes_per_token(avg_context_len, config, bytes_per_param)
    kv_write_bytes = kv_cache_bytes_per_token(1.0, config, bytes_per_param)
    return weight_bytes + batch_size * (kv_read_bytes + kv_write_bytes)


# ---------------------------------------------------------------------------
# DRAFT MODEL (speculative decoding) -- google/gemma-4-26B-A4B-it-assistant.
#
# Verified against the actual implementation (vllm/model_executor/models/
# gemma4_mtp.py: Gemma4MTP, Gemma4MultiTokenPredictor, Gemma4MTPAttention,
# Gemma4MTPMaskedEmbedder; vllm/v1/spec_decode/gemma4.py: Gemma4Proposer;
# vllm/v1/spec_decode/llm_base_proposer.py: SpecDecodeBaseProposer), not
# guessed from the checkpoint name -- same rigor as compute_active_params
# above. Facts that would silently produce a wrong formula if assumed:
#
# 1. Forward-pass multiplicities are NOT symmetric. Per verification
#    round: the draft model runs mtp_k SEQUENTIAL forward passes (one
#    initial pass + (mtp_k - 1) more in a loop, confirmed in
#    SpecDecodeBaseProposer.propose()) to produce mtp_k draft tokens; the
#    target model runs exactly ONE forward pass, verifying all mtp_k+1
#    candidate positions (the mtp_k draft tokens plus the bonus-eligible
#    position) at once (vllm/v1/worker/gpu_model_runner.py calls
#    self.model(...) once per step, then rejection_sampler consumes that
#    single pass's logits -- no further target forward passes).
#
# 2. The draft model has NO KV cache of its own. Gemma4MTPAttention is
#    Q-only (q_proj + o_proj + q_norm; no k_proj/v_proj/k_norm at all --
#    "the checkpoint has no v_proj" is true here too, but stronger: there
#    is no k_proj either). Each draft attention layer is wired via
#    kv_sharing_target_layer_name to alias the TARGET model's own KV
#    cache tensor directly (gpu_model_runner.py:
#    kv_caches[layer_name] = kv_caches[target_layer_name] -- a literal
#    tensor-object aliasing, not a copy), and Attention.forward() skips
#    the cache-WRITE op entirely whenever kv_sharing_target_layer_name is
#    set. So: the draft contributes zero KV write bytes, and its KV READ
#    bytes must be sized using the TARGET's per-layer-type head_dim/
#    kv_heads/kv_multiplier (that's the tensor actually being read), not
#    the draft's own attention config -- see
#    draft_kv_read_bytes_per_forward_pass(). _setup_gemma4_kv_sharing()
#    maps EVERY draft layer of a given attention type to "the last"
#    target layer of that type, so if the draft has more than one layer
#    of the same type, that one target layer's cache gets read multiple
#    times per forward pass -- each is still a real, separate HBM read
#    (not deduplicated), so the read cost is summed over the draft's OWN
#    layer count, not the target's.
#
# 3. Gemma4Proposer sets constant_draft_positions=True: all mtp_k draft
#    passes within one round attend to the SAME context length (context
#    does not grow across draft steps, unlike ordinary autoregressive
#    decoding) -- so draft KV read bytes per round is mtp_k times ONE
#    read at avg_context_len, not a growing series.
#
# 4. embed_tokens and lm_head are NOT both shared with the target, and
#    guessing either way would be wrong:
#    - embed_tokens (input token embedding, used only to build the
#      pre_projection input) IS runtime-aliased to the target's own
#      embedding table. SpecDecodeBaseProposer._maybe_share_embeddings()
#      takes an unconditional "MTP model" sharing branch specifically
#      because Gemma4MTP does not declare has_own_embed_tokens (that
#      attribute only exists on EAGLE-style models in this codebase) --
#      so embed_tokens contributes ZERO marginal params/bytes here,
#      already counted in the target's own compute_active_params().
#    - lm_head (used for the draft's own logits/centroid computation) is
#      explicitly NOT shared: Gemma4Proposer overrides
#      _maybe_share_lm_head() with a no-op, logged as "keeping draft
#      model's own lm_head (draft_dim != backbone_dim)" -- a real,
#      separate [vocab_size, hidden_size] weight at the draft's OWN
#      (smaller) hidden_size, not backbone_hidden_size.
#
# 5. The "centroid mechanism" (Gemma4MTPMaskedEmbedder, active when the
#    draft config's use_ordered_embeddings is set) replaces the full
#    lm_head matmul with a sparse gather: hidden_states are projected to
#    num_centroids cluster scores (a small, always-fully-read Linear),
#    top-K centroids are selected, and logits are computed only against
#    the num_selected = centroid_intermediate_top_k * (vocab_size //
#    num_centroids) tokens belonging to those clusters -- NOT the full
#    vocab_size rows of the lm_head table. This is architecturally the
#    same kind of sparsity as the target's top_k-of-num_experts MoE
#    routing (Fact #1 in the target section above), just applied to the
#    output projection instead of the MLP. use_ordered_embeddings
#    defaults to False (matching Gemma4MTP.__init__'s own getattr
#    default) if the draft checkpoint's config doesn't set it, in which
#    case the full [vocab_size, hidden_size] lm_head is read every pass.
#
# 6. The "backbone-hidden-state projection" is pre_projection
#    (Linear(2*backbone_hidden_size, hidden_size)) and post_projection
#    (Linear(hidden_size, backbone_hidden_size)): pre_projection
#    compresses [own-token-embedding ++ target-hidden-state] (both
#    backbone_hidden_size-wide -- embed_input_ids scales its embedding
#    to backbone_hidden_size via the target's convention, confirmed by
#    the 2*backbone_hidden_size input width) down to the draft's own
#    smaller hidden_size; post_projection expands back up to
#    backbone_hidden_size to feed the next draft step. Both are real,
#    always-fully-read weights. backbone_hidden_size is read off the
#    draft's OUTER config (getattr(draft_config, "backbone_hidden_size",
#    hidden_size)), not its .text_config, matching
#    Gemma4MultiTokenPredictor.__init__ exactly.
#
# 7. Per-layer MLP is dense-only (Gemma4MLP -- the SAME class the target
#    model uses for its own dense-MLP block, imported directly from
#    gemma4.py) -- there is no MoE/router at the draft level.
#
# 8. Target model KV WRITE bytes (not read) were previously uncounted
#    entirely, for both spec-decode and non-spec-decode alike, but this
#    is fixed ONLY for the spec_decode=True path here (see
#    bytes_per_decode_step_spec_decode) since the non-spec path must stay
#    numerically unchanged. Every verification round writes K/V for all
#    mtp_k+1 candidate positions UNCONDITIONALLY during the single target
#    forward pass -- attention must compute K/V for a position before it
#    can produce the logits used to reject it. Rejected positions are
#    only logically rolled back afterward (the scheduler decrements
#    request.num_computed_tokens by num_rejected -- see
#    vllm/v1/core/sched/scheduler.py -- and
#    single_type_kv_cache_manager.py explicitly documents that "a
#    request's blocks may be allocated for draft tokens which are later
#    rejected"); the physical bytes are left stale until the next round's
#    slot_mapping overwrites them, not erased. So target KV write bytes
#    scale with mtp_k+1, not with however many tokens end up accepted.
# ---------------------------------------------------------------------------


def compute_draft_active_params(draft_config) -> dict[str, int]:
    """Active parameter count for ONE of the draft model's mtp_k
    sequential forward passes -- see the DRAFT MODEL section above for
    the source-verified architecture this implements. `draft_config` is
    transformers.AutoConfig.from_pretrained(MODEL_ASSISTANT); same
    .text_config nesting handling as compute_active_params, except
    backbone_hidden_size/use_ordered_embeddings/num_centroids/
    centroid_intermediate_top_k are read off the OUTER config (not
    .text_config), matching Gemma4MultiTokenPredictor.__init__ and
    Gemma4MTP.__init__ exactly -- they are wrapper-level fields, not
    text-generation-config fields, in the actual source."""
    cfg = _get_text_config(draft_config)

    hidden_size = cfg.hidden_size
    backbone_hidden_size = getattr(draft_config, "backbone_hidden_size", hidden_size)
    n_heads = cfg.num_attention_heads
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    head_dim = cfg.head_dim
    global_head_dim = getattr(cfg, "global_head_dim", head_dim)
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    use_ordered_embeddings = getattr(draft_config, "use_ordered_embeddings", False)
    num_centroids = getattr(draft_config, "num_centroids", 2048)
    centroid_top_k = getattr(draft_config, "centroid_intermediate_top_k", 32)

    def attn_params(this_head_dim: int) -> int:
        # Q-only: no k_proj/v_proj at all (see DRAFT MODEL fact #2).
        q = hidden_size * (n_heads * this_head_dim)
        o = (n_heads * this_head_dim) * hidden_size
        return q + o

    attn_total = 0
    for layer_idx in range(n_layers):
        is_full = layer_types[layer_idx] == "full_attention"
        this_head_dim = global_head_dim if is_full else head_dim
        attn_total += attn_params(this_head_dim)

    dense_mlp_params = (
        hidden_size * (2 * intermediate_size) + intermediate_size * hidden_size
    )
    mlp_total = n_layers * dense_mlp_params

    pre_projection_params = 2 * backbone_hidden_size * hidden_size
    post_projection_params = hidden_size * backbone_hidden_size

    if use_ordered_embeddings and num_centroids:
        centroids_params = hidden_size * num_centroids
        vocab_size_per_centroid = vocab_size // num_centroids
        num_selected = centroid_top_k * vocab_size_per_centroid
        lm_head_active_params = num_selected * hidden_size
    else:
        centroids_params = 0
        lm_head_active_params = vocab_size * hidden_size

    embedding_params = 0  # shared with target -- see DRAFT MODEL fact #4

    return {
        "attn_params": attn_total,
        "dense_mlp_params": mlp_total,
        "pre_projection_params": pre_projection_params,
        "post_projection_params": post_projection_params,
        "centroids_params": centroids_params,
        "lm_head_active_params": lm_head_active_params,
        "embedding_params": embedding_params,
        "total_active_params": (
            attn_total
            + mlp_total
            + pre_projection_params
            + post_projection_params
            + centroids_params
            + lm_head_active_params
            + embedding_params
        ),
    }


def draft_kv_read_bytes_per_forward_pass(
    avg_context_len: float,
    target_config,
    draft_config,
    bytes_per_param: int = BYTES_PER_PARAM_BF16,
) -> float:
    """KV cache bytes the draft model reads (from the TARGET's physical
    cache -- see DRAFT MODEL fact #2) for ONE of its mtp_k forward
    passes. Sums over the DRAFT's own layer_types/num_hidden_layers (one
    read per draft layer, including repeats when multiple draft layers
    of the same type map to the same target layer -- fact #2), sizing
    each read with the TARGET's per-type head_dim/kv_heads/kv_multiplier
    (the physical tensor's real shape) but capping at the DRAFT's own
    sliding_window (the cap is enforced by whichever attention module
    issues the read, which is configured from the draft's own config --
    Gemma4MTPAttention.__init__ passes `config.sliding_window if
    self.is_sliding else None` using the draft's own config, not the
    target's, even though it's reading target-shaped data)."""
    draft_cfg = _get_text_config(draft_config)
    draft_layer_types = draft_cfg.layer_types
    draft_n_layers = draft_cfg.num_hidden_layers
    draft_sliding_window = getattr(draft_cfg, "sliding_window", None)

    total = 0.0
    for layer_idx in range(draft_n_layers):
        is_full = draft_layer_types[layer_idx] == "full_attention"
        unit_bytes = _kv_shape_bytes_for_layer_type(is_full, target_config, bytes_per_param)

        ctx = avg_context_len
        if not is_full and draft_sliding_window:
            ctx = min(ctx, draft_sliding_window)

        total += unit_bytes * ctx

    return total


def bytes_per_decode_step_spec_decode(
    target_active_params: dict[str, int],
    avg_context_len: float,
    target_config,
    batch_size: int,
    mtp_k: int,
    draft_config=None,
    draft_active_params: dict[str, int] | None = None,
    bytes_per_param: int = BYTES_PER_PARAM_BF16,
) -> float:
    """Total HBM bytes moved for ONE verification round across the whole
    batch, speculative-decoding-aware. Returns EXACTLY
    bytes_per_decode_step()'s output, unchanged, whenever mtp_k <= 0 or
    draft_config is None -- the non-spec-decode path is untouched.

    Adds three terms on top of the existing target-only base (see DRAFT
    MODEL facts #1-#8 above for why each is shaped this way):

    - Target KV WRITE bytes: mtp_k+1 candidate positions get K/V written
      every round regardless of acceptance (fact #8), scaled by
      batch_size like the existing read term. base already includes ONE
      write (bytes_per_decode_step's own write term, correct for the
      mtp_k=0/ordinary-decode case), so only mtp_k MORE writes are added
      here to reach the full mtp_k+1 total. Reuses
      kv_cache_bytes_per_token(1.0, target_config, ...) -- ctx=1 gives
      exactly the per-position write footprint, since a write's
      per-layer-type shape/multiplier is identical to a read's.
    - Draft model WEIGHT bytes: read once per draft forward pass, shared
      across the whole batch (same reasoning as the target's own weight
      bytes) -- there are mtp_k such passes per round (fact #1).
    - Draft model KV READ bytes: mtp_k passes, each reading
      draft_kv_read_bytes_per_forward_pass, each scaled by batch_size
      (every concurrent request has its own position in the shared
      cache). No draft KV write term exists (fact #2)."""
    base = bytes_per_decode_step(
        target_active_params, avg_context_len, target_config, batch_size, bytes_per_param
    )
    if mtp_k <= 0 or draft_config is None:
        return base

    if draft_active_params is None:
        draft_active_params = compute_draft_active_params(draft_config)

    target_kv_write_bytes_per_position = kv_cache_bytes_per_token(
        1.0, target_config, bytes_per_param
    )
    additional_target_write_bytes = mtp_k * target_kv_write_bytes_per_position * batch_size

    draft_weight_bytes = mtp_k * bytes_per_token(draft_active_params, bytes_per_param)

    draft_kv_read_bytes_per_pass = draft_kv_read_bytes_per_forward_pass(
        avg_context_len, target_config, draft_config, bytes_per_param
    )
    draft_kv_read_bytes = mtp_k * draft_kv_read_bytes_per_pass * batch_size

    return base + additional_target_write_bytes + draft_weight_bytes + draft_kv_read_bytes


def flops_per_token_spec_decode(
    target_active_params: dict[str, int],
    avg_context_len: float,
    target_config,
    mtp_k: int,
    draft_config=None,
    draft_active_params: dict[str, int] | None = None,
) -> float:
    """Total FLOPs for ONE verification round (target's single pass plus
    the draft's mtp_k passes) -- a PER-ROUND figure, matching
    bytes_per_decode_step_spec_decode's framing, NOT a per-accepted-token
    figure. Returns EXACTLY flops_per_token()'s output, unchanged,
    whenever mtp_k <= 0 or draft_config is None (flops_per_token()'s
    "2 * active_params" convention is exactly correct there: at mtp_k=0
    there is no verification round, one forward pass really does process
    exactly one token).

    Callers feeding this into compute_mfu (which expects FLOPs per
    accepted output token, multiplied by aggregate tokens/sec) must
    divide by mean_accept_length themselves -- the same round-to-
    accepted-token conversion run_pipeline.py already applies for MBU's
    decode_steps_per_second -- since this function only knows the
    architecture, not runtime-measured acceptance statistics. Callers
    must also NOT multiply this (already divided) per-accepted-token
    figure by a token rate that includes PREFILL tokens (total_tokens/
    elapsed) -- prefill tokens never go through a verification round or
    draft model at all, so blending them into the same rate silently
    misattributes the mean_accept_length division and the draft's FLOPs
    to tokens that shouldn't carry either. See run_pipeline.py's
    evaluate(), which computes prefill and decode MFU contributions
    separately for exactly this reason.

    When mtp_k >= 1, the TARGET's own contribution is scaled by
    (mtp_k + 1), not left at flops_per_token()'s 1-position figure:
    the target's single verification pass actually processes mtp_k+1
    candidate positions at once (fact #1), so its real FLOPs are
    (mtp_k+1) times what flops_per_token() computes for one position --
    both the "2 * active_params" term (batched matmul FLOPs scale with
    the number of positions in the batch) and the attention term (each
    of the mtp_k+1 positions independently attends to the cached
    context; attending among the mtp_k+1 positions themselves is a much
    smaller additional quadratic term, not modeled separately here,
    since avg_context_len is typically >> mtp_k+1).

    Draft FLOPs reuse flops_per_token() verbatim against the draft's own
    active params/config: the attention FLOPs term only needs n_heads/
    head_dim (both the draft's own), not kv_heads, so this reuse is
    exact, not approximate (see DRAFT MODEL fact #2 -- the draft has no
    kv_heads of its own to begin with)."""
    target_pass_flops = flops_per_token(target_active_params, avg_context_len, target_config)
    if mtp_k <= 0 or draft_config is None:
        return target_pass_flops

    target_round_flops = (mtp_k + 1) * target_pass_flops

    if draft_active_params is None:
        draft_active_params = compute_draft_active_params(draft_config)

    draft_flops_per_pass = flops_per_token(draft_active_params, avg_context_len, draft_config)
    return target_round_flops + mtp_k * draft_flops_per_pass


def compute_mfu(tokens_per_second: float, flops_per_tok: float, peak_tflops_bf16: float) -> float:
    achieved_flops = tokens_per_second * flops_per_tok
    peak_flops = peak_tflops_bf16 * 1e12
    return achieved_flops / peak_flops if peak_flops > 0 else 0.0


def compute_mbu(tokens_per_second: float, bytes_per_tok: int, peak_bandwidth_gbps: float) -> float:
    achieved_bytes_per_sec = tokens_per_second * bytes_per_tok
    peak_bytes_per_sec = peak_bandwidth_gbps * 1e9
    return achieved_bytes_per_sec / peak_bytes_per_sec if peak_bytes_per_sec > 0 else 0.0
