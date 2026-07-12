#!/usr/bin/env python3
"""Unit tests for hardware_metrics.py, focused on the KV-cache-bytes /
batch-scaling fix (bytes_per_token() previously omitted KV cache reads
entirely, and the caller flat-summed weight+KV bytes instead of scaling
the KV term by batch_size -- see hardware_metrics.py docstrings on
bytes_per_token / kv_cache_bytes_per_token / bytes_per_decode_step).

Uses a small hand-built fake config (SimpleNamespace) instead of a real
HF AutoConfig so these run fast and offline -- only the attributes
hardware_metrics.py actually reads are set, matching
compute_active_params/flops_per_token's own field access.
"""

from __future__ import annotations

from types import SimpleNamespace

import hardware_metrics as hm


def make_config(
    layer_types,
    attention_k_eq_v: bool = False,
    global_kv_heads: int | None = None,
    sliding_window: int | None = 16,
) -> SimpleNamespace:
    """A small 128-hidden-dim, 4-head fake Gemma4-shaped text config.
    global_kv_heads defaults to the same value as num_key_value_heads so
    tests can isolate the attention_k_eq_v multiplier effect without also
    changing the KV head count."""
    kv_heads = 2
    return SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        layer_types=layer_types,
        num_hidden_layers=len(layer_types),
        head_dim=32,
        num_key_value_heads=kv_heads,
        global_head_dim=64,
        num_global_key_value_heads=global_kv_heads if global_kv_heads is not None else kv_heads,
        attention_k_eq_v=attention_k_eq_v,
        intermediate_size=256,
        enable_moe_block=True,
        moe_intermediate_size=64,
        num_experts=8,
        top_k_experts=2,
        vocab_size=1000,
        sliding_window=sliding_window,
    )


MIXED_LAYERS = ["sliding_attention", "sliding_attention", "full_attention", "full_attention"]
FULL_ONLY = ["full_attention"]


# ---------------------------------------------------------------------------
# BUG 1: kv_cache_bytes_per_token
# ---------------------------------------------------------------------------

def test_kv_cache_bytes_matches_manual_per_layer_calc():
    """Sliding layers: 2x (K+V) * kv_heads * head_dim * ctx * bytes_per_param,
    capped at sliding_window. Full layers (k_eq_v=False): same but with
    global_head_dim. Hand-computed for ctx=8 (below the sliding_window=16
    cap, so no capping kicks in yet)."""
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    ctx = 8

    sliding_per_layer = 2 * 2 * 32 * ctx * hm.BYTES_PER_PARAM_BF16  # kv_heads=2, head_dim=32
    full_per_layer = 2 * 2 * 64 * ctx * hm.BYTES_PER_PARAM_BF16  # kv_heads=2, global_head_dim=64
    expected = 2 * sliding_per_layer + 2 * full_per_layer

    assert hm.kv_cache_bytes_per_token(ctx, cfg) == expected


def test_kv_cache_bytes_sliding_window_caps_context():
    """A context far beyond sliding_window must not inflate the sliding
    layers' contribution -- only the full_attention layers should scale
    with avg_context_len past the cap."""
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False, sliding_window=16)

    at_cap = hm.kv_cache_bytes_per_token(16, cfg)
    way_past_cap = hm.kv_cache_bytes_per_token(4096, cfg)

    # Only the two full_attention layers should have grown.
    full_layer_growth = 2 * (2 * 2 * 64 * (4096 - 16) * hm.BYTES_PER_PARAM_BF16)
    assert way_past_cap - at_cap == full_layer_growth


def test_kv_cache_bytes_k_eq_v_halves_full_layer_multiplier():
    """attention_k_eq_v=True means V is derived from K, not read from a
    separate cache -- multiplier drops from 2 to 1 for full_attention
    layers. Isolate this from the "KV head count also changes" effect by
    keeping global_kv_heads == num_key_value_heads in both variants."""
    ctx = 32
    cfg_k_eq_v_false = make_config(FULL_ONLY, attention_k_eq_v=False, global_kv_heads=2)
    cfg_k_eq_v_true = make_config(FULL_ONLY, attention_k_eq_v=True, global_kv_heads=2)

    bytes_false = hm.kv_cache_bytes_per_token(ctx, cfg_k_eq_v_false)
    bytes_true = hm.kv_cache_bytes_per_token(ctx, cfg_k_eq_v_true)

    assert bytes_true == bytes_false / 2


def test_kv_cache_bytes_k_eq_v_also_changes_kv_head_count():
    """When attention_k_eq_v is set, compute_active_params drops the KV
    head count to num_global_key_value_heads -- kv_cache_bytes_per_token
    must follow the same rule (not just halve the multiplier), otherwise
    it silently diverges from the params/FLOPs accounting for the same
    layers."""
    ctx = 32
    # global_kv_heads deliberately different from num_key_value_heads.
    cfg = make_config(FULL_ONLY, attention_k_eq_v=True, global_kv_heads=6)

    expected = 1 * 6 * 64 * ctx * hm.BYTES_PER_PARAM_BF16  # multiplier=1, kv_heads=6 (global)
    assert hm.kv_cache_bytes_per_token(ctx, cfg) == expected


def test_kv_cache_bytes_respects_text_config_nesting():
    """Same _get_text_config dereferencing as compute_active_params /
    flops_per_token: a config with a nested .text_config must give the
    same result as calling directly on the inner config."""
    inner = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    wrapped = SimpleNamespace(text_config=inner)

    assert hm.kv_cache_bytes_per_token(64, wrapped) == hm.kv_cache_bytes_per_token(64, inner)


# ---------------------------------------------------------------------------
# BUG 2: bytes_per_decode_step batch scaling
# ---------------------------------------------------------------------------

def test_bytes_per_decode_step_scales_kv_term_by_batch_not_weights():
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    active_params = hm.compute_active_params(cfg)
    ctx = 64

    weight_bytes = hm.bytes_per_token(active_params)
    kv_bytes = hm.kv_cache_bytes_per_token(ctx, cfg)

    for batch_size in (1, 8, 64):
        total = hm.bytes_per_decode_step(active_params, ctx, cfg, batch_size)
        assert total == weight_bytes + batch_size * kv_bytes
        # Flat-summing (the bug) would give weight_bytes + kv_bytes
        # regardless of batch_size -- confirm we're not doing that except
        # trivially at batch_size=1.
        if batch_size > 1:
            assert total != weight_bytes + kv_bytes


def test_short_context_batch_one_kv_term_is_small_relative_to_weights():
    """batch_size=1, short context: KV bytes should be a small fraction of
    weight bytes (weights dominate decode-step traffic at low concurrency
    / short context)."""
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    active_params = hm.compute_active_params(cfg)
    ctx = 8

    weight_bytes = hm.bytes_per_token(active_params)
    total = hm.bytes_per_decode_step(active_params, ctx, cfg, batch_size=1)
    kv_contribution = total - weight_bytes

    assert 0 < kv_contribution < 0.05 * weight_bytes


def test_long_context_large_batch_kv_term_dominates():
    """Long context + large batch: KV bytes should become the dominant
    term, not a rounding error next to weight bytes."""
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    active_params = hm.compute_active_params(cfg)
    ctx = 4096

    weight_bytes = hm.bytes_per_token(active_params)
    total = hm.bytes_per_decode_step(active_params, ctx, cfg, batch_size=64)
    kv_contribution = total - weight_bytes

    assert kv_contribution > 10 * weight_bytes


# ---------------------------------------------------------------------------
# End-to-end MBU sanity: must stay <= 1.0 (never re-introduce the >100%
# bug that motivated the original batch-size-divisor fix).
# ---------------------------------------------------------------------------

def _mbu_for(ctx: float, batch_size: int, decode_steps_per_second: float) -> float:
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    active_params = hm.compute_active_params(cfg)
    bytes_per_step = hm.bytes_per_decode_step(active_params, ctx, cfg, batch_size)
    peak_bandwidth_gbps = hm.GPU_SPECS["NVIDIA A100-SXM4-80GB"]["peak_bandwidth_gbps"]
    return hm.compute_mbu(decode_steps_per_second, bytes_per_step, peak_bandwidth_gbps)


def test_mbu_stays_bounded_short_context_batch_one():
    # 100 output tokens over a 5s decode window, no spec decode -> 20 steps/sec.
    mbu = _mbu_for(ctx=8, batch_size=1, decode_steps_per_second=20.0)
    assert 0 < mbu <= 1.0


def test_mbu_stays_bounded_long_context_large_batch():
    # 4096 output tokens/request over a 60s decode window -> ~68 steps/sec.
    mbu = _mbu_for(ctx=4096, batch_size=64, decode_steps_per_second=4096 / 60)
    assert 0 < mbu <= 1.0


def test_mbu_with_kv_term_is_higher_than_weights_only_mbu():
    """Sanity check that the KV term is actually being counted: MBU
    computed with the combined bytes_per_decode_step must exceed what
    the old weights-only bytes_per_token would have given, for a
    long-context/large-batch case where the KV term is significant."""
    cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    active_params = hm.compute_active_params(cfg)
    ctx = 4096
    batch_size = 64
    decode_steps_per_second = 4096 / 60
    peak_bandwidth_gbps = hm.GPU_SPECS["NVIDIA A100-SXM4-80GB"]["peak_bandwidth_gbps"]

    weights_only_bytes = hm.bytes_per_token(active_params)
    combined_bytes = hm.bytes_per_decode_step(active_params, ctx, cfg, batch_size)

    mbu_weights_only = hm.compute_mbu(decode_steps_per_second, weights_only_bytes, peak_bandwidth_gbps)
    mbu_combined = hm.compute_mbu(decode_steps_per_second, combined_bytes, peak_bandwidth_gbps)

    assert mbu_combined > mbu_weights_only
    assert mbu_combined <= 1.0


# ---------------------------------------------------------------------------
# DRAFT MODEL (speculative decoding) -- Gemma4 MTP, see hardware_metrics.py's
# DRAFT MODEL section (verified against vllm/model_executor/models/
# gemma4_mtp.py + vllm/v1/spec_decode/{gemma4,llm_base_proposer}.py).
# ---------------------------------------------------------------------------

def make_draft_config(
    layer_types,
    backbone_hidden_size: int = 128,
    use_ordered_embeddings: bool = False,
    num_centroids: int = 8,
    centroid_intermediate_top_k: int = 2,
    sliding_window: int | None = 16,
) -> SimpleNamespace:
    """A small 32-hidden-dim (draft's own, deliberately smaller than the
    128-dim target fixture's hidden_size) fake Gemma4 MTP config. No
    num_key_value_heads/global_head_dim K/V fields needed for
    compute_draft_active_params itself (Q-only, no K/V projections) --
    only draft_kv_read_bytes_per_forward_pass needs the TARGET's config
    for KV shape, not the draft's own."""
    return SimpleNamespace(
        hidden_size=32,
        backbone_hidden_size=backbone_hidden_size,
        num_attention_heads=4,
        layer_types=layer_types,
        num_hidden_layers=len(layer_types),
        head_dim=8,
        global_head_dim=16,
        intermediate_size=64,
        vocab_size=1000,
        use_ordered_embeddings=use_ordered_embeddings,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=centroid_intermediate_top_k,
        sliding_window=sliding_window,
    )


DRAFT_LAYERS = ["sliding_attention", "full_attention"]


def test_draft_active_params_matches_manual_calc_no_centroids():
    """use_ordered_embeddings=False: lm_head is a full [vocab_size,
    hidden_size] matmul, no centroids term."""
    cfg = make_draft_config(DRAFT_LAYERS, use_ordered_embeddings=False)
    params = hm.compute_draft_active_params(cfg)

    # Q-only attn: q=hidden*(n_heads*head_dim), o=(n_heads*head_dim)*hidden, no k/v.
    sliding_attn = 32 * (4 * 8) + (4 * 8) * 32       # head_dim=8
    full_attn = 32 * (4 * 16) + (4 * 16) * 32        # global_head_dim=16
    attn_total = sliding_attn + full_attn
    assert attn_total == 6144

    dense_mlp = 32 * (2 * 64) + 64 * 32
    mlp_total = 2 * dense_mlp
    assert mlp_total == 12288

    pre_projection = 2 * 128 * 32
    post_projection = 32 * 128
    assert pre_projection == 8192
    assert post_projection == 4096

    lm_head = 1000 * 32  # full vocab, no centroids
    assert lm_head == 32000

    expected_total = attn_total + mlp_total + pre_projection + post_projection + 0 + lm_head + 0
    assert params["total_active_params"] == expected_total == 62720
    assert params["centroids_params"] == 0
    assert params["embedding_params"] == 0  # shared with target, zero marginal cost


def test_draft_active_params_centroids_shrink_lm_head():
    """use_ordered_embeddings=True: lm_head active bytes shrink to a
    sparse gather (num_selected rows), not the full vocab -- and a small
    always-read centroids Linear is added. Net effect should be a
    SMALLER total than the no-centroids variant, since the lm_head
    savings dwarf the small centroids addition."""
    cfg_no_centroids = make_draft_config(DRAFT_LAYERS, use_ordered_embeddings=False)
    cfg_centroids = make_draft_config(
        DRAFT_LAYERS, use_ordered_embeddings=True, num_centroids=8, centroid_intermediate_top_k=2
    )

    params_no_centroids = hm.compute_draft_active_params(cfg_no_centroids)
    params_centroids = hm.compute_draft_active_params(cfg_centroids)

    vocab_size_per_centroid = 1000 // 8
    num_selected = 2 * vocab_size_per_centroid
    expected_lm_head = num_selected * 32
    expected_centroids = 32 * 8

    assert params_centroids["lm_head_active_params"] == expected_lm_head == 8000
    assert params_centroids["centroids_params"] == expected_centroids == 256
    assert params_centroids["total_active_params"] == 38976
    assert params_centroids["total_active_params"] < params_no_centroids["total_active_params"]


def test_draft_active_params_is_lightweight_relative_to_target():
    """The draft model should be meaningfully smaller than the target --
    matches the module docstring's "lightweight decoder" framing -- but
    not vanishingly small either."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    draft_cfg = make_draft_config(DRAFT_LAYERS, use_ordered_embeddings=False)
    draft_params = hm.compute_draft_active_params(draft_cfg)

    assert 0 < draft_params["total_active_params"] < target_params["total_active_params"]
    ratio = draft_params["total_active_params"] / target_params["total_active_params"]
    assert 0.01 < ratio < 0.5


def test_draft_kv_read_bytes_uses_target_shape_but_draft_cap():
    """Shape (head_dim/kv_heads/multiplier) must come from the TARGET
    config (that's the physical tensor being read); the sliding_window
    CAP must come from the DRAFT's own config (that's what the draft's
    own attention module is configured with) -- deliberately using
    different sliding_window values on target (16) vs draft (4) to
    prove the cap doesn't leak from the wrong config."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False, sliding_window=16)
    draft_cfg = make_draft_config(DRAFT_LAYERS, sliding_window=4)
    ctx = 100

    # sliding layer: target kv_heads=2, head_dim=32, multiplier=2 -> unit=256/token,
    # capped at DRAFT's sliding_window=4 (not target's 16).
    sliding_contribution = 2 * 2 * 32 * 2 * 4  # kv_multiplier*kv_heads*head_dim*bytes*ctx
    # full layer: target kv_heads=2, global_head_dim=64, multiplier=2 (k_eq_v=False),
    # uncapped (full_attention layers are never capped).
    full_contribution = 2 * 2 * 64 * 2 * 100

    expected = sliding_contribution + full_contribution
    assert hm.draft_kv_read_bytes_per_forward_pass(ctx, target_cfg, draft_cfg) == expected

    # Sanity: capping at draft's smaller window must give LESS than an
    # uncapped draft config would, proving the cap is actually applied.
    draft_cfg_uncapped = make_draft_config(DRAFT_LAYERS, sliding_window=None)
    uncapped = hm.draft_kv_read_bytes_per_forward_pass(ctx, target_cfg, draft_cfg_uncapped)
    assert uncapped > expected


def test_flops_per_token_spec_decode_k0_matches_target_only_exactly():
    """Regression test: mtp_k=0 (or draft_config=None) must return
    EXACTLY flops_per_token()'s own output -- the non-spec-decode path
    must not change numerically."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    ctx = 64

    base = hm.flops_per_token(target_params, ctx, target_cfg)

    draft_cfg = make_draft_config(DRAFT_LAYERS)
    draft_params = hm.compute_draft_active_params(draft_cfg)

    assert hm.flops_per_token_spec_decode(target_params, ctx, target_cfg, 0, draft_cfg, draft_params) == base
    assert hm.flops_per_token_spec_decode(target_params, ctx, target_cfg, 5, None, None) == base


def test_flops_per_token_spec_decode_adds_draft_passes_exactly():
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    draft_cfg = make_draft_config(DRAFT_LAYERS)
    draft_params = hm.compute_draft_active_params(draft_cfg)
    ctx = 64
    mtp_k = 5

    base = hm.flops_per_token(target_params, ctx, target_cfg)
    draft_flops_per_pass = hm.flops_per_token(draft_params, ctx, draft_cfg)
    expected = base + mtp_k * draft_flops_per_pass

    actual = hm.flops_per_token_spec_decode(
        target_params, ctx, target_cfg, mtp_k, draft_cfg, draft_params
    )
    assert actual == expected
    assert actual > base  # draft's contribution is nonzero


def test_bytes_per_decode_step_spec_decode_k0_matches_base_exactly():
    """Regression test: mtp_k=0 (or draft_config=None) must return
    EXACTLY bytes_per_decode_step()'s own output."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    ctx = 64
    batch_size = 8

    base = hm.bytes_per_decode_step(target_params, ctx, target_cfg, batch_size)

    draft_cfg = make_draft_config(DRAFT_LAYERS)
    draft_params = hm.compute_draft_active_params(draft_cfg)

    assert hm.bytes_per_decode_step_spec_decode(
        target_params, ctx, target_cfg, batch_size, 0, draft_cfg, draft_params
    ) == base
    assert hm.bytes_per_decode_step_spec_decode(
        target_params, ctx, target_cfg, batch_size, 5, None, None
    ) == base


def test_bytes_per_decode_step_spec_decode_k5_matches_closed_form():
    """k=5: verify the combined total matches the exact closed-form sum
    of base + target KV write bytes ((k+1)*batch_size scaled) + draft
    weight bytes (k scaled, NOT batch-scaled) + draft KV read bytes (k
    AND batch scaled) -- this is BUG 5 (write bytes) plus the k/batch_size
    threading, checked as one precise equality rather than just a
    monotonicity check."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    draft_cfg = make_draft_config(DRAFT_LAYERS)
    draft_params = hm.compute_draft_active_params(draft_cfg)
    ctx = 64
    batch_size = 8
    mtp_k = 5

    base = hm.bytes_per_decode_step(target_params, ctx, target_cfg, batch_size)
    target_write_unit = hm.kv_cache_bytes_per_token(1.0, target_cfg)
    draft_weight_unit = hm.bytes_per_token(draft_params)
    draft_kv_unit = hm.draft_kv_read_bytes_per_forward_pass(ctx, target_cfg, draft_cfg)

    expected = (
        base
        + (mtp_k + 1) * target_write_unit * batch_size
        + mtp_k * draft_weight_unit
        + mtp_k * draft_kv_unit * batch_size
    )

    actual = hm.bytes_per_decode_step_spec_decode(
        target_params, ctx, target_cfg, batch_size, mtp_k, draft_cfg, draft_params
    )
    assert actual == expected
    assert actual > base


def test_bytes_per_decode_step_spec_decode_write_term_scales_with_k():
    """The write-byte term specifically (isolated by holding batch_size=1
    and comparing consecutive k values) must grow by exactly ONE more
    write-unit per additional k -- (k+1) scaling, not k or k+2."""
    target_cfg = make_config(MIXED_LAYERS, attention_k_eq_v=False)
    target_params = hm.compute_active_params(target_cfg)
    draft_cfg = make_draft_config(DRAFT_LAYERS)
    draft_params = hm.compute_draft_active_params(draft_cfg)
    ctx = 64
    batch_size = 1

    totals = [
        hm.bytes_per_decode_step_spec_decode(
            target_params, ctx, target_cfg, batch_size, k, draft_cfg, draft_params
        )
        for k in (1, 2, 3)
    ]

    target_write_unit = hm.kv_cache_bytes_per_token(1.0, target_cfg)
    draft_weight_unit = hm.bytes_per_token(draft_params)
    draft_kv_unit = hm.draft_kv_read_bytes_per_forward_pass(ctx, target_cfg, draft_cfg)
    per_k_increment = target_write_unit + draft_weight_unit + draft_kv_unit

    assert totals[1] - totals[0] == per_k_increment
    assert totals[2] - totals[1] == per_k_increment


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
