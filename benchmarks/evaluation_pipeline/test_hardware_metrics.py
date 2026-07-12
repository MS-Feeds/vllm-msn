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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
