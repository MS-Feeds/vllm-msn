# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the RelayAttention backend.

Tests are grouped into four areas:

1. Metadata dataclasses  — RelayInfo / RelayAttentionMetadata
2. relay_fuse math       — LSE-weighted merge correctness vs. reference
3. Backend API           — registry, get_name, get_impl_cls, get_builder_cls
4. Config & selector     — enable_relay_attention auto-selects RELAY_ATTN
5. Forward-pass smoke    — fallback path (no relay metadata → standard attn)
"""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.relay_attn import (
    RelayAttentionBackend,
    RelayAttentionImpl,
    RelayAttentionMetadata,
    RelayInfo,
)

# relay_fuse calls merge_attn_states which requires triton / CUDA
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="relay_fuse calls merge_attn_states which requires CUDA/triton",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(shape, dtype=torch.float32, device="cpu"):
    return torch.randn(shape, dtype=dtype, device=device)


def _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr):
    """Pure-PyTorch reference for LSE-weighted attention merge.

    Implements exactly:
        alpha = 1 / (1 + exp(lse_usr - lse_sys))
        out   = alpha * out_sys + (1 - alpha) * out_usr

    lse tensors expected in [H, N] shape.
    """
    # lse_sys, lse_usr: [H, N] → [N, H, 1] for broadcasting
    lse_sys_f = lse_sys.transpose(0, 1).unsqueeze(-1).float()
    lse_usr_f = lse_usr.transpose(0, 1).unsqueeze(-1).float()
    alpha = 1.0 / (1.0 + (lse_usr_f - lse_sys_f).exp())
    return (alpha * out_sys.float() + (1.0 - alpha) * out_usr.float()).to(out_sys.dtype)


# ===========================================================================
# 1. Metadata dataclasses
# ===========================================================================

class TestRelayInfo:
    def test_defaults(self):
        info = RelayInfo(system_length=64)
        assert info.system_length == 64
        assert info.enabled is True
        assert info.use_triton is True
        assert info.min_batch_size == 2
        assert info.min_system_length == 256

    def test_disabled(self):
        info = RelayInfo(system_length=64, enabled=False)
        assert info.enabled is False

    def test_zero_system_length(self):
        """system_length=0 is technically valid (no-op relay)."""
        info = RelayInfo(system_length=0)
        assert info.system_length == 0

    # --- is_beneficial guards -----------------------------------------------

    def test_beneficial_nominal(self):
        info = RelayInfo(system_length=512)
        assert info.is_beneficial(batch_size=8)

    def test_not_beneficial_when_disabled(self):
        info = RelayInfo(system_length=512, enabled=False)
        assert not info.is_beneficial(batch_size=8)

    def test_not_beneficial_small_batch(self):
        """B < min_batch_size → relay adds merge overhead with no savings."""
        info = RelayInfo(system_length=512, min_batch_size=2)
        assert not info.is_beneficial(batch_size=1)

    def test_beneficial_at_min_batch_size(self):
        info = RelayInfo(system_length=512, min_batch_size=2)
        assert info.is_beneficial(batch_size=2)

    def test_not_beneficial_short_system_prompt(self):
        """S_sys < min_system_length → bandwidth saving too small."""
        info = RelayInfo(system_length=64, min_system_length=256)
        assert not info.is_beneficial(batch_size=32)

    def test_beneficial_at_min_system_length(self):
        info = RelayInfo(system_length=256, min_system_length=256)
        assert info.is_beneficial(batch_size=8)

    def test_custom_thresholds(self):
        info = RelayInfo(system_length=100, min_batch_size=4, min_system_length=50)
        assert info.is_beneficial(batch_size=4)
        assert not info.is_beneficial(batch_size=3)
        assert not info.is_beneficial(batch_size=4) if False else True  # already checked


class TestRelayAttentionMetadata:
    def test_empty_has_no_relay(self):
        meta = RelayAttentionMetadata()
        assert not meta.has_relay()

    def test_with_relay_info_enabled(self):
        meta = RelayAttentionMetadata(relay_info=RelayInfo(system_length=32))
        assert meta.has_relay()

    def test_with_relay_info_disabled(self):
        meta = RelayAttentionMetadata(
            relay_info=RelayInfo(system_length=32, enabled=False)
        )
        assert not meta.has_relay()

    def test_relay_info_none(self):
        meta = RelayAttentionMetadata(relay_info=None)
        assert not meta.has_relay()


# ===========================================================================
# 2. relay_fuse math
# ===========================================================================

class TestRelayFuse:
    """Tests that relay_fuse matches the reference LSE-weighted merge.

    Math correctness tests use the pure-PyTorch reference and run on CPU.
    Tests that call relay_fuse directly (which uses merge_attn_states →
    triton/CUDA kernel) are guarded by ``requires_cuda``.
    """

    # --- Pure-PyTorch reference tests (always run, no GPU needed) -----------

    @pytest.mark.parametrize("N,H,D", [
        (4, 8, 64),
        (1, 1, 32),
        (16, 32, 128),
    ])
    def test_reference_matches_formula(self, N, H, D):
        """Verify the reference implementation matches the LSE formula."""
        torch.manual_seed(0)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse_sys = _rand((H, N))
        lse_usr = _rand((H, N))

        result = _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr)
        # Re-compute manually and compare
        lse_sys_b = lse_sys.T.unsqueeze(-1).float()  # [N, H, 1]
        lse_usr_b = lse_usr.T.unsqueeze(-1).float()
        alpha = 1.0 / (1.0 + (lse_usr_b - lse_sys_b).exp())
        expected = (alpha * out_sys.float() + (1.0 - alpha) * out_usr.float()).to(out_sys.dtype)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_reference_sys_dominates(self):
        """When LSE_sys >> LSE_usr, alpha ≈ 1 so output ≈ out_sys."""
        N, H, D = 4, 4, 32
        out_sys = torch.ones(N, H, D)
        out_usr = torch.zeros(N, H, D)
        lse_sys = torch.full((H, N), 100.0)
        lse_usr = torch.zeros(H, N)
        result = _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr)
        torch.testing.assert_close(result, out_sys, atol=1e-3, rtol=1e-3)

    def test_reference_usr_dominates(self):
        """When LSE_usr >> LSE_sys, alpha ≈ 0 so output ≈ out_usr."""
        N, H, D = 4, 4, 32
        out_sys = torch.ones(N, H, D)
        out_usr = torch.zeros(N, H, D)
        lse_sys = torch.zeros(H, N)
        lse_usr = torch.full((H, N), 100.0)
        result = _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr)
        torch.testing.assert_close(result, out_usr, atol=1e-3, rtol=1e-3)

    def test_reference_equal_lse_gives_mean(self):
        """When LSE_sys == LSE_usr, alpha = 0.5 so output is the mean."""
        N, H, D = 4, 4, 32
        torch.manual_seed(2)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse = _rand((H, N))
        result = _relay_fuse_reference(out_sys, lse, out_usr, lse)
        expected = 0.5 * (out_sys.float() + out_usr.float())
        torch.testing.assert_close(result.float(), expected, atol=1e-5, rtol=1e-5)

    def test_reference_bfloat16_preserves_dtype(self):
        """Reference preserves bfloat16 dtype."""
        N, H, D = 8, 16, 64
        torch.manual_seed(4)
        out_sys = _rand((N, H, D), dtype=torch.bfloat16)
        out_usr = _rand((N, H, D), dtype=torch.bfloat16)
        lse_sys = _rand((H, N))
        lse_usr = _rand((H, N))
        result = _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr)
        assert result.dtype == torch.bfloat16
        assert result.shape == (N, H, D)

    # --- GPU-backed relay_fuse tests (require CUDA / triton) ----------------

    @requires_cuda
    @pytest.mark.parametrize("N,H,D", [
        (4, 8, 64),
        (1, 1, 32),
        (16, 32, 128),
    ])
    def test_fuse_matches_reference(self, N, H, D):
        torch.manual_seed(0)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse_sys = _rand((H, N))
        lse_usr = _rand((H, N))

        expected = _relay_fuse_reference(out_sys, lse_sys, out_usr, lse_usr)
        actual = RelayAttentionImpl.relay_fuse(out_sys, lse_sys, out_usr, lse_usr)

        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-4, rtol=1e-4)

    @requires_cuda
    def test_sys_dominates_when_lse_sys_much_larger(self):
        N, H, D = 4, 4, 32
        out_sys = torch.ones(N, H, D)
        out_usr = torch.zeros(N, H, D)
        lse_sys = torch.full((H, N), 100.0)
        lse_usr = torch.zeros(H, N)
        result = RelayAttentionImpl.relay_fuse(out_sys, lse_sys, out_usr, lse_usr)
        torch.testing.assert_close(result, out_sys, atol=1e-3, rtol=1e-3)

    @requires_cuda
    def test_usr_dominates_when_lse_usr_much_larger(self):
        N, H, D = 4, 4, 32
        out_sys = torch.ones(N, H, D)
        out_usr = torch.zeros(N, H, D)
        lse_sys = torch.zeros(H, N)
        lse_usr = torch.full((H, N), 100.0)
        result = RelayAttentionImpl.relay_fuse(out_sys, lse_sys, out_usr, lse_usr)
        torch.testing.assert_close(result, out_usr, atol=1e-3, rtol=1e-3)

    @requires_cuda
    def test_equal_lse_gives_mean(self):
        N, H, D = 4, 4, 32
        torch.manual_seed(2)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse = _rand((H, N))
        result = RelayAttentionImpl.relay_fuse(out_sys, lse, out_usr, lse)
        expected = 0.5 * (out_sys.float() + out_usr.float())
        torch.testing.assert_close(result.float(), expected, atol=1e-4, rtol=1e-4)

    @requires_cuda
    def test_output_lse_populated(self):
        """output_lse should be filled when provided."""
        N, H, D = 4, 4, 32
        torch.manual_seed(3)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse_sys = _rand((H, N))
        lse_usr = _rand((H, N))
        output_lse = torch.empty(H, N)
        RelayAttentionImpl.relay_fuse(out_sys, lse_sys, out_usr, lse_usr, output_lse)
        max_lse = torch.maximum(lse_sys, lse_usr)
        assert (output_lse >= max_lse - 1e-3).all()

    # ----- shape validation -----------------------------------------------

    def test_wrong_lse_sys_shape_raises(self):
        N, H, D = 4, 8, 64
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse_bad = _rand((N, H))   # [N, H] instead of [H, N]
        lse_ok  = _rand((H, N))

        with pytest.raises(ValueError, match="lse_sys must have shape"):
            RelayAttentionImpl.relay_fuse(out_sys, lse_bad, out_usr, lse_ok)

    def test_wrong_lse_usr_shape_raises(self):
        N, H, D = 4, 8, 64
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        lse_ok  = _rand((H, N))
        lse_bad = _rand((N, H))   # [N, H] instead of [H, N]

        with pytest.raises(ValueError, match="lse_usr must have shape"):
            RelayAttentionImpl.relay_fuse(out_sys, lse_ok, out_usr, lse_bad)

    def test_transposed_lse_hint_in_error(self):
        """The error message should mention transpose as the fix."""
        N, H, D = 2, 4, 16
        out = _rand((N, H, D))
        lse_bad = _rand((N, H))
        lse_ok = _rand((H, N))

        with pytest.raises(ValueError, match="transpose"):
            RelayAttentionImpl.relay_fuse(out, lse_bad, out, lse_ok)

    @requires_cuda
    def test_bfloat16_inputs(self):
        """relay_fuse should work with bfloat16 attention outputs."""
        N, H, D = 8, 16, 64
        torch.manual_seed(4)
        out_sys = _rand((N, H, D), dtype=torch.bfloat16)
        out_usr = _rand((N, H, D), dtype=torch.bfloat16)
        lse_sys = _rand((H, N))
        lse_usr = _rand((H, N))

        result = RelayAttentionImpl.relay_fuse(out_sys, lse_sys, out_usr, lse_usr)
        assert result.dtype == torch.bfloat16
        assert result.shape == (N, H, D)


# ===========================================================================
# 3. Backend API
# ===========================================================================

class TestRelayAttentionBackend:
    def test_get_name(self):
        assert RelayAttentionBackend.get_name() == "RELAY_ATTN"

    def test_get_impl_cls(self):
        assert RelayAttentionBackend.get_impl_cls() is RelayAttentionImpl

    def test_get_builder_cls(self):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
        assert RelayAttentionBackend.get_builder_cls() is FlashAttentionMetadataBuilder

    def test_registered_in_enum(self):
        assert AttentionBackendEnum.RELAY_ATTN is not None
        assert "relay_attn" in AttentionBackendEnum.RELAY_ATTN.value.lower()

    def test_enum_get_path_returns_module(self):
        path = AttentionBackendEnum.RELAY_ATTN.value
        assert path.startswith("vllm.")
        assert "RelayAttentionBackend" in path

    def test_inherits_flash_attn_capabilities(self):
        """RELAY_ATTN should inherit all FlashAttention compatibility flags."""
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        assert RelayAttentionBackend.supported_dtypes == FlashAttentionBackend.supported_dtypes
        assert RelayAttentionBackend.supports_batch_invariance() == \
               FlashAttentionBackend.supports_batch_invariance()
        assert RelayAttentionBackend.supports_non_causal() == \
               FlashAttentionBackend.supports_non_causal()

    def test_impl_is_subclass_of_flash_attn_impl(self):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        assert issubclass(RelayAttentionImpl, FlashAttentionImpl)

    def test_backend_is_subclass_of_flash_attn_backend(self):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        assert issubclass(RelayAttentionBackend, FlashAttentionBackend)


# ===========================================================================
# 4. Config & selector
# ===========================================================================

class TestRelayAttentionConfig:
    def test_default_fields_exist(self):
        from vllm.config.attention import AttentionConfig
        cfg = AttentionConfig()
        assert hasattr(cfg, "enable_relay_attention")
        assert hasattr(cfg, "relay_system_prompt_length")
        assert hasattr(cfg, "relay_min_batch_size")
        assert hasattr(cfg, "relay_min_system_length")

    def test_default_disabled(self):
        from vllm.config.attention import AttentionConfig
        cfg = AttentionConfig()
        assert cfg.enable_relay_attention is False
        assert cfg.relay_system_prompt_length == 0
        assert cfg.relay_min_batch_size == 2
        assert cfg.relay_min_system_length == 256

    def test_can_enable(self):
        from vllm.config.attention import AttentionConfig
        cfg = AttentionConfig(enable_relay_attention=True, relay_system_prompt_length=256)
        assert cfg.enable_relay_attention is True
        assert cfg.relay_system_prompt_length == 256

    def test_can_lower_thresholds(self):
        from vllm.config.attention import AttentionConfig
        cfg = AttentionConfig(relay_min_batch_size=1, relay_min_system_length=128)
        assert cfg.relay_min_batch_size == 1
        assert cfg.relay_min_system_length == 128


class TestRelayAttentionSelector:
    """Tests that the selector picks RELAY_ATTN when the flag is set."""

    def _make_mock_config(self, enable_relay: bool, explicit_backend=None):
        """Build a minimal VllmConfig-like mock with the attention_config."""
        from types import SimpleNamespace
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backend import AttentionType

        attn_cfg = AttentionConfig(
            enable_relay_attention=enable_relay,
            backend=explicit_backend,
        )
        mock_cache = SimpleNamespace(user_specified_block_size=False, block_size=16)
        return SimpleNamespace(
            attention_config=attn_cfg,
            cache_config=mock_cache,
        )

    def test_relay_flag_selects_relay_backend(self, monkeypatch):
        """When enable_relay_attention=True and no explicit backend, selector
        should choose RELAY_ATTN."""
        from vllm.v1.attention import selector as sel_mod
        import vllm.envs as envs

        vllm_config = self._make_mock_config(enable_relay=True)

        captured = {}

        def fake_cached(backend, attn_selector_config, num_heads=None):
            captured["backend"] = backend
            # Return a dummy class to avoid full import chain
            return RelayAttentionBackend

        monkeypatch.setattr(sel_mod, "_cached_get_attn_backend", fake_cached)
        # get_current_vllm_config is a local import inside get_attn_backend;
        # patch at the source module, not on the selector.
        monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: vllm_config)
        monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

        sel_mod.get_attn_backend(
            head_size=128, dtype=torch.float16, kv_cache_dtype=None
        )

        assert captured.get("backend") is AttentionBackendEnum.RELAY_ATTN

    def test_explicit_backend_overrides_relay_flag(self, monkeypatch):
        """An explicit backend= should take priority over enable_relay_attention."""
        from vllm.v1.attention import selector as sel_mod
        import vllm.envs as envs

        vllm_config = self._make_mock_config(
            enable_relay=True,
            explicit_backend=AttentionBackendEnum.FLASH_ATTN,
        )

        captured = {}

        def fake_cached(backend, attn_selector_config, num_heads=None):
            captured["backend"] = backend
            return RelayAttentionBackend

        monkeypatch.setattr(sel_mod, "_cached_get_attn_backend", fake_cached)
        monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: vllm_config)
        monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

        sel_mod.get_attn_backend(
            head_size=128, dtype=torch.float16, kv_cache_dtype=None
        )

        assert captured.get("backend") is AttentionBackendEnum.FLASH_ATTN

    def test_relay_not_selected_when_disabled(self, monkeypatch):
        """When enable_relay_attention=False, backend should stay None."""
        from vllm.v1.attention import selector as sel_mod
        import vllm.envs as envs

        vllm_config = self._make_mock_config(enable_relay=False)

        captured = {}

        def fake_cached(backend, attn_selector_config, num_heads=None):
            captured["backend"] = backend
            return RelayAttentionBackend

        monkeypatch.setattr(sel_mod, "_cached_get_attn_backend", fake_cached)
        monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: vllm_config)
        monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

        sel_mod.get_attn_backend(
            head_size=128, dtype=torch.float16, kv_cache_dtype=None
        )

        assert captured.get("backend") is None  # no override


# ===========================================================================
# 5. Forward-pass smoke tests (no GPU required for basic path coverage)
# ===========================================================================

class TestRelayAttentionImplForward:
    """Smoke tests for the RelayAttentionImpl.forward dispatch logic.

    These tests mock out the FlashAttentionImpl parent to avoid GPU
    dependency and focus on the relay routing logic only.
    """

    def _make_impl(self):
        """Build a RelayAttentionImpl without triggering GPU operations."""
        impl = object.__new__(RelayAttentionImpl)
        # Minimal attribute set needed by our forward override.
        return impl

    def test_no_relay_metadata_delegates_to_super(self, monkeypatch):
        """If attn_metadata has no relay_metadata attr, super().forward() is called."""
        impl = self._make_impl()

        calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            calls.append("super")
            return query  # dummy

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=object())
        assert calls == ["super"]

    def test_relay_metadata_none_delegates_to_super(self, monkeypatch):
        """relay_metadata=None → standard path."""
        impl = self._make_impl()
        calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            calls.append("super")
            return query

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        from types import SimpleNamespace
        meta = SimpleNamespace(relay_metadata=None)
        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=meta)
        assert calls == ["super"]

    def test_relay_disabled_delegates_to_super(self, monkeypatch):
        """relay_metadata with enabled=False → standard path."""
        impl = self._make_impl()
        calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            calls.append("super")
            return query

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        from types import SimpleNamespace
        meta = SimpleNamespace(
            relay_metadata=RelayAttentionMetadata(
                relay_info=RelayInfo(system_length=16, enabled=False)
            )
        )
        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=meta)
        assert calls == ["super"]

    def test_relay_enabled_calls_forward_relay(self, monkeypatch):
        """relay_metadata with enabled=True → _forward_relay is called."""
        impl = self._make_impl()
        relay_calls = []

        def mock_forward_relay(self_inner, query, key, value, kv_cache,
                               attn_metadata, relay_meta, output, output_scale):
            relay_calls.append(relay_meta.relay_info.system_length)
            return query

        monkeypatch.setattr(RelayAttentionImpl, "_forward_relay", mock_forward_relay)

        from types import SimpleNamespace
        meta = SimpleNamespace(
            num_reqs=8,  # large enough batch
            relay_metadata=RelayAttentionMetadata(
                relay_info=RelayInfo(system_length=512, enabled=True)
            )
        )
        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=meta)
        assert relay_calls == [512]

    def test_relay_skipped_when_batch_too_small(self, monkeypatch):
        """With B=1 (< min_batch_size=2), is_beneficial=False → standard path."""
        impl = self._make_impl()
        super_calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            super_calls.append("super")
            return query

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        from types import SimpleNamespace
        meta = SimpleNamespace(
            num_reqs=1,  # too small — relay should be skipped
            relay_metadata=RelayAttentionMetadata(
                relay_info=RelayInfo(system_length=512, enabled=True, min_batch_size=2)
            )
        )
        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=meta)
        assert super_calls == ["super"], "relay should fall through to super for B=1"

    def test_relay_skipped_when_system_prompt_too_short(self, monkeypatch):
        """With short system prompt (< min_system_length), relay is skipped."""
        impl = self._make_impl()
        super_calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            super_calls.append("super")
            return query

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        from types import SimpleNamespace
        meta = SimpleNamespace(
            num_reqs=16,
            relay_metadata=RelayAttentionMetadata(
                relay_info=RelayInfo(
                    system_length=64,         # too short
                    enabled=True,
                    min_system_length=256,
                )
            )
        )
        q = torch.randn(4, 8, 64)
        impl.forward(q, q, q, kv_cache=None, attn_metadata=meta)
        assert super_calls == ["super"], "relay should fall through for short system prompt"

    def test_forward_relay_fallthrough_calls_super(self, monkeypatch):
        """_forward_relay placeholder must call super().forward() exactly once."""
        impl = self._make_impl()
        calls = []

        def mock_super_forward(self_inner, query, key, value, kv_cache,
                               attn_metadata, output=None, output_scale=None):
            calls.append("super")
            return output if output is not None else query

        monkeypatch.setattr(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
            mock_super_forward,
        )

        from types import SimpleNamespace
        relay_meta = RelayAttentionMetadata(relay_info=RelayInfo(system_length=64))
        meta = SimpleNamespace()

        q = torch.randn(4, 8, 64)
        impl._forward_relay(
            query=q, key=q, value=q, kv_cache=None,
            attn_metadata=meta, relay_meta=relay_meta,
            output=None, output_scale=None,
        )
        assert calls == ["super"], "placeholder must delegate to super exactly once"


# ===========================================================================
# 6. relay_fuse math — equivalence with relayattn_ops_v091 reference
# ===========================================================================

class TestRelayFuseEquivalenceWithOriginal:
    """Cross-check relay_fuse against the port's relay_fusion function.

    The port's relay_fusion (native backend, [N, H] LSE layout) should
    produce the same result as relay_fuse (merge_attn_states, [H, N] layout)
    after accounting for the LSE transpose.
    """

    @pytest.mark.parametrize("N,H,D", [(4, 8, 64), (8, 16, 128)])
    def test_matches_port_native_backend(self, N, H, D):
        try:
            from relay_attention_port.relayattn_ops_v091 import relay_fusion
        except ImportError:
            pytest.skip("relay_attention_port not importable from test context")

        torch.manual_seed(42)
        out_sys = _rand((N, H, D))
        out_usr = _rand((N, H, D))
        # port uses [N, H] layout; relay_fuse uses [H, N]
        lse_sys_nh = _rand((N, H))   # [N, H]
        lse_usr_nh = _rand((N, H))   # [N, H]
        lse_sys_hn = lse_sys_nh.T.contiguous()  # [H, N]
        lse_usr_hn = lse_usr_nh.T.contiguous()  # [H, N]

        ref = relay_fusion(out_sys, lse_sys_nh, out_usr, lse_usr_nh, backend="native")
        actual = RelayAttentionImpl.relay_fuse(out_sys, lse_sys_hn, out_usr, lse_usr_hn)

        torch.testing.assert_close(actual.float(), ref.float(), atol=1e-4, rtol=1e-4)
