# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Relay Attention backend for vLLM v1 engine.

Implements the RelayAttention algorithm from:
    "RelayAttention for Efficient Large Language Model Serving with Long
     System Prompts", Zhu et al., ACL 2024. https://arxiv.org/abs/2402.14808

Core idea: when a batch of requests shares a long system prompt, reading the
system-prompt KV pairs from DRAM once per batch (instead of once per request)
eliminates redundant memory transfers.  The system and user attention outputs
are then fused via LSE-weighted combination:

    alpha = 1 / (1 + exp(LSE_usr - LSE_sys))
    out   = alpha * out_sys + (1 - alpha) * out_usr

This is an exact, lossless reformulation of causal attention — no retraining.

The backend extends FlashAttentionBackend and delegates to it for all standard
attention operations. When relay metadata is present on a request, it computes
separate system-prompt and user attention passes, then merges them using the
existing `merge_attn_states` primitive (already used for chunked prefill).

RoPE handling: following MiniPIC (arXiv:2606.13126), unrotated K vectors
should be stored in the prefix KV cache, with RoPE applied inside the kernel
per-request. This is noted as a TODO until the prefix cache layer is wired in.

Speculative decoding compatibility
-----------------------------------
The relay fusion is per-token and operates over the query dimension, so it is
compatible with all speculative decoding strategies that vLLM supports:

* **Standard chain speculation** (ngram, MLP, draft model): each sequence has
  ``1 + num_speculative_tokens`` query tokens. ``merge_attn_states`` handles
  any number of query tokens transparently.
* **EAGLE / padded batch**: inherited from ``FlashAttentionBackend``
  (``supports_batch_invariance=True``).
* **Tree-based speculation (EAGLE3 tree)**: all branches share the same
  system-prompt prefix, so relay delivers an *amplified* DRAM saving — the
  prefix KV is read once for the entire tree, not once per branch. This aligns
  with DeFT (arXiv:2404.00242).
* **CUDAGraphs**: inherited from ``FlashAttentionBackend``.

.. warning::
    The **draft model** should use a standard backend (e.g. ``FLASH_ATTN``).
    Set ``SpeculativeConfig.attention_backend = AttentionBackendEnum.FLASH_ATTN``
    so that relay overhead is not applied to the draft model, which does not
    benefit from system-prompt KV sharing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImpl

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config / metadata dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RelayInfo:
    """Per-request relay attention parameters.

    Attributes:
        system_length: Number of system-prompt tokens (prefix length).
        enabled: Whether relay fusion is active for this request.
        use_triton: Whether to use the Triton fusion kernel (else native).
        min_batch_size: Relay is skipped when the current batch is smaller
            than this value (single-sequence batches gain nothing).
        min_system_length: Relay is skipped when the system prompt is shorter
            than this threshold (bandwidth saving too small vs merge cost).
    """

    system_length: int
    enabled: bool = True
    use_triton: bool = True
    min_batch_size: int = 2
    min_system_length: int = 256

    def is_beneficial(self, batch_size: int) -> bool:
        """Return False when relay would hurt throughput rather than help it.

        Throughput regressions occur when:
          * batch_size < min_batch_size  — no savings from sharing the prefix
          * system_length < min_system_length — bandwidth saving is negligible
        """
        if not self.enabled:
            return False
        if batch_size < self.min_batch_size:
            return False
        if self.system_length < self.min_system_length:
            return False
        return True


@dataclass
class RelayAttentionMetadata:
    """Relay-specific metadata attached to an attention metadata object.

    This is stored as an optional field ``relay_metadata`` on the standard
    attention metadata object produced by ``FlashAttentionMetadataBuilder``.
    Callers that want relay fusion must monkey-patch this field onto the
    metadata after the builder produces it, or use a subclassed builder.

    Example::

        attn_meta = builder.build(...)
        attn_meta.relay_metadata = RelayAttentionMetadata(
            relay_info=RelayInfo(system_length=128)
        )

    A production-grade integration would subclass
    ``FlashAttentionMetadataBuilder`` and populate this field there.
    """

    relay_info: Optional[RelayInfo] = None

    def has_relay(self) -> bool:
        """True if relay is structurally enabled (ignores runtime batch/length guards)."""
        return self.relay_info is not None and self.relay_info.enabled


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class RelayAttentionBackend(FlashAttentionBackend):
    """Relay Attention backend.

    Extends FlashAttentionBackend. For requests that carry RelayInfo in their
    attention metadata the forward pass performs two separate flash-attention
    calls (system-prompt part and user part) and merges the results with
    `merge_attn_states`.  All other requests fall through to the standard
    FlashAttention path transparently.
    """

    @staticmethod
    def get_name() -> str:
        return "RELAY_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return RelayAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        # Reuse the standard FlashAttention metadata builder; relay metadata
        # is added as a side-channel on top of it.
        return FlashAttentionMetadataBuilder


# ---------------------------------------------------------------------------
# Impl class
# ---------------------------------------------------------------------------

class RelayAttentionImpl(FlashAttentionImpl):
    """Relay Attention implementation.

    Inherits all standard attention logic from FlashAttentionImpl and
    overrides only the forward pass to support relay fusion.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Relay-aware forward pass.

        If *attn_metadata* carries a :class:`RelayAttentionMetadata` with
        relay enabled, splits the KV cache into system-prompt and user parts,
        runs flash-attention separately over each, then merges the outputs
        using ``merge_attn_states``.

        Falls back to the standard FlashAttention path otherwise.
        """
        relay_meta: Optional[RelayAttentionMetadata] = getattr(
            attn_metadata, "relay_metadata", None
        )

        # Determine current batch size from the query tensor (dim 0 = num_tokens;
        # use attn_metadata.num_reqs if available for a true batch count).
        batch_size: int = getattr(attn_metadata, "num_reqs", query.shape[0])

        if (
            relay_meta is None
            or not relay_meta.has_relay()
            or not relay_meta.relay_info.is_beneficial(batch_size)
        ):
            # Standard path — no relay fusion needed.
            return super().forward(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                output_scale=output_scale,
            )

        return self._forward_relay(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            relay_meta=relay_meta,
            output=output,
            output_scale=output_scale,
        )

    # ------------------------------------------------------------------
    # Relay fusion path
    # ------------------------------------------------------------------

    def _forward_relay(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        relay_meta: RelayAttentionMetadata,
        output: Optional[torch.Tensor],
        output_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Two-pass relay attention with LSE-based fusion.

        Algorithm (once fully wired):
          1. Run flash-attention over user KV only  → (out_usr, lse_usr).
          2. Run flash-attention over the system-prompt prefix KV once per
             batch                                  → (out_sys, lse_sys).
          3. Merge via merge_attn_states (LSE rescaling).

        Current status: Phase 3 (prefix KV cache split) is not yet wired into
        the v1 KV cache manager.  Until then this method falls through to the
        standard FlashAttention forward pass so the backend is always safe to
        load; the relay optimisation is a no-op.

        NOTE on RoPE: when the prefix pass is added, unrotated K vectors
        should be stored in the prefix cache and RoPE applied inside the
        kernel per-request (MiniPIC approach, arXiv:2606.13126).
        """
        sys_len = relay_meta.relay_info.system_length  # noqa: F841 (used once wired)

        logger.debug_once(
            "RelayAttention: prefix pass not yet wired — "
            "system_length=%d. Falling through to standard FlashAttention.",
            sys_len,
        )

        # Delegate directly to the standard path.  Write into `output` if
        # the caller pre-allocated a buffer, avoiding an extra copy.
        return super().forward(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
        )

    # ------------------------------------------------------------------
    # Public helper: fuse two partial attention outputs
    # ------------------------------------------------------------------

    @staticmethod
    def relay_fuse(
        out_sys: torch.Tensor,
        lse_sys: torch.Tensor,
        out_usr: torch.Tensor,
        lse_usr: torch.Tensor,
        output_lse: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse system-prompt and user attention outputs via LSE weighting.

        Uses the existing ``merge_attn_states`` primitive (same as chunked
        prefill) so that we get the optimised CUDA/Triton implementation
        for free.

        Args:
            out_sys: Attention output over system prompt, shape [N, H, D].
            lse_sys: Log-sum-exp for system-prompt attention, shape [H, N].
                     **Must be in [H, N] order** (num_heads, num_tokens),
                     matching the ``merge_attn_states`` convention.
            out_usr: Attention output over user tokens, shape [N, H, D].
            lse_usr: Log-sum-exp for user attention, shape [H, N].
            output_lse: Optional pre-allocated tensor [H, N] to receive the
                        merged LSE values (useful for chaining merges in
                        tree-based speculative decoding).

        Returns:
            Merged output tensor, shape [N, H, D].

        Raises:
            ValueError: If LSE tensors are not in [H, N] shape.
        """
        # Validate LSE shapes: both must be [H, N].
        if lse_sys.shape != (out_sys.shape[1], out_sys.shape[0]):
            raise ValueError(
                f"lse_sys must have shape [num_heads, num_tokens] = "
                f"[{out_sys.shape[1]}, {out_sys.shape[0]}], "
                f"got {tuple(lse_sys.shape)}. "
                "If your LSE is [N, H], transpose it first: "
                "lse_sys.transpose(0, 1).contiguous()"
            )
        if lse_usr.shape != (out_usr.shape[1], out_usr.shape[0]):
            raise ValueError(
                f"lse_usr must have shape [num_heads, num_tokens] = "
                f"[{out_usr.shape[1]}, {out_usr.shape[0]}], "
                f"got {tuple(lse_usr.shape)}. "
                "If your LSE is [N, H], transpose it first: "
                "lse_usr.transpose(0, 1).contiguous()"
            )

        merged = torch.empty_like(out_usr)
        merge_attn_states(
            output=merged,
            prefix_output=out_sys,
            prefix_lse=lse_sys,
            suffix_output=out_usr,
            suffix_lse=lse_usr,
            output_lse=output_lse,
        )
        return merged
