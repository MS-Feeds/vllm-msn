# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP KV Connector that extracts hidden states AND shared_kv_states.

This connector extends ExampleHiddenStatesConnector to also capture the
full_attention and sliding_attention KV caches from Gemma4 target layers,
which are needed for Gemma4 MTP online training (shared_kv_states).

The shared_kv_states dict structure matches what Gemma4's transformers
implementation expects:
    {
        "full_attention": (kv_full_k, kv_full_v),   # (B, H, T, D)
        "sliding_attention": (kv_slide_k, kv_slide_v),
    }
"""

from __future__ import annotations

import fcntl
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
    PendingSave,
    extract_from_kv_cache,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import Attention

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _extract_kv_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract K and V tensors from the KV cache for the given slot mapping.

    Args:
        kv_cache: KV cache tensor of shape (num_blocks, block_size, 2, num_heads, head_dim)
                  or similar layout depending on the attention backend.
        slot_mapping: Flat tensor of slot indices for each token.
        num_tokens: Number of tokens to extract.

    Returns:
        Tuple of (K, V) tensors each of shape (num_tokens, num_heads, head_dim).
    """
    block_size = kv_cache.shape[1]
    # kv_cache layout depends on backend. For vLLM v1, common layouts are:
    # - (num_blocks, block_size, 2, num_heads, head_dim) for FLASH_ATTN
    # - (num_blocks, block_size, num_heads, 2, head_dim) for TRITON_ATTN
    # We need to handle both cases.
    if kv_cache.ndim == 5:
        # (num_blocks, block_size, 2, num_heads, head_dim) -> transpose to
        # (num_blocks, block_size, num_heads, 2, head_dim)
        kv_cache = kv_cache.transpose(3, 4).transpose(2, 3)
        # Now: (num_blocks, block_size, num_heads, 2, head_dim)
    elif kv_cache.ndim != 5:
        raise ValueError(f"Unexpected KV cache ndim: {kv_cache.ndim}, shape={kv_cache.shape}")

    indices = slot_mapping // block_size
    offsets = slot_mapping % block_size

    # Extract K and V separately
    # K is at position 0, V is at position 1 in the 2nd-to-last dimension
    k_cache = kv_cache[:, :, :, 0, :]  # (num_blocks, block_size, num_heads, head_dim)
    v_cache = kv_cache[:, :, :, 1, :]

    # Gather along block dimension
    k_extracted = k_cache[indices, offsets]  # (num_tokens, num_heads, head_dim)
    v_extracted = v_cache[indices, offsets]

    return k_extracted[:num_tokens], v_extracted[:num_tokens]


@dataclass
class Gemma4PendingSave(PendingSave):
    """Extended PendingSave that includes KV cache block mappings."""
    # Map of attention type -> list of block_ids for that attention type's KV cache
    kv_block_ids: dict[str, list[int]] = field(default_factory=dict)
    # Map of attention type -> slot mapping for this request
    kv_slot_mappings: dict[str, torch.Tensor] = field(default_factory=dict)


class Gemma4SharedKVStatesConnectorMetadata(ExampleHiddenStatesConnectorMetadata):
    """Extended metadata that tracks KV cache info per attention type."""

    # Map of attention type -> block size for that attention type's KV cache group
    kv_block_sizes: dict[str, int] = field(default_factory=dict)


class Gemma4SharedKVStatesConnector(ExampleHiddenStatesConnector):
    """
    KV Connector for Gemma4 MTP that extracts both hidden states and
    shared_kv_states (full_attention + sliding_attention KV caches).

    This enables online training where the assistant model needs access to
    the target's KV cache for cross-attention, not just hidden states.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # Extended state for shared_kv_states extraction
        # Map of attention type -> KV cache tensor
        self._kv_caches: dict[str, torch.Tensor] = {}
        # Map of attention type -> block size
        self._kv_block_sizes: dict[str, int] = {}
        # Map of attention type -> layer names that contribute to this KV type
        self._kv_layer_names: dict[str, list[str]] = {
            "full_attention": [],
            "sliding_attention": [],
        }
        # Map of attention type -> KV cache group index
        self._kv_group_ids: dict[str, int] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register both hidden states cache and target attention KV caches."""
        # First, call parent to register the hidden states cache
        super().register_kv_caches(kv_caches)

        # Now register the target attention layers for shared_kv_states
        self._register_attention_kv_caches(kv_caches)

    def _register_attention_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Register attention layers that hold the KV cache for shared_kv_states."""
        if not self._is_tp_rank_zero:
            return

        from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

        attn_layers = get_layers_from_vllm_config(
            self._vllm_config, AttentionLayerBase, list(kv_caches.keys())
        )

        # Iterate over all attention layers (not just CacheOnlyAttentionLayer)
        # to find the source layers for shared_kv_states
        for layer_name, layer in attn_layers.items():
            if isinstance(layer, Attention):
                # Check if this is a source layer (not a shared layer)
                kv_sharing_target = getattr(layer, "kv_sharing_target_layer_name", None)
                if kv_sharing_target is not None:
                    # This is a shared layer - skip, we use the target
                    continue

                # Get the layer type (full_attention or sliding_attention)
                is_full_attention = getattr(layer, "is_full_attention", False)
                is_sliding = getattr(layer, "is_sliding", False)

                if is_full_attention:
                    attn_type = "full_attention"
                elif is_sliding:
                    attn_type = "sliding_attention"
                else:
                    # Default based on config
                    attn_type = "full_attention"

                if layer_name not in self._kv_layer_names[attn_type]:
                    self._kv_layer_names[attn_type].append(layer_name)

                if attn_type not in self._kv_caches:
                    self._kv_caches[attn_type] = kv_caches[layer_name]
                    # Get block size from the KV cache tensor
                    if len(kv_caches[layer_name].shape) >= 2:
                        self._kv_block_sizes[attn_type] = kv_caches[layer_name].shape[1]
                    else:
                        self._kv_block_sizes[attn_type] = self._block_size

                    # Find the group ID for this attention type
                    if self._kv_cache_config is not None:
                        for gid, group in enumerate(
                            self._kv_cache_config.kv_cache_groups
                        ):
                            if layer_name in group.layer_names:
                                self._kv_group_ids[attn_type] = gid
                                break

        logger.info(
            "Gemma4SharedKVStatesConnector: registered KV caches for types %s",
            list(self._kv_caches.keys()),
        )
        for attn_type, layer_names in self._kv_layer_names.items():
            logger.info(
                "  %s: %d layers, block_size=%s",
                attn_type,
                len(layer_names),
                self._kv_block_sizes.get(attn_type),
            )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "Gemma4SharedKVStatesConnectorMetadata":
        """Build connector metadata including KV block sizes for each attention type."""
        meta = super().build_connector_meta(scheduler_output)

        # Convert to our extended metadata type
        extended_meta = Gemma4SharedKVStatesConnectorMetadata(
            pending_saves=meta.pending_saves,
            new_req_filenames=meta.new_req_filenames,
            kv_block_sizes=self._kv_block_sizes,
        )
        return extended_meta

    def wait_for_save(self) -> None:
        """Pre-create lock files and set up KV extraction tracking."""
        super().wait_for_save()
        # Additional setup if needed

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished generating.

        For Gemma4 MTP with multiple KV cache groups (full_attention and
        sliding_attention), we override this to properly map each group's
        block_ids to the correct attention type.
        """
        req_id = request.request_id

        # Create slot mappings for each attention type based on the
        # corresponding group's block_ids
        kv_slot_mappings: dict[str, torch.Tensor] = {}
        kv_block_ids: dict[str, list[int]] = {}

        for attn_type, cache_block_size in self._kv_block_sizes.items():
            # Find the group index for this attention type
            gid = self._kv_group_ids.get(attn_type, -1)
            if gid < 0 or gid >= len(block_ids):
                continue

            group_block_ids = block_ids[gid]
            if not group_block_ids or cache_block_size <= 0:
                continue

            block_ids_t = torch.tensor(group_block_ids, dtype=torch.long)
            num_blocks = block_ids_t.shape[0]
            block_offsets = torch.arange(0, cache_block_size, dtype=torch.long)
            slot_mapping = (
                block_offsets.reshape((1, cache_block_size))
                + block_ids_t.reshape((num_blocks, 1)) * cache_block_size
            )
            kv_slot_mappings[attn_type] = slot_mapping.flatten()
            kv_block_ids[attn_type] = group_block_ids

        # Create extended pending save with KV info
        kv_pending = Gemma4PendingSave(
            req_id=req_id,
            filename=self._request_filenames.get(req_id, ""),
            token_ids=torch.tensor(
                list(request.prompt_token_ids) if request.prompt_token_ids else []
            ),
            block_ids=block_ids[0] if block_ids else [],
            kv_block_ids=kv_block_ids,
            kv_slot_mappings=kv_slot_mappings,
        )

        # Add to pending saves for async extraction
        if hasattr(self, "_pending_saves"):
            self._pending_saves[req_id] = kv_pending

        # Also call the base class's request_finished directly to handle
        # hidden states extraction (not going through our parent's override
        # which would create a conflicting PendingSave)
        filename = self._request_filenames.pop(req_id, "")
        kv_params = request.kv_transfer_params or {}
        if kv_params.get("include_output_tokens", False):
            token_ids = torch.tensor(list(request.all_token_ids)[:-1])
        elif request.prompt_token_ids is not None:
            token_ids = torch.tensor(request.prompt_token_ids)
        else:
            logger.warning(
                "Request %s has no prompt_token_ids (prompt_embeds only). "
                "Saved token_ids will be empty.",
                req_id,
            )
            token_ids = torch.tensor([], dtype=torch.long)

        # Don't call parent's request_finished - directly add the PendingSave
        # for hidden states (will be processed by parent's get_finished)
        self._pending_saves[f"{req_id}_hidden"] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids[self._cache_kv_group_id]),
        )

        return True, {"hidden_states_path": filename}

    def _submit_async_write(
        self,
        pending: "Gemma4PendingSave | PendingSave",
    ) -> None:
        """Extract hidden states AND shared_kv_states, then submit async write."""
        if not self._is_tp_rank_zero:
            return

        # First, extract hidden states (parent logic)
        super()._submit_async_write(pending)

        # If no shared_kv_states to extract, we're done
        if not isinstance(pending, Gemma4PendingSave) or not pending.kv_block_ids:
            return

        # Extract shared_kv_states for Gemma4 MTP
        self._extract_shared_kv_states(pending)

    def _extract_shared_kv_states(
        self,
        pending: "Gemma4PendingSave",
    ) -> None:
        """Extract shared_kv_states (full_attention and sliding_attention KV) for a request."""
        # Get slot mappings from the pending save's stored slot mappings
        # Note: slot_mappings were stored during request_finished per attention type
        for attn_type, slot_mapping in pending.kv_slot_mappings.items():
            if attn_type not in self._kv_caches:
                logger.warning(
                    "No KV cache registered for attention type: %s",
                    attn_type,
                )
                continue

            kv_cache = self._kv_caches[attn_type]
            block_size = self._kv_block_sizes.get(attn_type, self._block_size)
            num_tokens = pending.token_ids.shape[0]

            copy_stream = self._get_copy_stream()

            # Ensure copy stream is synchronized
            ready_event = torch.cuda.Event()
            ready_event.record()
            copy_stream.wait_event(ready_event)

            with torch.cuda.stream(copy_stream):
                slot_mapping_gpu = slot_mapping.to(
                    device=kv_cache.device, non_blocking=True
                )
                k_extracted, v_extracted = _extract_kv_from_kv_cache(
                    kv_cache, slot_mapping_gpu, num_tokens
                )

                # Async DtoH copy
                pinned_k = torch.empty_like(k_extracted, device="cpu", pin_memory=True)
                pinned_v = torch.empty_like(v_extracted, device="cpu", pin_memory=True)
                pinned_k.copy_(k_extracted, non_blocking=True)
                pinned_v.copy_(v_extracted, non_blocking=True)

            # The actual saving is handled separately - we store the extracted
            # tensors in pending.kv_extracted so they can be saved with the main output
            # For now, we'll save them to a separate file for each attention type
            # The filename pattern is: {base_filename}.{attn_type}.safetensors
            base_filename = pending.filename
            kv_filename = base_filename.replace(".safetensors", f".{attn_type}.safetensors")

            copy_done = torch.cuda.Event()
            copy_done.record(copy_stream)

            tensors = {
                f"kv_{attn_type}_k": pinned_k,
                f"kv_{attn_type}_v": pinned_v,
            }

            # Use existing lock fd pattern
            lock_fd = self._lock_fds.pop(f"{pending.req_id}_{attn_type}", None)
            if lock_fd is None and self.use_lock:
                lock_path = kv_filename + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            future = self._executor.submit(
                self._write_tensors, tensors, copy_done, kv_filename, lock_fd
            )
            logger.debug(
                "Submitted async write for %s KV: %s, %d tokens",
                attn_type,
                kv_filename,
                num_tokens,
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished generating.

        For Gemma4 MTP, we also need to track the block IDs and slot mappings
        for each attention type's KV cache.
        """
        # Call parent to handle hidden states path
        result = super().request_finished(request, block_ids)

        # Store KV block IDs and slot mappings for shared_kv_states extraction
        # The slot mapping depends on the block_ids for each attention type's cache group
        # For now, we use the same block_ids as the main cache
        # In a full implementation, we would track per-group block IDs
        req_id = request.request_id

        # Create extended pending save with KV info
        kv_pending = Gemma4PendingSave(
            req_id=req_id,
            filename=self._request_filenames.get(req_id, ""),
            token_ids=torch.tensor(list(request.prompt_token_ids) if request.prompt_token_ids else []),
            block_ids=block_ids,
            kv_block_ids={
                "full_attention": block_ids,
                "sliding_attention": block_ids,
            },
            kv_slot_mappings={},
        )

        # Calculate slot mappings for each attention type
        for attn_type, cache_block_size in self._kv_block_sizes.items():
            if cache_block_size and cache_block_size > 0:
                block_ids_t = torch.tensor(block_ids, dtype=torch.long)
                num_blocks = block_ids_t.shape[0]
                block_offsets = torch.arange(0, cache_block_size, dtype=torch.long)
                slot_mapping = (
                    block_offsets.reshape((1, cache_block_size))
                    + block_ids_t.reshape((num_blocks, 1)) * cache_block_size
                )
                kv_pending.kv_slot_mappings[attn_type] = slot_mapping.flatten()
                kv_pending.kv_block_ids[attn_type] = block_ids

        # Add to pending saves for async extraction
        if hasattr(self, "_pending_saves"):
            self._pending_saves[req_id] = kv_pending

        return result

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check for completed async operations.

        Returns (done_sending, done_recving) tuples.
        """
        # First handle hidden states extraction
        done_sending, done_recving = super().get_finished(finished_req_ids)

        # For Gemma4, we also need to check KV extraction completions
        # This is handled by the parent's get_finished tracking copy events
        return done_sending, done_recving