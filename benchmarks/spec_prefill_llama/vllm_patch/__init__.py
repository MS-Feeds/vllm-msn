"""SpecPrefill Algorithm 1 implementation, targeting this fork's vLLM V1 engine.

Status: the full algorithm (lines 1, 3-16, 18-20) is implemented, grounded
directly in this fork's verified V1 source (see ../EXPERIMENT_PLAN.md's
"Implementation status" and the two approved plans in
`c-users-v-dialan-desktop-protocol-specu-logical-fox.md`). Not yet run on
real hardware (no GPU on the machine this was written on) -- see
`../validate_proposer.py` and each new module's own "residual risk" notes
for what to check first.

This package replaces an earlier stub round that mirrored the paper's
reference repo's v0-era dual-`Worker`/`parallel_config.worker_cls` design
directly wrapping two full `Worker`s -- that was the wrong integration
point. The current design: `proposer.py` mirrors V1's own pattern for "run a
secondary model mid-step" (`vllm/v1/spec_decode/llm_base_proposer.py`,
`draft_model.py`) for the *speculator*; `worker.py`/`model_runner.py` use
the verified `parallel_config.worker_cls` extension point to swap in a
custom runner for the *target* model's request handling, without editing
any of vLLM's own files (see `model_runner.py`'s docstring for exactly why
that's safe).

Module map (Algorithm 1 pseudocode step numbers in parens):

    config.py             -- SpecConfig (keep_strategy, keep_kwargs,
                              look_ahead_cnt, pool_kernel_size). Pure
                              Python/YAML, engine-agnostic.
    prefill_split.py       -- (1) split_prefill_decode_requests, via
                              Request.is_prefill_chunk.
    proposer.py             -- (3-11) SpecPrefillProposer: speculator
                              loading, N-step lookahead loop with an
                              instance-level LlamaAttention query-capture
                              hook, TP gather, Q/K retrieval orchestration,
                              plus build_lookahead_metadata (shared metadata
                              helper for proposer.py's own callers).
    kv_cache_utils.py       -- (11, K half) generic per-layer K retrieval
                              from a model's own paged KV cache via
                              get_attention_context.
    scoring.py              -- (12, 14, 16) compute_attention_score,
                              aggregate_attention_score,
                              chunk_select_from_smoothed_attention. Pure
                              tensor math, engine/architecture-agnostic.
    pruning_registry.py     -- process-local PruneRecord store (T, orig_len
                              per pruned request_id), read by model_runner.py.
                              Pure Python, no vLLM dependency -- but note
                              it must be POPULATED inside the Worker's own
                              process (worker.py's register_prune_record),
                              not the driver's -- EngineCore always runs
                              out-of-process from the driver, confirmed on
                              real hardware, see worker.py's docstring.
    pruner.py               -- driver-facing entry point: runs lines 3-16
                              for one prompt via proposer.py/scoring.py,
                              pushes the PruneRecord into the Worker's
                              process via collective_rpc (NOT a direct
                              pruning_registry call -- see worker.py), then
                              calls LLMEngine.add_request with the pruned
                              prompt and cache_salt set (prefix-cache
                              isolation).
    model_runner.py          -- (18-19) SpecPrefillGPUModelRunner: thin
                              _prepare_inputs override that patches RoPE
                              positions in place for pruned requests only,
                              after the stock implementation runs
                              unmodified. Target-forward (line 20) is then
                              just the unmodified rest of execute_model().
    worker.py                -- wires model_runner.py in via the verified
                              parallel_config.worker_cls extension point.

Public re-exports below are for convenience only -- each module also works
standalone (see each file's own docstring for exact API/scope).

`kv_cache_utils.py`/`proposer.py`/`pruner.py`/`model_runner.py`/`worker.py`
import from vLLM's engine/distributed/model-executor internals, which pull
in vLLM's full runtime (e.g. `pyzmq`, `msgspec`) even just to import -- not
installed in every environment this package might be inspected from (e.g.
this repo's local Windows dev checkout has vLLM's *source* importable but
not its full runtime deps). `config.py`, `prefill_split.py`, `scoring.py`,
and `pruning_registry.py` have no such dependency and are importable/
testable anywhere. To keep that true even via `import vllm_patch` (which
would otherwise always execute this file first, pulling in every
submodule's imports transitively), the vLLM-dependent re-exports below are
guarded rather than unconditional.
"""

from .config import SpecConfig, get_spec_config, init_spec_config
from .prefill_split import split_prefill_decode_requests
from .pruning_registry import PruneRecord
from .pruning_registry import discard as discard_prune_record
from .pruning_registry import discard_finished as discard_finished_prune_records
from .pruning_registry import get as get_prune_record
from .pruning_registry import register as register_prune_record
from .scoring import (
    aggregate_attention_score,
    chunk_select_from_smoothed_attention,
    compute_attention_score,
)

__all__ = [
    "SpecConfig",
    "get_spec_config",
    "init_spec_config",
    "split_prefill_decode_requests",
    "compute_attention_score",
    "aggregate_attention_score",
    "chunk_select_from_smoothed_attention",
    "PruneRecord",
    "register_prune_record",
    "get_prune_record",
    "discard_prune_record",
    "discard_finished_prune_records",
]

try:
    from .kv_cache_utils import (
        gather_keys_for_slots,
        read_layer_keys,
        retrieve_keys_per_sample,
    )
    from .model_runner import SpecPrefillGPUModelRunner
    from .proposer import LookaheadMetadata, SpecPrefillProposer
    from .pruner import compute_pruned_prompt, prune_and_add_request
    from .worker import SpecPrefillWorker

    __all__ += [
        "SpecPrefillProposer",
        "LookaheadMetadata",
        "read_layer_keys",
        "gather_keys_for_slots",
        "retrieve_keys_per_sample",
        "compute_pruned_prompt",
        "prune_and_add_request",
        "SpecPrefillGPUModelRunner",
        "SpecPrefillWorker",
    ]
except ImportError:
    # vLLM's full runtime isn't importable in this environment -- the
    # engine-agnostic pieces above (config/prefill_split/scoring/
    # pruning_registry) still work. Import vllm_patch.proposer /
    # vllm_patch.worker etc. directly (where vLLM *is* installed) to get the
    # real ImportError instead of this silent skip.
    pass
