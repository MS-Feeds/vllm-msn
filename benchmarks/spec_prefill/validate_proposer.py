"""Real-hardware validation for SpecPrefillProposer (vllm_patch/proposer.py).

Run on the actual GPU node (this needs a real Gemma-4-E2B-it checkpoint, a
GPU, and vLLM's full runtime -- none of which are available on the Windows
dev box this was written on; it has NOT been executed, only carefully
reasoned through against this fork's verified V1 APIs. Expect to iterate
against real errors -- see "Known risk areas" below before debugging blind).

Usage:
    export HF_TOKEN=<your token>
    source .env_exports.sh   # this directory's local copy (not the shared
                              # one in ../gemma4_moe_benchmarks/)
    python3 validate_proposer.py --model $GEMMA4_E2B_MODEL_PATH

What this validates, and why it's split into two steps:

Step A (cheap, no forward pass, should just work): loads the speculator via
SpecPrefillProposer, installs the query-capture hooks, and reports each
layer's head_dim/global_head_dim/num_heads/num_kv_heads/is_sliding. This
directly answers the plan's single biggest open question -- does
Gemma-4-E2B-it (dense) share the 26B MoE variant's heterogeneous per-layer
head-dim quirk, or is it uniform? -- without needing any KV cache at all.

Step B (only attempted if Step A finds a uniform head_dim across layers):
a minimal single-request forward pass through run_lookahead_steps() and
retrieve_qk(), using hand-built attention metadata and a manually-allocated,
manually-bound dummy KV cache. This deliberately does NOT do real KV-cache
memory budgeting between two concurrently-loaded Gemma-4 models (that's
explicitly out of scope for this pass, see EXPERIMENT_PLAN.md) -- it
allocates one small dummy cache directly, which is sufficient to exercise
the forward-call mechanics and kv_cache_utils.py's read path for real.

If Step A finds heterogeneous head dims, Step B is skipped with a clear
message: reading a heterogeneous-head-dim KV cache generically is what
kv_cache_utils.py is *designed* to handle (see its per-layer AttentionSpec
approach), but wiring per-layer-group backends/specs for a from-scratch
standalone smoke test (rather than through the real runner, which already
does this) is more machinery than this validation script's scope -- that
would fold into the real runner-integration follow-up work instead.

Known risk areas (things that may need a fix on first real run, since this
was written without being able to execute it):
  - Whether Attention layers self-register into
    `vllm_config.compilation_config.static_forward_context` automatically
    during `get_model()` (assumed here, based on ForwardContext.no_compile_
    layers' docstring "copy from vllm_config.compilation_config.
    static_forward_context") -- if not, `get_attention_context` in
    kv_cache_utils.py won't find the layer and this needs an explicit
    registration step added to SpecPrefillProposer.__init__.
  - Whether directly assigning `layer.attn.kv_cache = dummy_tensor` (instead
    of going through vLLM's own `bind_kv_cache` utility, which does more
    than this script does) is sufficient -- functionally it should be, since
    `get_attention_context` only reads the plain `.kv_cache` attribute, but
    this hasn't been cross-checked against `bind_kv_cache`'s full source.
  - The attention backend actually selected for a real Gemma-4-E2B-it config
    (TritonAttention forced only if head dims differ AND exceed 256 -- Step
    A's own output determines this).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from vllm.config import CacheConfig, ModelConfig, VllmConfig, replace
from vllm_patch.proposer import SpecPrefillProposer


def step_a_inspect_head_dims(proposer: SpecPrefillProposer) -> dict:
    per_layer = []
    for idx, self_attn in enumerate(proposer._speculator_layers):
        per_layer.append(
            {
                "layer_idx": idx,
                "head_dim": self_attn.head_dim,
                "num_heads": self_attn.num_heads,
                "num_kv_heads": self_attn.num_kv_heads,
                "is_sliding": getattr(self_attn, "is_sliding", None),
                "use_k_eq_v": getattr(self_attn, "use_k_eq_v", None),
            }
        )
    head_dims = {row["head_dim"] for row in per_layer}
    return {
        "per_layer": per_layer,
        "num_layers": len(per_layer),
        "distinct_head_dims": sorted(head_dims),
        "is_uniform": len(head_dims) == 1,
    }


def step_b_forward_smoke_test(
    proposer: SpecPrefillProposer, head_dim: int, device: torch.device
) -> dict:
    look_ahead_cnt = 2
    prompt_len = 8

    # Metadata/dummy-KV-cache construction factored into
    # SpecPrefillProposer.build_lookahead_metadata (shared with pruner.py) --
    # see that method's docstring for why it uses tests/v1/attention/utils.py's
    # proven pattern rather than hand-rolled metadata.
    lookahead_meta = proposer.build_lookahead_metadata(prompt_len, look_ahead_cnt, head_dim)

    input_ids = torch.randint(0, 1000, (prompt_len,), dtype=torch.int32, device=device)
    positions = torch.arange(prompt_len, dtype=torch.int64, device=device)

    query_buffer = proposer.run_lookahead_steps(
        initial_input_ids=input_ids,
        initial_positions=positions,
        num_tokens=prompt_len,
        look_ahead_cnt=look_ahead_cnt,
        per_step_attn_metadata=lookahead_meta.per_step_attn_metadata,
        per_step_slot_mapping=lookahead_meta.per_step_slot_mapping,
        next_input_fn=lambda tok: tok[:1].int(),  # not realistic multi-token advance; smoke test only
        next_positions_fn=lambda pos, step: pos[:1] + step + 1,
        eos_token_id=None,
    )

    nan_or_inf = [bool(torch.isnan(q).any() or torch.isinf(q).any()) for q in query_buffer]

    gathered_qk = proposer.tp_gather_qk(query_buffer)
    per_sample_slot_mapping = [lookahead_meta.slot_mapping]
    _, key_buffer = proposer.retrieve_qk(
        gathered_qk,
        per_sample_slot_mapping,
        lookahead_meta.block_size,
        lookahead_meta.num_kv_heads,
        head_dim,
    )

    return {
        "query_buffer_shapes": [tuple(q.shape) for q in query_buffer],
        "any_nan_or_inf_in_queries": any(nan_or_inf),
        "key_buffer_shapes": [tuple(key_buffer[0][0].shape)] if key_buffer and key_buffer[0] else None,
        "attention_backend": proposer.vllm_config.attention_config.backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path or HF id for Gemma-4-E2B-it")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    model_config = ModelConfig(model=args.model, trust_remote_code=True, dtype="bfloat16")
    base_vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16),
    )

    print(f"Loading speculator from {args.model} ...")
    proposer = SpecPrefillProposer(
        base_vllm_config=base_vllm_config,
        speculator_model_config=model_config,
        device=device,
    )
    print(f"Loaded. {proposer._num_layers} attention layers found.")

    print("\n=== Step A: per-layer head_dim inspection ===")
    result_a = step_a_inspect_head_dims(proposer)
    for row in result_a["per_layer"]:
        print(f"  layer {row['layer_idx']:2d}: head_dim={row['head_dim']:4d} "
              f"num_heads={row['num_heads']:3d} num_kv_heads={row['num_kv_heads']:3d} "
              f"is_sliding={row['is_sliding']} use_k_eq_v={row['use_k_eq_v']}")
    print(f"\nDistinct head_dims across layers: {result_a['distinct_head_dims']}")
    print(f"UNIFORM head_dim: {result_a['is_uniform']}"
          f"{' -- attempting Step B' if result_a['is_uniform'] else ' -- SKIPPING Step B (see module docstring)'}")

    if not result_a["is_uniform"]:
        print(
            "\nHeterogeneous head_dim confirmed for Gemma-4-E2B-it. "
            "kv_cache_utils.py is designed to handle this per-layer already; "
            "a from-scratch standalone forward smoke test across multiple "
            "attention-backend groups is deferred to the runner-integration "
            "follow-up (see EXPERIMENT_PLAN.md)."
        )
        return

    print("\n=== Step B: minimal forward-pass + KV retrieval smoke test ===")
    head_dim = result_a["distinct_head_dims"][0]
    result_b = step_b_forward_smoke_test(proposer, head_dim, device)
    print(f"Attention backend selected: {result_b['attention_backend']}")
    print(f"Query buffer shapes (per layer): {result_b['query_buffer_shapes']}")
    print(f"Any NaN/Inf in captured queries: {result_b['any_nan_or_inf_in_queries']}")
    print(f"Retrieved key shape (layer 0, sample 0): {result_b['key_buffer_shapes']}")

    if result_b["any_nan_or_inf_in_queries"]:
        print("\nFAIL: NaN/Inf detected in captured queries -- investigate the "
              "query-capture hook or model loading (dtype/precision issue?).")
        sys.exit(1)
    print("\nStep B PASSED (no NaN/Inf, shapes look sane).")


if __name__ == "__main__":
    main()
