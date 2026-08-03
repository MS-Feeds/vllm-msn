# Implementation Plan: `benchmarks/rlm_specprefill/`

## Context

[../rlm/rlm_specprefill_ablation_plan.md](../rlm/rlm_specprefill_ablation_plan.md) (untracked, sitting in `benchmarks/rlm/`) lays out the *experimental design* for a scoped ablation: does SpecPrefill (attention-transfer-based prefill token pruning) actually speed up an RLM (Recursive Language Model) pipeline on massive (>131K-token) contexts, isolated from every other pipeline variable? That document specifies what to measure (three arms A/B/C, a 4-term latency model, an Amdahl-bound go/no-go checkpoint on `f`) but says nothing about how to actually build it against this repo's code. This directory, and this plan, are that missing implementation layer.

Two things already exist that this ties together, but which have **zero code coupling today**:
- `benchmarks/rlm/` — a vendored copy of the upstream `alexzhang13/rlm` OSS package (RLM orchestration: root model + REPL + recursive sub-calls). Treat as a dependency, not something to edit in place.
- `benchmarks/spec_prefill_llama/` — a from-scratch SpecPrefill port for vLLM's V1 engine targeting Llama-3.1-8B (target) + Llama-3.2-1B (speculator), wired in via a `worker_cls` extension point. Status: **code-complete but never run on real GPU hardware.**

`benchmarks/rlm_specprefill/` (sibling to both) is the orchestration layer that makes RLM's output feed into SpecPrefill-gated (or not) target-model calls, plus the instrumentation/dataset/calibration work the ablation doc calls for. The machine this was built on (Windows, no GPU, `vllm` not even installed) cannot execute the actual experiment — the deliverable for that pass was a complete, GPU-ready harness plus everything about it that *is* verifiable without a GPU, ending in a `REPRODUCE.md` runbook for whoever runs it on a GPU node next.

### Decisions already made (user-confirmed, not open questions)

1. **Root model = hosted Claude** via the Anthropic API (`backend="anthropic"`), matching the already-working `../rlm/examples/quickstart_anthropic.py` and the live `ANTHROPIC_API_KEY` in `benchmarks/rlm/.env`. Root does not need local attention access — only the SpecPrefill target+speculator pair does. *Note: may change later if the team decides root also needs to be self-hosted.*
2. **Target/speculator = Llama-3.1-8B / Llama-3.2-1B**, using `benchmarks/spec_prefill_llama/`'s existing `vllm_patch/`. Originally built against `benchmarks/spec_prefill_qwen/`'s Qwen3-8B/Qwen3-1.7B pairing, then switched to this Llama pairing (see the "Switched from Qwen3 to Llama" note below for exactly what that changed). *Note: may change again later — e.g. to the Gemma4 pairing in `benchmarks/spec_prefill/`, or back to Qwen3.*
3. **Execution scope**: build the harness now; real GPU runs deferred to later, on a GPU node, following the runbook this plan produces.
4. **Network egress requirement dropped**: no sandboxing/firewall work. Use `environment="local"` (RLM's default in-process REPL) as-is, exactly like the existing quickstart examples.

### Load-bearing findings from direct code inspection (shape the design below)

- **`AnthropicClient` never forwards `sampling_args`/temperature** ([../rlm/rlm/clients/anthropic.py:41](../rlm/rlm/clients/anthropic.py) — `kwargs` only contains `model`, `max_tokens`, `messages`, `system`). So "fixed seed / greedy decoding" is not actually available for confound control on this backend — **cache-and-replay of RLM's evidence output is not a fallback option, it is the only mechanism**, and the harness is built around it as a hard requirement, not an optimization.
- **`max_budget` is a no-op for the Anthropic backend** — `_track_cost` ([../rlm/rlm/clients/anthropic.py:87](../rlm/rlm/clients/anthropic.py)) never sets `ModelUsageSummary.total_cost` (stays `None`), and `RLM`'s budget check computes `current_usage.total_cost or 0.0`, which is always `0.0`. Don't rely on it as a real guardrail; lean on `max_tokens` + `max_timeout` instead, and compute real cost post-hoc from token counts.
- **Confirmed live against the real Anthropic API during step 7 (`runner/smoke_test.py`): RLM's own max-iterations fallback path can crash with a bare `IndexError`.** When `max_iterations` is exhausted without the model setting `answer["ready"]=True`, `RLM._default_answer` ([../rlm/rlm/core/rlm.py:673](../rlm/rlm/core/rlm.py)) appends a synthetic "please provide a final answer" message with `role="assistant"` (almost certainly meant to be `"user"` — as written, Anthropic treats it as an assistant-message *prefill to continue*, not an instruction to follow) and sends it via `lm_handler.completion()`. At least one such call came back with an **empty `content` list**, which `AnthropicClient.completion()` doesn't guard against (`response.content[0].text`, [../rlm/rlm/clients/anthropic.py:47](../rlm/rlm/clients/anthropic.py)) — bare `IndexError`, uncaught. Hit twice in a handful of live smoke-test runs (not a rare edge case), on both a 1-needle and a 2-needle synthetic task, so this is real, observed non-determinism in whether RLM converges within `max_iterations` on this prompt design, not a contrived failure mode. **Not something to patch in the vendored `rlm` package** — instead, `runner/run_arm.py::collect_evidence_for_dataset` wraps each sample's evidence extraction in a try/except and skips-and-reports failures (same convention `spec_prefill_llama/predict_longbench_v2.py` uses for over-budget samples), so one non-converging sample doesn't crash an entire arm's evidence collection. Worth watching in aggregate once real sweeps run: if a nontrivial fraction of samples hit this, it's a signal the evidence-extraction prompt (`prompts/evidence_extraction.py`) needs tuning to converge faster, not just something to keep skipping past.
- **The offline-batched engine-driving pattern doesn't give per-request latency for free — it had to be added.** `predict_longbench_v2.py` (and, initially, this project's own `target_stage/vllm_offline_engine.py`) only tracked TTFT per request and aggregate elapsed time per batch, because that's all the pattern naturally exposes; there's no true "time from submission to finish" per request the way a live server's per-connection timing gives you, since all requests in a batch are submitted ~simultaneously and driven together. That's not good enough for `analysis/aggregate_metrics.py`'s per-sample `f` (target share of latency) — a mean/aggregate `f` isn't what the ablation doc asks for ("Report `f` as a distribution ... not a mean"). Fixed by extending `drive_engine_to_completion` to record a wall-clock timestamp the first time each request's output reports `finished=True`, and `answer_batch` to compute `generation_time_s` from that minus the batch's own start time — a real measured duration (batch-relative, not live-server-accurate, same caveat TTFT already carries), not an estimate.

## Directory layout

```
benchmarks/rlm_specprefill/
  README.md                          # orientation: links the experimental-design doc vs. this implementation
  IMPLEMENTATION_PLAN.md             # this file
  REPRODUCE.md                       # GPU-node runbook (modeled on spec_prefill_llama/REPRODUCE.md)
  .env_exports.sh                    # sources rlm/.env's ANTHROPIC_API_KEY + spec_prefill_llama paths

  configs/
    guardrails.yaml                  # max_depth, max_iterations, max_concurrent_subcalls, max_timeout, max_tokens, max_errors
    spec_config_always_on.yaml       # SpecConfig YAML for Arm B (same schema as vllm_patch/config.py's SpecConfig)
    n_min.json                       # written by calibration/sweep_n_min.py, consumed by target_stage/gate.py
    arms.yaml                        # per-arm target-generation params

  prompts/
    evidence_extraction.py           # system prompt + builder: RLM as retrieval front-end, not answerer
    target_answer.py                 # template for the harness's own final target-answering call

  eval_data/                        # named to avoid shadowing the installed `datasets` (HF) library
    schema.py                        # common {"id","context","question","answer","source"} record
    prep_longbench_v2_long.py        # spec_prefill_llama's prep script, filtered to length=="long" (>128k words) not "short"
    filter_by_token_length.py        # real Llama-tokenizer length measurement + >131K filter
    gen_synthetic_niah.py            # scaled-up multi-needle NIAH generator (generalizes quickstart_anthropic.py's haystack)
    longbench_v2_long_samples.jsonl  # (generated)
    synthetic_niah_samples.jsonl     # (generated)

  rlm_stage/
    evidence_rlm.py                  # constructs RLM(backend="anthropic", custom_system_prompt=..., guardrails), runs one query
    evidence_cache.py                # content-hash cache/replay layer — the confound-control mechanism
    timing_decomposition.py          # post-processes an RLMLogger trajectory into T_RLM_root / T_REPL_compute / T_RLM_subcalls

  target_stage/
    vllm_offline_engine.py           # adapted core of predict_longbench_v2.py: build_plain_target_engine(), build_specprefill_target_engine(), answer_one()
    gate.py                          # should_compress(n_tokens, n_min) -> bool
    route_queries.py                 # Arm C: partitions queries into plain/specprefill buckets before either engine loads

  runner/
    run_arm.py                       # CLI: run one arm (A/B/C) over a dataset -> results/<arm>/predictions.jsonl
    run_all_arms.py                  # sequences A -> B -> C (C = two sequential engine passes, never concurrent)
    smoke_test.py                    # no-GPU: evidence_rlm.py + evidence_cache.py end-to-end on small synthetic samples

  calibration/
    sweep_n_min.py                   # N_min crossover sweep against target+speculator alone, decoupled from live RLM
    transferability_check.py         # RLM-format vs. spec_prefill_llama's own benchmark-format keep-rate/quality comparison

  analysis/
    aggregate_metrics.py             # f-distribution, depth/fanout dist, gate split, r, accuracy/recall (reuses grade_longbench_v2.py)

  tests/
    test_timing_decomposition.py
    test_gate.py
    test_route_queries.py
    test_dataset_filtering.py        # not yet written — see Verification section for what's covered ad hoc instead
    test_evidence_cache.py
    test_vllm_offline_engine.py      # added beyond the original file list: percentile/resolve_max_num_batched_tokens/prompt-formatting, no vllm needed
    test_runner.py                   # added beyond the original file list: run_arm/run_all_arms validation + ordering, via monkeypatched run_evidence_extraction
    test_sweep_n_min.py              # added beyond the original file list: pooling/bucketing/crossover computation
    test_transferability_check.py    # added beyond the original file list: reference-curve loading/recall/comparison
    test_aggregate_metrics.py        # added beyond the original file list: aggregation against synthetic predictions.jsonl/timing.jsonl fixtures
    test_config_roundtrip.py         # not yet written — round-trips spec_config_always_on.yaml through SpecConfig.from_path
    test_smoke_anthropic.py          # not yet written as a pytest test — covered instead by rlm_stage/evidence_rlm.py --smoke-test and runner/smoke_test.py (both run directly, not via pytest, since they make real billed API calls)

  results/
    evidence_cache/                  # {hash}.json cached RLM evidence outputs
    A/  B/  C/                       # per-arm predictions.jsonl + timing.jsonl
  logs/                              # RLMLogger(log_dir=...) target, gitignored
```

Does not fold into `spec_prefill_llama/` or `rlm/` — it orchestrates both without owning either.

## Key design decisions

**1. RLM is repurposed into a retrieval front-end, not an answerer.** Use `custom_system_prompt` on the `RLM` constructor (`../rlm/rlm/core/rlm.py:62`) — must retain the `{custom_tools_section}` placeholder that `build_rlm_system_prompt` formats into it (`../rlm/rlm/utils/prompts.py:228`), or a `KeyError` results. Keep the REPL/tool description block adapted from `RLM_SYSTEM_PROMPT` (context, `llm_query`, `rlm_query`, `SHOW_VARS`, the `answer` dict mechanics); replace only the goal framing: the model is told it is *not* answering the question, but locating and returning the smallest set of verbatim excerpts a downstream model would need, as `answer["content"] = json.dumps({"excerpts": [...], "question": ...})`. Call via `rlm.completion(prompt=full_prompt)` where `full_prompt` folds the question and context together in one string, matching the repo's own working example (`quickstart_anthropic.py`) rather than relying on the `root_prompt` argument — `root_prompt`'s hardcoded "Answer the following: …" wording (baked into `build_rlm_system_prompt`) actively fights the retrieval framing, so it's simpler to skip it. Harness reads `RLMChatCompletion.response`, `json.loads()`s it with a graceful fallback (treat as one unstructured excerpt blob on parse failure — don't hard-fail the sample).

**2. Target calls bypass RLM's client abstraction entirely.** RLM's `backend="vllm"` routes through `OpenAIClient` hitting an HTTP server (`../rlm/rlm/clients/__init__.py:23`) — but SpecPrefill's only validated driver pattern is vLLM's **offline** `LLM` class, manually driving `engine.step()` (`../spec_prefill_llama/predict_longbench_v2.py`). Rather than building an HTTP shim or testing whether `worker_cls` survives `vllm serve` (both unvalidated, both extra work), `target_stage/vllm_offline_engine.py` drives the offline engine directly — adapted from (not importing) `predict_longbench_v2.py`'s `submit_baseline_requests`/`submit_pruned_requests`/TTFT extraction (`output.metrics.first_token_latency`, confirmed-correct per that file's own real-hardware note — not `first_token_ts - arrival_time`). Copy rather than import: that script has import-time side effects (creates a `results/` dir, hardcodes sweep constants as module globals) unsuited to being a library. Note also (found while reading that script in detail): it constructs `SpecConfig` directly in Python and passes it into `submit_pruned_requests`/the pruner driver call, rather than relying solely on the `SPEC_CONFIG_PATH` env-var singleton — `vllm_offline_engine.py` should follow that same direct-construction pattern rather than assuming the env-var path is what's exercised.

**3. "Two serving endpoints" (plain vs. SpecPrefill target) are sequential single-engine passes, never concurrent** — forced by `worker_cls` being fixed at `LLM()` construction time (no hot-swap). Arm A and B each build one engine and run every sample through it. Arm C computes the gate decision for every sample **before touching a GPU** (`target_stage/route_queries.py`, using the real Llama tokenizer on cached evidence), partitions into skip/compress buckets, runs the plain engine over skip, tears it down, builds the SpecPrefill engine, runs it over compress. `run_all_arms.py` encodes this ordering so at most one `LLM` instance is ever live.

**4. Cache-and-replay is the confound-control mechanism, not an option.** `rlm_stage/evidence_cache.py` keys on a hash of `(sample_id, context_digest, question, prompt_version, guardrail_config_digest)`. First run (naturally an Arm-A-shaped pass) executes `evidence_rlm.py` once and persists the full `RLMChatCompletion` — including `.metadata`/trajectory, needed by `timing_decomposition.py` for per-query reporting even on cache hits — to `results/evidence_cache/<hash>.json`. Arms B/C always call `evidence_cache.get_or_run(sample)` and hit the cache; RLM is never re-invoked for them.

**5. Guardrail caps map directly onto existing `RLM.__init__` params** — no wrapper needed for `max_depth`, `max_concurrent_subcalls` (default 4 already matches the ablation doc's own suggestion), `max_timeout`, `max_tokens`, `max_errors`. `max_budget` is kept in the config (harmless) but is advisory-only per the finding above. **No hard `max_total_calls` cap or per-iteration fan-out cap** — `on_subcall_start/complete` callbacks are fire-and-forget (`Callable[..., None]`, can't abort), so a real cap needs monkeypatching call sites; not worth the fragility while `max_tokens`+`max_timeout`+`max_errors` already bound worst case. Instead, realized call count and fan-out per iteration are logged for reporting (already visible via `len(REPLResult.rlm_calls)`); revisit a hard cap only if Arm-A-only data shows it's actually needed.

**6. Latency decomposition (`T_RLM_root`, `T_REPL_compute`, `T_RLM_subcalls`) needs no new instrumentation hooks — it's derivable from data `RLMLogger` already captures:**
- `T_RLM_root_i = iteration.iteration_time - sum(code_block.result.execution_time for code_block in iteration.code_blocks)` (root call happens before REPL execution in `RLM._completion_turn`, `../rlm/rlm/core/rlm.py:646`).
- `sum(call.execution_time for call in code_block.result.rlm_calls)` correctly reconstructs true wall-clock sub-call time whether calls were sequential or concurrent (batched calls' individual `execution_time` is `total_time / n`, so the sum recovers the true total) — this is the "critical path" the latency model needs.
- `T_REPL_compute_b = code_block.result.execution_time - sum(rlm_calls execution_time)`.
- TTFT (and, since step 9, real per-request `generation_time_s`) only exist target-side (Anthropic root calls are non-streaming) — captured entirely in `vllm_offline_engine.py::answer_batch()`.
- `T_REPL_network` is out of scope (network-egress requirement dropped).

**7. Eval set: LongBench v2 "long" bucket (primary) + scaled synthetic NIAH (secondary, for well-defined evidence-recall ground truth).** `eval_data/prep_longbench_v2_long.py` flips `../spec_prefill_llama/datasets/prep_longbench_v2.py`'s filter from `length == "short"` to `length == "long"`. (This project's own data directory is named `eval_data/`, not `datasets/`, specifically to avoid shadowing the installed `datasets` HF library once the project root is on `sys.path` — `prep_longbench_v2_long.py` itself calls `from datasets import load_dataset`, which would silently resolve to the wrong thing if the two names collided.) Token length must be measured with Llama's **real tokenizer** (`AutoTokenizer.from_pretrained(LLAMA31_8B_MODEL_PATH)`), not `rlm/utils/token_utils.py`'s tiktoken/char-estimate fallback — that utility exists for the Anthropic root's own compaction accounting against Anthropic's context limit, a different use case, and its `MODEL_CONTEXT_LIMITS` table has no Llama entry at all (checked directly — only OpenAI/Anthropic/Gemini/Qwen/Kimi/GLM keys are present), so it would silently fall back to a generic default, not even a wrong per-model guess. `gen_synthetic_niah.py` generalizes `../rlm/examples/quickstart_anthropic.py`'s single-needle haystack into a parametric, multi-needle, known-ground-truth generator — also the natural source for step 8's calibration sweep.

**8. `N_min` calibration stays decoupled from live RLM**, per the ablation doc's explicit requirement: `calibration/sweep_n_min.py` pools candidate texts (synthetic NIAH excerpts as a day-1 bootstrap, real cached RLM evidence once available), buckets by fixed token-count bins, feeds each bin through both the plain and SpecPrefill target engines, and writes the empirical crossover to `configs/n_min.json`. `transferability_check.py` then compares against `spec_prefill_llama`'s own LongBench-v2-format keep-rate/quality curve to flag divergence.

## Order of implementation

1. Scaffold directories + `configs/guardrails.yaml`, `configs/spec_config_always_on.yaml`, `REPRODUCE.md` skeleton, `README.md`. **[done]**
2. `eval_data/schema.py`, `prep_longbench_v2_long.py`, `filter_by_token_length.py`, `gen_synthetic_niah.py` — no GPU/API needed. **[done]**
3. `prompts/evidence_extraction.py` + `rlm_stage/evidence_rlm.py` — testable against the live Anthropic API now (small synthetic contexts). **[done]**
4. `rlm_stage/evidence_cache.py` — cache/replay keyed on a hash of (sample_id, context, question, prompt_version, guardrails). Unit-tested with an injected stub `run_fn` rather than a mocked `RLM` client (`RLM`'s `backend=` string routing through `rlm.clients.get_client` has no clean seam for substituting a mock without monkeypatching internals — see the module's own docstring). 7/7 tests pass (`tests/test_evidence_cache.py`). **[done]**
5. `rlm_stage/timing_decomposition.py` — unit-tested against hand-built trajectory fixtures (sequential sub-calls, batched-call reconstruction, single- and multi-level child-RLM recursion) validating the arithmetic identities in decision 6, then cross-checked against the real trajectory logged during step 3's live smoke test — the identity `t_rlm_root + t_repl_compute + t_rlm_subcalls == iteration_time` holds exactly on real Anthropic API data, and (as expected for a run that made 0 sub-calls) `total_calls == n_iterations`. 8/8 tests pass (`tests/test_timing_decomposition.py`). **[done]**
6. `target_stage/vllm_offline_engine.py`, `gate.py`, `route_queries.py` — adapted from `predict_longbench_v2.py`'s confirmed patterns (drive_engine_to_completion, original-vs-rewritten request-id handling, force-kept prefix/suffix around the prunable excerpts blob, `first_token_latency` TTFT, p90-based budget auto-sizing). All `vllm`/`torch`/`transformers` imports deliberately deferred to inside functions (matching that script's own convention) so the dataclasses/prompt-formatting/budget-sizing logic stays importable and testable without `vllm` installed — confirmed: all three modules import cleanly on this machine. `gate.py`/`route_queries.py` (pure token-count math) and the pure-Python pieces of `vllm_offline_engine.py` (percentile, resolve_max_num_batched_tokens, render_excerpts_text, build_prompt_pieces, the TargetQuery/TargetAnswer dataclasses) are unit-tested against a hand-written fake chat-template tokenizer (no network, no vllm) — 23/23 tests pass across `tests/test_vllm_offline_engine.py`, `test_gate.py`, `test_route_queries.py`. Engine construction and request submission (GPU-only) are not exercised here; see REPRODUCE.md's GPU-node validation steps. **[done]**
7. `runner/smoke_test.py`, then `run_arm.py`/`run_all_arms.py` — the RLM-evidence half (evidence collection + caching, `--dry-run`) is real-API-tested via the actual CLI: generated a tiny 2-sample dataset, ran `run_arm.py --arm A --dry-run` twice (first pass: 0/2 cache hits, real RLM calls; second pass: 2/2 cache hits, no RLM calls), and separately ran `runner/smoke_test.py` itself (which exercises `collect_evidence_for_dataset` directly) to a clean pass. `run_all_arms.py`'s arm-ordering/dedup logic (forces Arm A first regardless of requested order) is unit-tested via a monkeypatched `run_arm`. `run_arm`'s validation (unknown arm, empty dataset) is unit-tested directly. 7/7 tests pass (`tests/test_runner.py`). The target-call half (Arms A/B/C past `--dry-run`) is GPU-only, not exercised here. **[done — see the load-bearing finding below about a real upstream crash this surfaced and how it's handled]**
8. `calibration/sweep_n_min.py`, `transferability_check.py` — candidate-pooling/bucketing/truncation, crossover computation, reference-curve loading, and curve-comparison logic are all pure Python and unit-tested (23/23: 12 for `sweep_n_min`, 11 for `transferability_check`) against hand-built fixtures — including that the crossover requires SpecPrefill to keep winning at every larger bin (not just once, to avoid reporting noise as a real crossover) and that curve comparison matches to the nearest reference keep-percentage. The GPU-only sweep orchestration (`run_sweep`, `run_rlm_format_sweep`, both building real vLLM/SpecPrefill engines) is written, reviewed against `predict_longbench_v2.py`'s and `target_stage/vllm_offline_engine.py`'s confirmed patterns, and confirmed to import cleanly without `vllm` installed — but not executable on this machine. **[done]**
9. `analysis/aggregate_metrics.py` — unit-tested (20/20) against synthetic JSONL fixtures shaped exactly like `run_arm.py`'s real `predictions.jsonl`/`timing.jsonl` output, including edge cases a real sweep will actually hit: samples with a `timing.jsonl` row but no `predictions.jsonl` row (over-budget skips, or `--dry-run`), Arm C's mixed plain/SpecPrefill gate split derived purely from `n_prompt_tokens_kept`'s presence (no separate flag needed), and empty/dry-run arms rendering without crashing. Required first extending `target_stage/vllm_offline_engine.py`'s `answer_batch`/`drive_engine_to_completion` to record a REAL per-request `generation_time_s` (previously only TTFT was tracked, not enough to compute a genuine per-sample `f`) — see the load-bearing findings below. `score_accuracy`'s LongBench-v2 scoring is an explicitly-flagged approximation (substring match on free text, not `grade_longbench_v2.py`'s exact-letter extraction), since this project's target prompt asks for free text, not a letter — noted as follow-up work, not silently glossed over. **[done]**
10. `tests/` alongside each module above, full `pytest` pass. **[done — 85/85]**
11. Finalize `REPRODUCE.md`, sequencing: `spec_prefill_llama`'s own `validate_proposer.py`/`validate_runner_integration.py` (blocking for Arms B/C only, not A) → Arm-A-only pass → compute `f` (go/no-go checkpoint) → `sweep_n_min.py` → `transferability_check.py` → full A/B/C sweep → `aggregate_metrics.py`. **[done — see REPRODUCE.md]**

All 11 build-order steps are done. What's left is real execution on a GPU node, per REPRODUCE.md.

### Correction made during implementation: `datasets/` renamed to `eval_data/`

The directory originally named `datasets/` (per this plan's own layout above,
now corrected) was renamed to `eval_data/` after discovering it would
silently shadow the installed `datasets` (Hugging Face) library once this
project's root directory hits `sys.path` — `prep_longbench_v2_long.py`
itself calls `from datasets import load_dataset`, and `import datasets`
resolves to whichever matching directory/package `sys.path` finds first.
Confirmed directly: with the project root on `sys.path`, `import datasets`
resolved to our own empty local folder (no `__init__.py`, treated as a
namespace package) instead of the real library, with no error — exactly the
kind of silent failure that would only surface on a GPU node once `datasets`
(a real dependency per `spec_prefill_llama/REPRODUCE.md` step 1) is actually
installed. All references (`README.md`, `REPRODUCE.md`, `.gitignore`, this
file) were updated accordingly.

### Switched target/speculator from Qwen3 to Llama (post-step-9)

After all 11 build-order steps were done against `benchmarks/spec_prefill_qwen/`
(Qwen3-8B / Qwen3-1.7B), the target/speculator pairing was switched to
`benchmarks/spec_prefill_llama/` (Llama-3.1-8B / Llama-3.2-1B) — decision 2
above. This was **not** a pure find-and-replace; three things actually
differ between the two `spec_prefill_*` pipelines, confirmed by reading
`spec_prefill_llama/predict_longbench_v2.py` directly rather than assumed
from naming symmetry:

1. **`chunk_size=32` for Llama, not 64.** `spec_prefill_llama/predict_longbench_v2.py`'s
   own module docstring confirms this was set explicitly for Llama (not
   inherited by accident from the Qwen3/Gemma4 ports, which use 64) —
   `configs/spec_config_always_on.yaml` and `calibration/transferability_check.py`'s
   `run_rlm_format_sweep` default were both updated to `32`.
2. **No `enable_thinking` chat-template kwarg.** The Qwen3 pipeline's
   `render_chat`/`chat_wrapper_pieces` pass `enable_thinking=False` to
   suppress Qwen3's default reasoning-mode preamble; Llama's chat template
   has no equivalent concept (per `spec_prefill_llama/predict_longbench_v2.py`'s
   own `render_chat` docstring: passing it "would be meaningless"). Removed
   from `target_stage/vllm_offline_engine.py::chat_wrapper_pieces`.
3. **Env var names**: `QWEN3_MODEL_PATH`/`QWEN3_1_7B_MODEL_PATH` →
   `LLAMA31_8B_MODEL_PATH`/`LLAMA32_1B_MODEL_PATH` (both checkpoints gated
   on Hugging Face, unlike Qwen3's), `SPEC_PREFILL_QWEN_DIR` →
   `SPEC_PREFILL_LLAMA_DIR`, `ensure_spec_prefill_qwen_on_path` →
   `ensure_spec_prefill_llama_on_path`. Updated everywhere: `.env_exports.sh`,
   `configs/arms.yaml`, `runner/run_arm.py`/`run_all_arms.py`,
   `calibration/sweep_n_min.py`/`transferability_check.py`.

What did **not** need to change: `worker_cls="vllm_patch.worker.SpecPrefillWorker"`,
the `vllm_patch.pruner`/`vllm_patch.config`/`vllm_patch.proposer` import
paths, `predict_longbench_v2.py`'s request-submission/TTFT/budget-sizing
patterns this project's `target_stage/vllm_offline_engine.py` adapted from
— all identical across the Qwen3/Llama ports (confirmed by inspection, not
assumed). One inaccurate claim caught and fixed while doing this: an
earlier docstring said `rlm/utils/token_utils.py`'s `MODEL_CONTEXT_LIMITS`
table has a per-model entry for the target model (true for `"qwen3"`) —
checked directly for Llama and found no such entry exists at all, so the
claim was corrected rather than blindly word-swapped (see
`eval_data/filter_by_token_length.py`'s module docstring).

Status: unchanged from step 9's completion — still code-complete, still
**never run against a GPU or vLLM** (this machine has neither). All 85
tests re-run and still pass after the switch (the switch touched
env-var/config plumbing and one chat-template kwarg, not the underlying
logic any test exercises). `spec_prefill_llama/vllm_patch/` carries the
exact same "code-complete but unvalidated on real hardware" status
`spec_prefill_qwen/vllm_patch/` had — switching pipelines does not skip
`REPRODUCE.md` step 2's real-hardware validation requirement.

## Verification

- Run `spec_prefill_llama/test_vllm_patch.py` as a pre-flight sanity check before building on top of `vllm_patch/` (confirmed 20/20 passing on the machine this was built on).
- Dataset filtering/prep scripts (step 2) run and produce correctly-filtered JSONL — pure Python/data, no GPU needed. Confirmed: `prep_longbench_v2_long.py`'s filtering logic against fake rows, `gen_synthetic_niah.py`'s generator (every needle verified present in its generated context), and `filter_by_token_length.py`'s real-tokenizer counting end-to-end against a live HF tokenizer (`gpt2`, standing in for `LLAMA31_8B_MODEL_PATH` on this machine, which has no Llama checkpoint — Llama's own tokenizer is gated on Hugging Face and wasn't downloaded here either).
- `target_stage/gate.py` / `route_queries.py` — pure token-count math, no `vllm` import — 9/9 + 4/4 unit tests pass against a hand-written fake tokenizer, including the strict `N > N_min` boundary condition and full-coverage/no-duplication routing.
- `target_stage/vllm_offline_engine.py`'s pure-Python pieces (percentile, resolve_max_num_batched_tokens, render_excerpts_text, build_prompt_pieces, chat_wrapper_pieces, the TargetQuery/TargetAnswer/EngineHandle dataclasses) — 11/11 unit tests pass, confirming e.g. the question is never included in the prunable excerpts text, and budget auto-sizing is genuinely outlier-resistant (p90-based, not max-based). Engine construction / request submission (GPU-only) not exercised here.
- `rlm_stage/timing_decomposition.py` — 8/8 unit tests pass: hand-built fixtures confirming the arithmetic identities in decision 6 (no sub-calls, sequential sub-calls, batched-call wall-time reconstruction, single- and multi-level child-RLM recursion without double-counting), plus a cross-check against the real trajectory logged during step 3's live Anthropic API smoke test.
- `rlm_stage/evidence_cache.py` — 7/7 unit tests pass: hashing/hit-miss/force-refresh logic against an injected stub `run_fn` (no live API, no mocked `RLM` internals), including that the cache key is sensitive to context/question/guardrails/prompt-version changes and stable across dict key reordering, and that `EvidenceResult` round-trips through JSON (including the nested `RLMChatCompletion`) without loss.
- `runner/run_arm.py` / `run_all_arms.py` — 7/7 unit tests pass (validation errors, `--dry-run` timing-JSONL output, `max_samples` truncation, Arm-A-first ordering/dedup via a monkeypatched `run_arm`). Separately, real end-to-end validation via the actual CLI: `run_arm.py --arm A --dataset <tiny 2-sample dataset> --dry-run` run twice — first pass 0/2 cache hits (real RLM calls), second pass 2/2 cache hits (no RLM calls, response byte-identical) — and `runner/smoke_test.py` run directly to a clean pass.
- End-to-end evidence-extraction smoke tests against the **real** Anthropic API on small synthetic NIAH contexts: `rlm_stage/evidence_rlm.py --smoke-test` (RLM alone) and `runner/smoke_test.py` (RLM + caching together) both ran successfully — needle located, well-formed evidence JSON returned, secret value recovered exactly.
- **Real finding from those live runs, not just a hypothetical edge case**: RLM's own `_default_answer` max-iterations fallback ([../rlm/rlm/core/rlm.py:673](../rlm/rlm/core/rlm.py)) can crash `AnthropicClient.completion()` with a bare `IndexError` on an empty-content API response — hit twice across a handful of live smoke-test attempts (see the load-bearing findings above for the root cause). `collect_evidence_for_dataset` now skips-and-reports rather than crashing; `runner/smoke_test.py` retries (a fresh RLM attempt each time, not a cache replay) up to 3 times rather than treating one unlucky non-convergent run as a pipeline failure. This is real, observed model/prompt-convergence variance — worth monitoring in aggregate on real sweeps, not something "fixed" by this workaround alone.
- `calibration/sweep_n_min.py` — 12/12 unit tests pass: candidate pooling from both sources (evidence cache + synthetic NIAH), bucketing/truncation to exact bin sizes (never picking a random slice), the crossover requiring SpecPrefill to win at every larger bin (not just once), and `configs/n_min.json` round-tripping through `target_stage/gate.py::load_n_min`.
- `calibration/transferability_check.py` — 11/11 unit tests pass: reference-curve loading from `{exp_id}_result.json` files (sorted by keep-percentage, tolerant of missing experiments), needle-recall scoring, and curve comparison (nearest-keep-percentage matching, divergence flagging above threshold).
- `analysis/aggregate_metrics.py` — 20/20 unit tests pass against synthetic JSONL fixtures shaped exactly like `run_arm.py`'s real output: joining `predictions.jsonl`+`timing.jsonl` by id (including when a sample has no prediction at all — `--dry-run` or an over-budget skip), per-sample `f` computation and its distribution, Arm C's gate-split derived purely from `n_prompt_tokens_kept`'s presence, depth/fan-out distributions, and both accuracy-scoring paths (NIAH recall, LongBench-v2 best-effort substring match) — plus `render_arm_summary` confirmed not to crash on dry-run or empty-arm inputs (the classes of aggregate that would otherwise hit `None`-formatting `TypeError`s).
- `configs/spec_config_always_on.yaml` round-trips through `spec_prefill_llama/vllm_patch/config.py`'s `SpecConfig.from_path` — not yet directly tested (planned for `tests/test_config_roundtrip.py`, not yet written); the YAML schema was hand-verified against that dataclass's field names.
- Full `pytest` pass across `benchmarks/rlm_specprefill/tests/`: **85/85 passing** across all 9 test files.
- Explicitly **not** verifiable without a GPU node (flagged in `REPRODUCE.md` rather than silently skipped): anything constructing vLLM's offline `LLM`, `SpecPrefillWorker`/proposer/model_runner behavior, real TTFT/`generation_time_s` numbers, the `N_min` sweep's actual crossover, the transferability check's actual RLM-format recall curve, real accuracy/evidence-recall numbers at scale, GPU memory/throughput, and SpecPrefill's own basic correctness on Llama (never run on real hardware).
