# Batch Size x Speculative Decoding Cross-Sweep (High Range) — Experiment Plan

Status: **infrastructure implemented, sweep not yet run**. See "How to
run this suite" below.

## Goal

A higher-range companion to `EXPERIMENT_PLAN_MNS_SPEC_CROSS.md`'s
`cross` suite, covering a separate part of the `max_num_seqs` x MTP `k`
space:

- `max_num_seqs` in {128, 256, 512}
- MTP `k` in {4, 6, 8}
- spec decode **on** for every row

9 experiments total (3 x 3), IDs `Y001`-`Y009`.

## Relationship to the other suites

This is a **separate suite** (`cross_hi`, not more rows appended to the
existing `cross` suite), per explicit instruction — the two suites cover
different points in the same two-axis space rather than one combined
matrix, consistent with how `spec`/`batch`/`cross` are already kept
separate rather than merged. `--suite` selects exactly one; `--all`
never spans more than one suite.

| Suite | `max_num_seqs` | MTP `k` | spec_decode |
|---|---|---|:---:|
| `cross` (`X0xx`) | {128, 256} | {1, 3, 5} | on |
| `cross_hi` (`Y0xx`) | {128, 256, 512} | {4, 6, 8} | on |

## Why `mns=512` needs its own dataset (again)

Same reasoning as `BATCH_EXPERIMENTS`' `B006` and `CROSS_EXPERIMENTS`'
(implicit) reliance on the default 3 datasets: `batch_size =
min(len(prompts), max_num_seqs)` clamps to `len(prompts)` whenever the
configured cap exceeds the dataset size, silently turning a "mns=512"
experiment into an uncapped run. The default 3 datasets (198-300
prompts) are all too small for `mns=512` to bind. `Y007`-`Y009`
therefore override `datasets` to the same dedicated 554-prompt
LiveCodeBench sample `B006` uses (`datasets/livecodebench_mns512_samples.jsonl`)
— comfortably above 512, so the cap binds correctly there.

`Y001`-`Y006` (mns=128, 256) use the default 3 datasets, same as the
`cross` suite.

## Experiment matrix

| ID | `max_num_seqs` | MTP `k` | Datasets |
|---|---:|---:|---|
| Y001 | 128 | 4 | aime, gpqa_diamond, livecodebench |
| Y002 | 128 | 6 | " |
| Y003 | 128 | 8 | " |
| Y004 | 256 | 4 | " |
| Y005 | 256 | 6 | " |
| Y006 | 256 | 8 | " |
| Y007 | 512 | 4 | livecodebench_mns512 (dedicated) |
| Y008 | 512 | 6 | livecodebench_mns512 (dedicated) |
| Y009 | 512 | 8 | livecodebench_mns512 (dedicated) |

**Caveat inherited from `X004`-`X006`/`B005`**: `max_num_seqs=256`
exceeds `gpqa_diamond`'s 198 prompts, so the cap doesn't actually bind
for that one dataset at `Y004`-`Y006` — still a meaningful comparison
against the `mns=128` rows, just not a true 256-concurrency test for
`gpqa_diamond` specifically.

## Why this specific higher range

Higher MTP `k` (4, 6, 8) means a bigger draft batch and more candidate
positions verified per round — more per-round compute/memory pressure
than the `cross` suite's {1, 3, 5}. Crossing that against a higher
`max_num_seqs` range (up to 512, vs. `cross`'s max of 256) tests whether
the mns/k interaction (discussed in `EXPERIMENT_PLAN_MNS_SPEC_CROSS.md`
— does high concurrency saturate before or after high-k's extra
per-round cost matters) continues in the same direction at higher values
of both axes, or whether an inflection point appears somewhere between
the two suites' ranges.

## How to run this suite

```bash
./run_experiments.sh --suite cross_hi --all              # all of Y001-Y009
./run_experiments.sh --suite cross_hi Y007                # single config
python3 run_pipeline.py --suite cross_hi --list           # print the matrix
```

`--suite` defaults to `spec`, so this doesn't change any existing
invocation's behavior.

## Metrics captured per run

Same schema as the other suites (`CSV_FIELDS` in `run_pipeline.py`):
requests/sec, elapsed time, output tokens/sec, MFU, MBU, acceptance-rate/
mean-accept-length (populated here since spec decode is on), and
`max_num_seqs`.

## Expected signal

At the higher `k` values tested here, the draft model's own
compute/memory contribution per round is larger than anything the
`cross` suite exercises — if `mns=256→512` still shows a throughput gain
at `k=8` the way lower-`k`/lower-`mns` configs typically do, that
suggests headroom persists further into this higher range than the
`cross` suite alone could show. If the gain flattens or reverses before
reaching 512, that's the inflection point this suite exists to find.

## Open questions before running the sweep

1. Reps per (experiment x dataset) — default 2, matching the other
   suites; no reason assumed to change it here.
2. Whether a `Y010`+ dedicated-larger-sample experiment is ever wanted
   at `mns=1024`+ — would need a bigger sample than the current
   554-prompt one (see `EXPERIMENT_PLAN_MAX_NUM_SEQS.md`'s note on
   `prep_livecodebench.py`'s `FALLBACK_FILES` chain being fully
   exhausted at 554; a larger sample would need additional source
   files or a different dataset).
