# Batch Size x Speculative Decoding Cross-Sweep — Experiment Plan

Status: **infrastructure implemented, sweep not yet run**. See "How to
run this suite" below.

## Goal

Measure the combined effect of `max_num_seqs` and MTP draft length `k`
together, rather than each in isolation. Full grid:

- `max_num_seqs` in {128, 256}
- MTP `k` in {1, 3, 5}
- spec decode **on** for every row (unlike the `batch` suite, which
  holds it off)

6 experiments total (2 x 3).

## Relationship to the other two suites

This is the cross-sweep both sibling plans explicitly deferred:

- `EXPERIMENT_PLAN.md` (suite `spec`, `S0xx`) sweeps `k` alone, with
  `max_num_seqs` held at `DEFAULT_MAX_NUM_SEQS` (128) for every row.
- `EXPERIMENT_PLAN_MAX_NUM_SEQS.md` (suite `batch`, `B0xx`) sweeps
  `max_num_seqs` alone, with spec decode held **off** for every row —
  its own "Design decisions" section says explicitly: "Crossing
  `max_num_seqs` with the spec-decode/k axis ... would confound two
  variables at once. If a later cross-sweep (batch size x k) is wanted,
  that's a separate follow-up plan, not this one."

This is that follow-up. A third suite (`cross`, `X0xx`) rather than
folding these rows into either existing dict, for the same reason the
first two are already split: `--all` should never silently run more
than one kind of sweep in a single invocation.

A fourth suite, `cross_hi` (`Y0xx`, see `EXPERIMENT_PLAN_MNS_SPEC_CROSS_HI.md`),
covers a higher-range companion to this one: `max_num_seqs` in
{128, 256, 512} x MTP `k` in {4, 6, 8}.

## Why this axis combination matters

`max_num_seqs` and MTP `k` interact in the scheduler, not just
additively: more concurrent sequences (`max_num_seqs`) means more
verification rounds' worth of target+draft forward passes competing for
the same compute/memory each step, and higher `k` means each of those
rounds is more expensive (larger draft batch, more candidate positions
verified per step). Whether the two sweeps' separately-measured effects
(from `S0xx` and `B0xx`) simply add, or whether one saturates the
other's headroom (e.g. a high-`k` config might hit compute/memory limits
at a lower `max_num_seqs` than a low-`k` config would), is exactly what
neither existing suite alone can show.

## Experiment matrix

| ID | `max_num_seqs` | MTP `k` | spec_decode |
|---|---:|---:|:---:|
| X001 | 128 | 1 | on |
| X002 | 128 | 3 | on |
| X003 | 128 | 5 | on |
| X004 | 256 | 1 | on |
| X005 | 256 | 3 | on |
| X006 | 256 | 5 | on |

Datasets: all three existing datasets (aime, gpqa_diamond,
livecodebench — same `--datasets` default as the other suites), no
dedicated larger sample built for this suite.

**Caveat inherited from `B005`**: `max_num_seqs=256` exceeds
`gpqa_diamond`'s 198 prompts, so `batch_size = min(len(prompts),
max_num_seqs)` clamps to 198 for that one dataset — the cap doesn't
actually bind there. X004-X006 vs. gpqa_diamond are still a meaningful
comparison against X001-X003 (256 configured vs. effectively uncapped),
just not a true 256-concurrency test for that dataset specifically. AIME
and LiveCodeBench (300 prompts each) aren't affected — 256 binds
normally there.

## How to run this suite

```bash
./run_experiments.sh --suite cross --all              # all of X001-X006
./run_experiments.sh --suite cross X003                # single config
python3 run_pipeline.py --suite cross --list           # print the matrix
```

`--suite` defaults to `spec`, so this doesn't change any existing
invocation's behavior. All three suites (`spec`, `batch`, `cross`) share
one driver (`run_pipeline.py`) and are mutually exclusive per
invocation — `--all`/`--exp` always resolve against whichever suite
`--suite` selected.

## Metrics captured per run

Same schema as the other two suites (`CSV_FIELDS` in `run_pipeline.py`):
requests/sec, elapsed time, output tokens/sec, MFU, MBU, plus
acceptance-rate/mean-accept-length (populated here, unlike the `batch`
suite, since spec decode is on) and `max_num_seqs`.

## Expected signal

At `k=1`, going from mns=128 to 256 should behave similarly to the
`batch` suite's own mns sweep (low draft overhead per round). At `k=5`,
the higher per-round cost (larger draft batch, more verified positions)
may mean the mns=128→256 throughput gain is smaller than at `k=1`, or
could even reverse if compute/memory saturates earlier at high `k` —
that inflection (if any) is the primary thing this sweep is for.

## Open questions before running the sweep

1. Reps per (experiment x dataset) — default 2, matching the other two
   suites; no reason assumed to change it here.
2. Whether `gpqa_diamond`'s non-binding mns=256 rows (see caveat above)
   are worth a dedicated larger sample the way `B006` got one for the
   `batch` suite, or whether the AIME/LiveCodeBench rows alone are
   sufficient signal for this cross-sweep's purpose.
