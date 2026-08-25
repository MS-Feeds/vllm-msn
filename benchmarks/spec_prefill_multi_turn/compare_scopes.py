#!/usr/bin/env python3
"""Cross-tabs two runs of the same experiment rows against each other --
accuracy and FLOPs, split by keep rate x SCBench config x (turn 0 /
steady state).

Built for the question `--sparse-prefill` raises (does restricting a
turn's PREFILL, on top of its decode, pay for itself?), but it is not
specific to that: any two runs of the same `exp_id`s that differ in one
variable and were written with different `--output-suffix` values can be
fed to it.

## Why this exists rather than diffing two `*_result.json` files

Three things a side-by-side of published aggregates gets wrong, all of
which have already bitten this comparison at least once:

1. **The matched sets are not identical.** Two runs of the same rows can
   differ by a turn or two (a turn that generated nothing, a conversation
   that skipped). Comparing `overall` from each `result.json` then
   compares two different populations. This script re-scores BOTH
   prediction files through `grade_scbench.grade` and reports only the
   turns present and scorable in both -- a genuinely paired comparison.
2. **Per-turn FLOP means are dominated by turn 0.** On the SPARSE
   pipeline turn 0's cold prefill is ~99.7% of a run's target-prefill
   FLOPs, so `target_prefill_tflops_per_turn_mean` moves less between
   keep rates than it does between runs that happened to complete a
   different number of turns. (Observed: a run with 966 turns reported a
   HIGHER per-turn mean than one with 967, purely from the denominator.)
   Every FLOP figure here is steady-state, turn 0 excluded, over the
   common turn set.
3. **Turn 0 is a free control, and nobody was checking it.** Under the
   sparse pipeline's prefill scope, turn 0's prefill is dense either way,
   so its predictions MUST be byte-identical between the two runs. This
   script checks that and says so. A mismatch there is a bug -- the
   gather fired on a prefill it must not have -- not a result.

## Self-generated history makes late turns drift

Both arms feed the model's own output back as history, so once any turn's
generation diverges the two runs' contexts diverge too, and later turns
stop being a controlled A/B. The `drift` column reports that directly, as
the difference in `target_resident_len` (tokens resident entering the
turn) between the two runs. Small drift against a large resident is
harmless -- 21 tokens against 88k moved the measured prefill reduction by
~0.02% against an ~11.6% effect -- but it grows, so read it rather than
assume it.

Usage:
    python3 compare_scopes.py \\
        --exp SPARSE-k80-g32,SPARSE-k60-g32,SPARSE-k40-g32,SPARSE-k20-g32 \\
        --a-suffix=-full-dense --b-suffix=-full-pf \\
        --a-label decode-only --b-label sparse-prefill

    # Against a reference row that has no counterpart suffix (e.g. M000,
    # whose steady-state cost is the thing both scopes are judged against):
    python3 compare_scopes.py --exp SPARSE-k20-g32 \\
        --a-suffix=-full-dense --b-suffix=-full-pf --reference M000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grade_scbench import DEFAULT_SAMPLES, _read_jsonl, grade  # noqa: E402

DEFAULT_RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))
_TFLOP = 1e12

# `FlopBreakdown.STAGES`' three speculator stages, summed into one column --
# the target-side stages are what this comparison is about, and splitting
# the scorer three ways here would add width without adding a decision.
_SPEC_STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring")


def _predictions_path(results_dir: Path, exp_id: str, suffix: str) -> Path:
    return results_dir / f"{exp_id}{suffix}_predictions.jsonl"


def _scored_by_key(samples: list[dict], predictions: list[dict]) -> dict:
    """`grade`'s own per-turn rows, keyed by (conversation_id, turn_idx),
    keeping only turns that actually got a score.

    Re-scoring here rather than reading `*_result.json` is what makes the
    paired intersection possible at all -- `result.json` holds aggregates
    over one run's own matched set, with no way to recover which turns
    those were. It also means this script can never disagree with
    `grade_scbench.py` about a metric: there is only one implementation.
    """
    result = grade(samples, predictions)
    return {
        (row["conversation_id"], row["turn_idx"]): row
        for row in result["per_turn"]
        if row["score"] is not None
    }


def _flops_by_key(predictions: list[dict]) -> dict:
    return {(p["conversation_id"], p["turn_idx"]): p for p in predictions}


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _stage(row: dict, stage: str) -> float:
    flops = row.get("flops") or {}
    if stage == "spec":
        return sum(flops.get(s, 0) for s in _SPEC_STAGES) / _TFLOP
    return flops.get(stage, 0) / _TFLOP


class RunPair:
    """One exp_id's two runs, reduced to the turns scorable in both."""

    def __init__(self, samples, results_dir, exp_id, a_suffix, b_suffix):
        self.exp_id = exp_id
        self.paths = [
            _predictions_path(results_dir, exp_id, a_suffix),
            _predictions_path(results_dir, exp_id, b_suffix),
        ]
        preds = [_read_jsonl(p) for p in self.paths]
        self.scored = [_scored_by_key(samples, p) for p in preds]
        self.flops = [_flops_by_key(p) for p in preds]
        # The paired set: scorable in both runs AND carrying FLOP records
        # in both (a run predating the FLOP model would have neither).
        self.common = sorted(
            set(self.scored[0]) & set(self.scored[1])
            & set(self.flops[0]) & set(self.flops[1])
        )
        self.only_a = sorted(set(self.scored[0]) - set(self.scored[1]))
        self.only_b = sorted(set(self.scored[1]) - set(self.scored[0]))

    def keys(self, *, turn0: bool):
        return [k for k in self.common if (k[1] == 0) == turn0]

    def configs(self) -> list[str]:
        return sorted({self.scored[0][k]["config"] for k in self.common})

    def score(self, run: int, *, turn0: bool, config: str | None = None) -> float:
        return 100.0 * _mean(
            self.scored[run][k]["score"] for k in self.keys(turn0=turn0)
            if config is None or self.scored[run][k]["config"] == config
        )

    def flop(self, run: int, stage: str, *, turn0: bool) -> float:
        return _mean(
            _stage(self.flops[run][k], stage) for k in self.keys(turn0=turn0)
        )

    def turn0_identical(self) -> tuple[int, int]:
        keys = self.keys(turn0=True)
        same = sum(
            1 for k in keys
            if self.flops[0][k].get("pred") == self.flops[1][k].get("pred")
        )
        return same, len(keys)

    def max_drift(self) -> int:
        """Largest |resident-length difference| over the paired steady-state
        turns -- see module docstring's "Self-generated history" section.
        Zero when either run predates the `flop_inputs` bookkeeping."""
        drifts = []
        for k in self.keys(turn0=False):
            fa = self.flops[0][k].get("flop_inputs") or {}
            fb = self.flops[1][k].get("flop_inputs") or {}
            if "target_resident_len" in fa and "target_resident_len" in fb:
                drifts.append(abs(fb["target_resident_len"] - fa["target_resident_len"]))
        return max(drifts) if drifts else 0


def render(pairs, a_label, b_label, reference=None):
    out = []
    w = 78

    out.append("=" * w)
    out.append(f"  A = {a_label}     B = {b_label}")
    out.append("=" * w)

    # -- integrity ----------------------------------------------------------
    out.append("")
    out.append("## Integrity")
    out.append("")
    out.append(f"{'exp_id':<20} {'paired':>7} {'A-only':>7} {'B-only':>7} "
               f"{'turn0 identical':>17} {'max drift':>10}")
    for p in pairs:
        same, total = p.turn0_identical()
        flag = "" if same == total else "  <-- BUG"
        out.append(f"{p.exp_id:<20} {len(p.common):>7} {len(p.only_a):>7} "
                   f"{len(p.only_b):>7} {f'{same}/{total}':>17} {p.max_drift():>10}{flag}")
    out.append("")
    out.append("  turn 0 must be identical: its prefill is dense under both scopes,")
    out.append("  so any difference is the gather firing where it must not.")

    # -- accuracy -----------------------------------------------------------
    out.append("")
    out.append("## Accuracy, steady state (turns 1+, paired turns only)")
    out.append("")
    out.append(f"{'exp_id':<20} {'config':<18} {'n':>5} {'A':>8} {'B':>8} {'delta':>8}")
    for p in pairs:
        n_all = len(p.keys(turn0=False))
        for cfg in p.configs():
            n = sum(1 for k in p.keys(turn0=False) if p.scored[0][k]["config"] == cfg)
            if not n:
                continue
            sa, sb = p.score(0, turn0=False, config=cfg), p.score(1, turn0=False, config=cfg)
            out.append(f"{p.exp_id:<20} {cfg:<18} {n:>5} {sa:>8.2f} {sb:>8.2f} {sb - sa:>+8.2f}")
        sa, sb = p.score(0, turn0=False), p.score(1, turn0=False)
        out.append(f"{'':<20} {'ALL':<18} {n_all:>5} {sa:>8.2f} {sb:>8.2f} {sb - sa:>+8.2f}")

    out.append("")
    out.append("## Accuracy, turn 0 (control -- should not move)")
    out.append("")
    out.append(f"{'exp_id':<20} {'n':>5} {'A':>8} {'B':>8} {'delta':>8}")
    for p in pairs:
        sa, sb = p.score(0, turn0=True), p.score(1, turn0=True)
        out.append(f"{p.exp_id:<20} {len(p.keys(turn0=True)):>5} "
                   f"{sa:>8.2f} {sb:>8.2f} {sb - sa:>+8.2f}")

    # -- flops --------------------------------------------------------------
    out.append("")
    out.append("## FLOPs per turn, steady state (TFLOP, turn 0 excluded)")
    out.append("")
    out.append(f"{'exp_id':<20} {'run':<16} {'spec':>8} {'prefill':>9} "
               f"{'decode':>8} {'total':>9} {'vs A':>8}")
    for p in pairs:
        ta = p.flop(0, "total", turn0=False)
        for run, label in ((0, a_label), (1, b_label)):
            t = p.flop(run, "total", turn0=False)
            rel = "" if run == 0 else f"{100 * (t / ta - 1):>+7.2f}%"
            out.append(f"{p.exp_id if run == 0 else '':<20} {label:<16} "
                       f"{p.flop(run, 'spec', turn0=False):>8.3f} "
                       f"{p.flop(run, 'target_prefill', turn0=False):>9.3f} "
                       f"{p.flop(run, 'target_decode', turn0=False):>8.3f} "
                       f"{t:>9.3f} {rel:>8}")
        pa, pb = (p.flop(r, "target_prefill", turn0=False) for r in (0, 1))
        out.append(f"{'':<20} {'-> prefill':<16} {'':>8} "
                   f"{100 * (pb / pa - 1):>+8.1f}%")
    if reference is not None:
        out.append("")
        out.append(f"{reference.exp_id:<20} {'reference':<16} "
                   f"{reference.flop(0, 'spec', turn0=False):>8.3f} "
                   f"{reference.flop(0, 'target_prefill', turn0=False):>9.3f} "
                   f"{reference.flop(0, 'target_decode', turn0=False):>8.3f} "
                   f"{reference.flop(0, 'total', turn0=False):>9.3f}")

    # -- turn 0 cost --------------------------------------------------------
    out.append("")
    out.append("## Turn 0 cost, for scale (TFLOP/turn)")
    out.append("")
    out.append(f"{'exp_id':<20} {'turn0 total':>13} {'steady total':>14} "
               f"{'turn0 share of run':>20}")
    for p in pairs:
        t0 = p.flop(1, "total", turn0=True)
        st = p.flop(1, "total", turn0=False)
        n0, ns = len(p.keys(turn0=True)), len(p.keys(turn0=False))
        share = 100 * (t0 * n0) / (t0 * n0 + st * ns) if (n0 or ns) else float("nan")
        out.append(f"{p.exp_id:<20} {t0:>13.1f} {st:>14.3f} {share:>19.2f}%")
    out.append("")
    out.append("  A steady-state saving is a fraction of the right-hand column,")
    out.append("  not of the run. Read both before calling a scope a win.")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True,
                        help="Comma-separated exp_id(s), e.g. SPARSE-k80-g32,SPARSE-k20-g32.")
    parser.add_argument("--a-suffix", default="-full-dense",
                        help="--output-suffix of run A. A suffix starting with '-' "
                             "must use the '=' form: --a-suffix=-full-dense.")
    parser.add_argument("--b-suffix", default="-full-pf")
    parser.add_argument("--a-label", default="decode-only")
    parser.add_argument("--b-label", default="sparse-prefill")
    parser.add_argument("--reference", default=None,
                        help="Optional exp_id (e.g. M000) whose steady-state cost both "
                             "runs are judged against. Read from its --a-suffix file, "
                             "falling back to no suffix.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Also write the rendered report here.")
    args = parser.parse_args()

    samples = _read_jsonl(args.samples)

    pairs = []
    for exp_id in [x.strip() for x in args.exp.split(",") if x.strip()]:
        try:
            pair = RunPair(samples, args.results_dir, exp_id, args.a_suffix, args.b_suffix)
        except FileNotFoundError as exc:
            print(f"[compare_scopes] skipping {exp_id}: missing {exc.filename}")
            continue
        if not pair.common:
            print(f"[compare_scopes] skipping {exp_id}: no turns scorable in both runs")
            continue
        pairs.append(pair)

    if not pairs:
        parser.error("no exp_id had both runs present with overlapping scorable turns")

    reference = None
    if args.reference:
        for suffix in (args.a_suffix, ""):
            path = _predictions_path(args.results_dir, args.reference, suffix)
            if path.exists():
                # Same file twice: RunPair's pairing machinery collapses to
                # a single run, which is all a reference row needs.
                reference = RunPair(samples, args.results_dir, args.reference, suffix, suffix)
                break
        if reference is None:
            print(f"[compare_scopes] --reference {args.reference}: no predictions file found")

    report = render(pairs, args.a_label, args.b_label, reference)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\n[compare_scopes] wrote {args.output}")


if __name__ == "__main__":
    main()
