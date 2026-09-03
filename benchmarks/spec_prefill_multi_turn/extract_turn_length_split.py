#!/usr/bin/env python3
"""Splits each arm's steady-state turn into FIXED-per-turn and PER-TOKEN
cost, and reports the gap as a function of generation length.

Why this rather than reading `seconds_per_turn_excl_turn0_mean` directly:
that number mixes two things that scale differently, so comparing it across
runs with different `--max-tokens` compares different mixtures. The
speculator, the delta prefill and the driver RPCs are paid ONCE per turn
regardless of how many tokens come out; only decode scales with the answer.

    turn(T) = fixed + per_token * T

Separating them is what makes "would longer turns favour the sparse path?"
answerable: `fixed` is what longer turns amortize, `per_token` is what they
multiply. A method can only win at long generation if `per_token` is
negative -- amortizing a per-token penalty just converges to it.
"""
import csv
import sys

FIXED_STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring",
                "target_prefill", "driver_overhead")


def load(path, exp_ids):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = {}
    for exp in exp_ids:
        matching = [r for r in rows if r["exp_id"] == exp]
        if matching:
            out[exp] = matching[-1]
    return out


def split(row):
    def f(key):
        raw = (row.get(key) or "").strip()
        return float(raw) if raw else 0.0
    fixed = sum(f(f"{s}_seconds_per_turn_excl_turn0_mean") for s in FIXED_STAGES)
    decode = f("target_decode_seconds_per_turn_excl_turn0_mean")
    tokens = f("out_len_mean")
    return fixed, (decode / tokens if tokens else 0.0), tokens, fixed + decode


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/all_runs.csv"
    baseline = sys.argv[2] if len(sys.argv) > 2 else "M000"
    sparse = sys.argv[3] if len(sys.argv) > 3 else "SPARSE-k20-g32-masked"

    rows = load(path, [baseline, sparse])
    missing = [e for e in (baseline, sparse) if e not in rows]
    if missing:
        print(f"missing rows: {missing}")
        return 2

    print(f"{'row':<26} {'out_len':>8} {'fixed/turn':>12} {'per-token':>12} {'turn':>9}")
    print("-" * 72)
    stats = {}
    for exp in (baseline, sparse):
        fixed, per_tok, tokens, turn = split(rows[exp])
        stats[exp] = (fixed, per_tok, tokens, turn)
        print(f"{exp:<26} {tokens:>8.1f} {fixed:>11.3f}s {per_tok * 1000:>9.2f}ms "
              f"{turn:>8.3f}s")

    b_fixed, b_pt, _, _ = stats[baseline]
    s_fixed, s_pt, s_tokens, _ = stats[sparse]
    d_fixed = s_fixed - b_fixed
    d_pt = s_pt - b_pt

    print(f"\ndifference: fixed {d_fixed:+.3f}s per-turn, "
          f"per-token {d_pt * 1000:+.2f}ms")
    print(f"  gap(T) = {d_fixed:.3f} {d_pt * 1000:+.5f}ms * T")

    if d_pt > 0:
        print("\n  per-token cost is POSITIVE: longer turns shrink the gap in "
              "RELATIVE terms but never close it.")
        print(f"  asymptotic penalty as T -> inf: {d_pt / b_pt * 100:.1f}%")
    else:
        print(f"\n  per-token cost is NEGATIVE: break-even at "
              f"T = {-d_fixed / d_pt:,.0f} tokens/turn.")

    print(f"\n{'tokens/turn':>12} {'gap':>9} {'% of baseline turn':>20}")
    for T in (64, 128, 256, 512, 1024, 2048):
        gap = d_fixed + d_pt * T
        base = b_fixed + b_pt * T
        print(f"{T:>12} {gap:>8.3f}s {gap / base * 100:>19.1f}%")
    print(f"\n  (measured at T={s_tokens:.1f} this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
