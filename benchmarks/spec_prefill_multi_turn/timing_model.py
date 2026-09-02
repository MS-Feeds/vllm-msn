"""Per-stage wall-clock attribution for one turn.

The companion to `flops_model.py`'s `FlopBreakdown`, and it exists because
the two were asymmetric in a way that hid the result that matters: FLOPs
have been stage-split since the FLOP model landed, while time was a single
number per turn. That is why the speculator's 17.5% FLOP share was known and
its LATENCY share was not -- and the latency share is the one that decides
whether the sparse path can ever be faster.

## The specific measurement bug this fixes

`seconds_per_turn_excl_turn0_mean` does not measure the same thing in the
two arms it is used to compare:

  - `run_baseline` starts its turn clock at `predict_scbench.py:1434` and
    stops at `:1524` -- query render, `add_request`, prefill, decode, and
    nothing else. It makes ZERO `collective_rpc` calls per turn.
  - `run_sparse_attention` starts at `:1996` and stops at `:2215`, which
    additionally covers the ENTIRE speculator pass (a whole-conversation
    prefill plus `1 + look_ahead_cnt` decodes on a second engine), a Python
    `sorted(set(...))` translation over the whole selection, hand-building
    an `EngineCoreRequest`, and FIVE `collective_rpc` round trips -- the
    first of which msgpack-serializes the entire selection.

So the sparse path's measured "+35% per turn" is a sum of at least four
independent effects, only one of which is the sparse decode override. Any
optimization graded on that number is being graded on noise it does not
control.

## Why `driver_overhead` is a residual and not a measurement

The stages that CAN be timed directly are timed directly. Everything else in
the turn -- RPC round trips, msgpack serialization, position translation,
request construction -- is charged to `driver_overhead`, computed as
`turn_seconds - sum(measured stages)`.

That makes the breakdown EXHAUSTIVE by construction: stages always sum to
the turn's wall clock, so `check_flop_verification.py`'s "stages sum to
total" check has a direct analogue here, and un-instrumented cost shows up
as a visible number instead of vanishing. A residual that grows when a stage
shrinks means the work moved rather than went away.

A NEGATIVE residual is meaningful and is deliberately not clamped: it means
two stages overlapped, i.e. something is being double-counted, which is a
bug in the instrumentation rather than a fast turn.

Deliberately vLLM-free and dependency-free so it is unit-testable in the
same CPU-only environment `test_vllm_patch.py` runs in, exactly as
`flops_model.py` and `scoring.py`'s pure helpers are.
"""

from dataclasses import dataclass

#: Measured stages, in the order a turn executes them. Mirrors
#: `FlopBreakdown.STAGES` exactly so the two breakdowns can be read
#: side by side, plus the residual.
MEASURED_STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring",
                   "target_prefill", "target_decode")
STAGES = MEASURED_STAGES + ("driver_overhead",)


@dataclass
class TimeBreakdown:
    """Seconds per stage for one turn, or summed over many.

    `spec_*` are zero for the M000 baseline, which has no speculator -- the
    same convention `FlopBreakdown` uses, so a baseline row's speculator
    columns read 0.0 rather than blank.
    """

    spec_prefill: float = 0.0
    spec_lookahead: float = 0.0
    spec_scoring: float = 0.0
    target_prefill: float = 0.0
    target_decode: float = 0.0
    driver_overhead: float = 0.0

    STAGES = STAGES
    MEASURED_STAGES = MEASURED_STAGES

    @property
    def total(self) -> float:
        return sum(getattr(self, s) for s in self.STAGES)

    @property
    def measured_total(self) -> float:
        """Everything except the residual -- what was actually instrumented."""
        return sum(getattr(self, s) for s in self.MEASURED_STAGES)

    @property
    def speculator_total(self) -> float:
        return self.spec_prefill + self.spec_lookahead + self.spec_scoring

    @property
    def speculator_fraction(self) -> float:
        """The latency analogue of `FlopBreakdown.speculator_fraction`.

        These two are NOT interchangeable and the difference is the point:
        the speculator can be 17.5% of a turn's FLOPs and a quite different
        share of its seconds, because the two engines run at different
        efficiencies and, today, strictly one after the other.
        """
        total = self.total
        return self.speculator_total / total if total else 0.0

    def __iadd__(self, other: "TimeBreakdown") -> "TimeBreakdown":
        for stage in self.STAGES:
            setattr(self, stage, getattr(self, stage) + getattr(other, stage))
        return self

    def __add__(self, other: "TimeBreakdown") -> "TimeBreakdown":
        out = TimeBreakdown(**{s: getattr(self, s) for s in self.STAGES})
        out += other
        return out


def breakdown_with_residual(turn_seconds: float, **measured) -> TimeBreakdown:
    """Build a breakdown whose stages sum EXACTLY to `turn_seconds`.

    `measured` carries whichever of `MEASURED_STAGES` the caller was able to
    time; anything omitted is zero. The remainder becomes
    `driver_overhead`.

    Raises on an unknown stage name rather than silently dropping it -- a
    typo'd keyword would otherwise land the whole stage in the residual and
    look like driver cost, which is precisely the kind of misattribution
    this module exists to stop.
    """
    unknown = set(measured) - set(MEASURED_STAGES)
    if unknown:
        raise ValueError(
            f"unknown timing stage(s) {sorted(unknown)}; valid stages are "
            f"{list(MEASURED_STAGES)}"
        )
    bd = TimeBreakdown(**measured)
    bd.driver_overhead = turn_seconds - bd.measured_total
    return bd
