"""Decomposes an RLM trajectory into the ../../rlm_specprefill_ablation_plan.md
LATENCY MODEL's `T_RLM_root` / `T_REPL_compute` / `T_RLM_subcalls` terms.

Per IMPLEMENTATION_PLAN.md decision 6, this needs NO new instrumentation
hooks -- every number here is derivable from data `RLMLogger` already
captures (`../../rlm/rlm/logger/rlm_logger.py`,
`../../rlm/rlm/core/types.py`). The arithmetic identities, confirmed against
`RLM._completion_turn` (`../../rlm/rlm/core/rlm.py:646`, where the root LM
call happens before REPL execution within one iteration):

  T_RLM_root_i    = iteration_time_i - sum(code_block.execution_time for code_block in iteration_i)
  T_subcalls_b    = sum(rlm_call.execution_time for rlm_call in code_block.rlm_calls)
  T_REPL_compute_b = code_block.execution_time - T_subcalls_b

Summed across an iteration's code blocks, then across a trajectory's
iterations, this gives per-query T_RLM_root / T_REPL_compute /
T_RLM_subcalls. Batched sub-calls' individual `execution_time` values are an
even split of the batch's true wall time (`total_time / n`,
`../../rlm/rlm/core/lm_handler.py:130`) -- summing them back together
recovers the batch's true total wall time exactly, whether the underlying
calls ran sequentially or concurrently, which is why T_subcalls_b is a
correct "critical path" figure and not an overcount.

IMPORTANT: `T_RLM_root`/`T_REPL_compute`/`T_RLM_subcalls` computed from the
TOP-level (root, depth=0) trajectory ALONE are already the complete,
correctly-scoped 3 non-target terms of `T_total = T_RLM_root +
T_RLM_subcalls + T_REPL + T_target`. A `rlm_query()` call that spawns a
child RLM has its ENTIRE recursive cost (the child's own root turns, REPL
compute, and further sub-calls) already folded into that single
`rlm_call.execution_time` at the PARENT's level -- do not also recurse into
the child's own trajectory and add its `t_rlm_root`/`t_repl_compute` into
the parent's totals, or that cost gets double-counted. Recursion into a
child's nested `.metadata` trajectory (`decompose_trajectory`'s own
recursion below) exists ONLY to report `max_realized_depth` / `total_calls`
across the whole call tree -- a reporting concern, separate from the
T_total decomposition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IterationTiming:
    iteration_index: int
    iteration_time: float
    t_rlm_root: float
    t_repl_compute: float
    t_rlm_subcalls: float
    n_subcalls: int  # fan-out realized in this iteration (len of all code blocks' rlm_calls)


@dataclass
class TrajectoryTiming:
    """One RLM instance's own decomposition (this level only -- see module
    docstring for why `t_rlm_root`/`t_repl_compute`/`t_rlm_subcalls` must
    NOT be summed with `children`'s own values when computing T_total).
    `children` and `max_realized_depth`/`total_calls` ARE recursive, for
    depth/fan-out/call-count reporting only.
    """

    depth: int
    n_iterations: int
    t_rlm_root: float
    t_repl_compute: float
    t_rlm_subcalls: float
    n_direct_subcalls: int
    max_realized_depth: int
    total_calls: int
    per_iteration: list[IterationTiming] = field(default_factory=list)
    children: list["TrajectoryTiming"] = field(default_factory=list)


def decompose_iteration(iteration: dict[str, Any], index: int) -> IterationTiming:
    iteration_time = iteration.get("iteration_time") or 0.0

    t_repl_compute = 0.0
    t_subcalls = 0.0
    n_subcalls = 0
    sum_block_execution_time = 0.0

    for code_block in iteration.get("code_blocks", []):
        result = code_block.get("result", {})
        block_execution_time = result.get("execution_time") or 0.0
        rlm_calls = result.get("rlm_calls", [])
        block_subcalls_time = sum(c.get("execution_time") or 0.0 for c in rlm_calls)

        sum_block_execution_time += block_execution_time
        t_repl_compute += block_execution_time - block_subcalls_time
        t_subcalls += block_subcalls_time
        n_subcalls += len(rlm_calls)

    t_rlm_root = iteration_time - sum_block_execution_time

    return IterationTiming(
        iteration_index=index,
        iteration_time=iteration_time,
        t_rlm_root=t_rlm_root,
        t_repl_compute=t_repl_compute,
        t_rlm_subcalls=t_subcalls,
        n_subcalls=n_subcalls,
    )


def decompose_trajectory(trajectory: dict[str, Any], depth: int = 0) -> TrajectoryTiming:
    """`trajectory` is the `{"run_metadata": ..., "iterations": [...]}` shape
    `RLMLogger.get_trajectory()` returns (same shape as
    `RLMChatCompletion.metadata`, and as reconstructed by
    `load_trajectory_from_jsonl` below)."""
    iterations = trajectory.get("iterations", [])
    per_iteration = [decompose_iteration(it, i) for i, it in enumerate(iterations)]

    t_rlm_root = sum(it.t_rlm_root for it in per_iteration)
    t_repl_compute = sum(it.t_repl_compute for it in per_iteration)
    t_rlm_subcalls = sum(it.t_rlm_subcalls for it in per_iteration)
    n_direct_subcalls = sum(it.n_subcalls for it in per_iteration)

    max_realized_depth = depth
    # Real LM-call count at this level: one root/orchestrator call per
    # iteration, regardless of how many sub-calls that turn also made.
    total_calls = len(iterations)
    children: list[TrajectoryTiming] = []

    for iteration in iterations:
        for code_block in iteration.get("code_blocks", []):
            for rlm_call in code_block.get("result", {}).get("rlm_calls", []):
                child_metadata = rlm_call.get("metadata")
                if child_metadata:
                    # This rlm_call spawned a full child RLM -- recurse for
                    # depth/call-count reporting (see module docstring: NOT
                    # for re-summing t_rlm_root/t_repl_compute).
                    child_timing = decompose_trajectory(child_metadata, depth=depth + 1)
                    children.append(child_timing)
                    max_realized_depth = max(max_realized_depth, child_timing.max_realized_depth)
                    total_calls += child_timing.total_calls
                else:
                    # Leaf call (llm_query, or rlm_query that fell back to a
                    # leaf at max_depth): exactly one real LM call.
                    total_calls += 1

    return TrajectoryTiming(
        depth=depth,
        n_iterations=len(iterations),
        t_rlm_root=t_rlm_root,
        t_repl_compute=t_repl_compute,
        t_rlm_subcalls=t_rlm_subcalls,
        n_direct_subcalls=n_direct_subcalls,
        max_realized_depth=max_realized_depth,
        total_calls=total_calls,
        per_iteration=per_iteration,
        children=children,
    )


def load_trajectory_from_jsonl(path: Path) -> dict[str, Any]:
    """Reconstructs the `{"run_metadata": ..., "iterations": [...]}` shape
    from an `RLMLogger(log_dir=...)` JSONL file (a flat sequence of
    `{"type": "metadata", ...}` / `{"type": "iteration", ...}` lines) --
    the on-disk form of the same data `RLMLogger.get_trajectory()` returns
    in-memory, so `decompose_trajectory` works identically on either
    source.
    """
    run_metadata: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_type = entry.pop("type", None)
            entry.pop("timestamp", None)  # not part of the in-memory trajectory shape
            if entry_type == "metadata":
                run_metadata = entry
            elif entry_type == "iteration":
                entry.pop("iteration", None)  # the JSONL-only 1-based sequence counter
                iterations.append(entry)

    return {"run_metadata": run_metadata, "iterations": iterations}
