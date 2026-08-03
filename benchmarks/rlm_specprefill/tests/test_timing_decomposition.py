"""Tests for rlm_stage/timing_decomposition.py.

Validates the arithmetic identities from that module's docstring against
hand-built trajectory fixtures (covering sequential sub-calls, batched
sub-calls, and nested child-RLM recursion -- paths the real trajectory
captured during the step-3 smoke test happened not to exercise, since that
run never called llm_query/rlm_query). See test_real_smoke_test_trajectory
below for the cross-check against that real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlm_stage.timing_decomposition import (  # noqa: E402
    decompose_iteration,
    decompose_trajectory,
    load_trajectory_from_jsonl,
)

THIS_DIR = Path(__file__).resolve().parent.parent
REAL_SMOKE_TEST_LOG = THIS_DIR / "logs" / "rlm_2026-07-31_16-29-35_5e177917.jsonl"


def _code_block(execution_time: float, rlm_calls: list[dict] | None = None) -> dict:
    return {
        "code": "...",
        "result": {
            "stdout": "",
            "stderr": "",
            "locals": {},
            "execution_time": execution_time,
            "rlm_calls": rlm_calls or [],
            "final_answer": None,
        },
    }


def _leaf_call(execution_time: float) -> dict:
    """A plain llm_query() call, or an rlm_query() that fell back to a leaf
    call at max_depth -- either way, no nested `metadata` trajectory."""
    return {
        "root_model": "leaf-model",
        "prompt": "...",
        "response": "...",
        "usage_summary": {"model_usage_summaries": {}},
        "execution_time": execution_time,
        "metadata": None,
    }


def _child_rlm_call(execution_time: float, child_trajectory: dict) -> dict:
    """An rlm_query() call that spawned a full child RLM -- carries the
    child's own trajectory as nested `metadata`, per RLM._subcall
    (../../rlm/rlm/core/rlm.py:824)."""
    return {
        "root_model": "child-model",
        "prompt": "...",
        "response": "...",
        "usage_summary": {"model_usage_summaries": {}},
        "execution_time": execution_time,
        "metadata": child_trajectory,
    }


def test_decompose_iteration_no_subcalls():
    """No code blocks with sub-calls: all REPL time is compute, all
    remaining iteration time is root."""
    iteration = {
        "iteration_time": 10.0,
        "code_blocks": [_code_block(execution_time=2.0)],
    }
    timing = decompose_iteration(iteration, index=0)

    assert timing.t_rlm_root == 8.0  # 10.0 - 2.0
    assert timing.t_repl_compute == 2.0  # 2.0 - 0.0
    assert timing.t_rlm_subcalls == 0.0
    assert timing.n_subcalls == 0


def test_decompose_iteration_with_sequential_subcalls():
    """One code block containing two sequential llm_query() calls: their
    execution_time is subtracted out of the block's own execution_time to
    get T_REPL_compute, and summed to get T_RLM_subcalls."""
    iteration = {
        "iteration_time": 10.0,
        "code_blocks": [
            _code_block(
                execution_time=6.0,  # 6.0s total for this block...
                rlm_calls=[_leaf_call(2.0), _leaf_call(3.0)],  # ...5.0s of which was 2 LM calls
            )
        ],
    }
    timing = decompose_iteration(iteration, index=0)

    assert timing.t_rlm_root == 4.0  # 10.0 - 6.0
    assert timing.t_repl_compute == 1.0  # 6.0 - (2.0 + 3.0)
    assert timing.t_rlm_subcalls == 5.0  # 2.0 + 3.0
    assert timing.n_subcalls == 2


def test_decompose_iteration_batched_subcalls_reconstruct_true_wall_time():
    """Batched calls' individual execution_time is an even split of the
    batch's true wall time (total_time / n, ../../rlm/rlm/core/lm_handler.py:130).
    Summing them back must recover the batch's true total -- this is the
    'critical path' guarantee the latency model depends on (module
    docstring): whether 4 calls ran sequentially (4x2s=8s) or concurrently
    (batch wall time 8s, each call's logged execution_time = 8/4 = 2s),
    the reconstructed T_RLM_subcalls is the same 8.0, matching whichever
    actually happened.
    """
    batch_wall_time = 8.0
    n_calls = 4
    per_call_time = batch_wall_time / n_calls  # 2.0, as LMHandler fabricates it

    iteration = {
        "iteration_time": 20.0,
        "code_blocks": [
            _code_block(
                execution_time=9.0,  # 8.0s of batch + 1.0s of surrounding Python
                rlm_calls=[_leaf_call(per_call_time) for _ in range(n_calls)],
            )
        ],
    }
    timing = decompose_iteration(iteration, index=0)

    assert timing.t_rlm_subcalls == batch_wall_time  # reconstructed exactly
    assert timing.t_repl_compute == 1.0  # 9.0 - 8.0
    assert timing.t_rlm_root == 11.0  # 20.0 - 9.0
    assert timing.n_subcalls == n_calls


def test_decompose_trajectory_sums_across_iterations():
    trajectory = {
        "iterations": [
            {"iteration_time": 5.0, "code_blocks": [_code_block(1.0)]},
            {"iteration_time": 7.0, "code_blocks": [_code_block(2.0, [_leaf_call(1.5)])]},
        ]
    }
    timing = decompose_trajectory(trajectory)

    assert timing.depth == 0
    assert timing.n_iterations == 2
    assert timing.t_rlm_root == 4.0 + 5.0  # (5-1) + (7-2)
    assert timing.t_repl_compute == 1.0 + 0.5  # 1.0 + (2.0 - 1.5)
    assert timing.t_rlm_subcalls == 0.0 + 1.5
    assert timing.n_direct_subcalls == 1
    assert timing.max_realized_depth == 0
    assert timing.total_calls == 2 + 1  # 2 root turns + 1 leaf subcall


def test_decompose_trajectory_no_code_blocks_attributes_full_time_to_root():
    """An iteration where the model didn't emit any ```repl``` block (e.g.
    answered directly, or errored before executing code): all of
    iteration_time is root time."""
    trajectory = {"iterations": [{"iteration_time": 3.0, "code_blocks": []}]}
    timing = decompose_trajectory(trajectory)

    assert timing.t_rlm_root == 3.0
    assert timing.t_repl_compute == 0.0
    assert timing.t_rlm_subcalls == 0.0


def test_decompose_trajectory_recurses_into_child_rlm_for_depth_and_calls():
    """A child RLM's own root/repl time must NOT be added into the parent's
    t_rlm_root/t_repl_compute (module docstring's core warning) -- the
    child's entire cost is already opaquely captured in the parent's
    t_rlm_subcalls via the rlm_call's own execution_time. Recursion is only
    used for max_realized_depth/total_calls.
    """
    child_trajectory = {
        "iterations": [
            {"iteration_time": 4.0, "code_blocks": [_code_block(1.0)]},  # child's own t_root=3, t_repl=1
        ]
    }
    parent_trajectory = {
        "iterations": [
            {
                "iteration_time": 10.0,
                "code_blocks": [
                    _code_block(
                        execution_time=6.0,  # the child call's execution_time (4.0) is opaque at this level
                        rlm_calls=[_child_rlm_call(4.0, child_trajectory)],
                    )
                ],
            }
        ]
    }

    timing = decompose_trajectory(parent_trajectory)

    # Parent-level decomposition uses ONLY the opaque 4.0s for the child call
    # -- it must not equal the parent's t_root (10-6=4) plus the child's own
    # internal t_root (3.0) summed together anywhere in these three fields.
    assert timing.t_rlm_root == 4.0  # 10.0 - 6.0
    assert timing.t_repl_compute == 2.0  # 6.0 - 4.0
    assert timing.t_rlm_subcalls == 4.0  # the child call's own opaque execution_time
    assert timing.n_direct_subcalls == 1

    # Recursion IS used for these two:
    assert timing.max_realized_depth == 1  # child reached depth 1
    assert timing.total_calls == 1 + 1  # parent's 1 root turn + child's 1 root turn (leaf-free child)

    # The child's own local decomposition is available for inspection, but
    # kept separate from the parent's totals above.
    assert len(timing.children) == 1
    child_timing = timing.children[0]
    assert child_timing.depth == 1
    assert child_timing.t_rlm_root == 3.0
    assert child_timing.t_repl_compute == 1.0


def test_decompose_trajectory_multi_level_nesting_tracks_max_depth():
    """depth=0 -> depth=1 -> depth=2, confirming max_realized_depth and
    total_calls propagate correctly through more than one level of
    recursion (RLM's own max_depth cap allows up to a few levels)."""
    grandchild_trajectory = {"iterations": [{"iteration_time": 1.0, "code_blocks": []}]}
    child_trajectory = {
        "iterations": [
            {
                "iteration_time": 5.0,
                "code_blocks": [
                    _code_block(3.0, rlm_calls=[_child_rlm_call(1.0, grandchild_trajectory)])
                ],
            }
        ]
    }
    parent_trajectory = {
        "iterations": [
            {
                "iteration_time": 8.0,
                "code_blocks": [_code_block(6.0, rlm_calls=[_child_rlm_call(6.0, child_trajectory)])],
            }
        ]
    }

    timing = decompose_trajectory(parent_trajectory)

    assert timing.max_realized_depth == 2
    # total_calls: parent(1 root) + child(1 root) + grandchild(1 root) = 3
    assert timing.total_calls == 3
    assert timing.children[0].children[0].depth == 2


def test_real_smoke_test_trajectory_arithmetic_identity_holds():
    """Cross-checks the module against the REAL trajectory logged during
    the step-3 evidence-extraction smoke test (a live Anthropic API call) --
    confirms decompose_trajectory doesn't just satisfy hand-built fixtures
    but also round-trips correctly against genuine RLMLogger output.
    Skips gracefully if that log file isn't present (e.g. a fresh checkout
    before the smoke test has been run).
    """
    if not REAL_SMOKE_TEST_LOG.exists():
        import pytest

        pytest.skip(f"{REAL_SMOKE_TEST_LOG} not present -- run rlm_stage/evidence_rlm.py --smoke-test first")

    trajectory = load_trajectory_from_jsonl(REAL_SMOKE_TEST_LOG)
    assert trajectory["run_metadata"]["backend"] == "anthropic"
    assert len(trajectory["iterations"]) > 0

    timing = decompose_trajectory(trajectory)

    # The core arithmetic identity must hold exactly for every iteration:
    # whatever the iteration's wall time was, it decomposes fully into
    # root + repl_compute + subcalls with nothing left over and nothing
    # double-counted.
    for it in timing.per_iteration:
        reconstructed = it.t_rlm_root + it.t_repl_compute + it.t_rlm_subcalls
        assert abs(reconstructed - it.iteration_time) < 1e-6

    # This particular smoke-test run made no llm_query/rlm_query calls (the
    # model solved it with pure Python string search), so total_calls should
    # equal exactly the root-turn count with no sub-calls layered in.
    assert timing.n_direct_subcalls == 0
    assert timing.total_calls == timing.n_iterations
    assert timing.max_realized_depth == 0

    # Sanity bound: root LM call latency should dominate a Claude Haiku
    # call over a REPL doing trivial string search, not the other way
    # around -- if this ever flips, something about the decomposition
    # (or the underlying data) changed in a way worth investigating.
    assert timing.t_rlm_root > timing.t_repl_compute
