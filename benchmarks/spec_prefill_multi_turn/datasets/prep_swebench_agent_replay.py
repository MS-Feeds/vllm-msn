#!/usr/bin/env python3
"""Builds multi-turn conversations by REPLAYING recorded SWE-bench agent
trajectories, in the exact sample schema `predict_scbench.py` already consumes
-- ONE ACTION-CYCLE PER TURN.

## Why this dataset exists

Every dataset this pipeline has today fails the setting the method is actually
for. SCBench's steady-state delta is `d ~ 70` tokens against `o = 512` (d:o
~ 1:7), so the prefill saving lands in decode where it does not convert to
seconds. `prep_longbench_v2_multiturn.py` fixes `d` but its turns are
INDEPENDENT documents, and `prep_mmmu_multiturn.py` says outright that it is a
mechanism regression test rather than a selection-quality benchmark -- both for
the same reason: when every turn's answer lives entirely in that turn's own
force-kept query, accuracy is flat across the whole keep-rate grid no matter how
much is discarded, and the row proves nothing.

An agent trajectory has the property none of them have: turn 30 depends on a
file read at turn 5. That is `EXPERIMENT_PLAN.md`'s motivating question #2 --
does compression generalize, or overfit to the latest question -- and it is the
gap `RELATED_WORK.md` names in the literature, since H2O/StreamingLLM/Quest/
KVzip/HeadKV are all evaluated on one long prompt.

The per-turn economics are also the right shape. Against
`SPECULATION_ECONOMICS.md`'s win condition `(d + o)(1 - r - k) > 12r`
(practically `d > 0.45*o + 5.5`), an agent step's delta is the previous tool
output plus the model's action -- a few hundred to a few thousand tokens --
against an output that is one bash command, not a 512-token essay. And because
agent turns are SMALL, `T` can be large, which is the direction that helps:
sparse prefill can only remove attention to PRIOR turns, so the removable share
is `(T-1)/T` -- 80% at the paper's 5 turns, 67% at the 3-turn packed LongBench
file, ~94% at T=16 here.

What that does NOT buy is a bigger speedup. The paper's saving comes from a
~25k-token turn attending ~100k of history, and its own scaling fit is explicit
that only the history-scaling term shrinks with keep-rate. Agent turns are one
to two orders of magnitude smaller, so the absolute per-turn saving is much
smaller even though the removable SHARE is higher. What this dataset tests is
whether selection quality survives a long dependent conversation.

## What this measures, and what it does not

Replay is TEACHER-FORCED: every turn's input is the observation the DENSE run
actually received, so the sparse arm is snapped back onto the dense trajectory
at each step. That is deliberate -- it is what makes the arms paired, so the
`(config, turn_idx)` breakdown means something and a 30-row keep-rate sweep is
affordable. It is the same trade `EXPERIMENT_PLAN.md` decision #1 already makes
for golden-context mode.

The cost: this CANNOT see divergence compounding -- whether sparsity sends the
agent off the rails over many steps. Nothing here is evidence that the method
"works with agents"; that claim needs the live loop, which is a separate
experiment with its own endpoints. **A score from this file is action agreement
under teacher forcing, not a SWE-bench resolve rate, and must never be reported
as one.**

## Mapping onto the existing schema

`datasets/normalize_swebench_trajs.py` has already hoisted the system message
out and guaranteed strict user/assistant alternation, so the mapping is
positional:

    system                              ->  context
    messages[2i]   user  (observation)  ->  turns[i]["input"]
    messages[2i+1] asst  (action)       ->  turns[i]["answer"]

Turn 0's "observation" is the issue statement -- the task the agent was handed.
`context` is the scaffold's system prompt, non-empty on purpose for the same
reason the other two packers use a preamble: `ConversationState`'s turn-0
candidate pool is built from it (see conversation_state.py's ledger
construction), and an empty one leaves nothing to score at turn 0.

## Token accounting is borrowed, not reimplemented

`render_turn_query` and the chat wrapper helpers are imported from
`predict_scbench.py`, and `Budget` by explicit path from
`prep_longbench_v2_multiturn.py`. This is load-bearing rather than tidiness:
`render_turn_query` is the only place the `"\\n\\nQuestion {n}: ...\\nAnswer
{n}:"` framing exists, so this packer's token counts ARE the driver's token
counts by construction, and `Budget` encodes the driver's real pre-flight
including the `(T-1)*O` resident-output term. A copied format string or a second
budget would drift silently, and the failure mode -- the driver's pre-flight
firing mid-conversation -- `break`s out of the turn loop from that turn onward,
changing `num_turns` and de-comparabilising the row with nothing but
`num_skipped_too_large` to show for it.

**The `Question N:`/`Answer N:` framing is kept even though it reads oddly here**
(`"Question 3: <returncode>0</returncode><output>..."`). Forking the renderer to
say `Observation:`/`Action:` would be more natural and is not worth it: both
arms get identical framing, so the comparison is internally valid, and the
moment this file's rendering diverges from the driver's its budget arithmetic
stops being the driver's. The same argument `predict_scbench.py`'s own docstring
makes for SCBench applies -- these numbers are comparable to EACH OTHER, not to
anything published.

## Filter, never truncate

Four independent drops, all reported separately so a thin yield is visible:

  - **an action with no single bash block** -- the scaffold rejects such a
    generation and re-prompts, so the exchange is real, but it cannot serve as
    a grading target. Dropping the trajectory whole keeps the conversation
    contiguous; skipping the turn would renumber every later slot.

  - **fewer than `--turns-per-conv` steps** -- a trajectory is used as a PREFIX
    of length exactly `T`, because `Budget` assumes a uniform turn count and
    the headline columns are means over turns. A prefix of a real agent run is
    still a coherent conversation, unlike a truncated document. The cost is that
    late-trajectory behaviour, where context is longest, is under-sampled;
    complement it with a second file at larger `T`, the way the 3x40k LongBench
    file complements the 5-turn one.
  - **an observation over `--max-observation-tokens`** -- the main measurement
    control. Every headline column (`seconds_per_turn_excl_turn0_mean` and its
    per-stage twins) is a MEAN OVER TURNS, so a conversation mixing a 200-token
    `ls` with a 30k-token `cat` has variance that swamps the effect being
    measured. Bounding the spread matters more here than in either other
    packer, because agent observations are far more skewed than documents.
  - **an action over `--max-action-tokens`** -- specific to this dataset and
    easy to miss. `Budget` reserves `(T-1)*O` for resident generated output,
    and `M-k*-g*` feeds the GOLDEN answer forward via `render_golden_answer`.
    In the LongBench/MMMU files an answer is a single letter, so this could
    never bind; here it is a full THOUGHT-plus-command block. An action longer
    than the generation cap also makes the metric unfair -- the arm cannot emit
    a match it has no room to generate.

Usage:
    python3 datasets/normalize_swebench_trajs.py --traj-dir ./sweb_out

    python3 datasets/prep_swebench_agent_replay.py \\
        --tokenizer $TARGET_MODEL_PATH \\
        --turns-per-conv 16 --max-tokens 512 \\
        --max-observation-tokens 4000 \\
        --target-max-num-batched-tokens 130560 \\
        --speculator-max-num-batched-tokens 131063 \\
        --seed 42 \\
        --output datasets/swebench_agent_replay.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent

sys.path.insert(0, str(_PKG_ROOT))

from predict_scbench import (  # noqa: E402
    chat_turn_boundary_pieces,
    chat_wrapper_pieces,
    render_turn_query,
)
#: The grader's own action parser, imported rather than restated so "what
#: counts as an action" has exactly one definition. It is what lets this script
#: guarantee that every reference it emits is scorable, which in turn is what
#: lets `agent_action_match` treat a 0.0 as a statement about the PREDICTION
#: rather than an ambiguity about the ground truth. `grade_scbench.py` imports
#: nothing outside the standard library, so this pulls in no vLLM.
from grade_scbench import extract_agent_action  # noqa: E402
from vllm_patch.model_structure import load_tokenizer  # noqa: E402

DEFAULT_TRAJECTORIES = _HERE / "swebench_trajectories.jsonl"
DEFAULT_OUTPUT = _HERE / "swebench_agent_replay.jsonl"
DEFAULT_CONFIG_NAME = "swebench_agent"
DEFAULT_ID_PREFIX = "swea"


def _load_lbv2_module():
    """The LongBench v2 builder, by explicit file path -- for `Budget` and
    `_histogram`.

    Imported rather than reimplemented for the reason
    `prep_mmmu_multiturn.py::_load_budget_class` gives: `Budget` encodes the
    driver's real pre-flight arithmetic including the `(T-1)*O` resident-output
    term, and a second copy would drift. Addressed by path because several
    sibling benchmark directories share the base name `prep_longbench_v2*`
    and an import-by-name would resolve to whichever is on `sys.path` first.
    """
    path = _HERE / "prep_longbench_v2_multiturn.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected the LongBench v2 multi-turn builder at {path}, which "
            "supplies the Budget class this script reuses. Without it the "
            "driver's pre-flight arithmetic would have to be restated here "
            "and could drift from it."
        )
    spec = importlib.util.spec_from_file_location("_lbv2mt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_trajectories(path: Path) -> list[dict]:
    """Reads `normalize_swebench_trajs.py`'s output and re-checks the two
    invariants this script indexes on, rather than trusting the file.

    The file is hand-editable and may have been produced by an older version
    of the normalizer; a silently mis-paired row here would score every later
    turn against the wrong action, which no exception downstream would catch.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No trajectory file at {path}. Run "
            "datasets/normalize_swebench_trajs.py --traj-dir <mini-swe-agent "
            "output dir> first."
        )
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            system = (row.get("system") or "").strip()
            if not system:
                raise ValueError(
                    f"{path}:{lineno}: empty `system`. It becomes the "
                    "conversation preamble, which must be non-empty so "
                    "ConversationState's turn-0 candidate pool is non-empty."
                )
            roles = [m.get("role") for m in messages]
            if roles != ["user", "assistant"] * (len(roles) // 2) or not roles:
                raise ValueError(
                    f"{path}:{lineno}: `messages` does not alternate "
                    f"user/assistant (saw {roles[:6]!r}...). The positional "
                    "mapping this script uses would misalign every turn."
                )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Turn construction
# ---------------------------------------------------------------------------


def build_turns(traj: dict, turns_per_conv: int, tok) -> list[dict]:
    """The first `turns_per_conv` action-cycles of one trajectory, each
    measured with the driver's own renderer AT ITS TRUE SLOT INDEX.

    Measuring at the true slot from the start (rather than at slot 0 and
    re-measuring later, as the LongBench builder does) costs nothing here
    because turns are never reordered: a trajectory's order is semantic, so
    there is no grouping or shuffling step that could move a turn to a
    different slot.
    """
    messages = traj["messages"]
    turns: list[dict] = []
    for slot in range(turns_per_conv):
        observation = messages[2 * slot]["content"]
        action = messages[2 * slot + 1]["content"]
        turns.append(
            {
                "input": observation,
                "answer": action,
                "query_tokens": len(render_turn_query(tok, slot, {"input": observation})),
                # `render_golden_answer` encodes `f" {answer}"`; measured the
                # same way so the comparison against --max-action-tokens is
                # against what the driver would actually feed forward.
                "answer_tokens": len(tok.encode(f" {action}", add_special_tokens=False)),
            }
        )
    return turns


def build_conversations(
    trajectories: list[dict],
    budget_for,
    tok,
    turns_per_conv: int,
    max_observation_tokens: int,
    max_action_tokens: int,
    verbose_every: int = 50,
) -> tuple[list[dict], dict]:
    """Returns `(conversations, drop_counts)`.

    Each conversation carries its OWN `Budget`, built from its own system
    prompt's length. The scaffold's system prompt is normally identical across
    instances, but nothing guarantees that, and a shared budget built from one
    of them would misreport `resident_len_at_last_turn` for the rest.
    """
    conversations: list[dict] = []
    drops = {
        "too_few_steps": 0,
        "action_unparseable": 0,
        "observation_too_large": 0,
        "action_too_large": 0,
        "failed_budget": 0,
    }

    for traj in trajectories:
        if len(traj["messages"]) // 2 < turns_per_conv:
            drops["too_few_steps"] += 1
            continue

        turns = build_turns(traj, turns_per_conv, tok)

        # Every reference must be scorable. A recorded action with no single
        # bash block is a real occurrence -- the scaffold rejects it and
        # re-prompts, and `normalize_swebench_trajs.py`'s alternation check
        # keeps that exchange -- but it cannot serve as a target for
        # `agent_action_match`. Dropping the trajectory whole keeps the
        # conversation contiguous; skipping the turn would renumber every later
        # slot and break the replay.
        if any(extract_agent_action(t["answer"]) is None for t in turns):
            drops["action_unparseable"] += 1
            continue
        if any(t["query_tokens"] > max_observation_tokens for t in turns):
            drops["observation_too_large"] += 1
            continue
        if any(t["answer_tokens"] > max_action_tokens for t in turns):
            drops["action_too_large"] += 1
            continue

        preamble_len = len(tok.encode(traj["system"], add_special_tokens=False))
        budget = budget_for(preamble_len)
        if not budget.fits([t["query_tokens"] for t in turns]):
            drops["failed_budget"] += 1
            continue

        conversations.append(
            {
                "instance_id": traj["instance_id"],
                "exit_status": traj.get("exit_status"),
                "num_steps_recorded": traj.get("num_steps"),
                "system": traj["system"],
                "turns": turns,
                "budget": budget,
            }
        )
        if verbose_every and len(conversations) % verbose_every == 0:
            print(f"[prep_swea]   built {len(conversations)} conversations...",
                  flush=True)

    return conversations, drops


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def build_rows(conversations: list[dict], config_name: str, id_prefix: str,
               max_tokens: int) -> list[dict]:
    """One JSONL row per conversation, in `predict_scbench.py`'s schema.

    `id` is not just a label: it is the vLLM prefix-cache salt, the speculator
    worker's slot-history key (`proposer.py`'s `"{salt}::turn{n}"` convention)
    AND the grader's join key. The `swea` prefix keeps it from colliding with
    `scbench_<config>-<n>`, `lbv2mt-<n>` or `mmmumt-<n>` even if two sample
    files are concatenated -- two files sharing an id would let a predictions
    file grade silently against the wrong turns.
    """
    rows = []
    for n, conv in enumerate(conversations):
        budget = conv["budget"]
        query_lens = [t["query_tokens"] for t in conv["turns"]]
        rows.append(
            {
                "id": f"{id_prefix}-{n:04d}",
                "config": config_name,
                "context": conv["system"],
                "turns": [
                    {
                        "input": t["input"],
                        "answer": t["answer"],
                        "query_tokens": t["query_tokens"],
                        "answer_tokens": t["answer_tokens"],
                    }
                    for t in conv["turns"]
                ],
                # Inert to both `predict_scbench.py` (which reads id/config/
                # context/turns[].input/turns[].answer only) and
                # `grade_scbench.py`. Recorded so a later run at a different
                # --max-tokens is caught by inspecting the file rather than by
                # a silent mid-run turn-loop break.
                "instance_id": conv["instance_id"],
                "exit_status": conv["exit_status"],
                "num_steps_recorded": conv["num_steps_recorded"],
                "total_prompt_tokens": sum(query_lens),
                "resident_len_at_last_turn": budget.target_check(query_lens),
                "reserved_output_tokens_per_turn": max_tokens,
                "target_budget": budget.target_budget,
                "speculator_budget": budget.spec_budget,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay recorded SWE-bench agent trajectories as multi-turn "
                    "conversations in predict_scbench.py's sample schema."
    )
    parser.add_argument("--tokenizer", required=True,
                        help="TARGET model path -- token counts must be the "
                             "target's. A different tokenizer selects a "
                             "different population of trajectories, and two "
                             "files built that way do not replay the same runs.")
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES,
                        help="Output of datasets/normalize_swebench_trajs.py.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME,
                        help="Value of the `config` field; must match the key "
                             "added to grade_scbench.py's _METRIC_BY_CONFIG.")
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX,
                        help="Keeps ids distinct from other sample files'. The "
                             "id is the prefix-cache salt, the speculator's "
                             "slot key and the grader's join key.")
    parser.add_argument("--turns-per-conv", type=int, default=16,
                        help="Fixed T. A trajectory with fewer steps is dropped "
                             "whole; a longer one is used as a T-step prefix. "
                             "Run normalize_swebench_trajs.py first -- it "
                             "prints the yield at several T.")
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=130560,
                        help="MUST equal the value passed to predict_scbench.py. "
                             "Set it to native_target - --max-tokens.")
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131063,
                        help="MUST equal the value passed to predict_scbench.py.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Per-turn generation cap. This dataset is SIZED "
                             "for this value; a later run at a larger one will "
                             "overflow the session.")
    parser.add_argument("--max-observation-tokens", type=int, default=4000,
                        help="Rendered per-turn ceiling. Bounds the per-turn `d` "
                             "spread, which the mean-over-turns headline columns "
                             "are sensitive to. Trajectories with a bigger turn "
                             "are DROPPED, never truncated.")
    parser.add_argument("--max-action-tokens", type=int, default=-1,
                        help="-1 uses --max-tokens. An action longer than the "
                             "generation cap cannot be matched by any arm and "
                             "breaks Budget's (T-1)*O reservation under the "
                             "golden-context M-k*-g* path.")
    parser.add_argument("--safety-tokens", type=int, default=512)
    parser.add_argument("--max-conversations", type=int, default=-1,
                        help="-1 keeps all. Applied after every filter, so a "
                             "small pilot file is a prefix of the full one.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Only used to shuffle trajectory order before the "
                             "--max-conversations cap, so a pilot is not all "
                             "one repository.")
    args = parser.parse_args()

    max_action_tokens = (args.max_action_tokens if args.max_action_tokens >= 0
                         else args.max_tokens)

    lbv2 = _load_lbv2_module()
    Budget, _histogram = lbv2.Budget, lbv2._histogram
    look_ahead = lbv2.LOOK_AHEAD_CNT

    tok = load_tokenizer(args.tokenizer)
    wrapper_before, wrapper_after = chat_wrapper_pieces(tok)
    turn_boundary = chat_turn_boundary_pieces(tok)

    def budget_for(preamble_len: int):
        return Budget(
            target_budget=args.target_max_num_batched_tokens,
            spec_budget=args.speculator_max_num_batched_tokens,
            turns_per_conv=args.turns_per_conv,
            max_tokens=args.max_tokens,
            safety_tokens=args.safety_tokens,
            wrapper_before=len(tok.encode(wrapper_before, add_special_tokens=False)),
            wrapper_after=len(tok.encode(wrapper_after, add_special_tokens=False)),
            turn_boundary=len(tok.encode(turn_boundary, add_special_tokens=False)),
            preamble_len=preamble_len,
        )

    try:
        trajectories = load_trajectories(args.trajectories)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[prep_swea] ERROR: {exc}", file=sys.stderr)
        return 2
    if not trajectories:
        print(f"[prep_swea] ERROR: {args.trajectories} is empty.", file=sys.stderr)
        return 2

    random.Random(args.seed).shuffle(trajectories)

    preamble_lens = {len(tok.encode(t["system"], add_special_tokens=False))
                     for t in trajectories}
    reference = budget_for(max(preamble_lens))
    print(f"[prep_swea] loaded {len(trajectories)} trajectories from "
          f"{args.trajectories}")
    print(f"[prep_swea] system-prompt lengths: {len(preamble_lens)} distinct, "
          f"max {max(preamble_lens):,} tokens")
    print("[prep_swea] budget (at the longest system prompt):")
    print(reference.describe())
    print(f"  -> per-turn ceiling    {args.max_observation_tokens:>9,}  "
          f"(--max-observation-tokens)")
    if reference.doc_budget <= 0:
        print("[prep_swea] ERROR: non-positive turn budget -- the wrapper/"
              "output/safety reservations already exceed the engine budgets. "
              "Check both --*-max-num-batched-tokens against the checkpoints' "
              "native context lengths, and --turns-per-conv.", file=sys.stderr)
        return 2
    per_turn_ceiling = reference.doc_budget // args.turns_per_conv
    if args.max_observation_tokens > per_turn_ceiling:
        print(f"[prep_swea] NOTE: --max-observation-tokens "
              f"({args.max_observation_tokens:,}) is above the budget's own "
              f"per-turn share ({per_turn_ceiling:,}), so the budget check "
              f"rather than the ceiling will be what drops conversations.")

    conversations, drops = build_conversations(
        trajectories, budget_for, tok, args.turns_per_conv,
        args.max_observation_tokens, max_action_tokens,
    )

    if args.max_conversations >= 0:
        conversations = conversations[:args.max_conversations]

    if not conversations:
        print(f"[prep_swea] ERROR: no conversations formed (drops={drops}). "
              f"Lower --turns-per-conv (normalize_swebench_trajs.py prints the "
              f"yield at several T), raise --max-observation-tokens, or record "
              f"more trajectories.", file=sys.stderr)
        return 2

    out_rows = build_rows(conversations, args.config_name, args.id_prefix,
                          args.max_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_turn = [t["query_tokens"] for r in out_rows for t in r["turns"]]
    actions = [t["answer_tokens"] for r in out_rows for t in r["turns"]]
    resident = [r["resident_len_at_last_turn"] for r in out_rows]
    print(f"[prep_swea] conversations={len(out_rows)} turns={len(per_turn)} "
          f"dropped_too_few_steps={drops['too_few_steps']} "
          f"dropped_action_unparseable={drops['action_unparseable']} "
          f"dropped_observation_too_large={drops['observation_too_large']} "
          f"dropped_action_too_large={drops['action_too_large']} "
          f"dropped_failed_budget={drops['failed_budget']}")
    print(_histogram(per_turn, "per-turn d (rendered query tokens)"))
    print(_histogram(actions, "per-turn action tokens (golden answer)"))
    print(_histogram(resident, "resident len at last turn"))

    # Turn 0 is the issue statement and every later turn is a tool output, so
    # the two populations are not comparable -- report them apart. Turn 0 is
    # also the turn `grade_scbench.py` scores separately (`overall_turn0`),
    # and under the sparse path its prefill is dense under both scopes.
    turn0 = [r["turns"][0]["query_tokens"] for r in out_rows]
    later = [t["query_tokens"] for r in out_rows for t in r["turns"][1:]]
    print(_histogram(turn0, "turn 0 only (issue statement)"))
    print(_histogram(later, "turns 1+ only (tool observations)"))

    # Report BOTH engines' headroom and say which is binding. Reporting only
    # the target's is actively misleading whenever the scorer has the smaller
    # context window -- the target can show six figures of slack while the
    # scorer sits a few thousand tokens from its ceiling.
    worst_target = max(r["resident_len_at_last_turn"] + args.max_tokens
                       for r in out_rows)
    worst_spec = max(
        conv["budget"].spec_check([t["query_tokens"] for t in conv["turns"]])
        + 1 + look_ahead
        for conv in conversations
    )
    t_head = args.target_max_num_batched_tokens - worst_target
    s_head = args.speculator_max_num_batched_tokens - worst_spec
    for name, worst, head, cap in (
        ("target    ", worst_target, t_head, args.target_max_num_batched_tokens),
        ("speculator", worst_spec, s_head, args.speculator_max_num_batched_tokens),
    ):
        mark = "  <-- BINDING" if head == min(t_head, s_head) else ""
        print(f"  {name} worst-case {worst:,} of {cap:,} -- headroom {head:,}{mark}")
    print(f"  mean d:o ratio  {statistics.mean(per_turn) / args.max_tokens:.1f}:1 "
          f"(the paper's LongBench-v2-MC config is ~49:1; SCBench was ~1:7)")
    print(f"  removable prefill-attention share (T-1)/T  "
          f"{(args.turns_per_conv - 1) / args.turns_per_conv:.1%}")
    print(f"[prep_swea] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
