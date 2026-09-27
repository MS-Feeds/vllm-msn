#!/usr/bin/env python3
"""Builds the TASK-ONLY samples file that seeds a live agentic run
(`predict_scbench.py --agentic`).

## Why this exists

`--agentic` reads turn 0's issue statement from the samples file and gets every
later turn from the sandbox. The replay file
(`prep_swebench_agent_replay.py`) also carries that issue statement at turn 0 --
but building it requires trajectories, and recording trajectories requires a
live run, which requires a samples file. This breaks that circle: it needs
nothing but the SWE-bench dataset itself.

    prep_swebench_tasks.py                  (this file, no trajectories needed)
        -> predict_scbench.py --agentic --exp M000     (records, dense)
        -> normalize_swebench_trajs.py                 (from the run's output)
        -> prep_swebench_agent_replay.py               (the replay dataset)

## Why record this way rather than with mini-swe-agent

Recording through THIS driver means the prompts the model sees while being
recorded are rendered by `render_turn_query` -- byte-identical to what the
replay will later feed it. mini-swe-agent renders its own chat format instead,
so a replay of its trajectories asks the model to reproduce actions it took
under a different prompt, and `M000`'s action-agreement lands well below the
~0.5 the replay gate wants. Recording here makes `M000` a genuine near-1.0
control: same model, same rendering, same tokens.

It also drops the mini-swe-agent dependency entirely, which matters on a node
whose only container runtime is udocker -- mini-swe-agent has no udocker
environment class, while `agent_sandbox.py` does.

## The system prompt is the contract

`context` is the agent's system prompt, and three things in it are load-bearing
rather than stylistic:

  - **Exactly one fenced ```bash block per reply.** `grade_scbench.py`'s
    `extract_agent_action` returns None for zero or two blocks, and
    `agentic.py` spends a turn on a format reminder when that happens. A prompt
    that invites prose plus two commands wastes turns.
  - **`submit` as the finish signal**, matching `agentic.py`'s
    `DEFAULT_SUBMIT_RE`. If the prompt says anything else, nothing ever
    matches, every conversation runs to `--max-turns`, and the submission rate
    is zero for a reason that has nothing to do with the model.
  - **No interactive commands.** There is no TTY; anything that waits for input
    hits the per-command timeout and burns a turn.

`context` must also be non-empty for an unrelated reason: `ConversationState`
builds turn 0's candidate pool from it.

Usage:
    python3 datasets/prep_swebench_tasks.py \\
        --tokenizer $GEMMA4_31B_MODEL_PATH \\
        --max-instances 40 \\
        --output datasets/swebench_tasks.jsonl
"""

from __future__ import annotations

import argparse
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
from vllm_patch.model_structure import load_tokenizer  # noqa: E402

DEFAULT_OUTPUT = _HERE / "swebench_tasks.jsonl"
DEFAULT_CONFIG_NAME = "swebench_agent"
#: Distinct from the replay file's `swea` -- the two are different datasets and
#: `id` is the prefix-cache salt, the speculator's slot key and the grader's
#: join key, so a collision would let one grade against the other's turns.
DEFAULT_ID_PREFIX = "swet"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"

#: See "The system prompt is the contract" above before editing. The submit
#: word must stay in sync with `agentic.py::DEFAULT_SUBMIT_RE`.
SYSTEM_PROMPT = """\
You are a software engineer working in a checked-out repository at /testbed.
You will be shown an issue to fix. You interact with the repository only by \
running shell commands, one at a time, and observing their output.

Rules:
- Reply with EXACTLY ONE bash command block per message, formatted as:

```bash
your_command_here
```

- Write nothing after the command block.
- Commands are non-interactive: there is no terminal attached, so never run \
anything that waits for input (no editors, no pagers, no prompts).
- Inspect before editing. Read the relevant files first.
- Make the smallest change that fixes the issue, and do not modify tests.
- When the fix is complete, reply with exactly:

```bash
submit
```
"""


def load_instances(dataset: str, split: str, cache_dir: Path, hf_token):
    """Resolves to the Hugging Face `datasets` package, not this directory --
    `datasets/` has no `__init__.py`, so under PEP 420 it is only a namespace
    portion and the regular package found later on `sys.path` wins."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prep_swetasks] loading {dataset} (split={split})", flush=True)
    return list(load_dataset(dataset, split=split, cache_dir=str(cache_dir),
                             token=hf_token))


def build_rows(instances, tok, config_name: str, id_prefix: str,
               max_task_tokens: int) -> tuple[list[dict], dict]:
    """One row per instance: the system prompt as `context`, the issue
    statement as the single turn."""
    rows: list[dict] = []
    drops = {"no_problem_statement": 0, "task_too_large": 0}

    for instance in instances:
        statement = (instance.get("problem_statement") or "").strip()
        instance_id = instance.get("instance_id")
        if not statement or not instance_id:
            drops["no_problem_statement"] += 1
            continue

        # Measured through the driver's own renderer, so this count IS the
        # count the driver will see for turn 0.
        query_tokens = len(render_turn_query(tok, 0, {"input": statement}))
        if query_tokens > max_task_tokens:
            drops["task_too_large"] += 1
            continue

        rows.append({
            "id": f"{id_prefix}-{len(rows):04d}",
            "config": config_name,
            "context": SYSTEM_PROMPT,
            "turns": [{"input": statement, "answer": ""}],
            # Inert to the driver, which reads id/config/context/turns only.
            # `instance_id` is the exception that matters: `agentic.py` uses it
            # to look up the container image for this conversation.
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "task_tokens": query_tokens,
        })

    return rows, drops


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the task-only samples file for a live agentic run."
    )
    parser.add_argument("--tokenizer", required=True,
                        help="TARGET model path -- token counts must be the "
                             "target's.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", type=Path, default=_HERE / ".cache")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX)
    parser.add_argument("--max-instances", type=int, default=-1,
                        help="-1 keeps all. Applied after shuffling, so a "
                             "pilot is not all one repository.")
    parser.add_argument("--max-task-tokens", type=int, default=8000,
                        help="Drop instances whose rendered issue statement "
                             "exceeds this. Turn 0 is the one turn that cannot "
                             "be truncated at run time, and a pathological "
                             "statement would eat the session budget before "
                             "the agent has done anything.")
    parser.add_argument("--seed", type=int, default=42)
    # Reporting only -- these must MATCH what predict_scbench.py is given, and
    # exist here so the worst-case session length is visible before a run
    # rather than as a mid-run retirement.
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-observation-tokens", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=130560)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131063)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    instances = load_instances(args.dataset, args.split, args.cache_dir, hf_token)
    if not instances:
        print("[prep_swetasks] ERROR: dataset is empty.", file=sys.stderr)
        return 2

    random.Random(args.seed).shuffle(instances)
    rows, drops = build_rows(instances, tok, args.config_name, args.id_prefix,
                             args.max_task_tokens)
    if args.max_instances >= 0:
        rows = rows[:args.max_instances]
    if not rows:
        print(f"[prep_swetasks] ERROR: no instances survived ({drops}). Raise "
              f"--max-task-tokens.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_tokens = [r["task_tokens"] for r in rows]
    wrapper_before, wrapper_after = chat_wrapper_pieces(tok)
    overhead = (len(tok.encode(wrapper_before, add_special_tokens=False))
                + len(tok.encode(wrapper_after, add_special_tokens=False)))
    boundary = len(tok.encode(chat_turn_boundary_pieces(tok),
                              add_special_tokens=False))
    context_len = len(tok.encode(SYSTEM_PROMPT, add_special_tokens=False))

    # The bound that replaces prep-time verification. A live run cannot know
    # observation lengths in advance, so the guarantee is arithmetic rather
    # than measured: every observation is capped, and the number of turns is
    # capped, so the session cannot exceed this.
    per_turn = args.max_observation_tokens + args.max_tokens + boundary + \
        len(tok.encode(wrapper_after, add_special_tokens=False))
    worst_case = context_len + overhead + max(task_tokens) + args.max_turns * per_turn

    print(f"[prep_swetasks] instances={len(rows)} drops={drops}")
    print(f"  system prompt          {context_len:>9,} tokens")
    print(f"  issue statement        min={min(task_tokens):,} "
          f"median={statistics.median(task_tokens):,.0f} max={max(task_tokens):,}")
    print(f"  worst-case session     {worst_case:>9,} tokens "
          f"(= context + task + {args.max_turns} x "
          f"({args.max_observation_tokens:,} obs + {args.max_tokens} gen))")
    for name, cap in (("target    ", args.target_max_num_batched_tokens),
                      ("speculator", args.speculator_max_num_batched_tokens)):
        head = cap - worst_case
        mark = "  <-- WILL RETIRE EARLY" if head < 0 else ""
        print(f"  {name} budget {cap:>9,} -- headroom {head:,}{mark}")
    if worst_case > min(args.target_max_num_batched_tokens,
                        args.speculator_max_num_batched_tokens):
        print("[prep_swetasks] NOTE: the worst case exceeds an engine budget. "
              "That is not fatal -- the driver's pre-flight retires a "
              "conversation cleanly at the turn it would overrun, and "
              "num_skipped_too_large is EXPECTED to be non-zero on live rows. "
              "Lower --max-turns or --max-observation-tokens to avoid it.")
    print(f"[prep_swetasks] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
