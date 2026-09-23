#!/usr/bin/env python3
"""Flattens mini-swe-agent SWE-bench trajectory files into ONE JSONL of linear
message histories, for `prep_swebench_agent_replay.py` to turn into
conversations in `predict_scbench.py`'s sample schema.

## Why this is a separate step

Recording is the only part of the agentic pipeline that needs Docker and a
running agent scaffold, and it happens ONCE. Everything downstream -- packing,
the keep-rate sweep, grading -- is pure CPU work against the file this script
writes. Keeping the two apart means a re-pack at a different `--turns-per-conv`
never re-runs an agent, and the expensive artifact is reviewable on its own.

## Input

`mini-extra swebench -o <out>/` writes, per instance:

    <out>/<instance_id>/<instance_id>.traj.json
    <out>/preds.json

The trajectory file is `agents/default.py::serialize`'s output merged with the
runner's extra dict:

    {"info": {"exit_status": str, "submission": str, "traceback": str,
              "exception_str": str, ...},
     "messages": [{"role": str, "content": str}, ...],
     "trajectory_format": "mini-swe-agent-1.1",
     "instance_id": str}

`messages` is the whole point: mini-swe-agent keeps "a completely linear
history -- every step of the agent just appends to the messages", with no
summarisation or compaction, so the list IS the conversation the model saw and
a replay of it is faithful by construction. A scaffold that compacted its
history would need that compaction reproduced here, and none of this would work.

## Output

`datasets/swebench_trajectories.jsonl`, one row per surviving instance:

    {"instance_id": str, "exit_status": str, "num_steps": int,
     "system": str, "messages": [{"role", "content"}, ...]}

`messages` has the leading system message REMOVED (it is hoisted into
`system`, which becomes the conversation preamble downstream) and is
guaranteed to alternate user/assistant starting with user, so
`prep_swebench_agent_replay.py` can index it without re-checking.

## Filter, never repair

A trajectory whose roles do not alternate is DROPPED with its instance id
printed, never reshaped into something that does. The alternation is what makes
the `messages[2i]`/`messages[2i+1]` mapping mean "observation i" and "action i";
silently patching a gap would misalign every turn after it, and the replay would
score the sparse model against the wrong action with nothing to show for it.

Same for a missing system message: `context` must be non-empty downstream
(`ConversationState`'s turn-0 candidate pool is built from it), and inventing a
preamble here would put text in front of the model that the recorded run never
saw.

## Exit statuses are reported, not guessed

This script does not hardcode which `info.exit_status` values are "good" --
that vocabulary belongs to the scaffold and changes with it. It drops only runs
that carry a `traceback`/`exception_str` (the harness itself failed, so the
trajectory is a fragment), prints the observed status distribution, and leaves
value-based filtering to `--exit-status`, which you set after looking at that
distribution.

Usage:
    python3 datasets/normalize_swebench_trajs.py --traj-dir ./sweb_out
    python3 datasets/normalize_swebench_trajs.py --traj-dir ./sweb_out \\
        --exit-status Submitted --min-steps 8
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEFAULT_OUTPUT = _HERE / "swebench_trajectories.jsonl"

#: Roles the mapping below assumes, in the order it assumes them.
_USER = "user"
_ASSISTANT = "assistant"
_SYSTEM = "system"


def _content_text(content) -> str:
    """mini-swe-agent writes plain strings, but the OpenAI message schema also
    permits a list of typed content blocks and litellm can hand one back. Join
    the text parts rather than `str()`-ing a list into the prompt."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def iter_traj_files(traj_dir: Path):
    """Both layouts mini-swe-agent has used: one directory per instance, and a
    flat directory of `*.traj.json`. Sorted so a run is reproducible."""
    yield from sorted(traj_dir.glob("*/*.traj.json"))
    yield from sorted(traj_dir.glob("*.traj.json"))


def normalize_one(path: Path) -> tuple[dict | None, str]:
    """Returns `(row, reason)`. `row` is None when the trajectory is dropped,
    and `reason` is the drop bucket (or "ok")."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[normalize_trajs] unreadable {path.name}: {exc}", file=sys.stderr)
        return None, "unreadable"

    info = raw.get("info") or {}
    instance_id = raw.get("instance_id") or path.name.split(".traj.json")[0]

    # The harness itself blew up -- whatever messages exist are a fragment of a
    # run that never finished, not a short run.
    if info.get("traceback") or info.get("exception_str"):
        return None, "harness_error"

    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "no_messages"

    roles = [m.get("role") for m in messages]
    if roles[0] != _SYSTEM:
        # See "Filter, never repair" in the module docstring.
        print(f"[normalize_trajs] {instance_id}: first message is "
              f"{roles[0]!r}, not {_SYSTEM!r} -- dropped", file=sys.stderr)
        return None, "no_system"

    system = _content_text(messages[0].get("content"))
    if not system.strip():
        print(f"[normalize_trajs] {instance_id}: empty system message -- dropped",
              file=sys.stderr)
        return None, "empty_system"

    body = messages[1:]
    # Strict alternation user, assistant, user, assistant, ... A trailing user
    # message (the observation the agent never got to act on) is dropped, since
    # a turn needs both halves.
    if len(body) % 2 == 1:
        body = body[:-1]
    expected = [_USER, _ASSISTANT] * (len(body) // 2)
    actual = [m.get("role") for m in body]
    if actual != expected:
        first_bad = next(
            (i for i, (a, e) in enumerate(zip(actual, expected)) if a != e),
            len(expected),
        )
        print(f"[normalize_trajs] {instance_id}: roles stop alternating at "
              f"body index {first_bad} (saw {actual[first_bad:first_bad + 3]!r}) "
              f"-- dropped", file=sys.stderr)
        return None, "bad_alternation"

    flat = [{"role": m.get("role"), "content": _content_text(m.get("content"))}
            for m in body]
    if not flat:
        return None, "no_steps"
    if any(not m["content"].strip() for m in flat):
        return None, "empty_message"

    return (
        {
            "instance_id": instance_id,
            "exit_status": info.get("exit_status"),
            "num_steps": len(flat) // 2,
            "system": system,
            "messages": flat,
        },
        "ok",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flatten mini-swe-agent trajectories into one JSONL."
    )
    parser.add_argument("--traj-dir", type=Path, required=True,
                        help="mini-extra swebench's -o/--output directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exit-status", default="",
                        help="Comma-separated allowlist of info.exit_status "
                             "values. Empty (default) keeps every status and "
                             "just reports the distribution -- set this only "
                             "after looking at that report.")
    parser.add_argument("--min-steps", type=int, default=1,
                        help="Drop trajectories with fewer than this many "
                             "complete (observation, action) pairs. The real "
                             "floor is --turns-per-conv at pack time; this is "
                             "only to keep obvious stubs out of the file.")
    args = parser.parse_args()

    if not args.traj_dir.is_dir():
        print(f"[normalize_trajs] ERROR: no such directory: {args.traj_dir}",
              file=sys.stderr)
        return 2

    allow = {s.strip() for s in args.exit_status.split(",") if s.strip()}

    rows: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    statuses: collections.Counter = collections.Counter()
    seen: set[str] = set()

    for path in iter_traj_files(args.traj_dir):
        row, reason = normalize_one(path)
        reasons[reason] += 1
        if row is None:
            continue
        if row["instance_id"] in seen:
            reasons["duplicate"] += 1
            continue
        statuses[str(row["exit_status"])] += 1
        if allow and str(row["exit_status"]) not in allow:
            reasons["exit_status_filtered"] += 1
            continue
        if row["num_steps"] < args.min_steps:
            reasons["too_few_steps"] += 1
            continue
        seen.add(row["instance_id"])
        rows.append(row)

    if not rows:
        print(f"[normalize_trajs] ERROR: no trajectories survived. Buckets: "
              f"{dict(reasons)}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    steps = sorted(r["num_steps"] for r in rows)
    print(f"[normalize_trajs] kept={len(rows)} buckets={dict(reasons)}")
    print(f"[normalize_trajs] steps per trajectory: min={steps[0]} "
          f"median={steps[len(steps) // 2]} max={steps[-1]}")
    # The number that decides --turns-per-conv: how many trajectories survive a
    # given T, since a trajectory shorter than T is dropped whole at pack time.
    print("[normalize_trajs] trajectories with at least T steps:")
    for t in (8, 12, 16, 24, 32):
        print(f"    T={t:>2}: {sum(1 for s in steps if s >= t):>4}")
    print("[normalize_trajs] exit_status distribution (before --exit-status):")
    for status, n in statuses.most_common():
        print(f"    {status}: {n}")
    print(f"[normalize_trajs] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
