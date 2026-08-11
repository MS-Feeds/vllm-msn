#!/usr/bin/env python3
"""Prepares SCBench samples for the multi-turn Top-K KV Cache Selection sweep
(see ../EXPERIMENT_PLAN.md). SCBench (arXiv:2412.10319, `microsoft/SCBench`
on Hugging Face) is a genuinely MULTI-TURN benchmark: each row is one long
shared context plus a list of 2-4 sequential turns (question + reference
answer, and for multiple-choice configs, options) -- unlike
`../../spec_prefill_llama/datasets/prep_longbench_v2.py`'s LongBench-v2
loader, this script writes ONE ROW PER CONTEXT, not one row per question,
because a conversation's turns are not independent samples (turn N's prompt
depends on turns 1..N-1 -- see ../vllm_patch/conversation_state.py).

Schema (confirmed directly against the live `microsoft/SCBench` dataset
card on Hugging Face, not assumed -- see REPRODUCE.md for how to
re-verify): each row is `{"id": int, "context": str, "multi_turns": [{"input":
str, "answer": str, "options": [str, ...]}, ...]}`, 2-4 turns per row.
`options` is only meaningful for multiple-choice configs (e.g.
scbench_choice_eng); empty/absent for free-form QA/retrieval/summarization
configs.

Per the approved plan's confirmed MVP scope, this script loads 3
representative configs (out of SCBench's 12), one per capability area:

- `scbench_qa_eng`  -- semantic retrieval / free-form QA
- `scbench_kv`      -- string/exact retrieval (synthetic key-value lookup)
- `scbench_summary` -- global-information tasks (summarization)

Writes `datasets/scbench_samples.jsonl`, one row per (config, context) pair:
{"id", "config", "context", "turns": [{"input", "answer", "options"}, ...]}
-- "turns" (not "multi_turns") only for local naming consistency with the
rest of this codebase; content is unchanged from the source.

Usage:
    python3 prep_scbench.py --max-keep-per-config -1   # keep all rows
    python3 prep_scbench.py --configs scbench_kv --max-keep-per-config 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HF_DATASET_NAME = "microsoft/SCBench"
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_OUTPUT = Path(__file__).parent / "scbench_samples.jsonl"

# Per the approved plan's confirmed MVP scope -- 3 of SCBench's 12 configs,
# one per capability area. The other 9
# (scbench_choice_eng/scbench_many_shot/scbench_mf/scbench_prefix_suffix/
# scbench_qa_chn/scbench_repoqa/scbench_repoqa_and_kv/
# scbench_summary_with_needles/scbench_vt) are valid --configs values too
# (this script does not hardcode a config allowlist), just not part of the
# default MVP sweep.
DEFAULT_CONFIGS = ["scbench_qa_eng", "scbench_kv", "scbench_summary"]

_REQUIRED_FIELDS = ["id", "context", "multi_turns"]
_MIN_TURNS = 2
_MAX_TURNS = 4


def _resolve_hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )


def load_scbench_config_rows(
    config_name: str, cache_dir: Path, hf_token: str | None
) -> list[dict]:
    """Loads one SCBench HF config's rows via the `datasets` library --
    same loading convention as prep_longbench_v2.py's `load_dataset(...)`
    call (SCBench ships as parquet via the `datasets` library, not a plain
    CSV)."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[prep_scbench] loading {HF_DATASET_NAME}/{config_name} "
        f"(cache_dir={cache_dir})",
        flush=True,
    )
    ds = load_dataset(
        HF_DATASET_NAME,
        config_name,
        split="test",
        cache_dir=str(cache_dir),
        token=hf_token,
    )
    rows = list(ds)
    if rows and not all(col in rows[0] for col in _REQUIRED_FIELDS):
        raise KeyError(
            f"Expected columns {_REQUIRED_FIELDS} not all found in "
            f"{HF_DATASET_NAME}/{config_name}. Available columns: "
            f"{sorted(rows[0].keys())} -- SCBench's real schema may have "
            f"changed since this script was written (confirmed against the "
            f"live dataset card at write time, see module docstring)."
        )
    return rows


def build_scbench_samples(
    config_name: str,
    rows: list[dict],
    max_keep: int = -1,
) -> list[dict]:
    """Formats each row into {"id", "config", "context", "turns"},
    validating the 2-4-turns-per-row invariant SCBench's own documentation
    states rather than assuming it silently holds."""
    samples: list[dict] = []
    skipped_bad_row = 0
    skipped_turn_count = 0

    for row in rows:
        context = (row.get("context") or "").strip()
        multi_turns = row.get("multi_turns") or []
        row_id = row.get("id")

        if not context or row_id is None:
            skipped_bad_row += 1
            continue
        if not (_MIN_TURNS <= len(multi_turns) <= _MAX_TURNS):
            skipped_turn_count += 1
            continue

        turns = []
        bad_turns = False
        for turn in multi_turns:
            turn_input = (turn.get("input") or "").strip()
            turn_answer = turn.get("answer")
            if not turn_input or turn_answer is None:
                bad_turns = True
                break
            turns.append(
                {
                    "input": turn_input,
                    "answer": turn_answer,
                    "options": turn.get("options") or [],
                }
            )
        if bad_turns:
            skipped_bad_row += 1
            continue

        samples.append(
            {
                "id": f"{config_name}-{row_id}",
                "config": config_name,
                "context": context,
                "turns": turns,
            }
        )

    print(
        f"[prep_scbench] {config_name}: loaded={len(rows)} kept={len(samples)} "
        f"skipped_bad_row={skipped_bad_row} skipped_turn_count={skipped_turn_count}"
    )

    if max_keep >= 0 and len(samples) > max_keep:
        samples = samples[:max_keep]

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCBench multi-turn eval samples")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help=f"Comma-separated SCBench HF configs to load. Default: {DEFAULT_CONFIGS}",
    )
    parser.add_argument(
        "--max-keep-per-config",
        type=int,
        default=-1,
        help="Cap on number of context rows PER CONFIG; -1 keeps all available rows.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = _resolve_hf_token(args.hf_token)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    all_samples: list[dict] = []
    for config_name in config_names:
        rows = load_scbench_config_rows(config_name, args.cache_dir, token)
        all_samples.extend(
            build_scbench_samples(config_name, rows, max_keep=args.max_keep_per_config)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep_scbench] wrote {len(all_samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
