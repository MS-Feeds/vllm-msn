#!/usr/bin/env python3
"""Offline quality-evaluation runner for Gemma4 on A100.

Sibling to run_config.py with an identical CLI (--config / --prompts / --reps)
so the AML submission flow works unchanged. It reuses build_llm_kwargs() from
run_config.py verbatim so ANY branch/config/setting can be scored identically
to how it benchmarks.

Task-agnostic, deterministic (greedy, temperature=0) scoring. v1 tasks:
  - gsm8k: free-form arithmetic; regex-extract final number, compare to gold.
  - mmlu : multi-choice; predict option letter, accuracy.
  - bbh  : selected deterministic subtasks; extract final answer, EM.

The eval dataset is chosen by a config key `_agentic_eval_dataset` (a JSONL
path). Each line is a task record with a `type` field plus task-specific
fields. Emits:
  - a JSON score report (per-task accuracy / EM) for the quality gate
  - a parity-format dump (list-of-lists of token_ids) so detect_degeneration.py
    and the tolerant compare work unchanged.

This file is strictly additive. It touches nothing in run_config.py / the
throughput path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from run_config import build_llm_kwargs, truncate_prompts


SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as dataset:
        for line in dataset:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            if len(records) == limit:
                break
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


# ---------------------------------------------------------------------------
# Prompt rendering (builds on the same chat-template path as the benchmark)
# ---------------------------------------------------------------------------
def render_prompt(tokenizer, task_type: str, record: dict) -> str:
    if task_type == "gsm8k":
        question = record["question"]
        user_msg = (
            f"Solve the following math problem step by step, then output the "
            f"final numeric answer as a bare number on its own line at the very "
            f"end. Do not include the word 'answer'.\n\nProblem:\n{question}"
        )
    elif task_type == "mmlu":
        question = record["question"]
        choices = record["choices"]
        labeled = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        user_msg = (
            f"Choose the correct option letter (A, B, C, or D) for the "
            f"following multiple-choice question. Output ONLY the option "
            f"letter.\n\nQuestion:\n{question}\n\nChoices:\n{labeled}"
        )
    elif task_type == "bbh":
        user_msg = (
            f"Answer the following question. Output ONLY your final answer at "
            f"the very end on its own line.\n\nQuestion:\n{record['input']}"
        )
    elif task_type == "tool_call":
        tools = record["tools"]
        desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}  "
            f"arguments: {json.dumps(t.get('parameters', {}), ensure_ascii=False)}"
            for t in tools
        )
        user_msg = (
            f"You are an agent that can call functions to answer the user's "
            f"request. Available functions and their JSON-schema arguments:\n"
            f"{desc}\n\n"
            f"Determine which single function to call and output EXACTLY one "
            f"JSON object with the keys 'name' (the function name) and "
            f"'arguments' (the arguments object), for example:\n"
            f'{{"name":"search","arguments":{{"query":"cats"}}}}\n\n'
            f"User request:\n{record['input']}"
        )
    elif task_type == "free":
        user_msg = record["prompt"]
    else:
        raise ValueError(f"Unknown eval task type: {task_type}")

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True,
        tokenize=False,
    )


# ---------------------------------------------------------------------------
# Task scorers (pure python, deterministic)
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_number(s: str) -> str:
    """Normalize a numeric string: strip leading zeros, drop trailing .0 / .00."""
    try:
        v = float(s)
    except ValueError:
        return s.strip().lower()
    if v == int(v):
        return str(int(v))
    # Strip trailing zeros from the fractional part (e.g. 1.50 -> 1.5).
    return format(v, "f").rstrip("0").rstrip(".")


def score_gsm8k(text: str, record: dict) -> bool:
    gold = _last_number(str(record["answer"]))
    if gold is None:
        return False
    pred = _last_number(text) or ""
    return _normalize_number(pred) == _normalize_number(gold)


def score_mmlu(text: str, record: dict) -> bool:
    gold_idx = int(record["answer"])
    gold_letter = chr(65 + gold_idx).lower()
    m = re.search(r"\b([A-Da-d])\b", text.strip())
    if m:
        return m.group(1).lower() == gold_letter
    return False


def score_bbh(text: str, record: dict) -> bool:
    gold = str(record["target"]).strip().lower()
    if not gold:
        return False
    pred = text.strip().lower()
    # Prefer the last token/line as the extracted answer.
    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    candidates = list(lines) + [pred]
    return any(cand == gold for cand in candidates)


def score_tool_call(text: str, record: dict) -> bool:
    """Agentic-foundation scorer: the model must emit a valid function call.

    Accepts either a bare JSON object or a JSON object wrapped in a
    ```json ``` fenced block. Matches the gold function name and every
    required argument (gold args == a subset check on the emitted arguments).
    The 'arguments' field may be a dict or a JSON-encoded string (as some
    chat templates emit).
    """
    gold_name = record["target"]["name"]
    gold_args = record["target"].get("arguments", {}) or {}

    s = text.strip()
    # Strip a ```json ... ``` fenced block if present.
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # Fall back to the last balanced JSON object in the text.
    try:
        obj = json.loads(s)
    except Exception:
        candidates = re.findall(r"\{[^{}]*\}", s)
        obj = None
        for cand in reversed(candidates):
            try:
                obj = json.loads(cand)
                break
            except Exception:
                continue
    if not isinstance(obj, dict):
        return False
    if obj.get("name") != gold_name:
        return False
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return False
    if not isinstance(args, dict):
        return False
    # Every required gold argument must be present and non-empty.
    for k, v in gold_args.items():
        if k not in args:
            return False
        if isinstance(v, (dict, list)) and v and not args[k]:
            return False
        if isinstance(v, str) and v and not str(args[k]).strip():
            return False
    return True


_SCORERS = {
    "gsm8k": score_gsm8k,
    "mmlu": score_mmlu,
    "bbh": score_bbh,
    "tool_call": score_tool_call,
}


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = os.environ["GEMMA4_TEXT_ONLY_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    dataset_path = config.get("_agentic_eval_dataset")
    if not dataset_path:
        raise RuntimeError(
            "run_quality.py requires '_agentic_eval_dataset' (path to a JSONL "
            "eval dataset) in the config"
        )
    dataset_path = Path(dataset_path)
    # Allow a bare filename to resolve next to the benchmark datasets dir.
    if not dataset_path.is_absolute():
        dataset_path = SCRIPT_DIR / "datasets" / dataset_path.name

    records = load_dataset(dataset_path, args.prompts)
    max_tokens = config.get("max_tokens", 768)

    # Group prompts by task type so scores can be reported per task.
    rendered = [
        render_prompt(tokenizer, rec.get("type", "gsm8k"), rec) for rec in records
    ]
    prompts = truncate_prompts(
        tokenizer, rendered, config["max_model_len"], max_tokens
    )

    print(f"Quality eval model: {model_path}", flush=True)
    print(f"Quality eval records: {len(records)}", flush=True)
    print(f"Quality eval dataset: {dataset_path}", flush=True)

    llm = LLM(**build_llm_kwargs(config))
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=0
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    # Build per-task results + a parity-format dump of completions.
    per_task: dict[str, dict] = {}
    dump: list[list[list[int]]] = []
    correct = 0
    for idx, (record, output) in enumerate(zip(records, outputs)):
        task_type = record.get("type", "gsm8k")
        comp_text = output.outputs[0].text
        comp_tokens = list(output.outputs[0].token_ids)
        dump.append([comp_tokens])

        scorer = _SCORERS.get(task_type)
        ok = bool(scorer(comp_text, record)) if scorer else None
        if ok:
            correct += 1
        task = per_task.setdefault(
            task_type, {"count": 0, "correct": 0, "score": 0.0}
        )
        task["count"] += 1
        if ok:
            task["correct"] += 1

    for task in per_task.values():
        task["score"] = round(task["correct"] / task["count"], 4) if task["count"] else 0.0

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": str(dataset_path),
        "num_records": len(records),
        "overall_accuracy": round(correct / len(records), 4) if records else 0.0,
        "per_task": per_task,
        "status": "ok",
    }

    # Emit parity-format dump (consumable by detect_degeneration.py + tolerant compare).
    dump_path = Path(config.get("_agentic_eval_output_path", "/tmp/eval_dump.json"))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(dump), encoding="utf-8")

    # Emit score report. A deterministic path (config key) is used by the AML
    # integration so results can be copied back exactly; otherwise fall back to
    # a timestamped local path.
    report_cfg = config.get("_agentic_eval_report_path")
    if report_cfg:
        report_path = Path(report_cfg)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        report_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"overall_accuracy: {report['overall_accuracy']:.4f}", flush=True)
    for name, task in sorted(per_task.items()):
        print(f"  {name}: {task['correct']}/{task['count']} = {task['score']:.4f}", flush=True)
    print(f"Parity dump: {dump_path}", flush=True)
    print(f"Report: {report_path}", flush=True)

    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
