#!/usr/bin/env python3
"""Offline multi-turn AGENTIC (tool-LOOP) quality evaluator for Gemma4 on A100.

Follow-on to run_quality.py's single-shot `tool_call` task. This one scores the
real agentic behavior: the model must (1) call the right tool with the right
arguments, (2) consume the tool's returned values, and (3) produce the correct
FINAL answer. A regression in any leg of the loop changes the score.

CRITICAL PATH NOTES (learned from an initial 0/10 gate):
  - Use llm.generate (NOT llm.chat + native tools=). The gemma4 tool-chat
    template in the offline llm.chat(path) does not emit parseable tool calls.
    The PROVEN path (run_quality.py tool_call, which scored 0.95/0.95) renders
    the JSON-schema tools as TEXT inside the user prompt via tokenizer
    apply_chat_template + llm.generate.
  - So we drive a two-turn loop with plain chat turns:
      turn 1 user-building-prompt (tools as text) -> model emits a JSON
            tool call which we parse;
      we inject the simulator's canned return as a follow-up *user* message
            (role=user, since gemma it has no native tool role in this path);
      turn 2 -> model reads the injected values and answers.
  - A regression in tool-selection, argument value, tool-output-reading, or
    final reasoning changes the score.

CLI mirrors run_config/run_quality: --config / --prompts / --reps. Reuses
build_llm_kwargs(); reads the dataset path from `_agentic_eval_dataset`.
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


def load_dataset(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as dataset:
        for line in dataset:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) == limit:
                break
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def _norm(s: str) -> str:
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Parse a model tool-call emission: a bare JSON object, with JSON keys name+arguments
# ---------------------------------------------------------------------------
def parse_tool_calls(text: str) -> list[dict]:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        cands = re.findall(r"\{[^{}]*\}", s)
        for c in reversed(cands):
            try:
                obj = json.loads(c)
                break
            except Exception:
                continue
    if isinstance(obj, dict) and obj.get("name"):
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return [{"name": obj["name"], "arguments": args or {}}]
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        out = []
        for tc in obj["tool_calls"]:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name:
                out.append({"name": name, "arguments": args or {}})
        return out
    if isinstance(obj, list):
        out = []
        for tc in obj:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name:
                out.append({"name": name, "arguments": args or {}})
        return out
    return []


# ---------------------------------------------------------------------------
# Text-render the tool schemas into the user prompt (proven run_quality.py style).
# ---------------------------------------------------------------------------
def render_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        name = t["name"]
        desc = t.get("description", "")
        params = json.dumps(t.get("parameters", {}), ensure_ascii=False)
        lines.append(f"- {name}: {desc}  arguments: {params}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Arg-aware simulator: simulator[tool] = {json(args_subset): {output: str}}
# ---------------------------------------------------------------------------
def sim_output(sim: dict, args: dict) -> str:
    all_nested = all(isinstance(v, dict) and "output" in v for v in sim.values())
    if all_nested:
        for key, slot in sim.items():
            try:
                want = json.loads(key)
            except Exception:
                continue
            if all(args.get(k) == v for k, v in want.items()):
                return slot["output"]
        return json.dumps({"error": "no sim slot for args"}, ensure_ascii=False)
    return sim.get("output", "")


def render_prompt(tokenizer, record: dict, stage: str, injected: str | None = None) -> str:
    """Build a chat turn. stage='ask' -> first turn; stage='answer' -> injected tool results."""
    if stage == "ask":
        tool_text = render_tools(record["tools"])
        text = (
            "You are an agent that can call functions to answer the user's request. "
            "Available functions and their JSON-schema arguments:\n"
            f"{tool_text}\n\n"
            "First, decide the function call(s) needed and output EXACTLY one JSON "
            "object with keys 'name' (function name) and 'arguments' (the arguments "
            f"object). Call it now, nothing else:\n\n"
            f"User request:\n{record['input']}"
        )
    else:  # answer stage: we present the tool's returned values, model must answer
        text = (
            "Here are the results returned by the function call(s) you made. "
            "Read them carefully, do any needed arithmetic/filtering, and answer "
            "the original user request with ONLY the final answer.\n\n"
            f"Function results:\n{injected}\n\n"
            f"Original user request:\n{record['input']}"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def run_tool_loop(llm, tokenizer, sampling, record) -> dict:
    sim = record.get("simulator", {})

    # Turn 1: emit tool calls (text-rendered tools)
    p1 = render_prompt(tokenizer, record, "ask")
    out1 = llm.generate([p1], sampling, use_tqdm=False)[0]
    round1_text = out1.outputs[0].text.strip()
    calls = parse_tool_calls(round1_text)
    called = bool(calls)
    call_names = [c["name"] for c in calls]

    # Execute simulator for each valid call (dedupe same-name calls, keep outputs)
    results = []
    for c in calls:
        if c["name"] in sim:
            results.append(sim_output(sim[c["name"]], c.get("arguments", {})))
        else:
            results.append(f"ERROR: unknown function {c['name']}")
    injected = "\n".join(results) if results else "(no calls were recognized)"

    # Turn 2: model reads results and gives the final answer
    p2 = render_prompt(tokenizer, record, "answer", injected)
    out2 = llm.generate([p2], sampling, use_tqdm=False)[0]
    final_text = out2.outputs[0].text.strip()

    gold = _norm(record["target"]["answer"])
    lines = [ln.strip() for ln in final_text.splitlines() if ln.strip()]
    correct = bool(called) and (any(_norm(ln) == gold for ln in lines) or _norm(final_text) == gold)

    return {
        "called": called,
        "call_names": call_names,
        "correct": correct,
        "round1": round1_text,
        "injected": injected,
        "final": final_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = os.environ["GEMMA4_TEXT_ONLY_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    dataset_path = config.get("_agentic_eval_dataset")
    if not dataset_path:
        raise RuntimeError("run_quality_agentic.py requires '_agentic_eval_dataset'")
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = SCRIPT_DIR / "datasets" / dataset_path.name
    records = load_dataset(dataset_path, args.prompts)

    max_tokens = config.get("max_tokens", 768)
    print(f"Agentic tool-loop model: {model_path}", flush=True)
    print(f"Agentic tool-loop records: {len(records)}", flush=True)
    print(f"Agentic tool-loop dataset: {dataset_path}", flush=True)

    prompts = truncate_prompts(tokenizer, [record["input"] for record in records], config["max_model_len"], max_tokens)
    llm = LLM(**build_llm_kwargs(config))
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=0)

    correct = 0
    details = []
    for idx, record in enumerate(records):
        res = run_tool_loop(llm, tokenizer, sampling, record)
        if res["correct"]:
            correct += 1
        details.append({"idx": idx, **res})
        print(f"  [{idx}] called={res['called']} names={res['call_names']} correct={res['correct']}", flush=True)

    per_task = {"tool_loop": {"count": len(records), "correct": correct, "score": round(correct / len(records), 4)}}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": str(dataset_path),
        "num_records": len(records),
        "overall_accuracy": round(correct / len(records), 4) if records else 0.0,
        "per_task": per_task,
        "status": "ok",
    }

    report_cfg = config.get("_agentic_eval_report_path")
    if report_cfg:
        report_path = Path(report_cfg)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        report_path = SCRIPT_DIR / "eval_results" / f"agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"overall_accuracy: {report['overall_accuracy']:.4f}", flush=True)
    print(f"tool_loop: {per_task['tool_loop']['correct']}/{per_task['tool_loop']['count']} = {per_task['tool_loop']['score']:.4f}", flush=True)
    print(f"Report: {report_path}", flush=True)

    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
