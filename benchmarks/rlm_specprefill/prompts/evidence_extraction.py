"""Repurposes RLM into a retrieval front-end: given a massive context and a
question, it searches/chunks/queries sub-LLMs to locate relevant evidence,
but does NOT synthesize a final prose answer. That's left to the separate
target-model call in target_stage/ (see IMPLEMENTATION_PLAN.md decision 1
for why this split exists -- arms A/B/C differ only in what happens to this
evidence on its way to the target, so RLM's own output must be an
intermediate artifact, not a finished answer).

EVIDENCE_SYSTEM_PROMPT is adapted from
../../rlm/rlm/utils/prompts.py's RLM_SYSTEM_PROMPT -- same REPL/tool
description (context, llm_query, rlm_query, SHOW_VARS, the `answer` dict
mechanics), same `{custom_tools_section}` placeholder (required: `RLM`
passes this through `custom_system_prompt` to
`build_rlm_system_prompt`, which calls `.format(custom_tools_section=...)`
on it -- omitting the placeholder raises `KeyError` even when no custom
tools are configured). Only the goal framing changes: "answer" is redefined
to mean "produce evidence," not "produce a final response." RLM's own
`ORCHESTRATOR_ADDENDUM` (delegation/sub-call-budget guidance) is still
appended automatically by `build_rlm_system_prompt` when `orchestrator=True`
(RLM's default) -- it's generic search/delegation guidance that applies
equally well to evidence-gathering, so it isn't duplicated here.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

# Bump when EVIDENCE_SYSTEM_PROMPT's shape changes (affects `answer["content"]`'s
# expected JSON structure) -- rlm_stage/evidence_cache.py includes this in its
# cache key so a prompt-design change invalidates stale cached evidence
# instead of silently reusing evidence gathered under different instructions.
#
# v1 -> v2 (2026-08-05): confirmed on real hardware against a self-hosted
# Qwen3-Coder-480B root (root_backend='vllm') that asking the model to
# hand-author "a JSON string" directly produces a real parse-failure rate
# (~1/3 of samples in one small run) -- sometimes it just answers directly
# with no JSON wrapper at all, sometimes it emits a Python-dict-repr string
# (single quotes, invalid JSON) with an EMPTY excerpts list, a real evidence
# loss, not just a formatting miss. RLM's own core loop (rlm/core/rlm.py +
# rlm/environments/local_repl.py's exec()) already avoids exactly this class
# of failure for its own `answer` mechanism by having the model manipulate a
# live Python object via real code rather than emit free-text that has to be
# parsed -- v2 extends that same guarantee one level deeper: instead of
# asking the model to type out JSON text by hand, it's now asked to build a
# native Python dict in the REPL and call `json.dumps(...)` on it, so
# syntactic validity comes from Python's own stdlib serializer, not from the
# model correctly imitating JSON formatting in prose.
PROMPT_VERSION = "v2"

EVIDENCE_SYSTEM_PROMPT = textwrap.dedent(
    """You are a Recursive Language Model (RLM) acting as a RETRIEVAL FRONT-END, not an answerer: a language model with a prompt, and a very important context stored in a Python REPL related to that prompt.

You are NOT trying to answer the question yourself. Your job is to locate and return the smallest set of verbatim excerpts from `context` that a downstream model would need to answer the question, with enough surrounding text that each excerpt is self-contained. Do not synthesize a final prose answer, summarize away the specifics, or guess — just find and return the evidence.

You can iteratively interact with a Python REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have gathered sufficient evidence.

To use the REPL, you need to write code in ```repl``` blocks; the REPL persists across turns. Available in the REPL:
- `context`: the important, potentially very long information related to the prompt (typically `str` or `list[str]`).
- `llm_query(prompt: str, model: str | None = None) -> str`: a single sub-LLM completion. Use for extraction, summarization, or Q&A over a chunk of text. Sub-LLM context window ≈ 500K chars.
- `llm_query_batched(prompts: list[str], model=None) -> list[str]`: concurrently call several LLM calls in parallel over a list of prompts; same order out as in.
- `rlm_query(prompt, model=None)` / `rlm_query_batched(prompts, model=None)`: recursive RLM sub-calls for evidence-gathering subtasks that themselves need multi-step search. Fall back to `llm_query` / `llm_query_batched` when recursion is disabled.
- `SHOW_VARS() -> str`: list every variable currently in the REPL.
- `answer`: dict initialized to `{{"content": "", "ready": False}}`. To submit your evidence, `import json`, build a Python dict, and serialize it with `json.dumps(...)` to set `answer["content"]` (see "Submitting your evidence" below), then `answer["ready"] = True` inside a ```repl``` block.
{custom_tools_section}

REPL outputs over ~20K characters are truncated, so for longer payloads slice `context` and pass slices through `llm_query` rather than `print`-ing them whole. The REPL is NOT a Jupyter cell — only `print(...)` output (stdout) is shown back to you between turns; a bare expression on the last line is silently discarded. Always wrap inspections in `print(...)`.

As a general strategy, start by probing your context to understand its shape (e.g. print a few lines, count them). Then use the REPL to search, chunk, and query sub-LLMs to locate the passages relevant to the question below — treat this exactly like a retrieval task, not a question-answering task.

Submitting your evidence:
When (and only when) you have located sufficient evidence, build a Python dict of this exact shape:
{{"excerpts": [{{"text": "<verbatim or lightly-trimmed excerpt>", "loc_hint": "<where in context this came from, e.g. a section heading or approximate line range>"}}], "question": "<the original question, copied verbatim>"}}
then set `answer["content"] = json.dumps(that_dict)` and `answer["ready"] = True`. Do NOT hand-type the JSON text yourself — always construct it as a real Python dict/list first and pass it through `json.dumps(...)`, so the result is guaranteed valid JSON regardless of quote characters or special characters inside the excerpt text. Keep excerpts verbatim from `context` wherever possible (do not paraphrase the evidence itself) and include enough surrounding text that each excerpt is self-contained without the rest of `context`. Prefer a small number of well-chosen excerpts over exhaustively dumping large sections — a downstream model still has to read whatever you return.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the output, then continue on the next turn. Do not flip `answer["ready"] = True` on turn 1 without first inspecting `context`.
"""
)


def build_evidence_prompt(question: str, context: str) -> str:
    """Folds question + context into RLM's single `prompt` argument.

    Deliberately does NOT use RLM's `root_prompt` argument (`RLM.completion`,
    ../../rlm/rlm/core/rlm.py:326) even though it exists for exactly this
    "give the root a small anchor prompt alongside the huge context" use
    case: `root_prompt` gets baked into `build_rlm_system_prompt`'s metadata
    message as the hardcoded literal "Answer the following: {root_prompt}"
    (../../rlm/rlm/utils/prompts.py:238), which actively fights this
    module's retrieval framing. Folding the question into `prompt` directly
    also matches the repo's own validated working example
    (../../rlm/examples/quickstart_anthropic.py), rather than relying on a
    less-exercised code path.
    """
    return (
        "RETRIEVAL TASK: gather evidence for the following question. Do not "
        f"answer it yourself — see the system prompt.\n\nQUESTION: {question}\n\n"
        f"CONTEXT:\n{context}"
    )


def parse_evidence_response(raw_response: str) -> dict[str, Any]:
    """Parses `RLMChatCompletion.response` (== `answer["content"]`) into the
    `{"excerpts": [...], "question": ...}` shape EVIDENCE_SYSTEM_PROMPT asks
    for, with a graceful fallback for malformed JSON.

    LLM-emitted JSON is not 100% reliable (truncation, stray prose before/
    after the JSON, minor syntax slips) — per IMPLEMENTATION_PLAN.md
    decision 1, a parse failure should degrade to treating the whole raw
    response as one unstructured excerpt, not crash the sample.
    """
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict) and "excerpts" in parsed:
            return {
                "excerpts": parsed.get("excerpts", []),
                "question": parsed.get("question"),
                "parse_error": False,
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "excerpts": [{"text": raw_response, "loc_hint": None}],
        "question": None,
        "parse_error": True,
    }
