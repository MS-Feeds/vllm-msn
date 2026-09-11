#!/usr/bin/env python3
"""Builds synthetic multi-turn conversations from LongBench v2 single-document
QA, in the exact sample schema `predict_scbench.py` already consumes -- ONE
DOCUMENT PER TURN, rather than SCBench's one shared context plus tiny per-turn
queries.

## Why this dataset exists

Measured on `SPARSE-k20-g32-masked` vs `M000` (Gemma-4-31B TP=2 / Gemma-4-E2B,
`out_len=512`, steady-state turns), the sparse mechanism's FLOP savings land in
the stage where FLOPs do not convert to time:

    target_decode    -24.882 TFLOP  ->  ~0 seconds   (weight-streaming bound)
    target_prefill    -5.064 TFLOP  ->  ~-0.15 s     (compute bound -- converts)
    speculator        +9.036 TFLOP  ->  +0.846 s

Backing `target_prefill = 9.236 TFLOP` out at 131.2 GFLOP/token puts SCBench's
steady-state prompt delta at `d ~ 70` tokens against `o = 512` -- a d:o ratio of
about 1:7, so almost all of the saving lands in decode and is thrown away. This
dataset inverts that ratio (`d ~ 25k`, d:o ~ 49:1) so the saving lands in
prefill, and asks the one question that follows: DOES THE PREFILL SAVING
CONVERT TO SECONDS AT A REALISTIC `d`?

It changes only the data. Nothing in `predict_scbench.py` or `vllm_patch/`
branches on dataset identity -- the sole per-dataset dispatch in the pipeline is
`grade_scbench.py`'s `_METRIC_BY_CONFIG`, which gains one entry
(`longbench_v2_mc`) for multiple-choice letter scoring.

## Mapping onto the existing schema

`predict_scbench.py` reads `{id, config, context, turns[{input, answer}]}` and
nothing else, so:

  - `context`  = a short shared PREAMBLE (~60 tokens), not a document. Non-empty
    on purpose: it keeps `ConversationState`'s turn-0 candidate pool non-empty
    (see conversation_state.py's ledger construction) and it makes all T turns
    structurally identical, since `build_turn_delta_ids` renders turn 0 as
    `chat_before + context + query + chat_after` and turn N as
    `turn_boundary + query + chat_after` -- the same shape, differing by the
    preamble alone. That symmetry is what lets `*_mean` and `*_excl_turn0_mean`
    be read against each other here, unlike on SCBench where turn 0 carries the
    entire context.
  - `turns[i]["input"]` = one complete LongBench v2 prompt block (instruction +
    document + question + 4 lettered choices), rendered by the SINGLE-TURN
    pipeline's own `_PROMPT_TEMPLATE` so the block is byte-identical to what
    `../spec_prefill_llama/predict_longbench_v2.py` submits.
  - `turns[i]["answer"]` = "A".."D".
  - `id` = `lbv2mt-NNNN`. This is the `conversation_salt`: vLLM's prefix-cache
    salt AND the speculator worker's slot-history key (proposer.py's
    `"{salt}::turn{n}"` convention). The prefix guarantees no collision with
    SCBench's `scbench_<config>-<n>` ids even if both sample files are
    concatenated.

## Filter, never truncate

Documents are kept whole. A document too large to be one turn of a conversation
is DROPPED, not cut down. Packing one huge document with four small ones would
give a single conversation an 8x spread in per-turn `d`, and every headline
column (`seconds_per_turn_excl_turn0_mean` and its per-stage twins) is a MEAN
OVER TURNS -- that variance would swamp the effect being measured. Grouping is
therefore size-banded: documents of similar length share a conversation.

## Token accounting is borrowed, not reimplemented

`render_turn_query`, `chat_wrapper_pieces` and `chat_turn_boundary_pieces` are
imported from `predict_scbench.py` rather than reproduced here. That is
load-bearing: `render_turn_query` is the only place the
`"\\n\\nQuestion {n}: ...\\nAnswer {n}:"` framing exists, so this packer's token
counts ARE the driver's token counts by construction. A copied format string
would drift silently, and the failure mode it would cause -- the driver's
pre-flight firing mid-conversation -- `break`s out of the turn loop from that
turn onward, changing `num_turns` and de-comparabilising the row with nothing
but `num_skipped_too_large` to show for it.

Importing `predict_scbench` pulls in `flops_model`/`timing_model` (both
dependency-free) and no vLLM. It does have one module-scope side effect:
`OUT_DIR.mkdir(exist_ok=True)`, i.e. it creates `results/` relative to the
current working directory.

## Long-document variant: 3 turns of ~40-43k tokens

`--min-doc-tokens` adds a FLOOR to the existing ceiling, so a conversation can
be built from a narrow band of long documents instead of everything that fits.
The intended configuration is 3 turns per conversation, each turn one LongBench
v2 document of 40,000-42,981 rendered tokens:

  - **Why ~43k and not 45k.** The ceiling is the SCORER, not the target. Both
    scorers in use (Gemma-4-E2B, Llama-3.2-1B) have a 131,072-token native
    context, and the scorer must see the WHOLE conversation to score it --
    `Budget.spec_check` is `C + sum(Q) + (T-1)*O`. At 3 turns that caps a
    turn at ~42,991 (scorer) / ~42,981 (target at 131,072), and 45,000/turn
    would need 137,097 tokens. Gemma-4-31B's own 262,144-token window does not
    help, because the scorer is the binding side. (An `EARLY-*` scorer, which
    is the target's own first N layers, inherits 262,144 and would lift this.)

  - **What it buys.** A higher d:o ratio (~84:1 against the 5-turn file's
    ~49:1), so decode -- and with it the output-length confound that dominated
    every LongBench timing comparison so far -- is a smaller share of each
    turn. And each question is posed over a single long document.

  - **What it costs -- read this before choosing it for a SPEED measurement.**
    Sparse prefill can only remove attention to PRIOR turns. A turn's own
    tokens are the contiguous force-kept tail and are always attended densely,
    and turn 0 has no prior at all. With turn `i` of length `d` attending
    `i*d` prior tokens, the removable share of prefill attention is
    `(T-1)/T` -- 80% at 5 turns, 67% at 3 -- and at a FIXED total conversation
    length `D` the removable attention is `(T-1)/(2T) * D^2`: 0.400 D^2 at 5
    turns, 0.333 D^2 at 3. **Fewer, longer turns give sparse prefill ~17% LESS
    to save at the same length.** The dataset is well suited to studying long
    single-document questions and to shrinking the decode confound; it is a
    harder test for the speed claim, not an easier one.

  - **Single documents can't fill it -- use `--pack-distractors`.** Measured
    on Gemma-4-31B's tokenizer over `short,medium`: 260 of 395 documents are
    longer than the ~43k ceiling (the median is 79.6k), and only 6 fall in
    [40,000, 42,981] -- 2 conversations. `--pack-distractors` instead builds
    each turn from ONE question and its document plus other questions'
    documents as distractors, packed into the band. Any document short enough
    to take at least one distractor can be a question, so yield is set by the
    ~100 documents under ~32k rather than the 6 inside the band. It changes the
    task to finding the relevant document among several; see `pack_turn`.

  - **Per-model token counts select different documents.** Lengths are
    measured with `--tokenizer`, which must be the TARGET's. Llama-3's
    tokenizer is less efficient on English than Gemma's, so a Llama prep of the
    same band picks a different, smaller set of documents. Gemma and Llama rows
    built this way do not answer the same questions.

`--id-prefix` keeps this file's conversation ids distinct from the 5-turn
file's. The id is the prefix-cache salt, the speculator's slot-history key and
the grader's join key, so two sample files sharing `lbv2mt-0000` would let a
predictions file grade silently against the wrong questions.

Usage (the original 5-turn file):
    python3 datasets/prep_longbench_v2_multiturn.py \\
        --tokenizer $GEMMA4_31B_MODEL_PATH \\
        --target-max-num-batched-tokens 130560 \\
        --speculator-max-num-batched-tokens 131063 \\
        --max-tokens 512 --seed 42 \\
        --output datasets/longbench_v2_multiturn.jsonl

Usage (3 turns x 40-43k, packed with distractors):
    python3 datasets/prep_longbench_v2_multiturn.py \\
        --tokenizer $GEMMA4_31B_MODEL_PATH \\
        --target-max-num-batched-tokens 131072 \\
        --speculator-max-num-batched-tokens 131063 \\
        --max-tokens 512 --turns-per-conv 3 \\
        --lengths short,medium --min-doc-tokens 40000 \\
        --pack-distractors --id-prefix lbv2mt3p --seed 42 \\
        --output datasets/longbench_v2_multiturn_3x40k_packed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
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
from vllm_patch.model_structure import load_tokenizer  # noqa: E402

#: The single-turn LongBench v2 prep, loaded by explicit file path rather than
#: by `sys.path` + `import prep_longbench_v2`. Two prep modules share that base
#: name across sibling benchmark dirs (`spec_prefill/`, `spec_prefill_llama/`,
#: `spec_prefill_qwen/`, ...), so an import-by-name would resolve to whichever
#: happened to be on the path first -- and they differ in CHUNK_SIZE and prompt
#: details. Addressing the file directly makes the provenance unambiguous.
_SIBLING_PREP = (
    _PKG_ROOT.parent / "spec_prefill_llama" / "datasets" / "prep_longbench_v2.py"
)

#: Reserved for the speculator's lookahead tail; mirrors `LOOK_AHEAD_CNT` in
#: `predict_scbench.py`, which sets
#: `speculator_max_model_len = spec_budget + 1 + LOOK_AHEAD_CNT`.
LOOK_AHEAD_CNT = 8

DEFAULT_OUTPUT = _HERE / "longbench_v2_multiturn.jsonl"
DEFAULT_CACHE_DIR = _HERE / ".cache"
DEFAULT_CONFIG_NAME = "longbench_v2_mc"

DEFAULT_PREAMBLE = (
    "You will be shown a series of documents. After each document there is a "
    "multiple-choice question about that document. Answer each question with "
    "a single letter (A, B, C, or D)."
)


def _load_sibling_prep():
    """Imports `../spec_prefill_llama/datasets/prep_longbench_v2.py` by path."""
    if not _SIBLING_PREP.is_file():
        raise FileNotFoundError(
            f"Expected the single-turn LongBench v2 prep at {_SIBLING_PREP}, "
            "which supplies the prompt template and the HF loader this script "
            "reuses. Without it the per-document block would have to be "
            "re-specified here and could drift from the single-turn pipeline's."
        )
    spec = importlib.util.spec_from_file_location(
        "_lbv2_singleturn_prep", _SIBLING_PREP
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------


class Budget:
    """The driver's own pre-flight arithmetic, evaluated ahead of time.

    `predict_scbench.py::run_sparse_attention` checks two independent budgets
    before each turn and `break`s out of the turn loop if either fails. At the
    last turn `N = T-1` of a T-turn conversation, with `Wb/Wa/Wt` the chat
    wrapper pieces, `C` the preamble, `Q_s` turn `s`'s rendered query and `O`
    the per-turn generation cap:

        target:      Wb + C + sum(Q) + T*Wa + (T-1)*Wt + (T-1)*O  <= target_budget
        speculator:  C + sum(Q) + (T-1)*O                         <= spec_budget

    The `(T-1)*O` terms are the generated output that STAYS RESIDENT in the
    resumable session (`target_resident_len += len(delta_ids) +
    len(actual_output_ids)`). SCBench's `d ~ 70` kept sessions far enough from
    the ceiling that this term never mattered; at `d ~ 25k` it is the term that
    decides whether a conversation fits.

    `doc_budget` deliberately reserves `T*O` rather than `(T-1)*O`, leaving one
    full generation of slack so the separate `max_model_len` ceiling
    (`target_budget + max_tokens`, clamped to the checkpoint's native context)
    is discharged too.
    """

    def __init__(self, *, target_budget, spec_budget, turns_per_conv,
                 max_tokens, safety_tokens, wrapper_before, wrapper_after,
                 turn_boundary, preamble_len):
        self.target_budget = target_budget
        self.spec_budget = spec_budget
        self.turns = turns_per_conv
        self.max_tokens = max_tokens
        self.safety = safety_tokens
        self.wb = wrapper_before
        self.wa = wrapper_after
        self.wt = turn_boundary
        self.preamble = preamble_len

        t, o = turns_per_conv, max_tokens
        self.target_overhead = self.wb + self.preamble + t * self.wa + (t - 1) * self.wt
        self.spec_overhead = self.preamble
        # `t * o` (not `(t-1) * o`) -- see class docstring.
        self.target_doc_budget = target_budget - self.target_overhead - t * o - safety_tokens
        self.spec_doc_budget = spec_budget - self.spec_overhead - t * o - safety_tokens
        self.doc_budget = min(self.target_doc_budget, self.spec_doc_budget)

    @property
    def binding_side(self) -> str:
        return "speculator" if self.spec_doc_budget <= self.target_doc_budget else "target"

    def target_check(self, query_lens: list[int]) -> int:
        t, o = self.turns, self.max_tokens
        return (self.wb + self.preamble + sum(query_lens)
                + t * self.wa + (t - 1) * self.wt + (t - 1) * o)

    def spec_check(self, query_lens: list[int]) -> int:
        return self.preamble + sum(query_lens) + (self.turns - 1) * self.max_tokens

    def fits(self, query_lens: list[int]) -> bool:
        return (self.target_check(query_lens) + self.max_tokens <= self.target_budget
                and self.spec_check(query_lens) + 1 + LOOK_AHEAD_CNT <= self.spec_budget)

    def describe(self) -> str:
        return (
            f"  target_budget          {self.target_budget:>9,}\n"
            f"  speculator_budget      {self.spec_budget:>9,}\n"
            f"  chat wrapper overhead  {self.target_overhead - self.preamble:>9,}"
            f"  (Wb={self.wb} Wa={self.wa}x{self.turns} Wt={self.wt}x{self.turns - 1})\n"
            f"  preamble               {self.preamble:>9,}\n"
            f"  resident output        {self.turns * self.max_tokens:>9,}"
            f"  ({self.turns} x {self.max_tokens})\n"
            f"  safety                 {self.safety:>9,}\n"
            f"  -> doc budget          {self.doc_budget:>9,}  (binding side: {self.binding_side})"
        )


# ---------------------------------------------------------------------------
# Document rendering / measurement
# ---------------------------------------------------------------------------


def build_documents(rows, prep, tok, lengths, verbose_every=200) -> list[dict]:
    """Renders each eligible LongBench v2 row into a turn `input` block and
    measures its rendered query length with the driver's own renderer.

    Row eligibility mirrors the single-turn prep's own predicate (non-empty
    question/context, four non-empty choices, answer in A-D) so the two
    pipelines draw from the same population.
    """
    docs: list[dict] = []
    skipped_length = 0
    skipped_bad = 0

    for idx, row in enumerate(rows):
        if row.get("length") not in lengths:
            skipped_length += 1
            continue

        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        choices = [
            (row.get(f"choice_{letter}") or "").strip()
            for letter in prep.CHOICE_LETTERS
        ]
        answer = (row.get("answer") or "").strip().upper()
        if not question or not context or not all(choices) or answer not in prep.CHOICE_LETTERS:
            skipped_bad += 1
            continue

        block = prep._PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            choice_A=choices[0],
            choice_B=choices[1],
            choice_C=choices[2],
            choice_D=choices[3],
        )
        # Measured at slot 0. The slot index only changes the digit in
        # "Question N:", worth 0-1 tokens; the exact per-slot lengths are
        # recomputed in `verify_conversations` before anything is emitted.
        q_tokens = len(render_turn_query(tok, 0, {"input": block}))

        docs.append(
            {
                "doc_id": str(row.get("_id")),
                # Raw fields, kept in memory for `--pack-distractors`, which
                # re-renders a turn around several documents. Never emitted:
                # `build_rows` copies an explicit field list, so the 5-turn
                # file's output is byte-identical with or without these.
                "context": context,
                "question": question,
                # Identity by CONTENT, not row id: two LongBench v2 questions
                # can share a document, and packing must never hand a target
                # its own document as a "distractor", nor show one document
                # twice within a conversation.
                "doc_key": hashlib.sha1(context.encode("utf-8")).hexdigest()[:16],
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "input": block,
                "answer": answer,
                "options": choices,
                "query_tokens": q_tokens,
            }
        )
        if verbose_every and len(docs) % verbose_every == 0:
            print(f"[prep_lbv2_mt]   tokenized {len(docs)} documents...", flush=True)

    print(
        f"[prep_lbv2_mt] documents: kept={len(docs)} "
        f"skipped_length_bucket={skipped_length} skipped_bad_row={skipped_bad}",
        flush=True,
    )
    return docs


def _histogram(values: list[int], label: str) -> str:
    if not values:
        return f"  {label}: (none)"
    ordered = sorted(values)
    deciles = [ordered[min(len(ordered) - 1, (len(ordered) * d) // 10)] for d in range(10)]
    return (
        f"  {label}: n={len(ordered)} min={ordered[0]:,} "
        f"median={statistics.median(ordered):,.0f} max={ordered[-1]:,}\n"
        f"    deciles: {', '.join(f'{v:,}' for v in deciles)}"
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def select_length_band(docs, min_doc_tokens: int, max_doc_tokens: int):
    """Split `docs` into those at or above the floor, and a count of those
    below it. Returns `(in_band, dropped_too_small)`.

    Only the FLOOR is applied here. The ceiling stays where it always was, in
    `group_into_conversations`, so the existing drop accounting and the
    size-banded windowing are untouched; a document above the ceiling is still
    reported as `dropped_doc_too_large` by that function.

    Raises if the band is empty by construction. `max_doc_tokens` is derived
    from the budget (`doc_budget // turns_per_conv`), so asking for a floor
    above it -- e.g. `--min-doc-tokens 45000` at 3 turns, where the scorer's
    131,072-token context caps a turn near 43k -- would otherwise surface as
    "no conversations formed", which blames the documents for what is
    actually a budget contradiction.

    A floor of 0 (the default) keeps every document, which is what makes the
    5-turn file reproduce exactly.
    """
    if min_doc_tokens > max_doc_tokens:
        raise ValueError(
            f"--min-doc-tokens {min_doc_tokens:,} is above the per-turn ceiling "
            f"{max_doc_tokens:,} (= document budget // --turns-per-conv), so no "
            f"document can be in band. The ceiling is set by whichever engine "
            f"budget binds -- usually the scorer's native context, which must "
            f"hold the whole conversation. Lower the floor, use fewer turns, "
            f"or lower --max-tokens."
        )
    in_band = [d for d in docs if d["query_tokens"] >= min_doc_tokens]
    return in_band, len(docs) - len(in_band)


def group_into_conversations(docs, budget: Budget, max_doc_tokens, rng,
                             turns_per_conv) -> tuple[list[list[dict]], int]:
    """Size-banded grouping: documents of similar length share a conversation.

    The seeded shuffle before the sort is not redundant with it. LongBench v2's
    own row order correlates with document length (code-repository rows run
    long), so sorting a dataset-ordered list would produce domain-pure bands and
    confound any conversation-level result with domain. Shuffling first, then
    sorting by length with an explicit `doc_id` tiebreak, gives bands that are
    length-homogeneous but domain-mixed, and is fully deterministic.
    """
    eligible = [d for d in docs if d["query_tokens"] <= max_doc_tokens]
    dropped_too_large = len(docs) - len(eligible)

    rng.shuffle(eligible)
    eligible.sort(key=lambda d: (-d["query_tokens"], d["doc_id"]))

    conversations: list[list[dict]] = []
    i = 0
    while i + turns_per_conv <= len(eligible):
        window = eligible[i:i + turns_per_conv]
        if sum(d["query_tokens"] for d in window) > budget.doc_budget:
            # The window's largest member is its first element (descending
            # sort), and it is the only one that can be the cause. Drop it and
            # re-form the window one position along -- O(n) overall, no
            # backtracking.
            i += 1
            continue
        # Order within a conversation is randomized so the largest document is
        # not systematically at turn 0, which would correlate turn index with
        # `d` and make the per-turn-index breakdowns unreadable.
        window = list(window)
        rng.shuffle(window)
        conversations.append(window)
        i += turns_per_conv

    return conversations, dropped_too_large


#: How packed documents are delimited inside the template's single
#: `{context}` slot. Numbered so a question can be answered by locating "the
#: relevant document" without the prompt ever saying which one it is.
PACK_DOC_HEADER = "Document {n}:\n"
PACK_DOC_JOINER = "\n\n"

#: Fields `--pack-distractors` adds to a turn. Emitted only when present, so
#: the unpacked path's rows gain no keys -- see `build_rows`.
PACK_FIELDS = ("num_docs", "target_doc_position", "distractor_doc_ids")


def render_packed_block(template, choice_letters, target, docs_in_order) -> str:
    """One turn's prompt block: `target`'s question and choices, posed over
    every document in `docs_in_order` (which includes the target's own)."""
    context = PACK_DOC_JOINER.join(
        PACK_DOC_HEADER.format(n=i + 1) + d["context"]
        for i, d in enumerate(docs_in_order)
    )
    choices = {f"choice_{letter}": text
               for letter, text in zip(choice_letters, target["options"])}
    return template.format(context=context, question=target["question"], **choices)


def pack_turn(target, pool, used_keys, floor, ceiling, rng, measure, render,
              sep_tokens, max_attempts=25):
    """Pack `target` with at least one distractor document so the rendered
    turn lands in `[floor, ceiling]`. Returns a dict, or None if it can't.

    **Why this exists.** LongBench v2 cannot supply 3-turn conversations of
    long single documents: the per-turn ceiling is pinned near 43k by the
    SCORER's 131,072-token context (it must hold the whole conversation), and
    260 of the 395 short+medium documents are longer than that. A 40k floor
    left 6 documents -- 2 conversations. Packing decouples turn length from
    document supply: any document under the ceiling can be a question, and
    other questions' documents fill the turn to length.

    **Always at least one distractor**, so every turn poses the same task --
    find the relevant document among several -- rather than mixing single-
    and multi-document turns, which would add per-turn variance to accuracy.
    A target too long to admit any distractor is dropped, not asked alone.

    Distractor choice is estimate-then-verify. `context_tokens` sums are
    close to, but not exactly, the rendered length (headers, boundary merges),
    so each candidate is rendered and re-measured with `measure` -- the
    driver's own renderer, same as every other length in this file -- and
    retried if it misses the band.

    The target's position among the documents is drawn uniformly and recorded,
    so position effects ("lost in the middle") can be checked afterwards
    rather than silently biasing the result.

    `measure` and `render` are injected so the policy is testable without a
    tokenizer.
    """
    base = target["query_tokens"] + sep_tokens      # target gets a header too
    gap_min, gap_max = floor - base, ceiling - base
    candidates = [
        d for d in pool
        if d["doc_key"] not in used_keys
        and d["doc_key"] != target["doc_key"]
        and d["context_tokens"] + sep_tokens <= gap_max
    ]
    if not candidates:
        return None

    for _ in range(max_attempts):
        rng.shuffle(candidates)
        chosen, total = [], 0
        for d in candidates:
            cost = d["context_tokens"] + sep_tokens
            if total + cost > gap_max:
                continue
            chosen.append(d)
            total += cost
            if total >= gap_min:
                break
        if not chosen or total < gap_min:
            continue
        position = rng.randrange(len(chosen) + 1)
        order = chosen[:position] + [target] + chosen[position:]
        block = render(target, order)
        n = measure(block)
        if floor <= n <= ceiling:
            return {"input": block, "query_tokens": n, "distractors": chosen,
                    "position": position, "num_docs": len(order)}
    return None


def group_packed_conversations(docs, budget: Budget, floor, ceiling, rng,
                               turns_per_conv, measure, render, sep_tokens):
    """Build `turns_per_conv`-turn conversations of packed turns. Returns
    `(conversations, dropped_unpackable)`.

    Two reuse rules, and they differ on purpose:

      - **Each question is asked at most once** across the whole file, so no
        answer is graded twice.
      - **Distractors may be reused ACROSS conversations but never WITHIN
        one.** Each conversation is its own persistent session; a document
        that appeared earlier in the SAME session would already be resident
        in its KV cache, so reusing it -- as a later distractor or as a later
        turn's own document -- would leak. Across sessions there is nothing to
        leak into.

    Size banding, which `group_into_conversations` needs to keep per-turn `d`
    homogeneous, is unnecessary here: every packed turn is already inside a
    ~3k-token band by construction.
    """
    queue = list(docs)
    rng.shuffle(queue)
    conversations: list[list[dict]] = []
    dropped_unpackable = 0

    while len(queue) >= turns_per_conv:
        used: set = set()
        turns: list[dict] = []
        i = 0
        while len(turns) < turns_per_conv and i < len(queue):
            target = queue[i]
            if target["doc_key"] in used:
                i += 1              # leave it for a later conversation
                continue
            packed = pack_turn(target, docs, used, floor, ceiling, rng,
                               measure, render, sep_tokens)
            queue.pop(i)            # asked at most once, packed or not
            if packed is None:
                dropped_unpackable += 1
                continue
            used.add(target["doc_key"])
            used.update(d["doc_key"] for d in packed["distractors"])
            turn = dict(target)
            turn["input"] = packed["input"]
            turn["query_tokens"] = packed["query_tokens"]
            turn["num_docs"] = packed["num_docs"]
            turn["target_doc_position"] = packed["position"]
            turn["distractor_doc_ids"] = [d["doc_id"] for d in packed["distractors"]]
            turns.append(turn)
        if len(turns) < turns_per_conv:
            break                   # the queue can no longer fill a conversation
        if budget.fits([t["query_tokens"] for t in turns]):
            conversations.append(turns)

    return conversations, dropped_unpackable


def verify_conversations(conversations, budget: Budget, tok) -> tuple[list[list[dict]], int]:
    """Re-measures every turn at its TRUE slot index and applies the driver's
    exact pre-flight formula, dropping any conversation that would fail.

    This is what guarantees `predict_scbench.py`'s own budget check never fires
    mid-run. It matters more than it looks: that check does not skip a
    conversation, it `break`s out of the turn loop from the failing turn onward,
    so the row is still written -- with fewer turns, a different conversation
    mix, and nothing but `num_skipped_too_large` to indicate it.
    """
    kept: list[list[dict]] = []
    dropped = 0
    for conv in conversations:
        exact = []
        for slot, doc in enumerate(conv):
            n = len(render_turn_query(tok, slot, {"input": doc["input"]}))
            doc = dict(doc)
            doc["query_tokens"] = n
            exact.append(doc)
        if budget.fits([d["query_tokens"] for d in exact]):
            kept.append(exact)
        else:
            dropped += 1
    return kept, dropped


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def build_rows(conversations, budget: Budget, config_name, preamble,
               id_prefix: str = "lbv2mt") -> list[dict]:
    rows = []
    for n, conv in enumerate(conversations):
        query_lens = [d["query_tokens"] for d in conv]
        rows.append(
            {
                "id": f"{id_prefix}-{n:04d}",
                "config": config_name,
                "context": preamble,
                "turns": [
                    {
                        "input": d["input"],
                        "answer": d["answer"],
                        "options": d["options"],
                        "doc_id": d["doc_id"],
                        "domain": d["domain"],
                        "sub_domain": d["sub_domain"],
                        "difficulty": d["difficulty"],
                        "length": d["length"],
                        "query_tokens": d["query_tokens"],
                        # Packing metadata, only on packed turns. Adding the
                        # keys unconditionally would change every row of the
                        # 5-turn file.
                        **{k: d[k] for k in PACK_FIELDS if k in d},
                    }
                    for d in conv
                ],
                # Inert to both `predict_scbench.py` (reads id/config/context/
                # turns[].input/turns[].answer only) and `grade_scbench.py`.
                # Recorded so a later run at a different --max-tokens is
                # caught by inspecting the file rather than by a silent
                # mid-run turn-loop break.
                "total_prompt_tokens": sum(query_lens),
                "resident_len_at_last_turn": budget.target_check(query_lens),
                "reserved_output_tokens_per_turn": budget.max_tokens,
                "target_budget": budget.target_budget,
                "speculator_budget": budget.spec_budget,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one-document-per-turn LongBench v2 conversations "
                    "in predict_scbench.py's sample schema."
    )
    parser.add_argument("--tokenizer", required=True,
                        help="Target model path -- token counts must be the "
                             "target's, e.g. $GEMMA4_31B_MODEL_PATH.")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME,
                        help="Value of the `config` field; must match the key "
                             "added to grade_scbench.py's _METRIC_BY_CONFIG.")
    parser.add_argument("--lengths", default="short",
                        help="Comma-separated LongBench v2 `length` buckets.")
    parser.add_argument("--turns-per-conv", type=int, default=5,
                        help="Matches SCBench's uniform 5 turns per conversation.")
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=130560,
                        help="MUST equal the value passed to predict_scbench.py. "
                             "Set it to native_target - --max-tokens.")
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131063,
                        help="MUST equal the value passed to predict_scbench.py. "
                             f"Set it to native_spec - {1 + LOOK_AHEAD_CNT}.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Per-turn generation cap. This dataset is SIZED "
                             "for this value; a later run at a larger one will "
                             "overflow the session.")
    parser.add_argument("--doc-budget-tokens", type=int, default=-1,
                        help="-1 derives it from the budgets above.")
    parser.add_argument("--max-doc-tokens", type=int, default=-1,
                        help="-1 derives it as doc_budget // turns-per-conv. "
                             "Documents above this are DROPPED, never truncated.")
    parser.add_argument("--safety-tokens", type=int, default=512)
    parser.add_argument("--min-doc-tokens", type=int, default=0,
                        help="Documents BELOW this rendered length are dropped, "
                             "so every turn is drawn from a band [floor, "
                             "ceiling]. 0 (default) keeps everything and "
                             "reproduces the 5-turn file exactly. For the "
                             "3-turn long-document variant use 40000 with "
                             "--lengths short,medium; see the module docstring "
                             "for why the ceiling there is ~43k, not 45k.")
    parser.add_argument("--pack-distractors", action="store_true",
                        help="Build every turn from ONE LongBench v2 question "
                             "and its document, plus other questions' "
                             "documents as distractors, packed until the turn "
                             "lands in [--min-doc-tokens, ceiling]. Use when "
                             "too few single documents fall in the band -- on "
                             "3 turns x 40k, single documents gave 2 "
                             "conversations. Changes the task to finding the "
                             "relevant document among several; accuracy is not "
                             "comparable to standard LongBench v2.")
    parser.add_argument("--id-prefix", default="lbv2mt",
                        help="Conversation id prefix. Give each sample file its "
                             "own: the id is the grader's join key, so two "
                             "files sharing ids can grade a predictions file "
                             "against the wrong questions without error.")
    parser.add_argument("--max-conversations", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preamble", default=DEFAULT_PREAMBLE)
    args = parser.parse_args()

    prep = _load_sibling_prep()
    lengths = {s.strip() for s in args.lengths.split(",") if s.strip()}

    print(f"[prep_lbv2_mt] loading tokenizer from {args.tokenizer}", flush=True)
    tok = load_tokenizer(args.tokenizer)

    wrapper_before, wrapper_after = chat_wrapper_pieces(tok)
    turn_boundary = chat_turn_boundary_pieces(tok)
    budget = Budget(
        target_budget=args.target_max_num_batched_tokens,
        spec_budget=args.speculator_max_num_batched_tokens,
        turns_per_conv=args.turns_per_conv,
        max_tokens=args.max_tokens,
        safety_tokens=args.safety_tokens,
        wrapper_before=len(tok.encode(wrapper_before, add_special_tokens=False)),
        wrapper_after=len(tok.encode(wrapper_after, add_special_tokens=False)),
        turn_boundary=len(tok.encode(turn_boundary, add_special_tokens=False)),
        preamble_len=len(tok.encode(args.preamble, add_special_tokens=False)),
    )
    if args.doc_budget_tokens >= 0:
        budget.doc_budget = args.doc_budget_tokens
    max_doc_tokens = (
        args.max_doc_tokens if args.max_doc_tokens >= 0
        else budget.doc_budget // args.turns_per_conv
    )

    print("[prep_lbv2_mt] budget:")
    print(budget.describe())
    print(f"  -> max doc tokens      {max_doc_tokens:>9,}")
    if budget.doc_budget <= 0:
        print("[prep_lbv2_mt] ERROR: non-positive document budget -- the "
              "wrapper/output/safety reservations already exceed the engine "
              "budgets. Check the two --*-max-num-batched-tokens values "
              "against the checkpoints' native context lengths.", file=sys.stderr)
        return 2

    rows = prep.load_longbench_v2_rows(args.cache_dir, prep._resolve_hf_token(args.hf_token))
    docs = build_documents(rows, prep, tok, lengths)
    if not docs:
        print("[prep_lbv2_mt] ERROR: no eligible documents.", file=sys.stderr)
        return 2

    print(_histogram([d["query_tokens"] for d in docs], "rendered query tokens (all kept docs)"))

    dropped_unpackable = 0
    if args.pack_distractors:
        if args.min_doc_tokens <= 0:
            print("[prep_lbv2_mt] ERROR: --pack-distractors needs "
                  "--min-doc-tokens > 0 -- it is the length each turn is "
                  "packed UP to.", file=sys.stderr)
            return 2
        try:
            select_length_band([], args.min_doc_tokens, max_doc_tokens)
        except ValueError as exc:
            print(f"[prep_lbv2_mt] ERROR: {exc}", file=sys.stderr)
            return 2
        eligible = [d for d in docs if d["query_tokens"] <= max_doc_tokens]
        dropped_too_large = len(docs) - len(eligible)
        dropped_too_small = 0
        for d in eligible:
            d["context_tokens"] = len(tok.encode(d["context"], add_special_tokens=False))
        sep_tokens = len(tok.encode(PACK_DOC_HEADER.format(n=10) + PACK_DOC_JOINER,
                                    add_special_tokens=False))

        def measure(block):
            return len(render_turn_query(tok, 0, {"input": block}))

        def render(target, order):
            return render_packed_block(prep._PROMPT_TEMPLATE, prep.CHOICE_LETTERS,
                                       target, order)

        print(f"[prep_lbv2_mt] packing: {len(eligible)} documents under the "
              f"{max_doc_tokens:,} ceiling are candidate questions AND "
              f"distractors; each turn packed to [{args.min_doc_tokens:,}, "
              f"{max_doc_tokens:,}]", flush=True)
        conversations, dropped_unpackable = group_packed_conversations(
            eligible, budget, args.min_doc_tokens, max_doc_tokens,
            random.Random(args.seed), args.turns_per_conv, measure, render,
            sep_tokens)
        return _finish(args, budget, tok, conversations, dropped_too_small,
                       dropped_too_large, dropped_unpackable, max_doc_tokens)

    try:
        docs, dropped_too_small = select_length_band(
            docs, args.min_doc_tokens, max_doc_tokens)
    except ValueError as exc:
        print(f"[prep_lbv2_mt] ERROR: {exc}", file=sys.stderr)
        return 2
    if args.min_doc_tokens > 0:
        in_band = sum(1 for d in docs if d["query_tokens"] <= max_doc_tokens)
        # The number that decides whether this variant is worth running, printed
        # BEFORE grouping so a thin band is visible even when grouping succeeds.
        print(f"[prep_lbv2_mt] band [{args.min_doc_tokens:,}, {max_doc_tokens:,}]: "
              f"{in_band} documents in band -> at most "
              f"{in_band // args.turns_per_conv} conversations "
              f"({dropped_too_small} below the floor)")

    rng = random.Random(args.seed)
    conversations, dropped_too_large = group_into_conversations(
        docs, budget, max_doc_tokens, rng, args.turns_per_conv
    )
    return _finish(args, budget, tok, conversations, dropped_too_small,
                   dropped_too_large, dropped_unpackable, max_doc_tokens)


def _finish(args, budget, tok, conversations, dropped_too_small,
            dropped_too_large, dropped_unpackable, max_doc_tokens) -> int:
    """Verify, emit and report -- shared by the single-document and packed
    paths, so both write and describe their output identically."""
    conversations, dropped_verify = verify_conversations(conversations, budget, tok)

    if args.max_conversations >= 0:
        conversations = conversations[:args.max_conversations]

    if not conversations:
        print("[prep_lbv2_mt] ERROR: no conversations formed. Too few "
              f"documents fall in [{args.min_doc_tokens:,}, {max_doc_tokens:,}] "
              f"to fill {args.turns_per_conv}-turn conversations. Widen the band "
              "(lower --min-doc-tokens), add a bucket via --lengths (e.g. "
              "short,medium), or use a smaller --max-tokens to raise the "
              "ceiling.", file=sys.stderr)
        return 2

    out_rows = build_rows(conversations, budget, args.config_name, args.preamble,
                          id_prefix=args.id_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_turn = [t["query_tokens"] for r in out_rows for t in r["turns"]]
    resident = [r["resident_len_at_last_turn"] for r in out_rows]
    print(f"[prep_lbv2_mt] conversations={len(out_rows)} "
          f"turns={len(per_turn)} "
          f"dropped_doc_too_small={dropped_too_small} "
          f"dropped_doc_too_large={dropped_too_large} "
          f"dropped_unpackable={dropped_unpackable} "
          f"dropped_failed_verify={dropped_verify}")
    print(_histogram(per_turn, "per-turn d (rendered query tokens)"))
    if args.pack_distractors:
        packed_turns = [t for r in out_rows for t in r["turns"]]
        ndocs = [t["num_docs"] for t in packed_turns]
        pos = [t["target_doc_position"] for t in packed_turns]
        print(f"  documents per turn: min={min(ndocs)} "
              f"mean={statistics.mean(ndocs):.1f} max={max(ndocs)}")
        print(f"  target position (0-based) counts: "
              + ", ".join(f"{k}:{pos.count(k)}" for k in sorted(set(pos))))
    print(_histogram(resident, "resident len at last turn"))

    # Report BOTH engines' headroom, and say which one is actually binding.
    # Reporting only the target's is actively misleading whenever the scorer
    # has the smaller context window (Gemma-4-31B is 262,144 against E2B's
    # 131,072): the target can show six figures of slack while the scorer sits
    # a few thousand tokens from its ceiling.
    per_conv_lens = [[t["query_tokens"] for t in r["turns"]] for r in out_rows]
    worst_target = max(budget.target_check(q) + budget.max_tokens for q in per_conv_lens)
    worst_spec = max(budget.spec_check(q) + 1 + LOOK_AHEAD_CNT for q in per_conv_lens)
    t_head = budget.target_budget - worst_target
    s_head = budget.spec_budget - worst_spec
    for name, worst, head, cap in (
        ("target    ", worst_target, t_head, budget.target_budget),
        ("speculator", worst_spec, s_head, budget.spec_budget),
    ):
        mark = "  <-- BINDING" if head == min(t_head, s_head) else ""
        print(f"  {name} worst-case {worst:,} of {cap:,} -- headroom {head:,}{mark}")
    print(f"  mean d:o ratio  {statistics.mean(per_turn) / args.max_tokens:.1f}:1 "
          f"(SCBench steady-state is ~1:7)")
    print(f"[prep_lbv2_mt] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
