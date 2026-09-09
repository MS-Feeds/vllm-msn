#!/usr/bin/env python3
"""Builds synthetic multi-turn conversations from MMMU, in the sample schema
`predict_scbench.py` consumes -- ONE MMMU QUESTION (with its images) PER TURN.

The multimodal sibling of `prep_longbench_v2_multiturn.py`, and deliberately
the same construction: that script packs one LongBench v2 DOCUMENT per turn to
make the per-turn delta `d` large; this one packs one MMMU question per turn
and gets a large `d` from the IMAGES instead.

## What this is for, and what it is NOT for

**It is for**: (a) end-to-end correctness of the multimodal port -- the only
workload in this benchmark that exercises image KV through the block gather at
all; (b) the latency question, since images make `d` large enough to clear the
win condition that SCBench cannot.

Per-turn `d` here is roughly `text + n_images * ~260` tokens, so a
single-image question runs `d ~ 370` and a multi-image one `d ~ 2000`. Against
`o = 512` the wall-clock condition (see SPECULATION_ECONOMICS.md, and the
prefill/decode split that qualifies it) wants `d > 0.45*o + 5.5 ~ 236`. Every
MMMU turn clears that; SCBench's `d ~ 70` does not.

**It is NOT a selection-quality benchmark, and a flat accuracy row here proves
nothing about sparsity.** MMMU items are mutually independent -- turn 5's
answer depends on turn 5's own question and images, which are FORCE-KEPT and
therefore never prunable. Nothing an earlier turn contributed is needed later,
so accuracy can stay flat across the entire keep-rate grid while the estimator
is discarding everything it is offered. That is the same trap as the flat
`scbench_summary` row (ACCURACY_IMPROVEMENTS.md's "two live explanations").

What a DROP here does mean is real and worth having: it says the sparse path
has corrupted something structural -- a partially-kept image span, a position
translation that slipped, image KV mishandled by the gather. Read this row as a
regression test on the mechanism, not as evidence about selection.

## Schema

Same `{id, config, context, turns[{input, answer}]}` shape, plus ONE new field:

  - `turns[i]["images"]` = image file paths, RELATIVE to the jsonl's own
    directory, in the order their markers appear in `input`.

`context` is a short shared preamble, not a document -- same reasoning as the
LongBench v2 builder: it keeps `ConversationState`'s turn-0 candidate pool
non-empty and makes all T turns structurally identical.

  - `id` = `mmmumt-NNNN`, the conversation salt. The prefix cannot collide with
    `scbench_<config>-<n>` or `lbv2mt-<n>` if the files are concatenated.

## Image markers: why items get dropped

MMMU embeds literal `<image 1>` .. `<image 7>` markers in the question and
option text, and stores the images in separate `image_1..image_7` columns. A
processor requires the number of markers in the text to EQUAL the number of
images passed, or it raises -- so this packer:

  - walks the markers in ORDER OF APPEARANCE and builds the image list to
    match, rather than trusting column order (a question may reference
    `<image 2>` before `<image 1>`, or skip `<image 1>` entirely);
  - DROPS any item whose markers and images do not correspond exactly (a
    marker with no image, or an image no marker references).

Filter, never truncate -- the same rule the LongBench v2 builder applies to
oversized documents, for the same reason.

## Token accounting MEASURES image cost rather than assuming it

`validate_mm_token_alignment.py` established on the Gemma 4 pair that per-image
token cost is driven by the image's SHAPE, not its size: 1:1 images cost 258
tokens at every resolution from 224x224 to 896x896, while 4:3 images cost 268 at
both 640x480 and 1024x768. The two groups' pixel-count ranges OVERLAP -- 896x896
(803k px) costs 258 while 640x480 (307k px) costs 268 -- so a budget keyed on
pixel count, longest edge, or a per-image constant is wrong in both directions.

So `ImageCoster` measures each image's real cost by running the processor, and
caches by `(width, height)` -- which is sound precisely BECAUSE the gate showed
cost is a function of shape. Without that finding the cache would be a guess.

Text accounting is borrowed, not reimplemented: `render_turn_query`,
`chat_wrapper_pieces`, `chat_turn_boundary_pieces` and `Budget` are imported so
this packer's numbers ARE the driver's numbers by construction.

Usage:
    python3 datasets/prep_mmmu_multiturn.py \\
        --processor $GEMMA4_31B_MODEL_PATH \\
        --target-max-num-batched-tokens 130560 \\
        --speculator-max-num-batched-tokens 131063 \\
        --turns-per-conv 5 --max-tokens 512 --seed 42 \\
        --output datasets/mmmu_multiturn.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
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

HF_DATASET_NAME = "MMMU/MMMU"
DEFAULT_CACHE_DIR = _HERE / ".cache"
DEFAULT_OUTPUT = _HERE / "mmmu_multiturn.jsonl"

#: Emitted as each row's `config`, and the key `grade_scbench.py`'s
#: `_METRIC_BY_CONFIG` must gain an entry for (-> `multiple_choice_letter`).
CONFIG_NAME = "mmmu_mc"

#: Non-empty on purpose -- see the LongBench v2 builder's reasoning about
#: keeping turn 0's candidate pool non-empty and all turns structurally
#: identical.
DEFAULT_PREAMBLE = (
    "The following are multiple-choice questions from college-level "
    "coursework. Each question may refer to one or more images."
)

_MARKER_RE = re.compile(r"<image\s+(\d+)\s*>")
_LETTERS = "ABCDEFGHIJ"


def _load_budget_class():
    """`Budget` from the LongBench v2 builder, by explicit file path.

    Imported rather than reimplemented: it encodes the driver's real
    pre-flight arithmetic including the `(T-1)*O` resident-output term, and a
    second copy would drift. Addressed by path for the same provenance reason
    that builder loads its own sibling prep by path.
    """
    path = _HERE / "prep_longbench_v2_multiturn.py"
    spec = importlib.util.spec_from_file_location("_lbv2mt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Budget


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def resolve_images(question: str, options: list[str], images: dict) -> tuple[str, list[str], list]:
    """Renumber `<image N>` markers canonically and collect matching images.

    Returns `(question, options, ordered_images)` with markers in BOTH the
    question and the options rewritten to `<image K>`, numbered by order of
    appearance across question-then-options, and `ordered_images` holding the
    PIL images in that same order.

    Substituting in both is load-bearing, not tidiness: MMMU puts markers in
    option text too, and step 4 converts `<image K>` into the model's real
    marker by position. Rewriting only the question would leave an option
    carrying a stale number while the image list had been renumbered around
    it -- the text would then name an image that is not where it says, which
    no exception would catch.

    Raises ValueError if markers and images do not correspond exactly -- the
    caller drops the item. Both directions are errors: a marker with no image
    makes the processor raise at run time, and an image no marker references
    is silently ignored by the processor, changing the token count this packer
    budgeted for.
    """
    seen: list[int] = []
    for text in [question, *options]:
        for match in _MARKER_RE.finditer(text):
            idx = int(match.group(1))
            if idx not in seen:
                seen.append(idx)

    if not seen:
        raise ValueError("no image markers -- not a multimodal item")

    missing = [i for i in seen if images.get(f"image_{i}") is None]
    if missing:
        raise ValueError(f"markers reference absent images: {missing}")

    present = [i for i in range(1, 8) if images.get(f"image_{i}") is not None]
    unreferenced = [i for i in present if i not in seen]
    if unreferenced:
        raise ValueError(f"images present but never referenced: {unreferenced}")

    renumber = {old: new for new, old in enumerate(seen, start=1)}

    def _sub(text: str) -> str:
        return _MARKER_RE.sub(lambda m: f"<image {renumber[int(m.group(1))]}>", text)

    ordered = [images[f"image_{i}"] for i in seen]
    return _sub(question), [_sub(o) for o in options], ordered


def render_question_block(question: str, options: list[str]) -> str:
    """MMMU's standard multiple-choice block.

    The trailing instruction is what makes `multiple_choice_letter` gradable:
    without it the model narrates and the letter has to be recovered from
    prose, which is exactly the parse failure `grade_scbench.py` scores as
    wrong rather than missing.
    """
    lines = [question.strip(), ""]
    for letter, opt in zip(_LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image token cost
# ---------------------------------------------------------------------------


class ImageCoster:
    """Real per-image token cost, measured and cached by (width, height).

    Caching by shape is sound because `validate_mm_token_alignment.py`
    established that cost is a function of shape, not size, on this model pair
    -- see the module docstring. If that ever stops holding, this cache is the
    first thing to invalidate.

    Cost is the MARGINAL length difference between a two-image and a one-image
    encode, i.e. what the image actually consumes in ledger positions,
    INCLUDING its boi/eoi markers. The bare placeholder-run length (2 fewer on
    the Gemma 4 pair) is not what the budget needs.
    """

    def __init__(self, processor, probe_text: str = "x"):
        self.proc = processor
        self.probe_text = probe_text
        self._cache: dict[tuple[int, int], int] = {}
        self.marker = self._resolve_marker()

    def _resolve_marker(self) -> str:
        for attr in ("image_token", "boi_token"):
            val = getattr(self.proc, attr, None)
            if isinstance(val, str) and val:
                return val
        tok = getattr(self.proc, "tokenizer", None)
        for literal in ("<image_soft_token>", "<image>", "<|image|>"):
            if tok is not None:
                tid = tok.convert_tokens_to_ids(literal)
                if isinstance(tid, int) and tid >= 0:
                    return literal
        raise RuntimeError(
            "could not resolve the processor's image marker string; "
            "run validate_mm_token_alignment.py, which reports it."
        )

    def _encode_len(self, images) -> int:
        out = self.proc(
            text=[self.marker * len(images) + self.probe_text],
            images=list(images),
            return_tensors="pt",
        )
        return int(out["input_ids"].shape[-1])

    def cost(self, image) -> int:
        key = (image.width, image.height)
        if key not in self._cache:
            self._cache[key] = self._encode_len([image, image]) - self._encode_len([image])
        return self._cache[key]

    def total(self, images) -> int:
        return sum(self.cost(im) for im in images)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def build_items(rows, tok, coster, max_item_tokens, verbose_every=200) -> list[dict]:
    """One packable item per usable MMMU question.

    `query_tokens` is the FULL per-turn cost -- rendered text plus every
    image's measured cost -- because that is what the driver's pre-flight
    compares against its budget.
    """
    items, dropped = [], {"not_mc": 0, "markers": 0, "too_large": 0, "bad_answer": 0}
    for n, row in enumerate(rows):
        if verbose_every and n and n % verbose_every == 0:
            print(f"[prep_mmmu] scanned {n} rows, kept {len(items)}", flush=True)

        if row.get("question_type") != "multiple-choice":
            dropped["not_mc"] += 1
            continue
        options = row.get("options") or []
        if isinstance(options, str):
            try:
                options = json.loads(options.replace("'", '"'))
            except Exception:
                dropped["not_mc"] += 1
                continue
        answer = (row.get("answer") or "").strip().upper()
        if len(options) < 2 or answer not in set(_LETTERS[: len(options)]):
            dropped["bad_answer"] += 1
            continue

        try:
            question, options, images = resolve_images(row["question"], options, row)
        except ValueError:
            dropped["markers"] += 1
            continue

        block = render_question_block(question, options)
        text_tokens = len(render_turn_query(tok, 0, {"input": block}))
        image_tokens = coster.total(images)
        total = text_tokens + image_tokens
        if total > max_item_tokens:
            dropped["too_large"] += 1
            continue

        items.append(
            {
                "input": block,
                "answer": answer,
                "options": options,
                "pil_images": images,
                "subject": row.get("subfield") or row.get("id", "").split("_")[1:2],
                "difficulty": row.get("topic_difficulty"),
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "n_images": len(images),
                "query_tokens": total,
            }
        )
    return items, dropped


def group_into_conversations(items, budget, turns_per_conv, rng) -> list[list[dict]]:
    """Size-banded grouping, then a per-conversation budget check.

    Banded for the same reason as the LongBench v2 builder: every headline
    column is a MEAN OVER TURNS, so an 8x spread in per-turn `d` inside one
    conversation would swamp the effect being measured. Sorting by
    `query_tokens` before chunking keeps each conversation's turns comparable.
    """
    ordered = sorted(items, key=lambda d: d["query_tokens"])
    convs, rejected = [], 0
    for i in range(0, len(ordered) - turns_per_conv + 1, turns_per_conv):
        chunk = ordered[i : i + turns_per_conv]
        lens = [d["query_tokens"] for d in chunk]
        if budget.target_check(lens) > budget.target_budget:
            rejected += 1
            continue
        if budget.spec_check(lens) > budget.spec_budget:
            rejected += 1
            continue
        convs.append(chunk)
    rng.shuffle(convs)
    return convs, rejected


def save_images(conversations, out_dir: Path) -> None:
    """Materialise PIL images to PNG beside the jsonl.

    PNG, not JPEG: the gate measured token cost from the image's SHAPE, and a
    lossy re-encode does not change shape -- but it does change pixels, and a
    dataset whose contents shift under recompression is not reproducible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, conv in enumerate(conversations):
        for t, item in enumerate(conv):
            paths = []
            for k, image in enumerate(item["pil_images"], start=1):
                name = f"mmmumt-{n:04d}-t{t}-img{k}.png"
                image.convert("RGB").save(out_dir / name)
                paths.append(f"{out_dir.name}/{name}")
            item["images"] = paths


def build_rows(conversations, budget, preamble) -> list[dict]:
    rows = []
    for n, conv in enumerate(conversations):
        query_lens = [d["query_tokens"] for d in conv]
        rows.append(
            {
                "id": f"mmmumt-{n:04d}",
                "config": CONFIG_NAME,
                "context": preamble,
                "turns": [
                    {
                        "input": d["input"],
                        "answer": d["answer"],
                        # The one schema addition. Paths are relative to the
                        # jsonl's own directory so the file stays portable.
                        "images": d["images"],
                        "options": d["options"],
                        "subject": d["subject"],
                        "difficulty": d["difficulty"],
                        "n_images": d["n_images"],
                        "text_tokens": d["text_tokens"],
                        "image_tokens": d["image_tokens"],
                        "query_tokens": d["query_tokens"],
                    }
                    for d in conv
                ],
                # Inert to the driver and the grader; recorded so a later run
                # at a different --max-tokens is caught by inspecting the file
                # rather than by a silent mid-run turn-loop break.
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processor", required=True,
                        help="Target checkpoint path -- the processor whose image token "
                             "cost is measured. Must be the SAME pair that passed "
                             "validate_mm_token_alignment.py.")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated MMMU subject configs. Default: all.")
    parser.add_argument("--split", default="validation",
                        help="dev (150) / validation (900) / test (10500). Default validation: "
                             "test answers are withheld.")
    parser.add_argument("--turns-per-conv", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Per-turn generation cap. MUST match the --max-tokens the "
                             "sweep runs with: it is reserved in the budget as resident "
                             "output, and a larger value at run time breaks the turn loop "
                             "mid-conversation.")
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=130560)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131063)
    parser.add_argument("--safety-tokens", type=int, default=256)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoProcessor

    rng = random.Random(args.seed)
    tok = load_tokenizer(args.processor)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    coster = ImageCoster(processor)
    print(f"[prep_mmmu] image marker = {coster.marker!r}")

    Budget = _load_budget_class()
    wb, wa = chat_wrapper_pieces(tok)
    wt = chat_turn_boundary_pieces(tok)
    preamble_len = len(tok.encode(DEFAULT_PREAMBLE, add_special_tokens=False))
    budget = Budget(
        target_budget=args.target_max_num_batched_tokens,
        spec_budget=args.speculator_max_num_batched_tokens,
        turns_per_conv=args.turns_per_conv,
        max_tokens=args.max_tokens,
        safety_tokens=args.safety_tokens,
        wrapper_before=len(wb),
        wrapper_after=len(wa),
        turn_boundary=len(wt),
        preamble_len=preamble_len,
    )
    print(f"[prep_mmmu] per-item token ceiling = {budget.doc_budget} "
          f"(binding side: {budget.binding_side})")

    configs = (args.configs.split(",") if args.configs
               else get_dataset_config_names(HF_DATASET_NAME))
    print(f"[prep_mmmu] {len(configs)} configs, split={args.split}")

    all_rows = []
    for config in configs:
        ds = load_dataset(HF_DATASET_NAME, config, split=args.split,
                          cache_dir=str(args.cache_dir))
        all_rows.extend(ds)
    print(f"[prep_mmmu] loaded {len(all_rows)} raw rows")

    items, dropped = build_items(all_rows, tok, coster, budget.doc_budget)
    print(f"[prep_mmmu] usable items: {len(items)}; dropped {dropped}")
    print(f"[prep_mmmu] image-cost cache: {len(coster._cache)} distinct shapes")
    if not items:
        print("[prep_mmmu] nothing usable -- refusing to write an empty file")
        return 1

    convs, rejected = group_into_conversations(items, budget, args.turns_per_conv, rng)
    if args.max_conversations:
        convs = convs[: args.max_conversations]
    print(f"[prep_mmmu] {len(convs)} conversations ({rejected} rejected by budget)")
    if not convs:
        print("[prep_mmmu] no conversation fits the budget -- refusing to write an "
              "empty file. Lower --turns-per-conv or --max-tokens.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_images(convs, args.output.parent / f"{args.output.stem}_images")
    rows = build_rows(convs, budget, DEFAULT_PREAMBLE)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_turn = [t["query_tokens"] for r in rows for t in r["turns"]]
    n_images = [t["n_images"] for r in rows for t in r["turns"]]
    print(f"[prep_mmmu] wrote {len(rows)} conversations -> {args.output}")
    print(f"[prep_mmmu] per-turn d: min={min(per_turn)} "
          f"median={sorted(per_turn)[len(per_turn) // 2]} max={max(per_turn)}")
    print(f"[prep_mmmu] images per turn: min={min(n_images)} max={max(n_images)}")
    print(f"[prep_mmmu] NOTE: add '{CONFIG_NAME}': multiple_choice_letter to "
          "grade_scbench.py's _METRIC_BY_CONFIG before grading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
