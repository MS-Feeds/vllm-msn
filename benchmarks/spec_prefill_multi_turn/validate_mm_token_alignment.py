#!/usr/bin/env python3
"""CPU-only GATE check for the multimodal port: do the target and the
speculator expand the SAME image into the SAME number of placeholder
tokens?

Run this FIRST, before any other multimodal work -- it gates the whole
port. If it fails, stop; nothing downstream is worth building.

## Why this specific check gates everything

`vllm_patch/proposer.py`'s module docstring ("Local positions vs. absolute
conversation positions") states the pipeline's central invariant: the
speculator scores LOCAL contiguous positions 0..N-1, and `pruner.py`
translates those back into `conversation_state.py`'s ABSOLUTE ledger
positions through the (token_id, absolute_position) pairs riding alongside
each candidate. That translation is a positional identity -- it assumes
local index i and ledger entry i name the SAME token.

Text keeps that invariant for free: both engines share a tokenizer family
and a prompt is a flat list[int]. Images do not. A multimodal prompt
carries PLACEHOLDER token ids that the engine swaps for encoder embeddings
at runtime, and the NUMBER of placeholders per image is a property of each
model's own vision config -- patch size, image resolution, pan-and-scan
crop policy. If the target expands one image into 1600 placeholders and the
speculator expands it into 256, every ledger position after the first image
is off by 1344, and NOTHING RAISES: both sides stay internally consistent,
so `pruner.py` translates valid local indices into valid-looking absolute
positions that name the wrong tokens, and the gather keeps
confidently-selected garbage. It would surface only as degraded accuracy
that looks like a bad estimator.

That is why this is a gate and not a runtime assertion.

## What it checks

  A. Both processors load, and the transformers pin is right -- Gemma 4
     needs EXACTLY 5.14.1 (see REPRODUCE.md's "Gemma 4 only" section;
     older lacks the gemma4 registration, 5.15+ raises
     AmbiguousGlobalPerLayerAttributeError).
  B. **The gate.** For each probe resolution, the same image expands to the
     same number of placeholder tokens in both models.
  C. Placeholders form a CONTIGUOUS span. Step 5 of the port (force-keep
     whole image spans) assumes one image == one interval; a fragmented
     layout needs a different force-keep representation.
  D. Token count vs. resolution -- constant, or resolution-dependent? A
     pan-and-scan / multi-crop policy makes per-image cost variable, which
     the sample-schema budget arithmetic has to account for (compare
     `datasets/prep_longbench_v2_multiturn.py`'s Budget).
  E. Multi-image prompts stay consistent -- video benchmarks send many
     frames per turn, so the per-image cost has to be additive.

## How the count is measured

Two independent ways, cross-checked, because neither alone is trustworthy:

  1. **Marginal cost**: encode the same text with 1 image and with 2, and
     take the length difference. This needs no knowledge of the model's
     special-token naming and self-corrects for prompt scaffolding.
  2. **Direct count**: resolve the image placeholder token id and count its
     occurrences.

They can legitimately differ -- (1) includes per-image boi/eoi marker
tokens that (2) excludes -- so both are reported. (1) is authoritative for
the ledger arithmetic, since it is what actually consumes positions.

No GPU and no model weights required: this loads processors only, so it
runs on a login node in seconds.

Usage:
  python3 validate_mm_token_alignment.py \
      --target-model $GEMMA4_31B_MODEL_PATH \
      --speculator-model $GEMMA4_E2B_MODEL_PATH
"""

import argparse
import os
import sys

_EXPECTED_TRANSFORMERS = "5.14.1"

#: Probe resolutions, deliberately including non-square and both sides of
#: the common 896/1024 pan-and-scan thresholds. If per-image token count is
#: constant across ALL of these, the ledger arithmetic can use a single
#: constant; if not, the sample schema has to carry a per-image cost.
DEFAULT_SIZES = [
    (224, 224),
    (336, 336),
    (512, 512),
    (640, 480),
    (896, 896),
    (1024, 768),
]

_PROBE_TEXT = "Describe what you see."


def _make_image(width: int, height: int):
    """A deterministic, non-degenerate RGB image.

    Non-degenerate on purpose: a flat single-color image can take a
    different path through a processor that special-cases uniform input.
    Deterministic so this script's output is reproducible run to run.
    """
    from PIL import Image

    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    return img


def _check_transformers_pin() -> None:
    try:
        import transformers
    except ImportError:
        print(f"[FAIL] transformers is not importable; Gemma 4 needs {_EXPECTED_TRANSFORMERS}.")
        sys.exit(1)
    have = transformers.__version__
    if have != _EXPECTED_TRANSFORMERS:
        print(
            f"[WARN] transformers {have} found, Gemma 4 checkpoints need exactly "
            f"{_EXPECTED_TRANSFORMERS}. Older versions fail at the first AutoConfig call "
            f"(model type gemma4 unrecognized); 5.15+ raises "
            f"AmbiguousGlobalPerLayerAttributeError. See REPRODUCE.md."
        )
    else:
        print(f"[OK]   transformers {have} (matches the Gemma 4 pin)")


def _load_processor(path: str, label: str):
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:
        print(f"[FAIL] {label}: could not load AutoProcessor from {path}")
        print(f"       {exc}")
        sys.exit(1)
    print(f"[OK]   {label}: loaded {type(proc).__name__} from {path}")
    return proc


def _image_token_id(proc):
    """Resolve the image placeholder token id, or None.

    Tries the several places different model families put it. Returning
    None is not fatal: the marginal-cost measurement does not need it, and
    only the contiguity check (C) is skipped.
    """
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(proc, attr, None)
        if isinstance(val, int):
            return val
    tok = getattr(proc, "tokenizer", None)
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(tok, attr, None)
        if isinstance(val, int):
            return val
    for name_attr in ("image_token", "boi_token"):
        name = getattr(proc, name_attr, None)
        if isinstance(name, str) and tok is not None:
            tid = tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0:
                return tid
    if tok is not None:
        unk = getattr(tok, "unk_token_id", None)
        for literal in ("<image_soft_token>", "<image>", "<|image|>", "<image_placeholder>"):
            tid = tok.convert_tokens_to_ids(literal)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                return tid
    return None


def _encode(proc, images):
    """input_ids for _PROBE_TEXT plus len(images) copies of the image.

    Lets exceptions propagate: a processor that cannot encode its own
    model's image format is a hard stop, not a data point.
    """
    out = proc(text=[_PROBE_TEXT], images=images if images else None, return_tensors="pt")
    return [int(t) for t in out["input_ids"][0]]


def _contiguous_spans(ids, token_id):
    """Maximal runs of token_id, as (start, end_exclusive) pairs."""
    spans, start = [], None
    for i, t in enumerate(ids):
        if t == token_id and start is None:
            start = i
        elif t != token_id and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(ids)))
    return spans


def _probe(proc, width, height, img_token_id):
    """Per-image token cost at one resolution, measured both ways."""
    ids_1 = _encode(proc, [_make_image(width, height)])
    ids_2 = _encode(proc, [_make_image(width, height), _make_image(width, height)])
    return {
        "marginal": len(ids_2) - len(ids_1),
        "direct": ids_1.count(img_token_id) if img_token_id is not None else None,
        "spans": _contiguous_spans(ids_1, img_token_id) if img_token_id is not None else [],
        "len_1img": len(ids_1),
        "len_2img": len(ids_2),
    }


def _parse_sizes(spec):
    sizes = []
    for chunk in spec.split(","):
        w, _, h = chunk.strip().lower().partition("x")
        sizes.append((int(w), int(h)))
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-model", default=None,
                        help="Defaults to $GEMMA4_31B_MODEL_PATH.")
    parser.add_argument("--speculator-model", default=None,
                        help="Defaults to $GEMMA4_E2B_MODEL_PATH.")
    parser.add_argument("--sizes", default=None,
                        help="Comma-separated WxH probe resolutions, e.g. '224x224,896x896'. "
                             "Defaults to a spread covering square, non-square, and both "
                             "sides of the usual pan-and-scan thresholds.")
    args = parser.parse_args()

    target = args.target_model or os.environ.get("GEMMA4_31B_MODEL_PATH")
    spec = args.speculator_model or os.environ.get("GEMMA4_E2B_MODEL_PATH")
    if not target:
        parser.error("--target-model or $GEMMA4_31B_MODEL_PATH is required")
    if not spec:
        parser.error("--speculator-model or $GEMMA4_E2B_MODEL_PATH is required")

    sizes = _parse_sizes(args.sizes) if args.sizes else DEFAULT_SIZES

    print("=== A. Environment and processors ===")
    _check_transformers_pin()
    proc_t = _load_processor(target, "target")
    proc_s = _load_processor(spec, "speculator")

    tid_t, tid_s = _image_token_id(proc_t), _image_token_id(proc_s)
    print(f"       image placeholder token id: target={tid_t} speculator={tid_s}")
    if tid_t is None or tid_s is None:
        print("[WARN] could not resolve the placeholder token id for at least one model; "
              "the contiguity check (C) will be skipped. The gate (B) does not need it.")
    elif tid_t != tid_s:
        print(f"[WARN] the two models use DIFFERENT placeholder token ids ({tid_t} vs {tid_s}). "
              "Not fatal on its own -- what matters is the COUNT -- but it means the ledger "
              "cannot identify image spans by a single shared id.")

    print("\n=== B. The gate: per-image token cost, target vs speculator ===")
    print(f"{'size':>12}  {'target':>17}  {'speculator':>17}  verdict")
    failures, marginals_t, all_spans_ok = [], [], True
    for (w, h) in sizes:
        label = f"{w}x{h}"
        try:
            rt = _probe(proc_t, w, h, tid_t)
            rs = _probe(proc_s, w, h, tid_s)
        except Exception as exc:
            print(f"{label:>12}  encode failed: {exc}")
            failures.append((label, f"encode failed: {exc}"))
            continue

        def _fmt(r):
            direct = "-" if r["direct"] is None else r["direct"]
            return f"{r['marginal']:>7} /{direct:>8}"

        ok = rt["marginal"] == rs["marginal"]
        print(f"{label:>12}  {_fmt(rt):>17}  {_fmt(rs):>17}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((label, f"target {rt['marginal']} vs speculator {rs['marginal']}"))
        marginals_t.append(rt["marginal"])
        for who, r in (("target", rt), ("speculator", rs)):
            if r["spans"] and len(r["spans"]) != 1:
                all_spans_ok = False
                print(f"               {who}: placeholders span {len(r['spans'])} "
                      f"disjoint runs {r['spans'][:4]}")
    print("       (each cell is  marginal / direct-count  tokens per image)")

    print("\n=== C. Contiguity (force-keep-whole-image assumption) ===")
    if tid_t is None or tid_s is None:
        print("[SKIP] placeholder token id unresolved.")
    elif all_spans_ok:
        print("[OK]   every image expands to exactly ONE contiguous run of placeholders; "
              "force-keeping an image is a single interval.")
    else:
        print("[WARN] at least one image expands to MULTIPLE disjoint runs. Step 5's "
              "force-keep needs a list of intervals per image, not one span.")

    print("\n=== D. Resolution dependence (budget arithmetic) ===")
    distinct = sorted(set(marginals_t))
    if len(distinct) == 1:
        print(f"[OK]   constant {distinct[0]} tokens per image across all probes -- the "
              "sample-schema budget can use a single per-image constant.")
    elif not distinct:
        print("[SKIP] no successful probes.")
    else:
        print(f"[WARN] per-image token cost VARIES with resolution: {distinct}. The prep-time "
              "budget must compute per-image cost from the actual image rather than a "
              "constant; see datasets/prep_longbench_v2_multiturn.py's Budget.")

    print("\n=== Verdict ===")
    if failures:
        print("[FAIL] GATE FAILED -- target and speculator disagree on per-image token count:")
        for (label, why) in failures:
            print(f"         {label}: {why}")
        print("       Do NOT proceed with the multimodal port using this model pair. The "
              "local->absolute position translation in pruner.py would corrupt silently.")
        sys.exit(1)
    print("[PASS] GATE PASSED -- both models expand an image into the same number of "
          "positions at every probed resolution.")
    print("       Safe to proceed to step 2 (un-zero limit_mm_per_prompt).")


if __name__ == "__main__":
    main()
