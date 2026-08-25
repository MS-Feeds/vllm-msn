#!/usr/bin/env python3
"""
Configurable vLLM inference script for ablation study experiments.
Accepts all configuration parameters as command-line arguments.
"""

import asyncio
import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from transformers import AutoTokenizer


@dataclass
class RequestRecord:
    """Per-request timing + token accounting for one inference call."""
    request_id: str
    input_index: int                    # original 0-based index in the prompts list
    prompt_chars: int
    submit_time: float                  # client wall clock when we called engine.generate
    first_output_time: Optional[float]  # client wall clock when first non-empty output arrived
    finish_time: Optional[float]        # client wall clock when generate() returned
    prompt_tokens: int = 0
    output_tokens: int = 0
    num_cached_tokens: int = 0          # prefix cache hits
    finish_reason: Optional[str] = None
    # Engine-reported (canonical TTFT; client-side is an approximation that
    # includes asyncio scheduling latency on the consumer side).
    engine_arrival_time: Optional[float] = None
    engine_first_token_latency: Optional[float] = None
    error: Optional[str] = None

    @property
    def e2e_latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.submit_time

    @property
    def ttft(self) -> Optional[float]:
        """Client-side time to first token (includes our submit overhead)."""
        if self.first_output_time is None:
            return None
        return self.first_output_time - self.submit_time

    @property
    def tpot(self) -> Optional[float]:
        """Mean time per output token in steady state (excludes prefill)."""
        if (
            self.first_output_time is None
            or self.finish_time is None
            or self.output_tokens < 2
        ):
            return None
        return (self.finish_time - self.first_output_time) / (self.output_tokens - 1)


def _summarize_distribution(samples: List[float], unit: str = "s") -> Dict[str, Any]:
    """Return mean / std / min / p50 / p90 / p95 / p99 / max for a list of values."""
    clean = [x for x in samples if x is not None]
    if not clean:
        return {"count": 0, "unit": unit}

    sorted_vals = sorted(clean)
    def pct(p: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        # Nearest-rank percentile (no interpolation)
        k = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    return {
        "count": len(clean),
        "unit": unit,
        "mean": statistics.fmean(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": sorted_vals[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": sorted_vals[-1],
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurable vLLM inference for ablation study"
    )

    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "fp8", "awq", "gptq"],
                        help="Quantization method")
    parser.add_argument("--kv_cache_dtype", type=str, default="auto",
                        help="KV cache dtype (auto, fp8_e5m2, fp8_e4m3)")

    # MTP configuration
    parser.add_argument("--speculative_model", type=str, default=None,
                        help="Path to MTP assistant model")
    parser.add_argument("--num_speculative_tokens", type=int, default=5,
                        help="Number of speculative tokens")

    # Memory configuration
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_num_seqs", type=int, default=128,
                        help="Maximum number of sequences")
    parser.add_argument("--max_num_batched_tokens", type=int, default=6144,
                        help="Maximum batched tokens")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model length (default 32768 = 32K; covers p95 of the "
                             "MAI test dataset, drops ~2.8%% of long-tail prompts. Realistic "
                             "for a single A100 80GB shared between model + KV. Gemma 4 "
                             "text-mode can support up to 262144 if you have headroom.)")

    # Performance configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graphs (eager mode)")
    parser.add_argument("--enable_cuda_graphs", action="store_true",
                        help="Enable CUDA graphs (ignored if enforce_eager)")

    # Input/Output
    parser.add_argument("--input_path", type=str, default=None,
                        help="Input file path (optional for smoke test)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_test_samples", type=int, default=100,
                        help="Number of samples for smoke test (if no input)")

    # Sampling configuration
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens to generate")

    return parser.parse_args()


async def _run_one(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    input_index: int,
    output_text_holder: Dict[str, str],
) -> RequestRecord:
    """Drive a single request to completion, recording per-request metrics."""
    record = RequestRecord(
        request_id=request_id,
        input_index=input_index,
        prompt_chars=len(prompt),
        submit_time=time.perf_counter(),
        first_output_time=None,
        finish_time=None,
    )

    final_output = None
    try:
        async for out in engine.generate(prompt, sampling_params, request_id):
            # First yield that carries any generated token marks "first output"
            if record.first_output_time is None and out.outputs and out.outputs[0].token_ids:
                record.first_output_time = time.perf_counter()
            final_output = out
    except Exception as e:
        record.error = f"{type(e).__name__}: {e}"
        record.finish_time = time.perf_counter()
        return record

    record.finish_time = time.perf_counter()

    if final_output is None:
        record.error = "no output"
        return record

    if final_output.prompt_token_ids is not None:
        record.prompt_tokens = len(final_output.prompt_token_ids)
    if getattr(final_output, "num_cached_tokens", None) is not None:
        record.num_cached_tokens = final_output.num_cached_tokens or 0

    if final_output.outputs:
        comp = final_output.outputs[0]
        record.output_tokens = len(comp.token_ids)
        record.finish_reason = comp.finish_reason
        output_text_holder[request_id] = comp.text or ""

    # Engine-side timing (vLLM v1 attaches RequestStateStats to RequestOutput.metrics)
    m = getattr(final_output, "metrics", None)
    if m is not None:
        record.engine_arrival_time = getattr(m, "arrival_time", None) or None
        record.engine_first_token_latency = (
            getattr(m, "first_token_latency", None) or None
        )

    return record


async def generate_all(
    engine: AsyncLLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    output_text_holder: Dict[str, str],
) -> List[RequestRecord]:
    """Submit all prompts concurrently — lets vLLM continuous batching schedule them."""
    # Use a single submission-time epoch so request_ids are guaranteed unique
    # AND humanly traceable to a single run; the i:06d component disambiguates
    # within the run.
    run_epoch_ms = int(time.time() * 1000)
    tasks = [
        asyncio.create_task(
            _run_one(
                engine,
                p,
                sampling_params,
                f"req_{i:06d}_{run_epoch_ms}",
                i,
                output_text_holder,
            )
        )
        for i, p in enumerate(prompts)
    ]
    return await asyncio.gather(*tasks)


def aggregate_metrics(
    records: List[RequestRecord],
    wall_clock_seconds: float,
) -> Dict[str, Any]:
    """Build the canonical metrics.json structure from per-request records."""
    finished = [r for r in records if r.error is None and r.finish_time is not None]
    failed = [r for r in records if r.error is not None]

    total_prompt_tokens = sum(r.prompt_tokens for r in finished)
    total_output_tokens = sum(r.output_tokens for r in finished)
    total_cached_tokens = sum(r.num_cached_tokens for r in finished)

    ttft_samples = [r.ttft for r in finished if r.ttft is not None]
    tpot_samples = [r.tpot for r in finished if r.tpot is not None]
    e2e_samples = [r.e2e_latency for r in finished if r.e2e_latency is not None]
    # Engine-side TTFT (more accurate; excludes our submit overhead)
    engine_ttft_samples = [
        r.engine_first_token_latency
        for r in finished
        if r.engine_first_token_latency is not None
    ]

    # Finish-reason histogram (across finished requests).
    finish_reasons: Dict[str, int] = {}
    for r in finished:
        key = r.finish_reason or "unknown"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1

    # Failed-error histogram, keyed by exception class. Length-validation
    # rejections (when a prompt > max_model_len) appear here as "ValueError",
    # so the operator can tell at a glance whether a failure spike is data
    # shape (truncation/length) vs. compute (CUDA OOM) vs. something else.
    failed_errors: Dict[str, int] = {}
    for r in failed:
        key = (r.error or "unknown").split(":", 1)[0].strip()
        failed_errors[key] = failed_errors.get(key, 0) + 1

    return {
        "wall_clock_seconds": wall_clock_seconds,
        "counts": {
            "total_requests": len(records),
            "finished": len(finished),
            "failed": len(failed),
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "prefix_cache_hit_rate": (
                total_cached_tokens / total_prompt_tokens
                if total_prompt_tokens else 0.0
            ),
            "finish_reasons": finish_reasons,
            "failed_errors": failed_errors,
        },
        "throughput": {
            "qps": len(finished) / wall_clock_seconds if wall_clock_seconds > 0 else 0.0,
            "output_tokens_per_sec": (
                total_output_tokens / wall_clock_seconds
                if wall_clock_seconds > 0 else 0.0
            ),
            "prompt_tokens_per_sec": (
                total_prompt_tokens / wall_clock_seconds
                if wall_clock_seconds > 0 else 0.0
            ),
            "total_tokens_per_sec": (
                (total_prompt_tokens + total_output_tokens) / wall_clock_seconds
                if wall_clock_seconds > 0 else 0.0
            ),
        },
        "latency": {
            "ttft_client_s": _summarize_distribution(ttft_samples, "s"),
            "ttft_engine_s": _summarize_distribution(engine_ttft_samples, "s"),
            "tpot_s": _summarize_distribution(tpot_samples, "s"),
            "e2e_s": _summarize_distribution(e2e_samples, "s"),
        },
    }


async def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("vLLM Configurable Inference")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Batch size: {args.max_num_seqs}")
    logger.info(f"GPU memory: {args.gpu_memory_utilization}")
    logger.info(f"CUDA graphs: {not args.enforce_eager}")
    logger.info(f"MTP: {args.speculative_model is not None}")
    logger.info("=" * 70)
    logger.info("")

    # Build engine args
    logger.info("Building AsyncEngineArgs...")

    # Handle enforce_eager / enable_cuda_graphs logic
    enforce_eager = args.enforce_eager
    if args.enable_cuda_graphs and not args.enforce_eager:
        enforce_eager = False

    speculative_config = None
    if args.speculative_model:
        speculative_config = {
            "model": args.speculative_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        speculative_config=speculative_config,
        disable_log_stats=False,
    )

    logger.info("Configuration:")
    logger.info(f"  dtype: {args.dtype}")
    logger.info(f"  quantization: {args.quantization}")
    logger.info(f"  kv_cache_dtype: {args.kv_cache_dtype}")
    logger.info(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
    logger.info(f"  max_num_seqs: {args.max_num_seqs}")
    logger.info(f"  max_num_batched_tokens: {args.max_num_batched_tokens}")
    logger.info(f"  enforce_eager: {enforce_eager}")
    logger.info(f"  speculative_model: {args.speculative_model}")
    logger.info("")

    # Initialize engine
    logger.info("Initializing AsyncLLMEngine...")
    init_start = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    init_time = time.perf_counter() - init_start
    logger.info(f"✓ Engine initialized in {init_time:.2f} seconds")
    logger.info("")

    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Dataset loading + filtering. We apply the model's actual chat template
    # (so the string we hand to vLLM is what its tokenizer will see) AND drop
    # any prompt whose token count > max_model_len - max_tokens. Doing this
    # here, rather than letting vLLM ValueError mid-run, means:
    #   1) Every experiment sees an identical input subset → comparable.
    #   2) Skip counts are recorded structurally in metrics.json → auditable.
    dataset_stats: Dict[str, Any] = {
        "input_path": args.input_path,
        "n_total_seen": 0,
        "n_skipped_format": 0,
        "n_skipped_length": 0,
        "n_loaded": 0,
        "max_input_tokens_threshold": args.max_model_len - args.max_tokens,
        "used_chat_template": False,
    }

    if args.input_path and Path(args.input_path).exists():
        logger.info(f"Loading prompts from {args.input_path}...")

        logger.info("Loading tokenizer (separately from engine) to apply chat "
                    "template + count tokens for pre-filtering...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        has_template = getattr(tokenizer, "chat_template", None) is not None
        dataset_stats["used_chat_template"] = has_template
        if not has_template:
            logger.warning(
                "Tokenizer has no chat_template; falling back to a naive "
                "'role: content' join. This will tokenize DIFFERENTLY from "
                "what vLLM does internally for chat-formatted models — "
                "absolute token counts and TTFT will be skewed."
            )

        max_input_tokens = dataset_stats["max_input_tokens_threshold"]
        logger.info(
            f"Pre-filter threshold: skip prompts whose tokenized length "
            f"exceeds max_model_len({args.max_model_len}) - "
            f"max_tokens({args.max_tokens}) = {max_input_tokens} tokens"
        )

        prompts = []
        warned_format = 0  # cap warning spam
        with open(args.input_path) as f:
            for line_no, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                dataset_stats["n_total_seen"] += 1

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    dataset_stats["n_skipped_format"] += 1
                    if warned_format < 5:
                        logger.warning(f"  line {line_no}: invalid JSON ({e}); skipped")
                        warned_format += 1
                    continue

                messages = data.get("messages") if isinstance(data, dict) else None
                if not isinstance(messages, list) or not messages:
                    dataset_stats["n_skipped_format"] += 1
                    if warned_format < 5:
                        logger.warning(
                            f"  line {line_no}: no 'messages' list; skipped"
                        )
                        warned_format += 1
                    continue

                # Apply chat template — gives the exact string vLLM will
                # tokenize, so our token count matches what vLLM would compute.
                if has_template:
                    try:
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception as e:
                        dataset_stats["n_skipped_format"] += 1
                        if warned_format < 5:
                            logger.warning(
                                f"  line {line_no}: chat template failed ({e}); skipped"
                            )
                            warned_format += 1
                        continue
                else:
                    formatted = "\n\n".join(
                        f"{m.get('role','')}: {m.get('content','')}"
                        for m in messages
                    )

                n_tokens = len(
                    tokenizer.encode(formatted, add_special_tokens=False)
                )
                if n_tokens > max_input_tokens:
                    dataset_stats["n_skipped_length"] += 1
                    continue

                prompts.append(formatted)

        dataset_stats["n_loaded"] = len(prompts)
        logger.info(
            f"  Dataset: seen={dataset_stats['n_total_seen']} "
            f"loaded={dataset_stats['n_loaded']} "
            f"skipped_format={dataset_stats['n_skipped_format']} "
            f"skipped_length={dataset_stats['n_skipped_length']}"
        )
        if dataset_stats["n_loaded"] == 0:
            logger.error(
                "No usable prompts after filtering. Check max_model_len / "
                "input_path / file format."
            )
            sys.exit(1)
    else:
        logger.info(
            f"No input file, generating {args.num_test_samples} synthetic prompts..."
        )
        prompts = [
            f"Write a short story about {topic}."
            for topic in [
                "artificial intelligence", "space exploration", "time travel",
                "ancient civilizations", "future cities", "ocean depths",
                "quantum physics", "virtual reality", "genetic engineering",
                "climate change"
            ] * (args.num_test_samples // 10 + 1)
        ][: args.num_test_samples]
        dataset_stats["n_total_seen"] = args.num_test_samples
        dataset_stats["n_loaded"] = len(prompts)

    logger.info(f"Total prompts to submit: {len(prompts)}")
    logger.info("")

    # Run inference
    logger.info("Starting inference...")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Submitting {len(prompts)} prompts to engine for continuous batching...")
    output_texts: Dict[str, str] = {}
    wall_start = time.perf_counter()
    records = await generate_all(engine, prompts, sampling_params, output_texts)
    wall_clock_seconds = time.perf_counter() - wall_start

    # Write generations. input_index is the row's position in the original
    # prompts list (i.e. line N of input_path) — keep it so generations can be
    # joined back to the source prompts without re-loading.
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps({
                "request_id": r.request_id,
                "input_index": r.input_index,
                "output": output_texts.get(r.request_id, ""),
                "num_tokens": r.output_tokens,
                "finish_reason": r.finish_reason,
                "error": r.error,
            }) + "\n")

    # Per-request metrics (one JSONL row per request, for plotting/post-hoc analysis)
    per_request_path = output_path.with_name("per_request_metrics.jsonl")
    with open(per_request_path, "w") as f:
        for r in records:
            row = asdict(r)
            row["ttft_s"] = r.ttft
            row["tpot_s"] = r.tpot
            row["e2e_latency_s"] = r.e2e_latency
            f.write(json.dumps(row) + "\n")

    # Aggregate metrics.json
    metrics = aggregate_metrics(records, wall_clock_seconds)
    metrics["dataset"] = dataset_stats
    # Build the speculative-config snapshot from the ORIGINAL CLI args, not
    # from our local `speculative_config` dict. vLLM v1 mutates the dict it
    # receives via AsyncEngineArgs(speculative_config=...) — after engine
    # init the dict has extra fields including a ModelConfig object that
    # json.dump can't serialize. Reconstructing from args.* keeps this
    # section safely JSON-clean.
    spec_snapshot = (
        {
            "model": args.speculative_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        if args.speculative_model
        else None
    )
    metrics["config"] = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": enforce_eager,
        "speculative_config": spec_snapshot,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    }
    metrics_path = output_path.with_name("metrics.json")
    with open(metrics_path, "w") as f:
        # default=str catches any other unexpectedly-non-JSON values that
        # might creep in via dataset_stats or future additions, so an
        # otherwise-successful run doesn't lose its metrics.json over a
        # serialization typo.
        json.dump(metrics, f, indent=2, default=str)

    # Comprehensive summary
    c = metrics["counts"]
    t = metrics["throughput"]
    lat = metrics["latency"]

    def _fmt(d: Dict[str, Any], scale: float = 1000.0, fmt: str = "{:.2f}") -> str:
        if d.get("count", 0) == 0:
            return "  (no samples)"
        return (
            f"  mean={fmt.format(d['mean'] * scale)}ms "
            f"std={fmt.format(d['stdev'] * scale)}ms "
            f"min={fmt.format(d['min'] * scale)}ms "
            f"p50={fmt.format(d['p50'] * scale)}ms "
            f"p90={fmt.format(d['p90'] * scale)}ms "
            f"p95={fmt.format(d['p95'] * scale)}ms "
            f"p99={fmt.format(d['p99'] * scale)}ms "
            f"max={fmt.format(d['max'] * scale)}ms "
            f"(n={d['count']})"
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("Inference Complete — Comprehensive Metrics")
    logger.info("=" * 80)
    logger.info(f"Wall clock: {wall_clock_seconds:.2f}s")
    ds = metrics.get("dataset", {})
    logger.info(
        f"Dataset: seen={ds.get('n_total_seen')} loaded={ds.get('n_loaded')} "
        f"skipped_format={ds.get('n_skipped_format')} "
        f"skipped_length={ds.get('n_skipped_length')} "
        f"(threshold={ds.get('max_input_tokens_threshold')} tokens, "
        f"chat_template={ds.get('used_chat_template')})"
    )
    logger.info(
        f"Requests: total={c['total_requests']} finished={c['finished']} failed={c['failed']}"
    )
    logger.info(
        f"Tokens: prompt={c['total_prompt_tokens']:,} "
        f"output={c['total_output_tokens']:,} "
        f"cached={c['total_cached_tokens']:,} "
        f"(prefix-cache hit rate: {c['prefix_cache_hit_rate'] * 100:.2f}%)"
    )
    if c["finish_reasons"]:
        logger.info(f"Finish reasons: {c['finish_reasons']}")
    if c.get("failed_errors"):
        logger.info(f"Failed-error types: {c['failed_errors']}")
    logger.info("")
    logger.info("Throughput:")
    logger.info(f"  QPS:                  {t['qps']:.4f} req/s")
    logger.info(f"  Output tokens/sec:    {t['output_tokens_per_sec']:.2f}")
    logger.info(f"  Prompt tokens/sec:    {t['prompt_tokens_per_sec']:.2f}")
    logger.info(f"  Total tokens/sec:     {t['total_tokens_per_sec']:.2f}")
    logger.info("")
    logger.info("Latency — TTFT (engine, RequestStateStats.first_token_latency) [CANONICAL]:")
    logger.info(_fmt(lat["ttft_engine_s"]))
    logger.info("Latency — TPOT (mean ms per output token, steady-state decode):")
    logger.info(_fmt(lat["tpot_s"]))
    logger.info("Latency — E2E (per-request end-to-end):")
    logger.info(_fmt(lat["e2e_s"]))
    logger.info("Latency — TTFT (client wall clock; includes asyncio submit overhead) [sanity check]:")
    logger.info(_fmt(lat["ttft_client_s"]))
    logger.info("")
    logger.info(f"Generations:           {output_path}")
    logger.info(f"Per-request metrics:   {per_request_path}")
    logger.info(f"Aggregate metrics:     {metrics_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
