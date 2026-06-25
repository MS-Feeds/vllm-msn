"""Parameter space definition for Gemma4 MoE serving optimization on A100 80GB.

vLLM version: 0.21.1rc1.dev270+g6cbe448ee

Fixed params (always use, not searchable):
  - quantization: "fp8"
  - enforce_eager: False (CUDA graphs on)
  - gpu_memory_utilization: 0.95
  - kv_cache_dtype: "auto"

Fixed env vars (A100 sm_80 constraints):
  - VLLM_USE_FLASHINFER_MOE_FP8=0 (requires sm_90+)
  - VLLM_USE_FLASHINFER_SAMPLER=0
  - VLLM_MOE_USE_DEEP_GEMM=0 (requires sm_90+)

Known crash on A100 (DO NOT use):
  - --moe-backend triton: Triton FP8 requires sm_89+ (supports_fp8 check)
  - --moe-backend humming: not in FP8 oracle mapping (only MXFP4)
  - --moe-backend deep_gemm: requires sm_90+ (Hopper/Blackwell only)
  - max_num_batched_tokens=32768: OOM on 80GB
  - kv_cache_dtype=fp8_e4m3: Triton fp8e4nv codegen needs sm_89+

Working on A100:
  - --moe-backend cutlass: TritonOrCutlassExperts (2017 tok/s, slightly slower)
  - --moe-backend marlin: Marlin kernel (sm_75+, designed for non-Ada FP8)
  - VLLM_HUMMING_MOE_GEMM_TYPE=grouped: bypasses FP8 oracle (2030 tok/s, best)
  - VLLM_TEST_FORCE_FP8_MARLIN=1: force Marlin FP8 MoE kernel
"""
from __future__ import annotations

import random

# --- Searchable parameter space ---
PARAM_SPACE = {
    "max_num_seqs": [64, 96, 128, 192, 256],
    "max_num_batched_tokens": [8192, 16384, 24576],
    "max_model_len": [16384, 24576],
    "spec_tokens": [3, 4, 5, 7, 9],
    "moe_backend": ["auto", "cutlass", "marlin"],
    "VLLM_HUMMING_MOE_GEMM_TYPE": ["indexed", "grouped"],
    "VLLM_HUMMING_USE_F16_ACCUM": ["0", "1"],
    "VLLM_TEST_FORCE_FP8_MARLIN": ["0", "1"],
    "enable_prefix_caching": [True, False],
    "enable_chunked_prefill": [True, False],
    "async_scheduling": [True, False],
}

# --- Fixed params (always applied) ---
FIXED_PARAMS = {
    "quantization": "fp8",
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "gpu_memory_utilization": 0.95,
}

# Parameters that are env vars (set before vllm import)
ENV_VAR_PARAMS = {
    "VLLM_HUMMING_MOE_GEMM_TYPE",
    "VLLM_HUMMING_USE_F16_ACCUM",
    "VLLM_TEST_FORCE_FP8_MARLIN",
}

# Fixed env vars for A100 80GB (sm_80)
A100_FIXED_ENV = {
    "VLLM_USE_FLASHINFER_MOE_FP8": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_MOE_USE_DEEP_GEMM": "0",
}

# E011 baseline config (current best: 2020 output_tps)
BASELINE_CONFIG = {
    **FIXED_PARAMS,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 16384,
    "max_model_len": 24576,
    "spec_tokens": 5,
}


def validate_config(cfg: dict) -> tuple[bool, str]:
    """Validate a config dict. Returns (is_valid, error_message)."""
    mns = cfg.get("max_num_seqs", 128)
    mnbt = cfg.get("max_num_batched_tokens", 16384)
    if mnbt < mns:
        return False, f"max_num_batched_tokens ({mnbt}) < max_num_seqs ({mns})"

    mml = cfg.get("max_model_len", 24576)
    if mml < 8192:
        return False, f"max_model_len ({mml}) too small for 8192-token outputs"

    spec = cfg.get("spec_tokens", 0)
    if spec > 0 and cfg.get("quantization") is None:
        return False, "MTP requires quantization=fp8 (assistant model is FP8)"

    # Known crashes on A100
    moe_be = cfg.get("moe_backend", "auto")
    if moe_be == "triton":
        return False, "moe_backend=triton crashes on A100 (Triton FP8 needs sm_89+)"
    if moe_be == "humming":
        return False, "moe_backend=humming crashes (not in FP8 oracle, only MXFP4)"
    if moe_be == "deep_gemm":
        return False, "moe_backend=deep_gemm crashes on A100 (needs sm_90+)"

    if mnbt >= 32768:
        return False, "max_num_batched_tokens=32768 OOMs on A100 80GB"

    if cfg.get("kv_cache_dtype") == "fp8_e4m3":
        return False, "fp8_e4m3 KV cache crashes on A100 (Triton fp8e4nv needs sm_89+)"

    return True, ""


def config_to_env_vars(cfg: dict) -> dict[str, str]:
    """Extract env var settings from config."""
    env = dict(A100_FIXED_ENV)
    for key in ENV_VAR_PARAMS:
        if key in cfg:
            env[key] = str(cfg[key])
    return env


def config_to_llm_kwargs(cfg: dict, scenario_cfg: dict) -> dict:
    """Convert config dict to vllm.LLM constructor kwargs."""
    kwargs = {
        "trust_remote_code": True,
        "max_model_len": cfg.get("max_model_len", 24576),
        "max_num_seqs": cfg.get("max_num_seqs", 128),
        "max_num_batched_tokens": cfg.get("max_num_batched_tokens",
                                          scenario_cfg.get("max_num_batched_tokens", 16384)),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.95),
        "enforce_eager": cfg.get("enforce_eager", False),
        "seed": 0,
    }
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    moe_backend = cfg.get("moe_backend")
    if moe_backend and moe_backend != "auto":
        kwargs["moe_backend"] = moe_backend
    if "enable_prefix_caching" in cfg:
        kwargs["enable_prefix_caching"] = cfg["enable_prefix_caching"]
    if "enable_chunked_prefill" in cfg:
        kwargs["enable_chunked_prefill"] = cfg["enable_chunked_prefill"]
    if "async_scheduling" in cfg:
        kwargs["async_scheduling"] = cfg["async_scheduling"]
    if cfg.get("attention_backend"):
        kwargs["attention_backend"] = cfg["attention_backend"]
    if cfg.get("hf_overrides"):
        kwargs["hf_overrides"] = cfg["hf_overrides"]
    if cfg.get("kv_cache_dtype_skip_layers"):
        kwargs["kv_cache_dtype_skip_layers"] = cfg["kv_cache_dtype_skip_layers"]
    return kwargs


def config_summary(cfg: dict) -> str:
    """One-line summary string for results.tsv."""
    parts = []
    q = cfg.get("quantization", "bf16") or "bf16"
    parts.append(q)
    if not cfg.get("enforce_eager", True):
        parts.append("CG")
    spec = cfg.get("spec_tokens", 0)
    if spec > 0:
        parts.append(f"MTP-k{spec}")
    parts.append(f"mns={cfg.get('max_num_seqs', 128)}")
    parts.append(f"mnbt={cfg.get('max_num_batched_tokens', 16384)}")
    mml = cfg.get("max_model_len", 24576)
    if mml != 24576:
        parts.append(f"mml={mml}")
    moe_be = cfg.get("moe_backend")
    if moe_be and moe_be != "auto":
        parts.append(f"moe={moe_be}")
    humming = cfg.get("VLLM_HUMMING_MOE_GEMM_TYPE")
    if humming is not None:
        parts.append(f"humming-{humming}")
    if cfg.get("VLLM_HUMMING_USE_F16_ACCUM") == "1":
        parts.append("f16acc")
    if cfg.get("VLLM_TEST_FORCE_FP8_MARLIN") == "1":
        parts.append("marlin-forced")
    if cfg.get("enable_prefix_caching") is False:
        parts.append("no-prefix-cache")
    if cfg.get("enable_chunked_prefill") is False:
        parts.append("no-chunked-pf")
    if cfg.get("kv_cache_dtype", "auto") == "fp8_e4m3":
        parts.append("kv-fp8")
    if cfg.get("async_scheduling") is True:
        parts.append("async-sched")
    return " | ".join(parts)


def parse_summary(summary: str) -> dict:
    """Reverse of config_summary(): parse a summary string back into a config dict."""
    cfg = dict(FIXED_PARAMS)
    cfg["max_num_seqs"] = 128
    cfg["max_num_batched_tokens"] = 16384
    cfg["max_model_len"] = 24576
    cfg["spec_tokens"] = 0

    parts = [p.strip() for p in summary.split("|")]
    for part in parts:
        if part == "fp8":
            cfg["quantization"] = "fp8"
        elif part == "bf16":
            cfg["quantization"] = None
        elif part == "CG":
            cfg["enforce_eager"] = False
        elif part.startswith("MTP-k"):
            cfg["spec_tokens"] = int(part[5:])
        elif part.startswith("mns="):
            cfg["max_num_seqs"] = int(part[4:])
        elif part.startswith("mnbt="):
            cfg["max_num_batched_tokens"] = int(part[5:])
        elif part.startswith("mml="):
            cfg["max_model_len"] = int(part[4:])
        elif part.startswith("moe="):
            cfg["moe_backend"] = part[4:]
        elif part.startswith("humming-"):
            cfg["VLLM_HUMMING_MOE_GEMM_TYPE"] = part[8:]
        elif part == "f16acc":
            cfg["VLLM_HUMMING_USE_F16_ACCUM"] = "1"
        elif part == "marlin-forced":
            cfg["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
        elif part == "no-prefix-cache":
            cfg["enable_prefix_caching"] = False
        elif part == "no-chunked-pf":
            cfg["enable_chunked_prefill"] = False
        elif part == "kv-fp8":
            cfg["kv_cache_dtype"] = "fp8_e4m3"
        elif part == "async-sched":
            cfg["async_scheduling"] = True
        elif part.startswith("mem="):
            cfg["gpu_memory_utilization"] = float(part[4:])
    return cfg


def reproduce_command(cfg: dict) -> str:
    """Generate a copy-paste command to reproduce a config on GPU."""
    env_vars = ["VLLM_HUMMING_MOE_GEMM_TYPE", "VLLM_HUMMING_USE_F16_ACCUM",
                "VLLM_TEST_FORCE_FP8_MARLIN"]
    lines = [f"unset {v}" for v in env_vars]
    lines.append("")

    import json
    cfg_clean = {k: v for k, v in cfg.items() if v is not None}
    json_str = json.dumps(cfg_clean, separators=(",", ":"))
    lines.append(f"echo '{json_str}' > /tmp/cfg_reproduce.json")
    lines.append("python run_experiment.py --config /tmp/cfg_reproduce.json --prompts 1000 --reps 1")
    return "\n".join(lines)


def config_to_serve_cmd(cfg: dict, model_path: str = "${MODEL_PATH}",
                        port: int = 8100) -> str:
    """Convert a config dict to a deployable 'vllm serve' command."""
    args = [
        f"vllm serve {model_path}",
        f"  --port {port}",
        "  --dtype auto",
        "  --trust-remote-code",
        f"  --max-model-len {cfg.get('max_model_len', 24576)}",
        f"  --max-num-seqs {cfg.get('max_num_seqs', 128)}",
        f"  --max-num-batched-tokens {cfg.get('max_num_batched_tokens', 16384)}",
        f"  --gpu-memory-utilization {cfg.get('gpu_memory_utilization', 0.95)}",
    ]
    if cfg.get("quantization"):
        args.append(f"  --quantization {cfg['quantization']}")
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        args.append(f"  --kv-cache-dtype {cfg['kv_cache_dtype']}")
    if not cfg.get("enforce_eager", False):
        args.append("  --no-enable-log-requests")
    else:
        args.append("  --enforce-eager")
    moe_backend = cfg.get("moe_backend")
    if moe_backend and moe_backend != "auto":
        args.append(f"  --moe-backend {moe_backend}")
    if cfg.get("enable_prefix_caching") is False:
        args.append("  --no-enable-prefix-caching")
    if cfg.get("enable_chunked_prefill") is False:
        args.append("  --no-enable-chunked-prefill")
    if cfg.get("async_scheduling") is True:
        args.append("  --async-scheduling")
    spec = cfg.get("spec_tokens", 0)
    if spec > 0:
        args.append(f"  --spec-model ${{ASSISTANT_MODEL_PATH}}")
        args.append(f"  --spec-tokens {spec}")

    cmd = " \\\n".join(args)

    env_lines = []
    for key in ENV_VAR_PARAMS:
        if key in cfg:
            env_lines.append(f"export {key}={cfg[key]}")
    for k, v in A100_FIXED_ENV.items():
        env_lines.append(f"export {k}={v}")

    env_block = "\n".join(env_lines)
    return f"# Environment variables\n{env_block}\n\n# Serve command\n{cmd}"


def generate_next_configs(history: list[dict], n: int = 4) -> list[dict]:
    """Generate the next batch of configs to try.

    Strategy:
    - If history is empty or small: start with systematic one-at-a-time variations from baseline
    - Otherwise: perturb the best-known config in unexplored directions
    """
    if not history:
        return _initial_exploration(n)

    valid_runs = [h for h in history if h.get("status") == "ok" and h.get("output_tps", 0) > 0]
    if not valid_runs:
        return _initial_exploration(n)

    best = max(valid_runs, key=lambda h: h["output_tps"])
    best_cfg = best.get("config") or parse_summary(best.get("config_summary", ""))

    tried_summaries = {h.get("config_summary", "") for h in history}
    candidates = []

    for param, values in PARAM_SPACE.items():
        for val in values:
            if val == best_cfg.get(param):
                continue
            candidate = {**best_cfg, **FIXED_PARAMS, param: val}
            valid, _ = validate_config(candidate)
            if valid and config_summary(candidate) not in tried_summaries:
                candidates.append(candidate)

    if len(candidates) <= n:
        return candidates

    random.shuffle(candidates)
    return candidates[:n]


def _initial_exploration(n: int) -> list[dict]:
    """First round: vary one parameter at a time from baseline."""
    configs = []
    variations = [
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"},
        {"async_scheduling": True},
        {"VLLM_TEST_FORCE_FP8_MARLIN": "1"},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped", "VLLM_HUMMING_USE_F16_ACCUM": "1"},
        {"enable_prefix_caching": False},
        {"enable_chunked_prefill": False},
        {"moe_backend": "marlin"},
        {"spec_tokens": 4},
        {"max_num_seqs": 192},
        {"max_num_seqs": 256},
        {"max_num_batched_tokens": 24576},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "indexed"},
        {"moe_backend": "cutlass"},
        {"max_num_seqs": 96},
    ]
    for v in variations[:n]:
        cfg = {**BASELINE_CONFIG, **v}
        valid, _ = validate_config(cfg)
        if valid:
            configs.append(cfg)
    return configs
