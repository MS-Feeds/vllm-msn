#!/usr/bin/env bash
# Quick sanity-check: verify BF16/FP16/FP32 GEMM work with the CUDA library path fix.
ENV_ROOT="/mnt/remote/guangtaow/conda_env/vllm_glm5_py312"
_NVIDIA_LIB="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$_NVIDIA_LIB" ]]; then
  export LD_LIBRARY_PATH="$_NVIDIA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "LD_LIBRARY_PATH starts with: ${LD_LIBRARY_PATH%%:*}"

"$ENV_ROOT/bin/python" - <<'PY'
import torch
print("PyTorch", torch.__version__, "  CUDA", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), " SM:", torch.cuda.get_device_capability(0))
ok = True
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    try:
        a = torch.randn(64, 64, dtype=dt, device="cuda:0")
        b = torch.randn(64, 64, dtype=dt, device="cuda:0")
        c = torch.mm(a, b)
        print(f"  PASS  {str(dt):25s} (64x64)x(64x64)  sum={c.sum().item():.3f}")
    except Exception as e:
        print(f"  FAIL  {str(dt):25s}  -> {e}")
        ok = False
print("ALL OK" if ok else "SOME FAILED")
PY
