# Environment Setup for Ablation Study

## ✅ New Environment Created: `vllm-ablation`

**Purpose:** Clean, isolated environment for running ablation study experiments

### Installation Details

**Created:** 2025-05-20
**Python:** 3.10
**vLLM:** 0.21.1rc1.dev253+g2514eda53 (editable install from this repo)
**PyTorch:** 2.11.0 (with CUDA 13.0 support)
**Status:** ✅ Ready to use

### Packages Installed

**Core:**
- vLLM (editable from /nvmedata/chenw/vllm-ra)
- PyTorch 2.11.0
- Transformers 5.8.1
- FlashInfer 0.6.11
- Triton 3.6.0

**Dependencies:**
- CUDA bindings (13.2.0)
- cuDNN, cuBLAS, cuSPARSE
- FastAPI, uvicorn (for serving)
- SentencePiece, tokenizers
- All vLLM requirements

### GPU Configuration

**Hardware:** 4× NVIDIA A100 80GB PCIe
**Usage:** Single GPU only (GPU 0)
**Configuration:** `CUDA_VISIBLE_DEVICES=0` (set in all scripts)

### Single GPU Enforcement

All experiment scripts are configured to use only GPU 0:

```bash
# In run_ablation_experiment.sh (line 328)
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/nvmedata/chenw/vllm-ra \
time ${PYTHON_CMD} "${PYTHON_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/inference.log"
```

This ensures:
- ✅ Only GPU 0 is visible to the process
- ✅ No accidental multi-GPU usage
- ✅ Matches production environment (single GPU deployment)
- ✅ Consistent results across experiments

### Activation

```bash
# Activate environment
source /root/miniconda3/bin/activate vllm-ablation

# Verify
python -c "import vllm; print(vllm.__version__)"
# Output: 0.21.1rc1.dev253+g2514eda53
```

### Scripts Updated

All scripts now use `vllm-ablation` environment:
- ✅ `run_ablation_experiment.sh`
- ✅ `test_ablation_setup.sh`
- ✅ `run_all_ablation_experiments.sh`

### Data Setup

**Test Dataset:** `/nvmedata/data/layer1_delta_1k_test.txt`
- Size: 29 MB
- Samples: 1,000
- Format: JSONL (vLLM messages format)
- Already configured in scripts

### Verification

```bash
# Test environment
cd /nvmedata/chenw/vllm-ra/examples
source /root/miniconda3/bin/activate vllm-ablation

# Run single GPU test
CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch
print(f'GPU count visible: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"N/A\"}')
"
# Expected output:
# GPU count visible: 1
# GPU 0: NVIDIA A100 80GB PCIe
```

### Running Experiments

**Quick test:**
```bash
cd /nvmedata/chenw/vllm-ra/examples

./run_ablation_experiment.sh \
    --exp E002 \
    --backend FLASHINFER \
    --batch 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp
```

**Full ablation study:**
```bash
./run_all_ablation_experiments.sh
```

### Important Notes

1. **Single GPU Only:**
   - All experiments use `CUDA_VISIBLE_DEVICES=0`
   - Matches production environment
   - No tensor parallelism (TP=1)

2. **Memory:**
   - A100 80GB provides ample headroom
   - Can run larger batches than originally planned (up to 384-512)
   - All optimizations (FP8, CUDA graphs, MTP) should fit comfortably

3. **vLLM Version:**
   - Using dev version from this repo
   - Editable install (-e) for development
   - Any code changes in repo are immediately reflected

4. **Test Data:**
   - 1,000 samples is good starting point
   - Can scale to 10K or 100K later
   - Real-world prompts (MAI profile extraction)

### Troubleshooting

**If GPU not detected:**
```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

**If vLLM import fails:**
```bash
# Reinstall in editable mode
cd /nvmedata/chenw/vllm-ra
pip install -e . --no-build-isolation
```

**If experiments fail:**
```bash
# Check logs
tail -100 experiment_results/E002/inference.log

# Check GPU memory
nvidia-smi
```

---

## Summary

✅ **Environment:** `vllm-ablation` (Python 3.10, vLLM 0.21.1, PyTorch 2.11.0)
✅ **GPU:** Single GPU (CUDA_VISIBLE_DEVICES=0, A100 80GB)
✅ **Data:** 1K test samples ready (`layer1_delta_1k_test.txt`)
✅ **Scripts:** All updated to use new environment
✅ **Status:** Ready to run experiments

**Next step:** Run experiments!
```bash
./run_all_ablation_experiments.sh
```
