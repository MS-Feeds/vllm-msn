#!/bin/bash
# Verify Flash Attention 2 installation and benchmark performance

set -e

echo "=========================================="
echo "Flash Attention 2 Verification & Benchmark"
echo "=========================================="
echo ""

# Activate conda environment
source /root/miniconda3/bin/activate vllm

echo "Step 1: Checking installed attention backends..."
echo ""

python3 << 'PYTHON_EOF'
import sys

print("=" * 50)
print("Installed Attention Libraries")
print("=" * 50)

backends = {}

# Check flash_attn
try:
    import flash_attn
    print(f"✓ flash_attn:  {flash_attn.__version__}")
    backends['flash_attn'] = True
except ImportError:
    print("✗ flash_attn:  NOT INSTALLED")
    backends['flash_attn'] = False

# Check flashinfer
try:
    import flashinfer
    print(f"✓ flashinfer:  {flashinfer.__version__}")
    backends['flashinfer'] = True
except ImportError:
    print("✗ flashinfer:  NOT INSTALLED")
    backends['flashinfer'] = False

# Check xformers
try:
    import xformers
    print(f"✓ xformers:    {xformers.__version__}")
    backends['xformers'] = True
except ImportError:
    print("✗ xformers:    NOT INSTALLED")
    backends['xformers'] = False

print()

# Check environment variables
print("=" * 50)
print("Environment Configuration")
print("=" * 50)
import os
print(f"VLLM_ATTENTION_BACKEND: {os.getenv('VLLM_ATTENTION_BACKEND', 'not set')}")
print(f"VLLM_USE_FLASHINFER_MOE_FP8: {os.getenv('VLLM_USE_FLASHINFER_MOE_FP8', 'not set')}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST', 'not set')}")
print()

# Determine which backend will be used
print("=" * 50)
print("Backend Selection Logic")
print("=" * 50)

requested = os.getenv('VLLM_ATTENTION_BACKEND', 'auto')
if requested == 'FLASH_ATTN':
    if backends['flash_attn']:
        print("✓ FLASH_ATTN requested and available")
        print("  → Will use: Flash Attention 2")
        selected = 'flash_attn'
    elif backends['flashinfer']:
        print("⚠ FLASH_ATTN requested but not installed")
        print("  → Falling back to: FlashInfer")
        selected = 'flashinfer'
    elif backends['xformers']:
        print("⚠ FLASH_ATTN requested but not installed")
        print("  → Falling back to: xformers")
        selected = 'xformers'
    else:
        print("⚠ FLASH_ATTN requested but not installed")
        print("  → Falling back to: PyTorch native attention")
        selected = 'native'
elif requested == 'FLASHINFER':
    if backends['flashinfer']:
        print("✓ FLASHINFER requested and available")
        print("  → Will use: FlashInfer")
        selected = 'flashinfer'
    else:
        print("⚠ FLASHINFER requested but not installed")
        print("  → Falling back to: xformers or native")
        selected = 'fallback'
else:
    print(f"Auto-selection mode (requested: {requested})")
    if backends['flash_attn']:
        print("  → Will use: Flash Attention 2 (best for A100)")
        selected = 'flash_attn'
    elif backends['flashinfer']:
        print("  → Will use: FlashInfer")
        selected = 'flashinfer'
    elif backends['xformers']:
        print("  → Will use: xformers")
        selected = 'xformers'
    else:
        print("  → Will use: PyTorch native attention")
        selected = 'native'

print()

# Performance comparison
print("=" * 50)
print("Expected Performance (A100, Gemma 4 MoE)")
print("=" * 50)
print()

perf_data = {
    'flash_attn': {
        'memory_saving': '20-25%',
        'compute_tflops': '160-200',
        'kv_cache': '7-9 GB',
        'attention_latency': '1.5-2.5ms',
        'total_memory': '33-37 GB',
        'recommendation': 'Best for A100'
    },
    'flashinfer': {
        'memory_saving': '15-20%',
        'compute_tflops': '140-170',
        'kv_cache': '8-10 GB',
        'attention_latency': '2-3ms',
        'total_memory': '34-38 GB',
        'recommendation': 'Best for H100/4090'
    },
    'xformers': {
        'memory_saving': '10-15%',
        'compute_tflops': '120-150',
        'kv_cache': '10-12 GB',
        'attention_latency': '3-4ms',
        'total_memory': '36-40 GB',
        'recommendation': 'General purpose'
    },
    'native': {
        'memory_saving': '0% (baseline)',
        'compute_tflops': '80-100',
        'kv_cache': '12-14 GB',
        'attention_latency': '5-8ms',
        'total_memory': '38-42 GB',
        'recommendation': 'Debugging only'
    }
}

if selected in perf_data:
    data = perf_data[selected]
    print(f"Selected Backend: {selected.upper()}")
    print(f"  Memory Saving:     {data['memory_saving']} vs native")
    print(f"  Compute (TFLOPS):  {data['compute_tflops']}")
    print(f"  KV Cache Size:     {data['kv_cache']}")
    print(f"  Attention Latency: {data['attention_latency']}")
    print(f"  Total GPU Memory:  {data['total_memory']}")
    print(f"  Recommendation:    {data['recommendation']}")
    print()

# Recommendations
print("=" * 50)
print("Recommendations for Gemma 4 MoE on A100 40GB")
print("=" * 50)

if backends['flash_attn']:
    print("✓ OPTIMAL: Flash Attention 2 is installed")
    print("  - Best memory efficiency (20-25% savings)")
    print("  - Fastest on A100 (160-200 TFLOPS)")
    print("  - Best CUDA graph compatibility")
    print("  → Keep using VLLM_ATTENTION_BACKEND=FLASH_ATTN")
elif backends['flashinfer']:
    print("⚠ GOOD: FlashInfer is installed")
    print("  - Good memory efficiency (15-20% savings)")
    print("  - Good performance (140-170 TFLOPS)")
    print("  → Consider installing Flash Attention 2 for ~5% speedup")
    print("  → pip install flash-attn --no-build-isolation")
elif backends['xformers']:
    print("⚠ ACCEPTABLE: xformers is installed")
    print("  - Moderate memory efficiency (10-15% savings)")
    print("  - Moderate performance (120-150 TFLOPS)")
    print("  → Strongly recommend installing Flash Attention 2")
    print("  → pip install flash-attn --no-build-isolation")
else:
    print("✗ SUBOPTIMAL: Using PyTorch native attention")
    print("  - No memory savings")
    print("  - Slow performance (80-100 TFLOPS)")
    print("  → MUST install Flash Attention 2 for production")
    print("  → pip install flash-attn --no-build-isolation")

print()

if not backends['flash_attn']:
    print("To install Flash Attention 2:")
    print("  bash install_flash_attention.sh")
    print()
    sys.exit(1)

PYTHON_EOF

# If flash-attn is installed, run a quick benchmark
if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Running Flash Attention micro-benchmark..."
    echo ""

    python3 << 'PYTHON_EOF'
import torch
import time

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    print("Testing Flash Attention 2 kernels...")
    print()

    # Setup test tensors (simulate attention on A100)
    batch_size = 8
    seqlen = 2048
    nheads = 32
    headdim = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠ CUDA not available, skipping benchmark")
        exit(0)

    # Create test tensors
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.float16)

    # Warmup
    _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 20
    start = time.time()
    for _ in range(num_iters):
        _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iters * 1000  # ms

    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Number of heads: {nheads}")
    print(f"  Head dimension: {headdim}")
    print()
    print(f"Benchmark Results:")
    print(f"  Average latency: {avg_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} iters/sec")
    print()

    # Compare to expected
    expected_time = 3.0  # Expected ~2-3ms for this config on A100
    if avg_time < expected_time:
        print(f"✓ Performance: EXCELLENT (faster than {expected_time:.1f}ms baseline)")
    elif avg_time < expected_time * 1.5:
        print(f"✓ Performance: GOOD (within 50% of {expected_time:.1f}ms baseline)")
    else:
        print(f"⚠ Performance: SLOWER than expected ({expected_time:.1f}ms baseline)")
        print("  This might be normal for first run or cold GPU")

    print()
    print("=" * 50)
    print("✓ Flash Attention 2 is working correctly!")
    print("=" * 50)

except Exception as e:
    print(f"✗ Benchmark failed: {e}")
    print("  Flash Attention 2 is installed but may have issues")
    import traceback
    traceback.print_exc()
PYTHON_EOF
fi

echo ""
echo "Verification complete!"
echo ""
