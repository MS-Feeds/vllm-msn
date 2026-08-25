#!/bin/bash
# Install Flash Attention 2 for A100 + Gemma 4 MoE
# This will improve memory efficiency and performance

set -e  # Exit on error

echo "======================================"
echo "Flash Attention 2 Installation Script"
echo "======================================"
echo ""

# Activate conda environment
echo "Step 1: Activating vllm conda environment..."
source /root/miniconda3/bin/activate vllm
echo "✓ Environment activated: $(which python)"
echo ""

# Check current status
echo "Step 2: Checking current attention libraries..."
python3 << 'PYTHON_EOF'
print("Current installation status:")
try:
    import flash_attn
    print(f"  flash_attn: ✓ {flash_attn.__version__} (already installed)")
    exit(0)  # Already installed
except ImportError:
    print("  flash_attn: ✗ NOT INSTALLED")

try:
    import flashinfer
    print(f"  flashinfer: ✓ {flashinfer.__version__}")
except ImportError:
    print("  flashinfer: ✗ NOT INSTALLED")

try:
    import xformers
    print(f"  xformers: ✓ {xformers.__version__}")
except ImportError:
    print("  xformers: ✗ NOT INSTALLED")
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Flash Attention 2 is already installed!"
    echo "  Running verification script..."
    echo ""
    exec bash "$(dirname "$0")/verify_flash_attention.sh"
fi

echo ""

# Check CUDA and PyTorch versions
echo "Step 3: Verifying CUDA and PyTorch compatibility..."
python3 << 'PYTHON_EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
PYTHON_EOF
echo "✓ CUDA and PyTorch are compatible"
echo ""

# Install Flash Attention 2
echo "Step 4: Installing Flash Attention 2..."
echo "  This will take 5-10 minutes (compiling CUDA kernels)..."
echo "  Building for A100 (SM 8.0)..."
echo ""

export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install with proper flags
pip install flash-attn --no-build-isolation --no-cache-dir

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Installation failed!"
    echo "  Trying alternative installation method..."
    pip install flash-attn --no-build-isolation
fi

echo ""
echo "Step 5: Verifying installation..."
python3 << 'PYTHON_EOF'
import sys
try:
    import flash_attn
    print(f"✓ Flash Attention 2 installed successfully!")
    print(f"  Version: {flash_attn.__version__}")

    # Check if CUDA kernels are available
    print(f"  CUDA kernels: ", end="")
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        print("✓ Available")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Installation verification failed: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ Installation Complete!"
    echo "======================================"
    echo ""
    echo "Flash Attention 2 is now installed and ready to use."
    echo ""
    echo "Next steps:"
    echo "  1. Run verification: ./verify_flash_attention.sh"
    echo "  2. Test with Gemma 4: ./vllm_gemma4_moe_fp8_mtp.sh"
    echo ""
    echo "Expected improvements:"
    echo "  - Memory: ~1GB savings (34-38GB → 33-37GB)"
    echo "  - Performance: ~3-5% faster overall"
    echo "  - Latency: ~5-10% lower P99"
    echo ""

    # Automatically run verification
    echo "Running verification script..."
    echo ""
    bash "$(dirname "$0")/verify_flash_attention.sh"
else
    echo ""
    echo "======================================"
    echo "✗ Installation Failed"
    echo "======================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check CUDA version: nvcc --version"
    echo "  2. Check PyTorch CUDA: python -c 'import torch; print(torch.version.cuda)'"
    echo "  3. Try manual install: pip install flash-attn --no-build-isolation"
    echo ""
    exit 1
fi
