#!/bin/bash
# ============================================================
# Gemmra — AMD MI300X Environment Setup
# Run this FIRST on the AMD Developer Cloud notebook
# ============================================================
#
# IMPORTANT: Do NOT create a Python venv before running this.
# The AMD cloud's system Python already has ROCm-enabled PyTorch.
# Creating a venv installs CPU-only PyTorch from PyPI, breaking GPU access.
#
# If you already created a venv:
#   deactivate
#   rm -rf <venv_dir>
# Then re-run this script with the system Python.
#
# Alternatively, if you MUST use a venv, run this script first (it will
# install ROCm PyTorch into the venv via the correct index URL).

set -e  # Exit on error

echo "=========================================="
echo "  Gemmra — Environment Setup"
echo "=========================================="

# Step 1: Set environment variables
echo "[1/7] Setting environment variables..."
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export HF_HUB_DISABLE_XET=1

# AMD/vLLM AITER optimizations (2-4x inference speedup on MI300X)
# Source: AMD ROCm Performance Tuning Guide, June 2026
export VLLM_ROCM_USE_AITER=1
export ROCM_AITER_FA=1
export HIP_FORCE_DEV_KERNARG=1
export SAFETENSORS_FAST_GPU=1

# Make persistent
grep -q 'HSA_OVERRIDE_GFX_VERSION' ~/.bashrc 2>/dev/null || {
    echo 'export HSA_OVERRIDE_GFX_VERSION=9.4.2' >> ~/.bashrc
    echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
    echo 'export VLLM_ROCM_USE_AITER=1' >> ~/.bashrc
    echo 'export ROCM_AITER_FA=1' >> ~/.bashrc
    echo 'export HIP_FORCE_DEV_KERNARG=1' >> ~/.bashrc
    echo 'export SAFETENSORS_FAST_GPU=1' >> ~/.bashrc
}

# Step 2: Detect if we're in a venv and fix PyTorch if needed
echo "[2/7] Checking Python environment..."
IN_VENV=false
if [ -n "$VIRTUAL_ENV" ]; then
    IN_VENV=true
    echo "  ⚠️  Running inside a virtual environment: $VIRTUAL_ENV"
    echo "  Will install ROCm PyTorch into the venv..."
fi

# Step 3: Ensure ROCm-compatible PyTorch
echo "[3/7] Ensuring ROCm-compatible PyTorch..."
GPU_OK=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")

if [ "$GPU_OK" = "no" ]; then
    echo "  ⚠️  Current PyTorch does NOT detect GPU (CPU-only or CUDA build)"
    echo "  Installing ROCm-enabled PyTorch..."
    
    # Detect ROCm version (handles both X.Y and X.Y.Z formats)
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || echo "6.2")
    ROCM_MAJOR=$(echo "$ROCM_VER" | cut -d. -f1)
    echo "  ROCm version detected: $ROCM_VER (major: $ROCM_MAJOR)"
    
    # Choose pip command: prefer uv pip (faster), fall back to regular pip
    if command -v uv &>/dev/null; then
        PIP_CMD="uv pip"
        echo "  Using uv pip (faster install)"
    else
        PIP_CMD="pip"
    fi
    
    # Install ROCm PyTorch from the correct index
    # Try detected version first, then fall back through known-good versions
    # ROCm 7.x is available on newer AMD cloud instances (June 2026+)
    echo "  Trying ROCm ${ROCM_VER} index first..."
    $PIP_CMD install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VER}" 2>/dev/null || \
    $PIP_CMD install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm7.2" 2>/dev/null || \
    $PIP_CMD install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm6.4" 2>/dev/null || \
    $PIP_CMD install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm6.3" 2>/dev/null || \
    $PIP_CMD install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm6.2" 2>/dev/null || {
        echo "  ❌ Could not install ROCm PyTorch automatically."
        echo "  Detected ROCm version: $ROCM_VER"
        echo "  Try manually:"
        echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${ROCM_VER}"
        echo "  Or check available versions at: https://download.pytorch.org/whl/"
        exit 1
    }
fi

# Step 4: Verify GPU
echo "[4/7] Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'ERROR: GPU not detected even after installing ROCm PyTorch!'
print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
print(f'  ✅ PyTorch: {torch.__version__}')
"

# Step 5: Install bitsandbytes (CRITICAL — must be before Unsloth)
echo "[5/7] Installing ROCm-compatible bitsandbytes..."
pip install --force-reinstall --no-cache-dir --no-deps \
  "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"

# Step 6: Install Unsloth + TRL + other dependencies
echo "[6/7] Installing Unsloth, TRL, and other dependencies..."
pip install "unsloth[amd]"
pip install git+https://github.com/huggingface/trl.git
pip install duckdb streamlit plotly pandas pyarrow requests lxml datasets pyyaml scikit-learn

# Step 7: Verify full stack
echo "[7/7] Verifying installation..."
python3 -c "
print('Checking imports...')
import torch; print(f'  ✅ PyTorch {torch.__version__} (GPU: {torch.cuda.is_available()})')
import bitsandbytes; print(f'  ✅ bitsandbytes {bitsandbytes.__version__}')
import unsloth; print('  ✅ Unsloth imported')
import trl; print(f'  ✅ TRL {trl.__version__}')
import duckdb; print(f'  ✅ DuckDB {duckdb.__version__}')
print()
print('All dependencies verified! ✅')
print()
print('Next steps:')
print('  1. Run the smoke test: python src/setup/smoke_test.py')
print('  2. Start training:    python src/training/01_sft_train.py')
print('     (Training data is already in the repo — no data pipeline needed)')
"

echo ""
echo "=========================================="
echo "  Setup complete! ✅"
echo "=========================================="
