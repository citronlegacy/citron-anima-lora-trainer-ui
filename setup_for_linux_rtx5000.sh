#!/usr/bin/env bash
# =============================================================================
# setup_for_linux_rtx5000.sh — Anima LoRA Trainer setup for RTX 5000-series
#
# RTX 5070 / 5080 / 5090 require PyTorch compiled with sm_120 (CUDA 12.8).
# Standard pip torch wheels lack this — this script installs the correct build.
#
# Ref: https://github.com/Stephensmetana/nvidia-rtx5070-pytorch-guide
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Anima LoRA Trainer — Linux Setup (RTX 5000-series)"
echo "============================================================"
echo "  Targeting: torch==2.9.1+cu128 (sm_120 / CUDA 12.8)"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# 1. Python venv (prefer python3.10, fall back to 3.11 / 3.12 / python3)
# -----------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3.10 python3.11 python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python 3 interpreter found. Install Python 3.10+ and try again."
    exit 1
fi

PY_VER=$("$PYTHON_BIN" --version 2>&1)
echo "Using Python: $PYTHON_BIN  ($PY_VER)"

if [ ! -d "venv" ]; then
    echo "[1/7] Creating virtual environment..."
    "$PYTHON_BIN" -m venv venv
    echo "      ✓ venv created."
else
    echo "[1/7] venv already exists — skipping creation."
fi

source venv/bin/activate
echo "      ✓ venv activated."

pip install --upgrade pip --quiet

# -----------------------------------------------------------------------------
# 2. Clone kohya-ss/sd-scripts
# -----------------------------------------------------------------------------
echo ""
if [ ! -d "sd-scripts" ]; then
    echo "[2/7] Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    echo "      ✓ sd-scripts cloned."
else
    echo "[2/7] sd-scripts already present — skipping clone."
fi

# -----------------------------------------------------------------------------
# 3. Install PyTorch 2.9.1+cu128 FIRST (before sd-scripts requirements)
#    This ensures the correct sm_120 build is installed and not overwritten.
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Installing PyTorch 2.9.1+cu128 for RTX 5000-series (sm_120)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
echo "      ✓ PyTorch cu128 installed."

# Verify sm_120 support
echo ""
echo "      Verifying GPU support..."
python - <<'PYEOF'
import torch
print(f"      PyTorch:  {torch.__version__}")
print(f"      CUDA:     {torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"      GPU:      {name}")
    print(f"      Compute:  sm_{cap[0]}{cap[1]}")
    x = torch.randn(100, 100, device='cuda')
    _ = x @ x
    print("      ✓ GPU tensor ops working.")
else:
    print("      WARNING: CUDA not available — check your drivers.")
import torchvision
print(f"      torchvision: {torchvision.__version__}")
PYEOF

# -----------------------------------------------------------------------------
# 4. Install sd-scripts requirements (must run from inside sd-scripts/)
#    torch is already installed above; pip will skip it.
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Installing sd-scripts requirements..."
pushd sd-scripts > /dev/null
pip install -r requirements.txt
popd > /dev/null
echo "      ✓ sd-scripts requirements installed."

# Re-pin torch to cu128 in case sd-scripts overwrote it
echo ""
echo "      Re-pinning PyTorch to cu128 (in case sd-scripts overwrote it)..."
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --quiet
echo "      ✓ PyTorch cu128 confirmed."

# -----------------------------------------------------------------------------
# 5. Install app requirements (gradio, toml)
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Installing app requirements..."
pip install -r requirements.txt
echo "      ✓ App requirements installed."

# -----------------------------------------------------------------------------
# 6. Configure accelerate (default single GPU)
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Configuring accelerate (default)..."
accelerate config default
echo "      ✓ accelerate configured."

# -----------------------------------------------------------------------------
# 7. Download Anima models (idempotent — skips if already present)
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Downloading Anima models (~5.6 GB total)..."
echo "      This may take a while depending on your connection."
echo ""

mkdir -p models/anima/dit
mkdir -p models/anima/text_encoder
mkdir -p models/anima/vae

DIT_PATH="models/anima/dit/anima-preview.safetensors"
QWEN_PATH="models/anima/text_encoder/qwen_3_06b_base.safetensors"
VAE_PATH="models/anima/vae/qwen_image_vae.safetensors"

if [ ! -f "$DIT_PATH" ]; then
    echo "  Downloading DiT model (4.18 GB)..."
    wget -c --show-progress \
        -O "$DIT_PATH" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors"
    echo "  ✓ DiT model downloaded."
else
    echo "  ✓ DiT model already present — skipping."
fi

if [ ! -f "$QWEN_PATH" ]; then
    echo "  Downloading Qwen3 text encoder (1.19 GB)..."
    wget -c --show-progress \
        -O "$QWEN_PATH" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
    echo "  ✓ Qwen3 text encoder downloaded."
else
    echo "  ✓ Qwen3 text encoder already present — skipping."
fi

if [ ! -f "$VAE_PATH" ]; then
    echo "  Downloading Qwen-Image VAE (254 MB)..."
    wget -c --show-progress \
        -O "$VAE_PATH" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
    echo "  ✓ VAE model downloaded."
else
    echo "  ✓ VAE model already present — skipping."
fi

echo ""
echo "============================================================"
echo "  Setup complete! (RTX 5000-series)"
echo "  Start the trainer with:  bash run_linux.sh"
echo "============================================================"
