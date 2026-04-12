#!/usr/bin/env bash
# =============================================================================
# setup_for_linux.sh — Anima LoRA Trainer local setup (Linux / macOS)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Anima LoRA Trainer — Linux Setup"
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

if [ ! -d ".venv" ]; then
    echo "[1/6] Creating virtual environment..."
    "$PYTHON_BIN" -m venv .venv
    echo "      ✓ .venv created."
else
    echo "[1/6] .venv already exists — skipping creation."
fi

source .venv/bin/activate
echo "      ✓ venv activated."

pip install --upgrade pip --quiet

# -----------------------------------------------------------------------------
# 2. Clone kohya-ss/sd-scripts
# -----------------------------------------------------------------------------
if [ ! -d "sd-scripts" ]; then
    echo ""
    echo "[2/6] Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    echo "      ✓ sd-scripts cloned."
else
    echo ""
    echo "[2/6] sd-scripts already present — skipping clone."
fi

# -----------------------------------------------------------------------------
# 3. Install sd-scripts requirements
#    Must run from inside sd-scripts/ because requirements.txt contains "-e ."
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing sd-scripts requirements..."
pushd sd-scripts > /dev/null
pip install -r requirements.txt
popd > /dev/null
echo "      ✓ sd-scripts requirements installed."

# -----------------------------------------------------------------------------
# 4. Install app requirements (gradio, toml)
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing app requirements..."
pip install -r requirements.txt
echo "      ✓ App requirements installed."

# -----------------------------------------------------------------------------
# 5. Configure accelerate (default single GPU)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Configuring accelerate (default)..."
accelerate config default
echo "      ✓ accelerate configured."

# -----------------------------------------------------------------------------
# 6. Download Anima models (idempotent — skips if already present)
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Downloading Anima support models (~1.4 GB total)..."
echo "      (The DiT base model will be downloaded automatically when you start training.)"
echo "      This may take a while depending on your connection."
echo ""

mkdir -p models/anima/dit
mkdir -p models/anima/text_encoder
mkdir -p models/anima/vae

QWEN_PATH="models/anima/text_encoder/qwen_3_06b_base.safetensors"
VAE_PATH="models/anima/vae/qwen_image_vae.safetensors"
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
echo "  Setup complete!"
echo "  Start the trainer with:  bash run_linux.sh"
echo "============================================================"
