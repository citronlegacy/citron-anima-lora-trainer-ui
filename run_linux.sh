#!/usr/bin/env bash
# =============================================================================
# run_linux.sh — Activate venv and launch Anima LoRA Trainer
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_for_linux.sh first."
    exit 1
fi

source .venv/bin/activate
echo "Starting Anima LoRA Trainer at http://127.0.0.1:7860 ..."
python app.py
