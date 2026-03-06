

# Anima LoRA Trainer — Local Gradio UI

> **The first local GUI for training LoRA adapters on the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model.**  
> Simple by design. Runs on less than 6 GB VRAM with default settings.

A lightweight Gradio web app that wraps [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) to make local Anima LoRA training accessible without writing a single line of code or using Google Colab.

**Key features:**
- 🖥️ Local — no Colab, no cloud, no time limits
- 🎛️ Simple UI — basic settings on one tab, advanced on another
- 💾 Low VRAM — train with under 6 GB using default settings
- 📋 Config persistence — all settings saved and reloaded automatically
- 📡 Live log streaming — watch training progress in real time

Based on the [citronlegacy/citron-colab-anima-lora-trainer](https://github.com/citronlegacy/citron-colab-anima-lora-trainer) Colab notebook.

---

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA (required — CPU training is not supported)
- `git` in PATH
- `wget` (Linux) or `curl` (Windows) for model downloads
- ~6 GB free disk space for models + ~2 GB for sd-scripts dependencies

---

## Quick Start

### Linux / macOS

```bash
# RTX 5000-series (5070, 5080, 5090) — uses torch==2.9.1+cu128 (sm_120)
bash setup_for_linux_rtx5000.sh

# All other GPUs (RTX 3000/4000 series, etc.)
bash setup_for_linux.sh

# 2. Launch the app (same for both)
bash run_linux.sh
```

### Windows

```bat
REM RTX 5000-series (5070, 5080, 5090) — uses torch==2.9.1+cu128 (sm_120)
setup_for_windows_rtx5000.bat

REM All other GPUs (RTX 3000/4000 series, etc.)
setup_for_windows.bat

REM Launch the app (same for both)
run_windows.bat
```

> **RTX 5070/5080/5090 users:** The standard setup script may install a PyTorch
> build that lacks `sm_120` kernel support, causing `no kernel image` errors at
> runtime. Always use the `_rtx5000` script for these GPUs.

Open your browser at **http://127.0.0.1:7860**

---

## Project Structure

```
local_anima_trainer_gradio/
├── app.py                           # Gradio UI and training orchestrator
├── requirements.txt                 # Gradio + toml (app deps only)
├── setup_for_linux.sh               # One-time setup — Linux/macOS (RTX 3000/4000 series)
├── setup_for_linux_rtx5000.sh       # One-time setup — Linux/macOS (RTX 5070/5080/5090)
├── setup_for_windows.bat            # One-time setup — Windows (RTX 3000/4000 series)
├── setup_for_windows_rtx5000.bat    # One-time setup — Windows (RTX 5070/5080/5090)
├── run_linux.sh                     # Launch app (Linux/macOS)
├── run_windows.bat                  # Launch app (Windows)
│
├── sd-scripts/                # kohya-ss/sd-scripts (cloned by setup)
├── models/
│   └── anima/
│       ├── dit/               # anima-preview.safetensors  (4.18 GB)
│       ├── text_encoder/      # qwen_3_06b_base.safetensors (1.19 GB)
│       └── vae/               # qwen_image_vae.safetensors  (254 MB)
├── configs/                   # Auto-generated TOML configs (per run)
├── logs/                      # Training logs (one file per run)
└── config.json                # Persistent UI settings (auto-saved)
```

---

## Dataset Format

Flat directory — no nested subfolders:

```
my_dataset/
  image001.png
  image001.txt    ← caption (comma-separated tags)
  image002.jpg
  image002.txt
  image003.webp
  image003.txt
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`

Example caption:
```
mycharname, 1girl, long blonde hair, blue eyes, high quality, detailed
```

---

## Training Workflow

1. Open **http://127.0.0.1:7860**
2. Fill in **Project Name**, **Image Directory**, and **Output Directory** on the **Training** tab.
3. Adjust **Network Dim/Alpha, Learning Rate, Epochs, Resolution, Repeats** as needed.
4. Switch to the **Advanced Settings** tab if you want to change optimizer, scheduler, batch size, etc.
5. Click **⚙️ Configure Training** — this validates your dataset, checks models, estimates steps, and generates the TOML config files.
6. Click **🚀 Start Training** — logs stream live into the Training Log box.
7. Trained LoRA (`.safetensors`) is saved to your Output Directory.

### Custom config

If you have a pre-existing training TOML, paste its path into the
**Override Training Config Path** field before clicking Start Training.
Leave it blank to use the last generated config.

---

## Settings Persistence

All settings (hyperparameters, paths, GPU selection) are automatically saved
to `config.json` when you click **Configure Training** and reloaded the next
time you launch the app.

---

## Advanced Settings Reference

| Setting | Default | Notes |
|---|---|---|
| Optimizer | AdamW8bit | 8-bit Adam saves VRAM |
| LR Scheduler | cosine_with_restarts | |
| LR Warmup Steps | 100 | |
| Batch Size | 1 | Increase if VRAM allows |
| Gradient Accumulation | 1 | Effective batch = batch × accum |
| Mixed Precision | bf16 | Requires Ampere+ (RTX 30xx/40xx) |
| Gradient Checkpointing | ✓ | Saves VRAM at cost of speed |
| Cache Latents | ✓ | Caches encoded images to disk |
| Cache Text Encoder | ✓ | Caches text embeddings to disk |
| Save Every N Epochs | 1 | Saves checkpoint each epoch |
| Keep Last N Checkpoints | 4 | Older checkpoints auto-deleted |
| Seed | 42 | |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA OOM | Lower `Network Dim` (try 8) and/or `Resolution` (try 512) |
| NaN loss | Lower the learning rate; ensure PyTorch ≥ 2.5 |
| "No images found" | Verify images are not named `.txt` only |
| `accelerate` not found | Make sure you're running via the `run_*.sh/bat` script |
| Model missing | Re-run `setup_for_linux.sh` / `setup_for_windows.bat` |

---

## Credits

- Original Colab notebook: [citronlegacy/citron-colab-anima-lora-trainer](https://github.com/citronlegacy/citron-colab-anima-lora-trainer)
- Training backend: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- Model: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
