"""
Anima LoRA Trainer — Local Gradio UI
"""

import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import toml

# ---------------------------------------------------------------------------
# Paths (all relative to the project root where app.py lives)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models" / "anima"
SD_SCRIPTS_DIR = ROOT / "sd-scripts"

DIT_MODEL = MODELS_DIR / "dit" / "anima-preview.safetensors"
QWEN3_MODEL = MODELS_DIR / "text_encoder" / "qwen_3_06b_base.safetensors"
VAE_MODEL = MODELS_DIR / "vae" / "qwen_image_vae.safetensors"
TRAIN_SCRIPT = SD_SCRIPTS_DIR / "anima_train_network.py"

BASE_MODEL_URLS = {
    "anima-preview": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors",
    "anima-preview3-base": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
}


def get_dit_model_path(base_model: str) -> Path:
    """Return the local Path for the selected base model's DiT weights."""
    filenames = {
        "anima-preview": "anima-preview.safetensors",
        "anima-preview3-base": "anima-preview3-base.safetensors",
    }
    return MODELS_DIR / "dit" / filenames.get(base_model, "anima-preview.safetensors")

CONFIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Project-local accelerate config — keeps use_cpu=false and mixed_precision=bf16
# scoped to this app only. See configs/accelerate_gpu.yaml to change these.
ACCELERATE_CONFIG = "app_configs/accelerate_gpu.yaml"

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Basic
    "project_name": "my_lora",
    "base_model": "anima-preview",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 20,
    "network_alpha": 20,
    "learning_rate": 0.0001,
    "max_train_epochs": 10,
    "resolution": 768,
    "repeats": 10,
    "caption_dropout": 0.1,
    "gpu_index": "0",
    # Advanced
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 1,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "noise_offset": 0.03,
    "multires_noise_discount": 0.3,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "num_cpu_threads_per_process": 1,
    # Internal
    "last_train_config": "",
    "last_dataset_config": "",
}

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.json, filling missing keys with defaults."""
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    """Persist config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    """Return a list of GPU choices. Falls back to ['0', '1'] if torch unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ["CPU (no CUDA detected)"]
        choices = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            choices.append(f"{i}: {name}")
        return choices if choices else ["0", "1"]
    except ImportError:
        return ["0", "1"]


GPU_CHOICES = detect_gpus()


def gpu_index_from_choice(choice: str) -> str:
    """Extract the numeric GPU index from a dropdown choice string."""
    if not choice:
        return "0"
    return str(choice).split(":")[0].strip()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    """
    Returns (image_count, missing_captions, warnings).
    Raises FileNotFoundError if directory doesn't exist.
    """
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    all_files = list(p.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
    txt_basenames = {f.stem for f in all_files if f.suffix.lower() == ".txt" and f.is_file()}

    missing = [f.name for f in image_files if f.stem not in txt_basenames]
    warnings = []

    if not image_files:
        warnings.append("No image files found in directory.")
    if missing:
        warnings.append(f"{len(missing)} image(s) are missing caption (.txt) files.")

    return len(image_files), missing, warnings


# ---------------------------------------------------------------------------
# TOML config generation (ported directly from the notebook)
# ---------------------------------------------------------------------------

def create_training_config(
    project_name, output_dir, dit_model_path, qwen3_model_path, vae_model_path,
    network_dim=20, network_alpha=20, learning_rate=1e-4, max_train_epochs=10,
    optimizer_type="AdamW8bit", lr_scheduler="cosine_with_restarts",
    lr_scheduler_num_cycles=1, lr_warmup_steps=100,
    train_batch_size=1, gradient_accumulation_steps=1, max_grad_norm=1.0,
    save_every_n_epochs=1, save_last_n_epochs=4,
    mixed_precision="bf16", gradient_checkpointing=True,
    seed=42, noise_offset=0.03, multires_noise_discount=0.3,
    timestep_sampling="sigmoid", discrete_flow_shift=1.0,
    cache_latents=True, cache_text_encoder_outputs=True,
    vae_chunk_size=64, vae_disable_cache=True,
) -> str:
    """Generate training TOML and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_training_{current_date}.toml"

    training_config = {
        "pretrained_model_name_or_path": str(dit_model_path),
        "qwen3": str(qwen3_model_path),
        "vae": str(vae_model_path),
        "network_module": "networks.lora_anima",
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "network_train_unet_only": True,
        "learning_rate": float(learning_rate),
        "optimizer_type": optimizer_type,
        "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_train_epochs": int(max_train_epochs),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "output_dir": str(output_dir),
        "output_name": project_name,
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "shuffle_caption": False,
        "caption_extension": ".txt",
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "training_comment": f"Anima LoRA - {datetime.now().strftime('%Y-%m-%d')}",
    }

    with open(config_path, "w") as f:
        toml.dump(training_config, f)

    return str(config_path)


def create_dataset_config(
    project_name, image_dir, resolution=768, repeats=5, caption_dropout_rate=0.1
) -> str:
    """Generate dataset TOML and return its path."""
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_dataset_{current_date}.toml"

    dataset_config = {
        "general": {
            "resolution": int(resolution),
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 4096,
        },
        "datasets": [
            {
                "resolution": int(resolution),
                "subsets": [
                    {
                        "num_repeats": int(repeats),
                        "image_dir": str(image_dir),
                        "caption_extension": ".txt",
                        "caption_dropout_rate": float(caption_dropout_rate),
                    }
                ],
            }
        ],
    }

    with open(config_path, "w") as f:
        toml.dump(dataset_config, f)

    return str(config_path)


# ---------------------------------------------------------------------------
# Configure Training handler
# ---------------------------------------------------------------------------

def configure_training(
    project_name, base_model, image_directory, output_directory,
    network_dim, network_alpha, learning_rate, max_train_epochs,
    resolution, repeats, caption_dropout, gpu_index_choice,
    # advanced
    optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
    train_batch_size, gradient_accumulation_steps, max_grad_norm,
    save_every_n_epochs, save_last_n_epochs, mixed_precision,
    gradient_checkpointing, seed, noise_offset, multires_noise_discount,
    timestep_sampling, discrete_flow_shift,
    cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
    num_cpu_threads_per_process,
) -> tuple[str, str, str]:
    """
    Validate dataset, generate TOML configs, save settings to config.json.
    Returns (status_message, last_train_config_path, last_dataset_config_path).
    """
    lines = []

    # --- Validate inputs ---
    if not project_name.strip():
        return "❌ Project name cannot be empty.", "", ""
    if not image_directory.strip():
        return "❌ Image directory cannot be empty.", "", ""
    if not output_directory.strip():
        return "❌ Output directory cannot be empty.", "", ""

    lines.append(f"Project:          {project_name}")
    lines.append(f"Image directory:  {image_directory}")
    lines.append(f"Output directory: {output_directory}")
    lines.append("")

    # --- Validate dataset ---
    try:
        n_images, missing, warnings = validate_dataset(image_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        return f"❌ {e}", "", ""

    lines.append(f"Images found:     {n_images}")
    if missing:
        lines.append(f"⚠ Missing captions ({len(missing)}):")
        for m in missing[:20]:
            lines.append(f"    • {m}")
        if len(missing) > 20:
            lines.append(f"    … and {len(missing) - 20} more")
    else:
        lines.append("✓ All images have caption files.")

    for w in warnings:
        lines.append(f"⚠ {w}")

    if n_images == 0:
        lines.append("")
        lines.append("❌ Cannot configure — no images found.")
        return "\n".join(lines), "", ""

    # --- Step estimate ---
    batch = max(int(train_batch_size), 1)
    grad = max(int(gradient_accumulation_steps), 1)
    spe = math.ceil((n_images * int(repeats)) / (batch * grad))
    total = spe * int(max_train_epochs)
    lines.append("")
    lines.append("── Step Estimate ─────────────────────────────────────")
    lines.append(f"  Steps per epoch: {spe}  ({n_images} imgs × {repeats} repeats)")
    lines.append(f"  Total steps:     {total}  ({spe} × {max_train_epochs} epochs)")
    lines.append("──────────────────────────────────────────────────────")

    # --- Validate models ---
    lines.append("")
    lines.append("Checking models...")
    dit_model = get_dit_model_path(base_model)
    missing_models = []
    for label, path in [("DiT", dit_model), ("Qwen3", QWEN3_MODEL), ("VAE", VAE_MODEL)]:
        if Path(path).exists():
            lines.append(f"  ✓ {label}: {path}")
        else:
            if label == "DiT":
                lines.append(f"  ℹ {label}: {path}")
                lines.append(f"      (will auto-download when training starts)")
            else:
                lines.append(f"  ✗ {label} missing: {path}")
                missing_models.append(label)
    if missing_models:
        lines.append("")
        lines.append(f"❌ Missing models: {', '.join(missing_models)}")
        lines.append("Run setup_for_linux.sh / setup_for_windows.bat to download them.")
        return "\n".join(lines), "", ""

    # --- Generate configs ---
    lines.append("")
    lines.append("Generating TOML configs...")
    try:
        train_cfg = create_training_config(
            project_name=project_name,
            output_dir=output_directory,
            dit_model_path=dit_model,
            qwen3_model_path=QWEN3_MODEL,
            vae_model_path=VAE_MODEL,
            network_dim=network_dim,
            network_alpha=network_alpha,
            learning_rate=learning_rate,
            max_train_epochs=max_train_epochs,
            optimizer_type=optimizer_type,
            lr_scheduler=lr_scheduler,
            lr_scheduler_num_cycles=lr_scheduler_num_cycles,
            lr_warmup_steps=lr_warmup_steps,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_every_n_epochs=save_every_n_epochs,
            save_last_n_epochs=save_last_n_epochs,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            seed=seed,
            noise_offset=noise_offset,
            multires_noise_discount=multires_noise_discount,
            timestep_sampling=timestep_sampling,
            discrete_flow_shift=discrete_flow_shift,
            cache_latents=cache_latents,
            cache_text_encoder_outputs=cache_text_encoder_outputs,
            vae_chunk_size=vae_chunk_size,
            vae_disable_cache=vae_disable_cache,
        )
        dataset_cfg = create_dataset_config(
            project_name=project_name,
            image_dir=image_directory,
            resolution=resolution,
            repeats=repeats,
            caption_dropout_rate=caption_dropout,
        )
    except Exception as e:
        lines.append(f"❌ Failed to generate configs: {e}")
        return "\n".join(lines), "", ""

    lines.append(f"  ✓ Training config: {train_cfg}")
    lines.append(f"  ✓ Dataset  config: {dataset_cfg}")

    # --- Save all settings to config.json ---
    cfg = {
        "project_name": project_name,
        "base_model": base_model,
        "image_directory": image_directory,
        "output_directory": output_directory,
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "learning_rate": float(learning_rate),
        "max_train_epochs": int(max_train_epochs),
        "resolution": int(resolution),
        "repeats": int(repeats),
        "caption_dropout": float(caption_dropout),
        "gpu_index": gpu_index_from_choice(gpu_index_choice),
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "seed": int(seed),
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "num_cpu_threads_per_process": int(num_cpu_threads_per_process),
        "last_train_config": train_cfg,
        "last_dataset_config": dataset_cfg,
    }
    save_config(cfg)

    lines.append("")
    lines.append("✓ Configuration complete — ready to train.")
    return "\n".join(lines), train_cfg, dataset_cfg


# ---------------------------------------------------------------------------
# Training runner (generator — streams logs live to Gradio)
# ---------------------------------------------------------------------------

def start_training(
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
):
    """
    Generator: yields growing log text as training runs.
    Uses last generated configs unless custom_config_path is provided.
    Saves log to ./logs/
    """
    log_lines: list[str] = []

    def emit(line: str):
        log_lines.append(line)
        return "\n".join(log_lines)

    # --- Auto-download DiT model if needed ---
    dit_model = get_dit_model_path(base_model)
    if not dit_model.exists():
        url = BASE_MODEL_URLS.get(base_model)
        if not url:
            yield emit(f"❌ Unknown base model: {base_model}")
            return
        yield emit(f"⏳ Downloading base model '{base_model}'...")
        yield emit(f"   This may take a few minutes. Future runs will skip this step.")
        yield emit(f"   Destination: {dit_model}")
        yield emit("")
        os.makedirs(dit_model.parent, exist_ok=True)
        try:
            dl_proc = subprocess.Popen(
                ["wget", "-c", "--show-progress", "-O", str(dit_model), url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            for line in iter(dl_proc.stdout.readline, ""):
                yield emit(line.rstrip("\n"))
            dl_proc.wait()
            if dl_proc.returncode != 0:
                yield emit(f"❌ Download failed (exit code {dl_proc.returncode})")
                return
            yield emit(f"✓ Base model downloaded successfully.")
            yield emit("")
        except FileNotFoundError:
            yield emit("❌ 'wget' not found. Install wget and try again.")
            return

    # --- Resolve config paths ---
    saved_cfg = load_config()

    train_cfg = custom_config_path.strip() if custom_config_path.strip() else saved_cfg.get("last_train_config", "")
    dataset_cfg = saved_cfg.get("last_dataset_config", "")

    if not train_cfg:
        yield emit("❌ No training config found. Run 'Configure Training' first, or provide a config path.")
        return
    if not Path(train_cfg).exists():
        yield emit(f"❌ Training config not found: {train_cfg}")
        return
    if not dataset_cfg:
        yield emit("❌ No dataset config found. Run 'Configure Training' first.")
        return
    if not Path(dataset_cfg).exists():
        yield emit(f"❌ Dataset config not found: {dataset_cfg}")
        return

    # --- Validate sd-scripts ---
    if not TRAIN_SCRIPT.exists():
        yield emit(f"❌ Training script not found: {TRAIN_SCRIPT}\nRun setup_for_linux.sh / setup_for_windows.bat first.")
        return

    # --- Validate GPU ---
    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    yield emit(f"Using GPU index: {gpu_idx}")

    # --- Build accelerate command ---
    threads = max(int(num_cpu_threads_per_process), 1)
    cmd = [
        "accelerate", "launch",
        "--config_file", str(ACCELERATE_CONFIG),
        "--num_cpu_threads_per_process", str(threads),
        "--gpu_ids", gpu_idx,
        str(TRAIN_SCRIPT),
        "--config_file", train_cfg,
        "--dataset_config", dataset_cfg,
    ]

    yield emit(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
    yield emit("")

    # --- Set up log file ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    project_name = saved_cfg.get("project_name", "run")
    log_file_path = LOGS_DIR / f"{project_name}_{timestamp}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["PYTHONUNBUFFERED"] = "1"

    # --- Launch subprocess ---
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            cwd=str(ROOT),
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError:
        yield emit("❌ 'accelerate' not found. Make sure the venv is activated and accelerate is installed.")
        return

    # --- Stream output ---
    with open(log_file_path, "w", encoding="utf-8", errors="ignore") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n\n")

        for line in iter(process.stdout.readline, ""):
            line = line.rstrip("\n")
            log_f.write(line + "\n")
            log_f.flush()
            yield emit(line)

    exit_code = process.wait()

    if exit_code == 0:
        yield emit(f"\n✓ Training completed successfully!\nLoRA saved to: {saved_cfg.get('output_directory', 'output dir')}\nLog saved to: {log_file_path}")
    else:
        yield emit(f"\n✗ Training failed (exit code: {exit_code})\nLog saved to: {log_file_path}")
        # OOM hint
        try:
            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            tail = "\n".join(result.stdout.splitlines()[-40:])
            if any(t in tail for t in ("Out of memory", "Killed process", "oom_reaper", "OOM")):
                yield emit("\n💡 OOM detected in kernel log. Try: network_dim=8 and/or resolution=512")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    cfg = load_config()

    # Resolve saved GPU choice label
    saved_gpu_idx = str(cfg.get("gpu_index", "0"))
    default_gpu = next(
        (c for c in GPU_CHOICES if c.startswith(saved_gpu_idx + ":")),
        GPU_CHOICES[0] if GPU_CHOICES else "0",
    )

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown(
            """# 🍋 Citron's Anima LoRA Trainer

    Super Simple Gradio UI for training LoRA adapters on the <a href="https://huggingface.co/circlestone-labs/Anima" target="_blank" rel="noopener noreferrer">Anima</a> diffusion model using <a href="https://github.com/kohya-ss/sd-scripts" target="_blank" rel="noopener noreferrer">kohya-ss/sd-scripts</a>.

    🚀 Runs on ~6 GB VRAM with default settings. 

    Created by <a href="https://x.com/Citron_Legacy" target="_blank" rel="noopener noreferrer">Citron Legacy</a>  Please check out the <a href="https://github.com/citronlegacy/citron-anima-lora-trainer-ui" target="_blank" rel="noopener noreferrer">Code in Git</a>
    """
        )

        # ── Shared state for last-generated config paths ──────────────────
        last_train_cfg = gr.State(cfg.get("last_train_config", ""))
        last_dataset_cfg = gr.State(cfg.get("last_dataset_config", ""))

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Training
            # ================================================================
            with gr.Tab("Training"):

                with gr.Group():
                    gr.Markdown("### Project & Paths")
                    with gr.Row():
                        project_name = gr.Textbox(
                            label="Project Name",
                            value=cfg["project_name"],
                            placeholder="my_lora",
                        )
                        gpu_dropdown = gr.Dropdown(
                            label="GPU",
                            choices=GPU_CHOICES,
                            value=default_gpu,
                        )
                    with gr.Row():
                        base_model_dropdown = gr.Dropdown(
                            label="Base Model",
                            choices=["anima-preview", "anima-preview3-base"],
                            value=cfg.get("base_model", "anima-preview"),
                            info="Select Base Model (Model will auto download when you click start training. this may take a few minutes but future runs will not need to download.)",
                        )
                    image_directory = gr.Textbox(
                        label="Image Directory (flat folder with images + .txt captions)",
                        value=cfg["image_directory"],
                        placeholder="/path/to/my_dataset",
                    )
                    output_directory = gr.Textbox(
                        label="Output Directory (where trained LoRA is saved)",
                        value=cfg["output_directory"],
                        placeholder="/path/to/output",
                    )

                with gr.Group():
                    gr.Markdown("### Network")
                    with gr.Row():
                        network_dim = gr.Number(
                            label="Network Dim",
                            value=cfg["network_dim"],
                            precision=0,
                            minimum=1,
                        )
                        network_alpha = gr.Number(
                            label="Network Alpha",
                            value=cfg["network_alpha"],
                            precision=0,
                            minimum=1,
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=cfg["learning_rate"],
                        )
                        max_train_epochs = gr.Number(
                            label="Max Epochs",
                            value=cfg["max_train_epochs"],
                            precision=0,
                            minimum=1,
                        )

                with gr.Group():
                    gr.Markdown("### Dataset")
                    with gr.Row():
                        resolution = gr.Number(
                            label="Resolution (px)",
                            value=cfg["resolution"],
                            precision=0,
                            minimum=64,
                        )
                        repeats = gr.Number(
                            label="Repeats",
                            value=cfg["repeats"],
                            precision=0,
                            minimum=1,
                        )
                        caption_dropout = gr.Slider(
                            label="Caption Dropout",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=cfg["caption_dropout"],
                        )

                gr.Markdown("---")
                gr.Markdown("### Config & Training")

                with gr.Row():
                    configure_btn = gr.Button("⚙️ Configure Training", variant="secondary", size="lg")
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

                custom_config_input = gr.Textbox(
                    label="Override Training Config Path (optional — leave blank to use last generated)",
                    value="",
                    placeholder="/path/to/custom_training_config.toml",
                )

                status_box = gr.Textbox(
                    label="Configuration Status",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                )

                log_box = gr.Textbox(
                    label="Training Log",
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    autoscroll=True,
                )

            # ================================================================
            # TAB 2 — Advanced Settings
            # ================================================================
            with gr.Tab("Advanced Settings"):
                gr.Markdown(
                    "_These settings are applied when you click **Configure Training**._\n\n"
                    "Defaults match the original notebook."
                )

                with gr.Group():
                    gr.Markdown("### Optimizer & Scheduler")
                    with gr.Row():
                        optimizer_type = gr.Dropdown(
                            label="Optimizer",
                            choices=["AdamW8bit", "AdamW", "Lion", "SGD", "Prodigy"],
                            value=cfg["optimizer_type"],
                        )
                        lr_scheduler = gr.Dropdown(
                            label="LR Scheduler",
                            choices=[
                                "cosine_with_restarts",
                                "cosine",
                                "linear",
                                "constant",
                                "constant_with_warmup",
                                "polynomial",
                            ],
                            value=cfg["lr_scheduler"],
                        )
                    with gr.Row():
                        lr_scheduler_num_cycles = gr.Number(
                            label="LR Scheduler Num Cycles",
                            value=cfg["lr_scheduler_num_cycles"],
                            precision=0,
                            minimum=1,
                        )
                        lr_warmup_steps = gr.Number(
                            label="LR Warmup Steps",
                            value=cfg["lr_warmup_steps"],
                            precision=0,
                            minimum=0,
                        )

                with gr.Group():
                    gr.Markdown("### Batch & Gradient")
                    with gr.Row():
                        train_batch_size = gr.Number(
                            label="Train Batch Size",
                            value=cfg["train_batch_size"],
                            precision=0,
                            minimum=1,
                        )
                        gradient_accumulation_steps = gr.Number(
                            label="Gradient Accumulation Steps",
                            value=cfg["gradient_accumulation_steps"],
                            precision=0,
                            minimum=1,
                        )
                        max_grad_norm = gr.Number(
                            label="Max Grad Norm",
                            value=cfg["max_grad_norm"],
                        )

                with gr.Group():
                    gr.Markdown("### Saving")
                    with gr.Row():
                        save_every_n_epochs = gr.Number(
                            label="Save Every N Epochs",
                            value=cfg["save_every_n_epochs"],
                            precision=0,
                            minimum=1,
                        )
                        save_last_n_epochs = gr.Number(
                            label="Keep Last N Checkpoints",
                            value=cfg["save_last_n_epochs"],
                            precision=0,
                            minimum=1,
                        )

                with gr.Group():
                    gr.Markdown("### Precision & Memory")
                    with gr.Row():
                        mixed_precision = gr.Dropdown(
                            label="Mixed Precision",
                            choices=["bf16", "fp16", "no"],
                            value=cfg["mixed_precision"],
                        )
                        vae_chunk_size = gr.Number(
                            label="VAE Chunk Size",
                            value=cfg["vae_chunk_size"],
                            precision=0,
                            minimum=1,
                        )
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(
                            label="Gradient Checkpointing",
                            value=cfg["gradient_checkpointing"],
                        )
                        cache_latents = gr.Checkbox(
                            label="Cache Latents",
                            value=cfg["cache_latents"],
                        )
                        cache_text_encoder_outputs = gr.Checkbox(
                            label="Cache Text Encoder Outputs",
                            value=cfg["cache_text_encoder_outputs"],
                        )
                        vae_disable_cache = gr.Checkbox(
                            label="VAE Disable Cache",
                            value=cfg["vae_disable_cache"],
                        )

                with gr.Group():
                    gr.Markdown("### Noise & Flow")
                    with gr.Row():
                        noise_offset = gr.Number(
                            label="Noise Offset",
                            value=cfg["noise_offset"],
                        )
                        multires_noise_discount = gr.Number(
                            label="Multires Noise Discount",
                            value=cfg["multires_noise_discount"],
                        )
                        timestep_sampling = gr.Dropdown(
                            label="Timestep Sampling",
                            choices=["sigmoid", "uniform", "logit_normal"],
                            value=cfg["timestep_sampling"],
                        )
                        discrete_flow_shift = gr.Number(
                            label="Discrete Flow Shift",
                            value=cfg["discrete_flow_shift"],
                        )

                with gr.Group():
                    gr.Markdown("### Misc")
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=cfg["seed"],
                            precision=0,
                        )
                        num_cpu_threads = gr.Number(
                            label="CPU Threads Per Process",
                            value=cfg["num_cpu_threads_per_process"],
                            precision=0,
                            minimum=1,
                        )

        # ── All advanced inputs collected for passing to configure_training ─
        adv_inputs = [
            optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
            train_batch_size, gradient_accumulation_steps, max_grad_norm,
            save_every_n_epochs, save_last_n_epochs, mixed_precision,
            gradient_checkpointing, seed, noise_offset, multires_noise_discount,
            timestep_sampling, discrete_flow_shift,
            cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
            num_cpu_threads,
        ]

        basic_inputs = [
            project_name, base_model_dropdown, image_directory, output_directory,
            network_dim, network_alpha, learning_rate, max_train_epochs,
            resolution, repeats, caption_dropout, gpu_dropdown,
        ]

        # ── Configure Training event ─────────────────────────────────────
        configure_btn.click(
            fn=configure_training,
            inputs=basic_inputs + adv_inputs,
            outputs=[status_box, last_train_cfg, last_dataset_cfg],
        )

        # ── Start Training event (generator → streaming) ─────────────────
        train_btn.click(
            fn=start_training,
            inputs=[custom_config_input, gpu_dropdown, num_cpu_threads, base_model_dropdown],
            outputs=[log_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
