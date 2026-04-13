@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Anima LoRA Trainer -- Windows Setup (RTX 5000-series)
echo.
echo   RTX 5070 / 5080 / 5090 require PyTorch sm_120 (CUDA 12.8)
echo   This script installs torch==2.9.1+cu128 from the cu128 index.
echo.
echo   Ref: github.com/Stephensmetana/nvidia-rtx5070-pytorch-guide
echo ============================================================
echo.

:: -----------------------------------------------------------------------------
:: 1. Python venv
:: -----------------------------------------------------------------------------
if not exist ".venv" (
    echo [1/7] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        echo        Make sure Python 3.10+ is installed and in your PATH.
        pause
        exit /b 1
    )
    echo       .venv created.
) else (
    echo [1/7] .venv already exists -- skipping creation.
)

call .venv\Scripts\activate.bat
echo       venv activated.

python -m pip install --upgrade pip --quiet

:: -----------------------------------------------------------------------------
:: 2. Clone kohya-ss/sd-scripts
:: -----------------------------------------------------------------------------
echo.
if not exist "sd-scripts" (
    echo [2/7] Cloning kohya-ss/sd-scripts...
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    if errorlevel 1 (
        echo ERROR: Failed to clone sd-scripts.
        echo        Make sure git is installed and in your PATH.
        pause
        exit /b 1
    )
    echo       sd-scripts cloned.
) else (
    echo [2/7] sd-scripts already present -- skipping clone.
)

:: -----------------------------------------------------------------------------
:: 3. Install PyTorch 2.9.1+cu128 FIRST (before sd-scripts requirements)
::    Ensures the correct sm_120 build is in place and not overwritten.
:: -----------------------------------------------------------------------------
echo.
echo [3/7] Installing PyTorch 2.9.1+cu128 for RTX 5000-series (sm_120)...
pip uninstall -y torch torchvision torchaudio 2>nul
pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo       PyTorch cu128 installed.

:: Verify GPU
echo.
echo       Verifying GPU support...
python -c "import torch; v=torch.__version__; c=torch.version.cuda; print(f'      PyTorch: {v}  CUDA: {c}'); g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(f'      GPU: {g}'); import torchvision; print(f'      torchvision: {torchvision.__version__}')"

:: -----------------------------------------------------------------------------
:: 4. Install sd-scripts requirements (must run from inside sd-scripts\)
:: -----------------------------------------------------------------------------
echo.
echo [4/7] Installing sd-scripts requirements...
pushd sd-scripts
pip install -r requirements.txt
popd
echo       sd-scripts requirements installed.

:: Re-pin torch to cu128 in case sd-scripts overwrote it
echo.
echo       Re-pinning PyTorch to cu128 (in case sd-scripts overwrote it)...
pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
echo       PyTorch cu128 confirmed.

:: -----------------------------------------------------------------------------
:: 5. Install app requirements (gradio, toml)
:: -----------------------------------------------------------------------------
echo.
echo [5/7] Installing app requirements...
pip install -r requirements.txt
echo       App requirements installed.

:: -----------------------------------------------------------------------------
:: 6. Configure accelerate
:: -----------------------------------------------------------------------------
echo.
echo [6/7] Configuring accelerate (default)...
accelerate config default
echo       accelerate configured.

:: -----------------------------------------------------------------------------
:: 7. Download Anima models (idempotent)
:: -----------------------------------------------------------------------------
echo.
echo [7/7] Downloading Anima support models (~1.4 GB total)...
echo       (The DiT base model will be downloaded automatically when you start training.)
echo       This may take a while depending on your connection.
echo.

if not exist "models\anima\dit"          mkdir models\anima\dit
if not exist "models\anima\text_encoder" mkdir models\anima\text_encoder
if not exist "models\anima\vae"          mkdir models\anima\vae

if not exist "models\anima\text_encoder\qwen_3_06b_base.safetensors" (
    echo   Downloading Qwen3 text encoder (1.19 GB)...
    curl -L -C - --progress-bar ^
        -o "models\anima\text_encoder\qwen_3_06b_base.safetensors" ^
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
    echo   Qwen3 text encoder downloaded.
) else (
    echo   Qwen3 text encoder already present -- skipping.
)

if not exist "models\anima\vae\qwen_image_vae.safetensors" (
    echo   Downloading Qwen-Image VAE (254 MB)...
    curl -L -C - --progress-bar ^
        -o "models\anima\vae\qwen_image_vae.safetensors" ^
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
    echo   VAE model downloaded.
) else (
    echo   VAE model already present -- skipping.
)

echo.
echo ============================================================
echo   Setup complete! (RTX 5000-series)
echo   Start the trainer with:  run_windows.bat
echo ============================================================
pause
