@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Anima LoRA Trainer -- Windows Setup
echo ============================================================
echo.

:: -----------------------------------------------------------------------------
:: 1. Python venv
:: -----------------------------------------------------------------------------
if not exist "venv" (
    echo [1/6] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        echo        Make sure Python 3.10 is installed and in your PATH.
        pause
        exit /b 1
    )
    echo       venv created.
) else (
    echo [1/6] venv already exists -- skipping creation.
)

call venv\Scripts\activate.bat
echo       venv activated.

python -m pip install --upgrade pip --quiet

:: -----------------------------------------------------------------------------
:: 2. Clone kohya-ss/sd-scripts
:: -----------------------------------------------------------------------------
echo.
if not exist "sd-scripts" (
    echo [2/6] Cloning kohya-ss/sd-scripts...
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    if errorlevel 1 (
        echo ERROR: Failed to clone sd-scripts.
        echo        Make sure git is installed and in your PATH.
        pause
        exit /b 1
    )
    echo       sd-scripts cloned.
) else (
    echo [2/6] sd-scripts already present -- skipping clone.
)

:: -----------------------------------------------------------------------------
:: 3. Install sd-scripts requirements
::    Must run from inside sd-scripts\ because requirements.txt contains "-e ."
:: -----------------------------------------------------------------------------
echo.
echo [3/6] Installing sd-scripts requirements...
pushd sd-scripts
pip install -r requirements.txt
popd
echo       sd-scripts requirements installed.

:: -----------------------------------------------------------------------------
:: 4a. Install correct PyTorch for RTX 50-series (sm_120 / CUDA 12.8)
::     sd-scripts requirements may pull a generic torch without sm_120 support.
::     We explicitly reinstall from the cu128 index.
:: -----------------------------------------------------------------------------
echo.
echo [4/6] Installing PyTorch 2.9.1+cu128 for RTX 50-series GPU (sm_120)...
pip uninstall -y torch torchvision torchaudio 2>nul
pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo       PyTorch cu128 installed.

:: -----------------------------------------------------------------------------
:: 4b. Install app requirements (gradio, toml)
:: -----------------------------------------------------------------------------
echo.
echo [4/6] Installing app requirements...
pip install -r requirements.txt
echo       App requirements installed.

:: -----------------------------------------------------------------------------
:: 5. Configure accelerate
:: -----------------------------------------------------------------------------
echo.
echo [5/6] Configuring accelerate (default)...
accelerate config default
echo       accelerate configured.

:: -----------------------------------------------------------------------------
:: 6. Download Anima models (idempotent)
:: -----------------------------------------------------------------------------
echo.
echo [6/6] Downloading Anima models (~5.6 GB total)...
echo       This may take a while depending on your connection.
echo.

if not exist "models\anima\dit"          mkdir models\anima\dit
if not exist "models\anima\text_encoder" mkdir models\anima\text_encoder
if not exist "models\anima\vae"          mkdir models\anima\vae

if not exist "models\anima\dit\anima-preview.safetensors" (
    echo   Downloading DiT model (4.18 GB)...
    curl -L -C - --progress-bar ^
        -o "models\anima\dit\anima-preview.safetensors" ^
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors"
    echo   DiT model downloaded.
) else (
    echo   DiT model already present -- skipping.
)

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
echo   Setup complete!
echo   Start the trainer with:  run_windows.bat
echo ============================================================
pause
