@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv" (
    echo ERROR: .venv not found. Run setup_for_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Anima LoRA Trainer at http://127.0.0.1:7860 ...
python app.py
pause
