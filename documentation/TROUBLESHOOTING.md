# Troubleshooting Tips

These tips are based on real user feedback for the **Citron Anima LoRA Trainer UI**.  
They cover the most common **first-run / installation issues** reported on Windows (especially with NVIDIA GPUs).  
Most of these are environment-specific but can be easily prevented.

## 1. Git is not in your PATH

**Problem**  
The installer or setup scripts fail because `git` cannot be found.

**Fix**  
1. Install Git for Windows if you haven’t already.  
2. Add Git to your system PATH:  
   - Search Windows for **“Edit the system environment variables”** → Open it.  
   - Click **Environment Variables** → Under **System variables**, find and edit **Path**.  
   - Add the Git `cmd` folder (usually `C:\Program Files\Git\cmd`).  
3. Restart your terminal / Command Prompt and verify with `git --version`.

## 2. “Package not found” after running install.bat

**Problem**  
The first installation appears to complete, but when you launch the trainer you get missing-package errors. Activating the venv and checking `pip list` shows the package, but reinstalling from `requirements.txt` still fails.

**Fix**  
- Always **activate the virtual environment** first:  
  ```cmd
  .venv\Scripts\activate
  ```
- If problems persist, do a **clean reinstall**:
  1. Delete the entire `.venv` folder.
  2. Delete any `__pycache__` folders and `*.pyc` files if present.
  3. Run `install.bat` again (it will recreate the venv from scratch).

This clears any corrupted or partially-cached dependency state.

## 3. Python version – Torch + CUDA not installing correctly

**Problem**  
The repo states **Python 3.10**, but on Python 3.10.x the `install.bat` script does **not** pull the correct Torch + CUDA wheels. Users saw it only download `torch 2.9.1+cu*` (cp312) on later runs, and training failed until Python was upgraded.

**Recommended solution**  
- Use **Python 3.12** (tested working: 3.12.19).  
- After installing Python 3.12, delete the old `.venv` folder and re-run `install.bat`.  
- The script will now correctly install the CUDA-enabled Torch version and all dependencies.

> **Note:** While the repository currently lists Python 3.10 as the target, Python 3.12 is strongly recommended for reliable Torch + CUDA installation on modern Windows systems.

## 4. UnicodeEncodeError (charmap codec) when starting training

**Problem**  
Training fails immediately with a traceback ending in:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 41-54: character maps to <undefined>
```

This happens inside `accelerate` / Kohya SD scripts when trying to print Japanese/kanji characters in the terminal.

**Fix** (Windows only)  
1. Open **Control Panel** → **Clock and Region** → **Region**.  
2. Go to the **Administrative** tab.  
3. Click **Change system locale** (or “Language for non-Unicode programs”).  
4. Check the box: **Beta: Use Unicode UTF-8 for worldwide language support**.  
5. Click OK and **restart your computer**.

After the restart the error disappears.  
(This is a system-wide setting and only affects programs that use the legacy non-Unicode code page.)

> Note: This issue did **not** appear when using plain Kohya-SS, but it surfaces in the GUI + web interface combination.

## Quick Checklist Before First Run

- [ ] Git is in PATH  
- [ ] Python 3.12 is installed (recommended)  
- [ ] `.venv` folder was deleted and `install.bat` was re-run cleanly  
- [ ] UTF-8 for non-Unicode programs is enabled (Windows)  
- [ ] Run the trainer from an **activated** venv (`.venv\Scripts\activate`)

Once these are resolved, training works excellently (users report ~3.33 s/it on RTX 3060 with the PC remaining fully usable).


## Community Shoutouts

- Huge thanks to **[ttaetherai](https://civitai.com/user/ttaetherai)** on Civitai for the detailed Windows + RTX 3060 feedback that made this troubleshooting guide possible!