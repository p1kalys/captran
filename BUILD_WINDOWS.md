# Windows Offline Build & Packaging Guide (CapTran)

This guide walks you through building a standalone, **100% offline Windows executable** for `CapTran`. Once built, the application requires **zero internet access** at runtime.

---

## 1. Prerequisites

- **Windows 10 / 11** (64-bit)
- **Python 3.11+** installed and added to PATH
- **Microsoft Visual C++ Redistributable (2015–2022)** installed

---

## 2. Step-by-Step Build Instructions

### Step 1: Set Up Python Environment & Dependencies

Open PowerShell in the project root directory:

```powershell
# Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all project & build dependencies
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

### Step 2: Download & Stage Offline Models (One-Time Build Step)

Run the automated model setup script:

```powershell
# Set up all 7 languages and small Whisper model
python scripts/setup_models.py --whisper-model small

# Or specify a subset of languages to minimize package size
python scripts/setup_models.py --languages ja hi en es --whisper-model base
```

This script stages all model weights locally inside the repository under `models/`:
1. **Silero VAD ONNX**: Stored at `models/silero_vad.onnx` (~2.3 MB).
2. **Faster-Whisper Models**: CTranslate2 weights downloaded to `models/whisper/` (e.g. `base`, `small`).
3. **Argos Translate Models**: Staged at `models/argos_packages/` (e.g. `ja_en.argosmodel`, `hi_en.argosmodel`, etc.).

---

### Step 3: Compile Standalone Executable with PyInstaller

Run PyInstaller using the bundled [`captran.spec`](captran.spec):

```powershell
pyinstaller captran.spec --clean --noconfirm
```

---

## 3. Output & Distribution

Once the build finishes, your self-contained distribution folder is located at:

```text
dist/captran/
├── captran.exe                 <-- Main Application Executable
├── models/                     <-- Bundled Offline Models (VAD, Whisper, Argos)
├── _internal/                  <-- PySide6, CTranslate2, ONNX, and PyAudio DLLs
└── ...
```

---

## 4. Verification on an Offline Machine

1. **Disconnect Internet / Wi-Fi** on your machine.
2. Navigate to `dist/captran/`:
   ```powershell
   cd dist\captran
   .\captran.exe
   ```
3. Play a local audio/video clip or start a meeting.
4. Verify the GUI Control Panel and floating Subtitle Overlay launch and stream live subtitles with **zero network requests**.

---

## 5. Command-Line Options in Packaged Executable

The packaged `.exe` supports all configuration flags:

```powershell
# Run with GUI Control Panel & Floating Overlay (Default)
.\captran.exe

# Run in headless terminal mode with Japanese -> English
.\captran.exe --cli --source-language ja --target-language en

# Run with Hindi -> English and latency diagnostics
.\captran.exe --cli --source-language hi --target-language en --debug-latency

# List WASAPI loopback audio devices
.\captran.exe --list-devices

# Select specific model and device
.\captran.exe --model-size tiny --device cpu --compute-type int8
```
