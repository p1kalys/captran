# CapTran: Offline Multilingual Live Captioner & Real-Time Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Hexagonal](https://img.shields.io/badge/Architecture-Hexagonal%20%2F%20Ports%20%26%20Adapters-green.svg)](ARCHITECTURE.md)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%20WASAPI-lightgrey.svg)](https://docs.microsoft.com/en-us/windows/win32/coreaudio/wasapi)
[![Tests: 97 Passing](https://img.shields.io/badge/Tests-97%20Passing-brightgreen.svg)](#running-tests--benchmarks)

**CapTran** is a high-performance, **100% offline, privacy-first multilingual live meeting captioner and audio translator** built with clean Hexagonal Architecture (Ports and Adapters).

It captures live system audio (meetings, browser tabs, video streams, YouTube, or podcasts) via Windows WASAPI Loopback, detects speech utterances in real time using local **Silero VAD**, transcribes speech with CTranslate2-accelerated **Faster-Whisper (INT8 optimized)** across 7 languages, translates offline with **Argos Translate** (via direct packages and intelligent English-pivot routing covering all 42 language pairs), and renders live side-by-side subtitles on a modern translucent, draggable overlay.

---

## 🌟 Key Features

- **100% Offline & Private:** Zero cloud API calls or runtime network traffic. All neural network inferences (VAD, STT, and Machine Translation) run entirely on-device.
- **Multilingual Live Captioning & Translation (7 Languages, 42 Pairs):**
  - Full bidirectional support across **Hindi (`hi`)**, **Japanese (`ja`)**, **English (`en`)**, **Spanish (`es`)**, **French (`fr`)**, **German (`de`)**, and **Korean (`ko`)**.
  - **Intelligent Pivot Translation:** Automatically uses direct translation packages when available, or seamless 2-hop pivot routing via English (e.g. `ja -> en -> de`, `hi -> en -> es`) to support all 42 language pair combinations.
- **Side-by-Side Dual Subtitles (`[SRC] -> [TGT]`):**
  - Displays original speech (`[JA]`, `[HI]`, etc.) and translated output (`[EN]`, `[DE]`, etc.) side-by-side in real time.
  - Live rolling interim streaming with visual pulsing indicator (`●`) that stabilizes into finalized subtitles.
- **Decoupled Multi-Threaded Pipeline with Backpressure:**
  - Independent producer-consumer worker threads for Audio Capture, VAD Segmentation, STT Transcription, and Machine Translation.
  - Smart interim snapshot coalescing prevents STT backlog drift during continuous speech while ensuring finalized utterances are never dropped.
- **Ultra-Low Latency Inference:**
  - Quantized **INT8 computation** on CPU (leveraging multi-core AVX2/AVX-512) and auto-detection of CUDA **float16** on NVIDIA GPUs.
  - Greedy decoding (`beam_size=1`) and adaptive VAD slicing (~250ms latency).
- **Custom Domain Vocabulary & Dictionary:**
  - Local JSON-backed dictionary (`~/.ja-en-captioner/custom_vocab.json`).
  - Passes prompt hints to Faster-Whisper and applies pre/post-translation substitutions with regex word-boundary isolation.
- **Diagnostic Latency Instrumentation:**
  - High-resolution per-stage timing tracking (Audio->VAD, STT, Translation, Display, E2E).
  - Background reporter generating running p50/p95 statistical summary tables every 30 seconds via `--debug-latency`.
- **System Audio Loopback:** Captures internal speaker, headphone, and Bluetooth headset audio directly without requiring a physical microphone.
- **Dual Presentation Modes:**
  - **Modern GUI:** Draggable, translucent, always-on-top glassmorphic overlay + dark-themed Control Panel with HTML sanitization.
  - **Headless CLI:** Terminal-based overwrite presenter with side-by-side formatting and timestamp logging.

---

## 🏛️ Pipeline Architecture

```mermaid
flowchart TD
    A["System Audio / Headset - WASAPI Loopback"] -->|Audio Queue| B["Silero VAD ONNX - Speech Segmenter"]
    B -->|STT Queue - Interim Coalescing| C["Faster-Whisper - Multilingual INT8/FP16"]
    C -->|Interim Streaming| E["Caption Presenter - Side-by-Side"]
    C -->|Translation Queue| D["Argos Translate - Direct & English-Pivot MT"]
    D -->|Final Subtitles| E
    E --> F1["PySide6 Floating Overlay - Source to Target"]
    E --> F2["Terminal CLI Presenter"]
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python Version:** Python 3.11 or higher
- **Hardware:** Modern multi-core CPU (Intel/AMD) or NVIDIA GPU (optional)

### 2. Clone & Setup Environment

```powershell
# Clone repository
git clone https://github.com/p1kalys/captran.git
cd captran

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -e .
```

### 3. Model Assets Setup (One-Time)

Download and stage local offline model weights (Silero VAD, multilingual Faster-Whisper, and Argos Translate language packages):

```powershell
# Set up all 7 languages and small Whisper model (with download & disk size estimation)
python scripts/setup_models.py

# Or set up a specific subset of languages (e.g. Japanese, Hindi, English)
python scripts/setup_models.py --languages ja hi en --whisper-model base

# Preview estimated download and disk usage without downloading
python scripts/setup_models.py --dry-run
```

---

## 💻 How to Run

### Mode 1: Graphical User Interface (Default)
Launch the desktop control window and draggable side-by-side subtitle overlay:

```powershell
python main.py
```
- **Select Audio Device:** Choose your default speakers or connected headphones/Bluetooth headset from the dropdown.
- **Select Model Size:** Choose between `tiny` (fastest), `base` (balanced real-time), and `small` (high accuracy).
- **Dual Subtitles:** Toggle side-by-side original and translated subtitles.
- **Start/Stop:** Click **Start Live Captioning** to begin streaming.

---

### Mode 2: Terminal / Headless CLI Mode

For developers, servers, or lightweight terminal usage:

```powershell
# Real-time Japanese to English streaming with latency diagnostics
python main.py --cli --source-language ja --target-language en --model-size base --debug-latency

# Hindi to English live captioning
python main.py --cli --source-language hi --target-language en --model-size small

# Spanish to German translation (via automatic English pivot)
python main.py --cli --source-language es --target-language de --model-size base

# High-accuracy transcription & direct English output
python main.py --cli --source-language ja --target-language en --model-size small
```

---

### Mode 3: Headset & Audio Device Selection
If using headphones or Bluetooth earbuds:

1. **List all available loopback capture devices:**
   ```powershell
   python main.py --list-devices
   ```
   *Example Output:*
   ```text
   Available Loopback Devices (2 found):
     [16] Speakers (Realtek(R) Audio) [Loopback] (2 ch, 48000 Hz)
     [17] Headphones (Boult Audio Airbass) [Loopback] (2 ch, 44100 Hz)
   ```

2. **Run specifically on your headset (e.g., Device 17):**
   ```powershell
   python main.py --cli --device-index 17 --source-language ja --target-language en --model-size base
   ```

---

## ⚙️ Command-Line Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--cli` | `False` | Run in terminal CLI mode instead of PySide6 GUI |
| `--source-language` | `ja` | Source spoken language (`hi`, `ja`, `en`, `es`, `fr`, `de`, `ko`) |
| `--target-language` | `en` | Target caption language (`hi`, `ja`, `en`, `es`, `fr`, `de`, `ko`) |
| `--list-devices` | `False` | Enumerate available WASAPI loopback audio capture devices and exit |
| `--device-index` | `Auto` | WASAPI loopback device index (auto-detects active audio output) |
| `--model-size` | `small` | Faster-Whisper model size (`tiny`, `base`, `small`) |
| `--device` | `auto` | Compute device for Whisper (`auto`, `cpu`, `cuda`) |
| `--compute-type` | `int8` | Model quantization (`int8`, `float16`, `float32`, `default`) |
| `--beam-size` | `1` | Beam search size (`1` for real-time greedy decoding) |
| `--vad-threshold` | `0.4` | Silero VAD speech activation sensitivity (0.1 to 0.9) |
| `--silence-timeout`| `250.0` | Silence duration (ms) to finalize an utterance chunk |
| `--task` | `transcribe`| Whisper task: `transcribe` (speech STT + Argos Translation for dual subtitles) or `translate` (direct EN) |
| `--debug-latency` | `False` | Enable high-resolution stage timing instrumentation and print running p50/p95 periodic reports |
| `--fallback-to-base`| `False` | Automatically fall back to base Whisper model on resource-constrained systems |
| `--no-timestamps` | `False` | Omit timestamp prefixes in CLI mode |

---

## 🛠️ Model Tools & Utilities

CapTran includes dedicated CLI scripts for managing offline models, checking language coverage, and tuning pipeline parameters:

- **Model Setup (`scripts/setup_models.py`):**
  Downloads, validates, and stages Silero VAD, multilingual Faster-Whisper, and Argos translation models with pre-download size estimation:
  ```powershell
  python scripts/setup_models.py --languages hi ja en es fr de ko --whisper-model small
  ```
- **Argos Coverage Matrix (`scripts/check_argos_coverage.py`):**
  Analyzes all 7x6 = 42 language pairs and prints a status matrix showing direct packages vs English-pivot routes:
  ```powershell
  python scripts/check_argos_coverage.py
  ```
- **VAD Parameter Sweep (`scripts/sweep_vad_params.py`):**
  Sweeps VAD silence timeouts (300ms, 500ms, 800ms) and interim window sizes (1.0s, 1.5s, 2.0s) against meeting recordings to balance latency vs accuracy:
  ```powershell
  python scripts/sweep_vad_params.py
  ```
- **Latency & Pipeline Benchmarks:**
  ```powershell
  # Measure latency breakdown per stage
  python scripts/bench_latency.py

  # Test multi-threaded pipeline backpressure and throughput
  python scripts/bench_pipeline.py
  ```

---

## 🧪 Running Tests & Benchmarks

CapTran includes an automated unit and integration test suite (97 tests) with zero external cloud dependencies:

```powershell
# Run full automated test suite (97 tests)
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run multilingual pipeline integration tests specifically
python -m pytest tests/test_multilingual_pipeline_integration.py -v
```

---

## 📦 Building Standalone Windows Executable

To bundle CapTran into a single portable offline folder with PyInstaller:

```powershell
# 1. Install development dependencies
pip install -e .[dev]

# 2. Stage offline model weights
python scripts/setup_models.py

# 3. Build executable
pyinstaller captran.spec --clean --noconfirm
```

The resulting standalone distribution will be generated inside `dist/captran/captran.exe`. See [BUILD_WINDOWS.md](BUILD_WINDOWS.md) for full packaging details.

---

## 📁 Repository Structure

```text
captran/
├── main.py                     # Primary Application Entry Point (CLI/GUI dispatch)
├── pyproject.toml              # Build configuration and project dependencies
├── LICENSE                     # MIT Open Source License
├── ARCHITECTURE.md             # Comprehensive Hexagonal Architecture documentation
├── BUILD_WINDOWS.md            # Windows Standalone PyInstaller build instructions
├── captran.spec                # PyInstaller build specification
├── models/                     # Local offline weights directory
│   ├── silero_vad.onnx         # Offline Silero VAD ONNX model
│   ├── whisper/                # CTranslate2 Faster-Whisper model weights
│   └── argos_packages/         # Staged offline Argos Translate language packages
├── scripts/                    # Utilities and benchmarking tools
│   ├── setup_models.py         # All-in-one model setup with size estimation
│   ├── check_argos_coverage.py # 7x6 language pair coverage and pivot analyzer
│   ├── sweep_vad_params.py     # VAD silence timeout & interim window parameter sweep
│   ├── prepare_offline_models.py # Model staging utility
│   ├── install_translation_model.py # Translation model installer
│   ├── bench_latency.py        # Per-stage latency benchmark
│   ├── bench_pipeline.py       # End-to-end pipeline backpressure benchmark
│   └── test_capture_windows.py # WASAPI loopback test capture utility
├── src/
│   ├── domain/                 # Pure domain entities, value objects & abstract ports
│   │   ├── entities.py         # AudioChunk, TranscriptSegment, CaptionSegment, Language
│   │   └── ports.py            # AudioSource, SpeechSegmenter, Transcriber, Translator, Presenter
│   ├── application/            # Orchestration & Use Cases
│   │   ├── live_caption_use_case.py # Decoupled multi-threaded pipeline with backpressure
│   │   └── latency_tracker.py       # Diagnostic stage timing and p50/p95 metrics
│   ├── infrastructure/         # Concrete adapters (WASAPI, VAD, Whisper, Argos, GUI, CLI)
│   │   ├── audio/              # WASAPI loopback capture adapter and factory
│   │   ├── stt/                # FasterWhisperTranscriber (CPU int8 / CUDA float16)
│   │   ├── translation/        # ArgosTranslateTranslator & PivotingTranslator (42 pairs)
│   │   ├── ui/                 # OverlayPresenter (PySide6) and ConsolePresenter
│   │   ├── vocabulary/         # JsonCustomVocabularyRepository adapter
│   │   └── settings/           # LocalSettingsRepository (~/.ja-en-captioner/config.json)
│   ├── composition_root.py     # Clean dependency injection root
│   ├── composition_root_cli.py # Multilingual CLI application assembler
│   └── composition_root_gui.py # PySide6 GUI application assembler
└── tests/                      # Automated test suite (97 Unit & Integration tests)
    ├── domain/                 # Domain entity & logic tests
    ├── application/            # Use case, backpressure & latency tracker tests
    ├── infrastructure/         # Audio, STT, Translation & UI adapter tests
    ├── data/                   # Bundled multilingual audio test samples
    └── test_multilingual_pipeline_integration.py # 7-language end-to-end integration tests
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
