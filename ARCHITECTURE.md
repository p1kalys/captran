# Architecture: CapTran (`captran`)

## Overview

`captran` is designed using **Hexagonal Architecture (Ports and Adapters)**. The application captures Japanese audio from system/meeting output, detects speech utterances via offline VAD, transcribes/translates it to English via local faster-whisper, and displays live subtitles—**with zero cloud or network dependencies** once offline models/assets are loaded locally.

---

## The Dependency-Direction Rule

The fundamental architectural constraint is:

> **All source code dependencies point inward toward the domain.**
> - Domain has **zero** external dependencies.
> - Application depends **only on Domain**.
> - Infrastructure depends on **Domain port interfaces**, **never** the reverse.
> - The Domain and Application layers are completely decoupled from third-party libraries, audio drivers, ML runtimes, OS subsystems, and UI frameworks.

```text
       +---------------------------------------------------+
       |                 INFRASTRUCTURE                    |
       |  (WASAPI Audio, Silero VAD, Faster-Whisper, PySide)|
       |                         |                         |
       |                         v (implements)            |
       |  +---------------------------------------------+  |
       |  |                 APPLICATION                 |  |
       |  |             (LiveCaptionUseCase)            |  |
       |  |                      |                      |  |
       |  |                      v                      |  |
       |  |  +---------------------------------------+  |  |
       |  |  |                DOMAIN                 |  |  |
       |  |  |   (Entities, Value Objects, Ports)    |  |  |
       |  |  +---------------------------------------+  |  |
       |  +---------------------------------------------+  |
       +---------------------------------------------------+
```

---

## Layer Responsibilities

### 1. `src/domain/`
- **Pure Python standard library only.**
- **Entities & Value Objects**: `AudioChunk`, `TranscriptSegment`, `CaptionSegment`, `Settings`, `PipelineStatus`.
- **Ports (Abstract Interfaces)**:
  - `AudioSource`: Interface for capturing live audio streams.
  - `SpeechSegmenter`: Interface for voice activity detection (VAD).
  - `Transcriber`: Interface for offline speech-to-text / direct translation.
  - `Translator`: Interface for machine translation.
  - `CaptionPresenter`: Interface for subtitle presentation (GUI overlay or CLI).
  - `SettingsRepository`: Interface for user settings persistence.

### 2. `src/application/`
- Contains use case orchestrators (`LiveCaptionUseCase`).
- Coordinates the pipeline:
  1. Receive audio chunk streams from `AudioSource`.
  2. Segment speech utterances via `SpeechSegmenter` (Silero VAD).
  3. Transcribe and translate speech chunks via `Transcriber` (Faster-Whisper).
  4. Dispatch final/interim subtitles to `CaptionPresenter`.
- Holds business rules, lifecycle controls (start, stop), and automatic reconnection with backoff.

### 3. `src/infrastructure/`
Contains concrete adapters that implement domain ports using specific offline tools:
- `infrastructure/audio/`: Windows WASAPI loopback audio capture via `pyaudiowpatch` and `AudioSourceFactory`.
- `infrastructure/audio/silero_vad.py`: Offline ONNX Silero VAD segmenter.
- `infrastructure/stt/faster_whisper_stt.py`: CTranslate2 INT8 faster-whisper direct speech translator.
- `infrastructure/translation/argos_translator.py`: Local Argos Translate model adapter.
- `infrastructure/ui/`: PySide6 transparent overlay presenter (`overlay_presenter.py`), desktop control panel (`control_window.py`), and CLI console presenter (`console_presenter.py`).
- `infrastructure/settings/`: Local JSON configuration repository (`local_settings_repository.py`).

### 4. Composition Roots
- `src/composition_root_gui.py`: Assembles adapters for the PySide6 desktop GUI.
- `src/composition_root_cli.py`: Assembles adapters for the headless terminal runner.
- `main.py`: Top-level CLI argument dispatcher.

---

## Directory Layout

```text
captran/
├── pyproject.toml
├── README.md
├── LICENSE
├── ARCHITECTURE.md
├── BUILD_WINDOWS.md
├── captran.spec
├── main.py
├── models/
│   ├── silero_vad.onnx
│   └── whisper/
├── scripts/
│   ├── prepare_offline_models.py
│   ├── install_translation_model.py
│   └── test_capture_windows.py
├── src/
│   ├── domain/
│   │   ├── entities.py
│   │   └── ports.py
│   ├── application/
│   │   └── live_caption_use_case.py
│   ├── infrastructure/
│   │   ├── audio/
│   │   ├── stt/
│   │   ├── translation/
│   │   ├── ui/
│   │   └── settings/
│   ├── composition_root.py
│   ├── composition_root_cli.py
│   └── composition_root_gui.py
└── tests/
    ├── domain/
    ├── application/
    ├── infrastructure/
    ├── test_composition_root.py
    ├── test_composition_root_cli.py
    └── test_composition_root_gui.py
```
