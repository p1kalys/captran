"""Composition root for the CLI terminal runner.

Wires together concrete infrastructure adapters into the LiveCaptionUseCase orchestrator:
AudioSourceFactory -> SileroVadSegmenter -> FasterWhisperTranscriber -> ArgosTranslateTranslator -> ConsoleCaptionPresenter.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.infrastructure.audio.factory import AudioSourceFactory
from src.infrastructure.audio.silero_vad import SileroVadSegmenter
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber
from src.infrastructure.translation.argos_translator import ArgosTranslateTranslator
from src.infrastructure.ui.console_presenter import ConsoleCaptionPresenter


@dataclass
class LiveCaptionerCLIApplication:
    """Container holding wired application orchestrator and concrete adapters."""

    use_case: LiveCaptionUseCase
    audio_source: object
    segmenter: SileroVadSegmenter
    transcriber: FasterWhisperTranscriber
    translator: ArgosTranslateTranslator
    presenter: ConsoleCaptionPresenter


def build_cli_application(
    audio_device_index: Optional[int] = None,
    chunk_duration_ms: int = 150,
    vad_model_path: str = "models/silero_vad.onnx",
    vad_threshold: float = 0.4,
    vad_min_speech_ms: float = 250.0,
    vad_silence_timeout_ms: float = 250.0,
    whisper_model_size: str = "small",
    whisper_device: str = "auto",
    whisper_compute_type: str = "int8",
    whisper_download_root: str = "models/whisper",
    whisper_task: str = "translate",
    whisper_beam_size: int = 1,
    show_timestamps: bool = True,
    caption_prefix: str = "[EN] ",
) -> LiveCaptionerCLIApplication:
    """Instantiate concrete adapters and assemble the LiveCaptionUseCase."""

    # 1. Audio Source (WASAPI loopback or fallback)
    audio_source = AudioSourceFactory.create_audio_source(
        device_index=audio_device_index,
        chunk_duration_ms=chunk_duration_ms,
        target_sample_rate=16000,
        use_loopback=True,
    )

    # 2. VAD Utterance Segmenter
    segmenter = SileroVadSegmenter(
        model_path=vad_model_path,
        sample_rate=16000,
        threshold=vad_threshold,
        min_speech_duration_ms=vad_min_speech_ms,
        silence_timeout_ms=vad_silence_timeout_ms,
    )

    # 3. Offline Speech-to-Text (CTranslate2 Whisper with direct translation or STT)
    transcriber = FasterWhisperTranscriber(
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
        download_root=whisper_download_root,
        task=whisper_task,
        beam_size=whisper_beam_size,
    )

    # 4. Offline Translator (Argos Translate for fallback/transcribe mode)
    translator = ArgosTranslateTranslator(
        from_code="ja",
        to_code="en",
    )

    # 5. Terminal UI Presenter
    presenter = ConsoleCaptionPresenter(
        show_timestamps=show_timestamps,
        prefix=caption_prefix,
    )

    # 6. Use Case Orchestrator
    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    return LiveCaptionerCLIApplication(
        use_case=use_case,
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )


def run_cli_captioner(app: LiveCaptionerCLIApplication) -> None:
    """Run the live captioner until interrupted (Ctrl+C)."""
    trans_mode = (
        "Whisper Native Direct JA->EN (High Accuracy)"
        if getattr(app.transcriber, "task", "translate") == "translate"
        else "Argos Translate [ja->en] (Local)"
    )
    print("\n" + "=" * 65)
    print("  CAPTRAN (Japanese ➔ English Live Offline Captioner [ja->en])")
    print("=" * 65)
    print(" * Audio Source: WASAPI System Loopback (16kHz mono)")
    print(" * VAD:          Silero VAD ONNX (Local)")
    print(f" * STT / Engine: faster-whisper [{app.transcriber.model_size}] (Local CTranslate2)")
    print(f" * Translation:  {trans_mode}")
    print(" * UI:           Terminal Overwrite Presenter")
    print("=" * 65)
    print("\n>>> LISTENING FOR JAPANESE AUDIO... (Press Ctrl+C to stop) <<<\n")

    app.use_case.start(run_in_background=True)

    try:
        while app.use_case.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping pipeline gracefully...")
    finally:
        app.use_case.stop(timeout=2.0)
        print("Pipeline stopped.")
