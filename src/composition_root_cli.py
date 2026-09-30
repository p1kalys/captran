"""Composition root for the CLI terminal runner.

Wires together concrete infrastructure adapters into the LiveCaptionUseCase orchestrator:
AudioSourceFactory -> SileroVadSegmenter -> FasterWhisperTranscriber -> ArgosTranslateTranslator -> ConsoleCaptionPresenter.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import Language
from src.infrastructure.audio.factory import AudioSourceFactory
from src.infrastructure.audio.silero_vad import SileroVadSegmenter
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber
from src.infrastructure.translation.pivoting_translator import PivotingTranslator
from src.infrastructure.ui.console_presenter import ConsoleCaptionPresenter
from src.infrastructure.vocabulary.json_custom_vocabulary import (
    JsonCustomVocabularyRepository,
)


@dataclass
class LiveCaptionerCLIApplication:
    """Container holding wired application orchestrator and concrete adapters."""

    use_case: LiveCaptionUseCase
    audio_source: object
    segmenter: SileroVadSegmenter
    transcriber: FasterWhisperTranscriber
    translator: PivotingTranslator
    presenter: ConsoleCaptionPresenter
    vocabulary: JsonCustomVocabularyRepository


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
    whisper_task: str = "transcribe",
    whisper_beam_size: int = 1,
    source_language: str = "ja",
    target_language: str = "en",
    show_timestamps: bool = True,
    caption_prefix: str = "[EN] ",
    debug_latency: bool = False,
    fallback_to_base: bool = False,
    custom_vocab_path: Optional[Path] = None,
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
        emit_interim=True,
    )

    # 3. Custom Domain Vocabulary Adapter
    vocabulary = JsonCustomVocabularyRepository(config_file_path=custom_vocab_path)
    initial_prompt = vocabulary.get_initial_prompt()

    # 4. Offline Speech-to-Text (CTranslate2 Whisper with direct translation or STT)
    transcriber = FasterWhisperTranscriber(
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
        download_root=whisper_download_root,
        task=whisper_task,
        beam_size=whisper_beam_size,
        fallback_to_base=fallback_to_base,
        initial_prompt=initial_prompt if initial_prompt else None,
    )

    # 5. Offline Translator (Argos Translate direct + pivoting)
    translator = PivotingTranslator()

    # 6. Terminal UI Presenter
    presenter = ConsoleCaptionPresenter(
        show_timestamps=show_timestamps,
        prefix=caption_prefix,
    )

    # 7. Use Case Orchestrator
    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        source_language=source_language,
        target_language=target_language,
        vocabulary=vocabulary,
        debug_latency=debug_latency,
    )

    return LiveCaptionerCLIApplication(
        use_case=use_case,
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        vocabulary=vocabulary,
    )


def run_cli_captioner(app: LiveCaptionerCLIApplication) -> None:
    """Run the live captioner until interrupted (Ctrl+C)."""
    src_code = app.use_case.source_language.value
    tgt_code = app.use_case.target_language.value
    trans_mode = (
        f"Whisper Native Direct {src_code.upper()}->{tgt_code.upper()}"
        if getattr(app.transcriber, "task", "translate") == "translate"
        else f"Argos Translate [{src_code}->{tgt_code}] (Local)"
    )
    print("\n" + "=" * 65)
    print(f"  CAPTRAN ({src_code.upper()} ➔ {tgt_code.upper()} Live Offline Captioner [{src_code}->{tgt_code}])")
    print("=" * 65)
    print(" * Audio Source: WASAPI System Loopback (16kHz mono)")
    print(" * VAD:          Silero VAD ONNX (Local)")
    print(f" * STT / Engine: faster-whisper [{app.transcriber.model_size}] (Local CTranslate2)")
    print(f" * Translation:  {trans_mode}")
    print(" * UI:           Terminal Overwrite Presenter")
    print("=" * 65)
    print(f"\n>>> LISTENING FOR {src_code.upper()} AUDIO... (Press Ctrl+C to stop) <<<\n")

    app.use_case.start(run_in_background=True)

    try:
        while app.use_case.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping pipeline gracefully...")
    finally:
        app.use_case.stop(timeout=2.0)
        print("Pipeline stopped.")
