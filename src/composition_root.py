"""Composition root for CapTran (Offline Live Captioner).

This is the only place in the application where concrete infrastructure
adapters are instantiated and injected into application use cases.
"""

from dataclasses import dataclass
from typing import Optional

from src.application.use_cases import LiveCaptioningUseCase
from src.infrastructure.audio.adapter import OfflineAudioCaptureAdapter
from src.infrastructure.stt.adapter import OfflineSpeechToTextAdapter
from src.infrastructure.translation.adapter import OfflineTranslationAdapter
from src.infrastructure.ui.adapter import TerminalCaptionDisplayAdapter


@dataclass
class ApplicationContainer:
    """Container holding wired application instances and their adapters."""

    use_case: LiveCaptioningUseCase
    audio_adapter: OfflineAudioCaptureAdapter
    stt_adapter: OfflineSpeechToTextAdapter
    translation_adapter: OfflineTranslationAdapter
    ui_adapter: TerminalCaptionDisplayAdapter


def build_live_captioner(
    stt_model_path: str = "whisper-small-ja",
    translation_model_path: str = "opus-mt-ja-en",
    sample_rate: int = 16000,
    verbose_ui: bool = False,
    debug_latency: bool = False,
) -> ApplicationContainer:
    """Wire concrete infrastructure adapters into the LiveCaptioningUseCase.

    Returns an ApplicationContainer ready for offline execution.
    """
    audio_adapter = OfflineAudioCaptureAdapter(sample_rate=sample_rate)
    stt_adapter = OfflineSpeechToTextAdapter(model_name_or_path=stt_model_path)
    translation_adapter = OfflineTranslationAdapter(
        model_name_or_path=translation_model_path
    )
    ui_adapter = TerminalCaptionDisplayAdapter(verbose=verbose_ui)

    use_case = LiveCaptioningUseCase(
        audio_capture=audio_adapter,
        speech_to_text=stt_adapter,
        translation=translation_adapter,
        caption_display=ui_adapter,
        debug_latency=debug_latency,
    )

    return ApplicationContainer(
        use_case=use_case,
        audio_adapter=audio_adapter,
        stt_adapter=stt_adapter,
        translation_adapter=translation_adapter,
        ui_adapter=ui_adapter,
    )


if __name__ == "__main__":
    app = build_live_captioner(verbose_ui=True)
    print("Starting ja-en-live-captioner...")
    app.use_case.start()
    # Provide sample audio chunk
    app.audio_adapter.enqueue_chunk(b"\x00\x00" * 16000)
    app.use_case.process_next_chunk()
    app.use_case.stop()
    print("Execution finished successfully.")
