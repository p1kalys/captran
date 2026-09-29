"""GUI Composition Root for JA->EN Live Captioner.

Wires together the ControlWindow, OverlayCaptionPresenter, LocalSettingsRepository, and
concrete infrastructure adapters into the LiveCaptionUseCase via dependency injection.
Settings are loaded on launch and saved to ~/.ja-en-captioner/config.json automatically.
Pipeline status events (running, reconnecting, degraded, error) are reflected in real-time.
"""

from dataclasses import replace
from pathlib import Path
import sys
from typing import Optional

try:
    from PySide6.QtCore import QObject, Qt, Signal, Slot
    from PySide6.QtWidgets import QApplication
except ImportError:
    QApplication = None  # type: ignore
    QObject = object     # type: ignore
    Signal = lambda *args: None  # type: ignore
    Slot = lambda *args: lambda fn: fn  # type: ignore

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import PipelineStatus, Settings
from src.domain.ports import SettingsRepository
from src.infrastructure.audio.factory import AudioSourceFactory
from src.infrastructure.audio.silero_vad import SileroVadSegmenter
from src.infrastructure.settings.local_settings_repository import (
    LocalSettingsRepository,
)
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber
from src.infrastructure.translation.argos_translator import ArgosTranslateTranslator
from src.infrastructure.ui.control_window import ControlWindow
from src.infrastructure.ui.overlay_presenter import OverlayCaptionPresenter
from src.infrastructure.vocabulary.json_custom_vocabulary import (
    JsonCustomVocabularyRepository,
)


class _StatusBridge(QObject):
    """Bridge for dispatching pipeline status updates from worker threads to Qt GUI thread."""

    status_received = Signal(object)


class LiveCaptionerGUIController:
    """Manages the lifecycle, configuration persistence, and execution of the GUI and background pipeline."""

    def __init__(
        self,
        settings_repository: Optional[SettingsRepository] = None,
        vocabulary_repository: Optional[object] = None,
        default_model_size: str = "small",
        vad_threshold: float = 0.4,
        vad_silence_timeout_ms: float = 500.0,
        debug_latency: bool = False,
    ) -> None:
        if QApplication is None:
            raise ImportError("PySide6 is required to run the GUI application.")

        # 1. Initialize Settings and Vocabulary Repositories
        self.settings_repo = settings_repository or LocalSettingsRepository()
        self.vocabulary_repo = vocabulary_repository or JsonCustomVocabularyRepository()
        self.settings: Settings = self.settings_repo.load()

        self.default_model_size = default_model_size
        self.vad_threshold = vad_threshold
        self.vad_silence_timeout_ms = vad_silence_timeout_ms
        self.debug_latency = debug_latency
        self._use_case: Optional[LiveCaptionUseCase] = None

        # 2. Discover available WASAPI audio loopback devices
        devices = AudioSourceFactory.list_available_devices()

        # 3. Status Bridge for thread-safe UI updates
        self._status_bridge = _StatusBridge()

        # 4. Instantiate UI Presenter & Widgets with loaded settings
        self.presenter = OverlayCaptionPresenter(
            max_history_lines=self.settings.max_history_lines,
            font_size=self.settings.overlay_font_size,
            opacity=self.settings.overlay_opacity,
            show_window=True,
        )
        self.presenter.apply_settings(self.settings)

        self.control_window = ControlWindow(devices=devices)
        self.control_window.apply_settings(self.settings)

        # 5. Connect Signals for Live Execution, Status, and Settings Persistence
        self._status_bridge.status_received.connect(
            self.control_window.update_status,
            Qt.ConnectionType.QueuedConnection,
        )
        self._status_bridge.status_received.connect(
            self.presenter.update_status,
            Qt.ConnectionType.QueuedConnection,
        )
        self.control_window.start_requested.connect(self.start_pipeline)
        self.control_window.stop_requested.connect(self.stop_pipeline)
        self.control_window.settings_changed.connect(self._on_control_settings_changed)
        self.presenter.widget.position_changed.connect(self._on_overlay_position_changed)

    def _on_status_callback(self, status: PipelineStatus) -> None:
        """Called by LiveCaptionUseCase worker thread."""
        self._status_bridge.status_received.emit(status)

    def _on_control_settings_changed(self, changes: dict) -> None:
        """Update and persist settings when user modifies control panel options."""
        device_idx = changes.get("selected_audio_device")
        model_size = changes.get("whisper_model_size", self.settings.whisper_model_size)
        dual_subtitles = changes.get("dual_subtitles", self.settings.dual_subtitles)

        self.settings = replace(
            self.settings,
            selected_audio_device=device_idx,
            whisper_model_size=model_size,
            dual_subtitles=dual_subtitles,
        )
        self.settings_repo.save(self.settings)
        self.presenter.apply_settings(self.settings)

    def _on_overlay_position_changed(self, x: int, y: int) -> None:
        """Update and persist settings when user drags the overlay."""
        self.settings = replace(
            self.settings,
            overlay_position_x=x,
            overlay_position_y=y,
        )
        self.settings_repo.save(self.settings)

    def start_pipeline(self, config: dict) -> None:
        """Dynamically assemble and launch the LiveCaptionUseCase pipeline in background."""
        self.stop_pipeline()

        device_index = config.get("device_index", self.settings.selected_audio_device)
        model_size = config.get("model_size", self.settings.whisper_model_size)

        # Update settings
        self.settings = replace(
            self.settings,
            selected_audio_device=device_index,
            whisper_model_size=model_size,
        )
        self.settings_repo.save(self.settings)

        # Construct concrete adapters
        audio_source = AudioSourceFactory.create_audio_source(
            device_index=device_index,
            target_sample_rate=16000,
            use_loopback=True,
        )
        segmenter = SileroVadSegmenter(
            model_path="models/silero_vad.onnx",
            sample_rate=16000,
            threshold=self.settings.vad_sensitivity,
            silence_timeout_ms=self.settings.vad_silence_timeout_ms,
            emit_interim=True,
        )
        transcriber = FasterWhisperTranscriber(
            model_size=model_size,
            device="auto",
            compute_type="auto",
            beam_size=1,
            task="transcribe",
            fallback_to_base=self.settings.fallback_to_base,
        )
        translator = ArgosTranslateTranslator(
            from_code="ja",
            to_code="en",
        )

        # Wire into Application Use Case via Dependency Injection
        self._use_case = LiveCaptionUseCase(
            audio_source=audio_source,
            segmenter=segmenter,
            transcriber=transcriber,
            translator=translator,
            presenter=self.presenter,
            vocabulary=self.vocabulary_repo,
            status_callback=self._on_status_callback,
            debug_latency=self.debug_latency,
        )

        # Clear existing overlay captions and start background worker
        self.presenter.clear()
        self._use_case.start(run_in_background=True)

    def stop_pipeline(self) -> None:
        """Cleanly stop the running use case and release hardware/model resources."""
        if self._use_case is not None:
            try:
                self._use_case.stop(timeout=1.0)
            except Exception:
                pass
            self._use_case = None

    def show(self) -> None:
        """Show both the control window and overlay."""
        self.control_window.show()
        self.presenter.show()

    def close(self) -> None:
        """Gracefully close all windows, persist current state, and stop pipeline."""
        self.stop_pipeline()
        # Save overlay position on exit
        pos_x, pos_y = self.presenter.get_position()
        self.settings = replace(
            self.settings,
            overlay_position_x=pos_x,
            overlay_position_y=pos_y,
        )
        self.settings_repo.save(self.settings)

        self.control_window.close()
        self.presenter.close()


def run_gui_application(
    settings_repository: Optional[SettingsRepository] = None,
    default_model_size: str = "small",
    vad_threshold: float = 0.4,
    vad_silence_timeout_ms: float = 500.0,
    debug_latency: bool = False,
) -> int:
    """Bootstrap and run the PySide6 live captioner application."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CapTran")

    controller = LiveCaptionerGUIController(
        settings_repository=settings_repository,
        default_model_size=default_model_size,
        vad_threshold=vad_threshold,
        vad_silence_timeout_ms=vad_silence_timeout_ms,
        debug_latency=debug_latency,
    )
    controller.show()

    app.aboutToQuit.connect(controller.close)
    return app.exec()
