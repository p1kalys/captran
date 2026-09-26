"""Unit tests for GUI composition root, ControlWindow, and Settings integration."""

from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from src.domain.entities import Settings
from src.infrastructure.settings.local_settings_repository import (
    LocalSettingsRepository,
)
from src.infrastructure.ui.control_window import ControlWindow


def test_control_window_interaction() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    devices = [
        {"index": 1, "name": "Headphones Loopback", "channels": 2, "defaultSampleRate": 48000}
    ]

    window = ControlWindow(devices=devices)
    assert window.device_combo.count() == 2  # Default + 1 device

    # Test apply_settings
    custom_settings = Settings(
        selected_audio_device=1,
        whisper_model_size="tiny",
    )
    window.apply_settings(custom_settings)
    assert window.model_combo.currentText() == "tiny"
    assert window.device_combo.currentData() == 1

    # Test start emission
    started_configs = []
    window.start_requested.connect(started_configs.append)
    window._on_toggle_clicked()

    assert len(started_configs) == 1
    assert started_configs[0]["device_index"] == 1
    assert started_configs[0]["model_size"] == "tiny"
    assert window._is_running is True

    # Test stop emission
    stop_called = []
    window.stop_requested.connect(lambda: stop_called.append(True))
    window._on_toggle_clicked()

    assert len(stop_called) == 1
    assert window._is_running is False


def test_gui_controller_settings_wiring() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        repo = LocalSettingsRepository(config_file_path=cfg_file)
        repo.save(Settings(whisper_model_size="tiny", overlay_opacity=0.9))

        from src.composition_root_gui import LiveCaptionerGUIController

        controller = LiveCaptionerGUIController(settings_repository=repo)
        assert controller.settings.whisper_model_size == "tiny"
        assert controller.control_window.model_combo.currentText() == "tiny"
        assert controller.presenter.widget.opacity == 0.9

        # Clean up
        controller.close()
