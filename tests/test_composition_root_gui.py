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

    # Verify 7 supported languages in dropdowns, no auto-detect
    assert window.source_lang_combo.count() == 7
    assert window.target_lang_combo.count() == 7
    codes = [window.source_lang_combo.itemData(i) for i in range(window.source_lang_combo.count())]
    assert codes == ["hi", "ja", "en", "es", "fr", "de", "ko"]
    assert "auto" not in codes

    # Test apply_settings with custom languages
    custom_settings = Settings(
        selected_audio_device=1,
        whisper_model_size="tiny",
        source_language_mode="es",
        target_language="fr",
        dual_subtitles=True,
    )
    window.apply_settings(custom_settings)
    assert window.model_combo.currentText() == "tiny"
    assert window.device_combo.currentData() == 1
    assert window.source_lang_combo.currentData() == "es"
    assert window.target_lang_combo.currentData() == "fr"
    assert window.dual_subtitles_check.isChecked() is True
    assert window.toggle_btn.isEnabled() is True

    # Test start emission contains selected languages
    started_configs = []
    window.start_requested.connect(started_configs.append)
    window._on_toggle_clicked()

    assert len(started_configs) == 1
    assert started_configs[0]["device_index"] == 1
    assert started_configs[0]["model_size"] == "tiny"
    assert started_configs[0]["source_language"] == "es"
    assert started_configs[0]["target_language"] == "fr"
    assert window._is_running is True

    # While running, dropdowns are disabled
    assert window.source_lang_combo.isEnabled() is False
    assert window.target_lang_combo.isEnabled() is False
    assert window.device_combo.isEnabled() is False
    assert window.model_combo.isEnabled() is False

    # Test stop emission
    stop_called = []
    window.stop_requested.connect(lambda: stop_called.append(True))
    window._on_toggle_clicked()

    assert len(stop_called) == 1
    assert window._is_running is False

    # After stopped, dropdowns are re-enabled
    assert window.source_lang_combo.isEnabled() is True
    assert window.target_lang_combo.isEnabled() is True

    # Change dropdown to new pair (e.g. ko -> de) and verify next start config
    window.source_lang_combo.setCurrentIndex(window.source_lang_combo.findData("ko"))
    window.target_lang_combo.setCurrentIndex(window.target_lang_combo.findData("de"))
    window._on_toggle_clicked()
    assert len(started_configs) == 2
    assert started_configs[1]["source_language"] == "ko"
    assert started_configs[1]["target_language"] == "de"
    window._on_toggle_clicked()  # stop again


def test_gui_controller_settings_wiring() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        repo = LocalSettingsRepository(config_file_path=cfg_file)
        repo.save(Settings(
            whisper_model_size="tiny",
            source_language_mode="de",
            target_language="ko",
            overlay_opacity=0.9,
            dual_subtitles=True,
        ))

        from src.composition_root_gui import LiveCaptionerGUIController

        controller = LiveCaptionerGUIController(settings_repository=repo)
        assert controller.settings.whisper_model_size == "tiny"
        assert controller.control_window.model_combo.currentText() == "tiny"
        assert controller.control_window.source_lang_combo.currentData() == "de"
        assert controller.control_window.target_lang_combo.currentData() == "ko"
        assert controller.presenter.widget.opacity == 0.9
        assert controller.presenter.widget.dual_subtitles is True

        # Clean up
        controller.close()

