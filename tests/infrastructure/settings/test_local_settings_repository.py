"""Unit tests for Settings value object and LocalSettingsRepository."""

from pathlib import Path
import tempfile
import pytest

from src.domain.entities import Settings
from src.infrastructure.settings.local_settings_repository import (
    LocalSettingsRepository,
)


def test_settings_defaults_and_immutability() -> None:
    settings = Settings()
    assert settings.whisper_model_size == "small"
    assert settings.overlay_opacity == 0.85
    assert settings.overlay_font_size == 16
    assert settings.vad_sensitivity == 0.4
    assert settings.vad_silence_timeout_ms == 500.0


def test_local_settings_repository_save_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = Path(tmp_dir) / "sub_dir" / "config.json"
        repo = LocalSettingsRepository(config_file_path=config_file)

        # 1. Non-existent file returns defaults
        loaded = repo.load()
        assert loaded.whisper_model_size == "small"
        assert loaded.selected_audio_device is None

        # 2. Save customized settings
        custom_settings = Settings(
            selected_audio_device=2,
            whisper_model_size="medium",
            overlay_opacity=0.9,
            overlay_font_size=20,
            overlay_position_x=150,
            overlay_position_y=60,
            vad_sensitivity=0.45,
            vad_silence_timeout_ms=600.0,
            max_history_lines=4,
        )
        repo.save(custom_settings)
        assert config_file.exists()

        # 3. Load back and assert equality
        reloaded = repo.load()
        assert reloaded.selected_audio_device == 2
        assert reloaded.whisper_model_size == "medium"
        assert reloaded.overlay_opacity == 0.9
        assert reloaded.overlay_font_size == 20
        assert reloaded.overlay_position_x == 150
        assert reloaded.overlay_position_y == 60
        assert reloaded.vad_sensitivity == 0.45
        assert reloaded.vad_silence_timeout_ms == 600.0
        assert reloaded.max_history_lines == 4
