"""Local JSON file implementation of the SettingsRepository domain port.

Persists user preferences (audio device, model size, overlay styling/position, VAD parameters)
to local disk at ~/.ja-en-captioner/config.json with zero cloud synchronization.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Optional

from src.domain.entities import Settings
from src.domain.ports import SettingsRepository


class LocalSettingsRepository(SettingsRepository):
    """Saves and loads Settings to/from a local JSON configuration file."""

    DEFAULT_CONFIG_PATH = Path.home() / ".ja-en-captioner" / "config.json"

    def __init__(self, config_file_path: Optional[Path] = None) -> None:
        self.config_path = (
            Path(config_file_path).resolve()
            if config_file_path is not None
            else self.DEFAULT_CONFIG_PATH.resolve()
        )

    def load(self) -> Settings:
        """Load settings from local JSON file. Returns default Settings if file does not exist."""
        if not self.config_path.exists():
            return Settings()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return Settings()

            # Safely filter only valid fields defined on Settings
            default_settings = Settings()
            valid_fields = set(asdict(default_settings).keys())
            filtered_kwargs = {k: v for k, v in data.items() if k in valid_fields}

            return Settings(**filtered_kwargs)
        except Exception as e:
            print(f"[WARNING] Failed to load config from {self.config_path} ({e}), using defaults.")
            return Settings()

    def save(self, settings: Settings) -> None:
        """Persist settings to local JSON file on disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(settings)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.config_path)
        except Exception as e:
            print(f"[ERROR] Failed to save config to {self.config_path}: {e}")
