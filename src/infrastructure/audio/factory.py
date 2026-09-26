"""AudioSourceFactory for instantiating platform-appropriate AudioSource adapters."""

import sys
from typing import Any, Dict, List, Optional

from src.domain.ports import AudioSource
from src.infrastructure.audio.adapter import OfflineAudioCaptureAdapter


class AudioSourceFactory:
    """Factory creating concrete AudioSource port implementations."""

    @staticmethod
    def list_available_devices() -> List[Dict[str, Any]]:
        """List available system audio capture devices."""
        if sys.platform == "win32":
            try:
                from src.infrastructure.audio.windows_loopback import (
                    WindowsLoopbackAudioSource,
                )

                return WindowsLoopbackAudioSource.list_loopback_devices()
            except Exception:
                return []
        return []

    @staticmethod
    def create_audio_source(
        device_index: Optional[int] = None,
        chunk_duration_ms: int = 150,
        target_sample_rate: int = 16000,
        use_loopback: bool = True,
    ) -> AudioSource:
        """Create the appropriate AudioSource adapter for the current system.

        Defaults to WindowsLoopbackAudioSource on Windows, or OfflineAudioCaptureAdapter fallback.
        """
        if sys.platform == "win32" and use_loopback:
            try:
                from src.infrastructure.audio.windows_loopback import (
                    WindowsLoopbackAudioSource,
                )

                return WindowsLoopbackAudioSource(
                    device_index=device_index,
                    chunk_duration_ms=chunk_duration_ms,
                    target_sample_rate=target_sample_rate,
                )
            except Exception as e:
                print(f"[WARNING] Could not initialize WindowsLoopbackAudioSource ({e}), using fallback.")

        # Fallback adapter
        return OfflineAudioCaptureAdapter(sample_rate=target_sample_rate)
