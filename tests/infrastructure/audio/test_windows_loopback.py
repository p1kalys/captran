"""Unit tests for WindowsLoopbackAudioSource."""

import sys
from unittest.mock import MagicMock, patch
import pytest

from src.infrastructure.audio.windows_loopback import WindowsLoopbackAudioSource


def test_windows_loopback_initialization() -> None:
    source = WindowsLoopbackAudioSource(
        device_index=1,
        chunk_duration_ms=200,
        target_sample_rate=16000,
    )
    assert source.chunk_duration_ms == 200
    assert source.target_sample_rate == 16000
    assert source._is_active is False


def test_audio_callback_downmixing_and_resampling() -> None:
    source = WindowsLoopbackAudioSource(
        chunk_duration_ms=100,
        target_sample_rate=16000,
    )
    source._is_active = True
    source._native_sample_rate = 48000
    source._native_channels = 2

    # 48000 Hz, stereo (2 channels), 16-bit (2 bytes) -> 48000 * 4 = 192,000 bytes/sec
    # 0.1s = 4800 frames = 9600 samples (stereo) = 19200 bytes
    dummy_stereo_pcm = b"\x00\x01\x00\x01" * 4800

    source._audio_callback(dummy_stereo_pcm, 4800, {}, 0)

    # We expect one chunk in the queue
    chunk = source._queue.get_nowait()
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    # 0.1s at 16000Hz = 1600 samples = 3200 bytes
    assert len(chunk.pcm_data) == 1600 * 2
