"""Unit tests for offline audio capture adapter."""

from src.infrastructure.audio.adapter import OfflineAudioCaptureAdapter


def test_audio_adapter_lifecycle() -> None:
    adapter = OfflineAudioCaptureAdapter(sample_rate=16000, channels=1, sample_width=2)
    assert adapter.is_active() is False

    adapter.start_capture()
    assert adapter.is_active() is True

    chunk = adapter.enqueue_chunk(b"\x00\x01" * 160)
    assert chunk.sample_rate == 16000

    read = adapter.read_chunk(timeout=0.1)
    assert read is not None
    assert read.data == b"\x00\x01" * 160

    adapter.stop_capture()
    assert adapter.is_active() is False
