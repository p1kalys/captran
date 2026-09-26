"""Unit tests for offline speech-to-text adapter."""

from src.domain.entities import AudioChunk
from src.infrastructure.stt.adapter import OfflineSpeechToTextAdapter


def test_offline_stt_transcription() -> None:
    adapter = OfflineSpeechToTextAdapter(model_name_or_path="whisper-test")
    chunk = AudioChunk(
        data=b"\x00\x00" * 8000,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )
    segments = adapter.transcribe(chunk)
    assert len(segments) == 1
    assert segments[0].language == "ja"
    assert segments[0].text == "こんにちは、世界！"


def test_offline_stt_empty_chunk() -> None:
    adapter = OfflineSpeechToTextAdapter()
    chunk = AudioChunk(
        data=b"",
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )
    segments = adapter.transcribe(chunk)
    assert len(segments) == 0
