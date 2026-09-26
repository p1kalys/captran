"""Unit tests for domain entities and value objects."""

from dataclasses import FrozenInstanceError
import pytest

from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    TranscriptSegment,
)


def test_audio_chunk_immutability_and_duration() -> None:
    raw_data = b"\x00\x00" * 16000  # 1 sec of 16kHz 16-bit mono
    chunk = AudioChunk(
        pcm_data=raw_data,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )
    assert chunk.duration_seconds == 1.0
    assert chunk.sample_rate == 16000
    assert chunk.timestamp > 0

    # Ensure immutability
    with pytest.raises(FrozenInstanceError):
        chunk.sample_rate = 8000  # type: ignore


def test_transcript_segment_immutability() -> None:
    segment = TranscriptSegment(
        text="こんにちは",
        is_final=True,
        start_time=0.0,
        end_time=1.2,
    )
    assert segment.text == "こんにちは"
    assert segment.is_final is True
    assert segment.language == "ja"

    with pytest.raises(FrozenInstanceError):
        segment.is_final = False  # type: ignore


def test_caption_segment_immutability() -> None:
    caption = CaptionSegment(
        text="Hello",
        is_final=True,
        start_time=0.0,
        end_time=1.2,
    )
    assert caption.text == "Hello"
    assert caption.is_final is True
    assert caption.language == "en"

    with pytest.raises(FrozenInstanceError):
        caption.text = "Changed"  # type: ignore
