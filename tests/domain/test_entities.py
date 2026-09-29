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


def test_custom_vocabulary_word_boundary_substitutions() -> None:
    from src.domain.entities import CustomVocabulary

    vocab = CustomVocabulary(
        prompt_terms=["CapTran", "FastAPI"],
        source_substitutions={"キャプ": "CapTran"},
        target_substitutions={"cat": "feline", "AI": "Artificial Intelligence"},
    )
    assert vocab.get_initial_prompt() == "CapTran, FastAPI"
    assert vocab.apply_source_substitutions("これはキャプです") == "これはCapTranです"

    # Alphanumeric target substitutions should not replace inside longer words like "catch" or "TRAIN"
    text = "The cat caught a mouse using AI tech and a CAT."
    res = vocab.apply_target_substitutions(text)
    assert res == "The feline caught a mouse using Artificial Intelligence tech and a feline."

    # Backslashes in replacement values should be preserved literally
    vocab_backslash = CustomVocabulary(
        target_substitutions={"dir": r"C:\Users\path\test", "regex_ref": r"\1\g<0>"},
    )
    res_bs = vocab_backslash.apply_target_substitutions("Check the dir and regex_ref here.")
    assert res_bs == r"Check the C:\Users\path\test and \1\g<0> here."
