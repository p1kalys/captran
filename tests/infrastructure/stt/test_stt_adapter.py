"""Unit tests for offline speech-to-text adapter."""

from src.domain.entities import AudioChunk, Language, TranscriptionSegment
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


def test_offline_stt_callback_positional_binding() -> None:
    calls = []

    def transcribe_pos(chunk: AudioChunk, lang: Language):
        calls.append((chunk, lang))
        return [TranscriptionSegment(text="pos test", language=lang)]

    adapter = OfflineSpeechToTextAdapter(transcribe_fn=transcribe_pos)
    assert adapter._invocation_mode == "positional"

    chunk = AudioChunk(data=b"\x00\x00" * 100)
    res = adapter.transcribe(chunk, source_language=Language.SPANISH)
    assert len(res) == 1
    assert res[0].text == "pos test"
    assert calls == [(chunk, Language.SPANISH)]


def test_offline_stt_callback_keyword_binding() -> None:
    calls = []

    def transcribe_kw(chunk: AudioChunk, *, source_language: Language = Language.JAPANESE):
        calls.append((chunk, source_language))
        return [TranscriptionSegment(text="kw test", language=source_language)]

    adapter = OfflineSpeechToTextAdapter(transcribe_fn=transcribe_kw)
    assert adapter._invocation_mode == "keyword"

    chunk = AudioChunk(data=b"\x00\x00" * 100)
    res = adapter.transcribe(chunk, source_language=Language.FRENCH)
    assert len(res) == 1
    assert res[0].text == "kw test"
    assert calls == [(chunk, Language.FRENCH)]


def test_offline_stt_callback_single_arg_binding() -> None:
    calls = []

    def transcribe_single(chunk: AudioChunk):
        calls.append(chunk)
        return [TranscriptionSegment(text="single test")]

    adapter = OfflineSpeechToTextAdapter(transcribe_fn=transcribe_single)
    assert adapter._invocation_mode == "none"

    chunk = AudioChunk(data=b"\x00\x00" * 100)
    res = adapter.transcribe(chunk, source_language=Language.GERMAN)
    assert len(res) == 1
    assert res[0].text == "single test"
    assert calls == [chunk]


def test_offline_stt_callback_exception_preservation() -> None:
    def transcribe_raising(chunk: AudioChunk, lang: Language):
        raise ValueError("Inner callback error")

    adapter = OfflineSpeechToTextAdapter(transcribe_fn=transcribe_raising)
    chunk = AudioChunk(data=b"\x00\x00" * 100)

    import pytest
    with pytest.raises(ValueError, match="Inner callback error"):
        adapter.transcribe(chunk)
