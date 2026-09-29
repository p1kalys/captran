"""Unit tests for FasterWhisperTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import wave
import pytest

from src.domain.entities import AudioChunk, TranscriptSegment
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber


def create_or_load_japanese_sample_wav(file_path: Path) -> Path:
    """Create or return bundled sample Japanese audio wav clip."""
    if file_path.exists():
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Generate 1.5 seconds of 16kHz 16-bit mono silence/tone placeholder if needed
    sample_rate = 16000
    total_samples = int(1.5 * sample_rate)
    pcm_data = b"\x00\x00" * total_samples

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    return file_path


def test_faster_whisper_transcriber_initialization() -> None:
    transcriber = FasterWhisperTranscriber(
        model_size="small",
        device="cpu",
        compute_type="int8",
        download_root="models/whisper",
    )
    assert transcriber.model_size == "small"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert transcriber.beam_size == 1


def test_faster_whisper_auto_device_detection() -> None:
    with patch("src.infrastructure.stt.faster_whisper_stt.is_cuda_available", return_value=True):
        transcriber = FasterWhisperTranscriber(device="auto", compute_type="auto")
        assert transcriber.device == "cuda"
        assert transcriber.compute_type == "float16"

    with patch("src.infrastructure.stt.faster_whisper_stt.is_cuda_available", return_value=False):
        transcriber = FasterWhisperTranscriber(device="auto", compute_type="auto")
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"


def test_faster_whisper_fallback_to_base_option() -> None:
    transcriber = FasterWhisperTranscriber(
        model_size="small",
        fallback_to_base=True,
    )
    assert transcriber.model_size == "base"
    assert transcriber.fallback_to_base is True


def test_faster_whisper_transcription_with_mock() -> None:
    transcriber = FasterWhisperTranscriber(model_size="tiny", task="transcribe")

    # Mock WhisperModel output
    mock_segment = MagicMock()
    mock_segment.text = "こんにちは、お元気ですか？"
    mock_segment.start = 0.0
    mock_segment.end = 1.5

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
    transcriber._model = mock_model_instance

    chunk = AudioChunk(
        pcm_data=b"\x00\x00" * 16000,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )

    results = transcriber.transcribe(chunk)
    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].text == "こんにちは、お元気ですか？"
    assert results[0].language == "ja"
    assert results[0].is_final is True


def test_faster_whisper_direct_translation_with_mock() -> None:
    transcriber = FasterWhisperTranscriber(model_size="tiny", task="translate")

    mock_segment = MagicMock()
    mock_segment.text = "Hello, how are you?"
    mock_segment.start = 0.0
    mock_segment.end = 1.5

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
    transcriber._model = mock_model_instance

    chunk = AudioChunk(
        pcm_data=b"\x00\x00" * 16000,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )

    results = transcriber.transcribe(chunk)
    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].text == "Hello, how are you?"
    assert results[0].language == "en"
    assert results[0].is_final is True


@pytest.mark.integration
def test_faster_whisper_transcription_live_or_bundled_audio() -> None:
    """Runs actual model transcription if faster_whisper runtime is installed."""
    try:
        import faster_whisper
    except ImportError:
        pytest.skip("faster-whisper is not installed")

    sample_wav = create_or_load_japanese_sample_wav(
        Path("tests/data/sample_japanese.wav")
    )

    transcriber = FasterWhisperTranscriber(
        model_size="tiny",  # Use tiny for quick test execution
        device="cpu",
        compute_type="int8",
    )

    with wave.open(str(sample_wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    chunk = AudioChunk(
        pcm_data=pcm,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )

    results = transcriber.transcribe(chunk)
    assert isinstance(results, list)
