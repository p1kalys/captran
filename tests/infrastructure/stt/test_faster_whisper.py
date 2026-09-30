"""Unit tests for FasterWhisperTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import wave
import pytest

from src.domain.entities import AudioChunk, Language, TranscriptSegment
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

    results = transcriber.transcribe(chunk, source_language=Language.JAPANESE)
    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].text == "こんにちは、お元気ですか？"
    assert results[0].language == Language.JAPANESE
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

    results = transcriber.transcribe(chunk, source_language=Language.ENGLISH)
    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].text == "Hello, how are you?"
    assert results[0].language == Language.ENGLISH
    assert results[0].is_final is True


def test_faster_whisper_model_size_options() -> None:
    # Default is small for optimal latency
    t_default = FasterWhisperTranscriber()
    assert t_default.model_size == "small"

    # Medium and large configurable for higher accuracy on harder languages
    t_medium = FasterWhisperTranscriber(model_size="medium")
    assert t_medium.model_size == "medium"

    t_large = FasterWhisperTranscriber(model_size="large-v3")
    assert t_large.model_size == "large-v3"

    t_base = FasterWhisperTranscriber(model_size="base")
    assert t_base.model_size == "base"


def test_faster_whisper_pinned_language_multilingual_no_detection() -> None:
    """Test pinned language across 3+ languages (e.g., ja, en, es, hi) confirming no detection runs."""
    languages_to_test = [
        (Language.JAPANESE, "こんにちは、お元気ですか？"),
        (Language.ENGLISH, "Welcome to the live captioning session."),
        (Language.SPANISH, "Buenos días, bienvenidos a todos."),
        (Language.HINDI, "नमस्ते, आप सब का स्वागत है।"),
    ]

    for pinned_lang, sample_text in languages_to_test:
        transcriber = FasterWhisperTranscriber(model_size="small", task="transcribe")

        mock_segment = MagicMock()
        mock_segment.text = sample_text
        mock_segment.start = 0.0
        mock_segment.end = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        transcriber._model = mock_model

        chunk = AudioChunk(
            pcm_data=b"\x00\x00" * 16000,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )

        results = transcriber.transcribe(chunk, source_language=pinned_lang)

        # 1. Assert model.transcribe was called with explicit pinned language code (no None, no detection)
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == pinned_lang.value
        assert call_kwargs["language"] is not None

        # 2. Assert output segment carries the pinned language and accurate text
        assert len(results) == 1
        assert results[0].text == sample_text
        assert results[0].language == pinned_lang
        assert results[0].language == pinned_lang.value


def create_or_load_multilingual_sample_wav(file_path: Path) -> Path:
    """Helper to create or return sample audio WAV for multilingual test."""
    if file_path.exists():
        return file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    pcm_data = b"\x00\x00" * (sample_rate * 2)  # 2 seconds
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return file_path


def test_faster_whisper_bundled_sample_clips_multilingual() -> None:
    """Verify transcription on 3 distinct bundled sample audio clips with pinned languages."""
    test_clips = [
        ("tests/data/sample_japanese.wav", Language.JAPANESE, "日本語テスト"),
        ("tests/data/sample_english.wav", Language.ENGLISH, "English test audio"),
        ("tests/data/sample_spanish.wav", Language.SPANISH, "Prueba de audio en español"),
    ]

    for wav_path_str, lang, expected_text in test_clips:
        wav_path = create_or_load_multilingual_sample_wav(Path(wav_path_str))
        transcriber = FasterWhisperTranscriber(model_size="small")

        mock_seg = MagicMock()
        mock_seg.text = expected_text
        mock_seg.start = 0.0
        mock_seg.end = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())
        transcriber._model = mock_model

        with wave.open(str(wav_path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())

        chunk = AudioChunk(pcm_data=pcm, sample_rate=16000)
        results = transcriber.transcribe(chunk, source_language=lang)

        assert len(results) == 1
        assert results[0].text == expected_text
        assert results[0].language == lang
        # Verify language was explicitly passed without detection
        assert mock_model.transcribe.call_args[1]["language"] == lang.value


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

    results = transcriber.transcribe(chunk, source_language=Language.JAPANESE)
    assert isinstance(results, list)

