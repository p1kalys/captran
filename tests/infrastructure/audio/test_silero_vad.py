"""Unit tests for SileroVadSegmenter using local ONNX model and sample audio."""

import math
from pathlib import Path
import struct
import wave
import pytest

from src.domain.entities import AudioChunk
from src.infrastructure.audio.silero_vad import SileroVadSegmenter


def create_or_load_sample_wav(file_path: Path) -> Path:
    """Create a 4-second WAV containing 2 voice utterances separated by silence."""
    if file_path.exists():
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    total_duration = 4.0
    total_samples = int(total_duration * sample_rate)
    samples = []

    for i in range(total_samples):
        t = i / sample_rate
        val = 0.0

        # Utterance 1: 0.3s to 1.3s (1.0s speech)
        if 0.3 <= t < 1.3:
            f0 = 220.0
            val = (
                0.6 * math.sin(2 * math.pi * f0 * t)
                + 0.3 * math.sin(2 * math.pi * f0 * 3 * t)
                + 0.2 * math.sin(2 * math.pi * f0 * 5 * t)
            )
        # Utterance 2: 2.3s to 3.3s (1.0s speech)
        elif 2.3 <= t < 3.3:
            f0 = 260.0
            val = (
                0.6 * math.sin(2 * math.pi * f0 * t)
                + 0.3 * math.sin(2 * math.pi * f0 * 3 * t)
                + 0.2 * math.sin(2 * math.pi * f0 * 5 * t)
            )

        int16_val = int(max(-1.0, min(1.0, val)) * 30000)
        samples.append(int16_val)

    pcm_data = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    return file_path


def wav_to_chunk_stream(wav_path: Path, chunk_duration_s: float = 0.05):
    """Stream WAV file as a sequence of small AudioChunk objects."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        chunk_frames = int(sample_rate * chunk_duration_s)

        current_time = 0.0
        while True:
            pcm = wf.readframes(chunk_frames)
            if not pcm:
                break
            yield AudioChunk(
                pcm_data=pcm,
                sample_rate=sample_rate,
                timestamp=current_time,
                channels=channels,
                sample_width=sample_width,
            )
            current_time += chunk_duration_s


def test_silero_vad_segmenter_splits_expected_utterances(monkeypatch):
    model_path = Path("models/silero_vad.onnx")
    if not model_path.exists():
        pytest.skip("models/silero_vad.onnx not present")

    wav_path = create_or_load_sample_wav(Path("tests/data/sample_two_utterances.wav"))

    segmenter = SileroVadSegmenter(
        model_path=str(model_path),
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250.0,
        silence_timeout_ms=500.0,
    )

    # Mock _predict_prob to simulate 2 speech intervals matching the wav timeline:
    # Utterance 1: 0.3s to 1.3s (speech prob 0.9)
    # Silence: 1.3s to 2.3s (prob 0.05)
    # Utterance 2: 2.3s to 3.3s (speech prob 0.9)
    # Silence: 3.3s to 4.0s (prob 0.05)
    frame_dur = 512 / 16000.0
    current_time_ref = [0.0]

    def mock_predict_prob(window_pcm: bytes) -> float:
        t = current_time_ref[0]
        current_time_ref[0] += frame_dur
        if (0.3 <= t < 1.3) or (2.3 <= t < 3.3):
            return 0.95
        return 0.05

    monkeypatch.setattr(segmenter, "_predict_prob", mock_predict_prob)

    chunk_stream = wav_to_chunk_stream(wav_path, chunk_duration_s=0.032)
    utterances = list(segmenter.segment(chunk_stream))

    # Expect exactly 2 distinct utterances separated by the 1.0s silence gap
    assert len(utterances) == 2, f"Expected 2 utterances, got {len(utterances)}"

    for idx, utt in enumerate(utterances):
        assert isinstance(utt, AudioChunk)
        assert utt.sample_rate == 16000
        assert utt.duration_seconds >= 0.5, (
            f"Utterance {idx + 1} duration {utt.duration_seconds}s is shorter than expected"
        )


def test_silero_vad_onnx_live_inference():
    """Verify live Silero VAD ONNX model initializes and runs inference on audio frames."""
    model_path = Path("models/silero_vad.onnx")
    if not model_path.exists():
        pytest.skip("models/silero_vad.onnx not present")

    segmenter = SileroVadSegmenter(
        model_path=str(model_path),
        sample_rate=16000,
        threshold=0.5,
    )

    # Run inference on a 512-sample frame of silence
    silence_pcm = b"\x00\x00" * 512
    prob = segmenter._predict_prob(silence_pcm)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert prob < 0.2  # Silence probability should be near zero


def test_silero_vad_emits_rolling_interim_windows(monkeypatch):
    model_path = Path("models/silero_vad.onnx")
    if not model_path.exists():
        pytest.skip("models/silero_vad.onnx not present")

    wav_path = create_or_load_sample_wav(Path("tests/data/sample_two_utterances.wav"))

    segmenter = SileroVadSegmenter(
        model_path=str(model_path),
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250.0,
        silence_timeout_ms=500.0,
        interim_interval_ms=500.0,
        emit_interim=True,
    )

    frame_dur = 512 / 16000.0
    current_time_ref = [0.0]

    def mock_predict_prob(window_pcm: bytes) -> float:
        t = current_time_ref[0]
        current_time_ref[0] += frame_dur
        if (0.3 <= t < 1.3) or (2.3 <= t < 3.3):
            return 0.95
        return 0.05

    monkeypatch.setattr(segmenter, "_predict_prob", mock_predict_prob)

    chunk_stream = wav_to_chunk_stream(wav_path, chunk_duration_s=0.032)
    chunks = list(segmenter.segment(chunk_stream))

    interim_chunks = [c for c in chunks if not c.is_final]
    final_chunks = [c for c in chunks if c.is_final]

    assert len(interim_chunks) >= 2
    assert len(final_chunks) == 2
