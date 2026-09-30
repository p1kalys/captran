"""Tests for latency tracking instrumentation and percentiles calculations."""

import time
from typing import List, Sequence

import pytest

from src.application.latency_tracker import (
    LatencyTracker,
    calculate_percentile,
)
from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    Language,
    TranscriptSegment,
)
from src.domain.ports import (
    AudioSource,
    CaptionPresenter,
    SpeechSegmenter,
    Transcriber,
    Translator,
)


def test_calculate_percentile_empty_and_single() -> None:
    assert calculate_percentile([], 50.0) == 0.0
    assert calculate_percentile([100.0], 50.0) == 100.0
    assert calculate_percentile([100.0], 95.0) == 100.0


def test_calculate_percentile_multiple_values() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p50 = calculate_percentile(data, 50.0)
    p95 = calculate_percentile(data, 95.0)

    assert p50 == pytest.approx(55.0, 0.1)
    assert p95 == pytest.approx(95.5, 0.1)


def test_latency_tracker_recording_and_stats() -> None:
    printed_lines = []
    tracker = LatencyTracker(enabled=True, print_callback=printed_lines.append)

    tracker.record(
        audio_to_vad_ms=200.0,
        transcription_ms=150.0,
        translation_ms=10.0,
        display_ms=5.0,
        total_pipeline_ms=165.0,
        e2e_latency_ms=365.0,
        text_preview="Hello world",
    )
    tracker.record(
        audio_to_vad_ms=300.0,
        transcription_ms=250.0,
        translation_ms=20.0,
        display_ms=15.0,
        total_pipeline_ms=285.0,
        e2e_latency_ms=585.0,
        text_preview="Second utterance",
    )

    stats = tracker.get_stage_stats()

    assert stats["Audio -> VAD"].count == 2
    assert stats["Audio -> VAD"].min == 200.0
    assert stats["Audio -> VAD"].max == 300.0
    assert stats["Audio -> VAD"].p50 == pytest.approx(250.0, 0.1)

    assert stats["STT (Faster-Whisper)"].p50 == pytest.approx(200.0, 0.1)
    assert stats["Translation"].p50 == pytest.approx(15.0, 0.1)
    assert stats["Caption Display"].p50 == pytest.approx(10.0, 0.1)

    table = tracker.format_summary_table()
    assert "CAPTRAN LATENCY SUMMARY" in table
    assert "Audio -> VAD" in table
    assert "STT (Faster-Whisper)" in table
    assert "Caption Display" in table

    # Ensure printed log callbacks occurred
    assert len(printed_lines) == 2
    assert "[LATENCY] Utterance #1" in printed_lines[0]
    assert "[LATENCY] Utterance #2" in printed_lines[1]


def test_latency_tracker_bounded_deque_and_disabled() -> None:
    # 1. Test bounded maxlen
    tracker = LatencyTracker(enabled=True, max_records=3)
    for i in range(10):
        tracker.record(10.0, 20.0, 5.0, 1.0, 26.0, 36.0, f"utterance {i}")
    assert len(tracker._records) == 3

    # 2. Test disabled recording (does not append to records)
    disabled_tracker = LatencyTracker(enabled=False)
    disabled_tracker.record(10.0, 20.0, 5.0, 1.0, 26.0, 36.0, "test")
    assert len(disabled_tracker._records) == 0


class DummyTranscriber(Transcriber):
    def transcribe(
        self,
        audio: AudioChunk,
        source_language: Language = Language.JAPANESE,
    ) -> Sequence[TranscriptSegment]:
        time.sleep(0.01)  # small simulated latency
        return [
            TranscriptSegment(
                text="こんにちは世界",
                is_final=True,
                language=source_language,
                start_time=0.0,
                end_time=1.0,
            )
        ]


class DummyTranslator(Translator):
    def translate(
        self,
        text: str,
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> str:
        time.sleep(0.005)
        return "Hello world"


class DummyPresenter(CaptionPresenter):
    def __init__(self) -> None:
        self.captions: List[CaptionSegment] = []

    def present(self, caption: CaptionSegment) -> None:
        time.sleep(0.002)
        self.captions.append(caption)


def test_live_caption_use_case_timing_instrumentation() -> None:
    tracker_prints = []
    tracker = LatencyTracker(enabled=True, print_callback=tracker_prints.append)

    use_case = LiveCaptionUseCase(
        transcriber=DummyTranscriber(),
        translator=DummyTranslator(),
        presenter=DummyPresenter(),
        latency_tracker=tracker,
    )

    utterance = AudioChunk(pcm_data=b"\x00\x00" * 8000, sample_rate=16000, timestamp=time.time() - 0.25)
    captions = use_case.process_utterance(utterance)

    assert len(captions) == 1
    assert captions[0].text == "Hello world"

    stats = tracker.get_stage_stats()
    assert stats["STT (Faster-Whisper)"].count == 1
    assert stats["STT (Faster-Whisper)"].min >= 9.0  # ~10ms
    assert stats["Translation"].min >= 4.0          # ~5ms
    assert stats["Caption Display"].min >= 1.0       # ~2ms
    assert stats["Audio -> VAD"].min >= 200.0        # ~250ms elapsed since timestamp

    assert len(tracker_prints) == 1
    assert "Audio->VAD=" in tracker_prints[0]
    assert "STT=" in tracker_prints[0]


def test_latency_tracker_reporter_thread_lifecycle() -> None:
    reports = []
    tracker = LatencyTracker(
        enabled=True,
        report_interval_s=0.05,
        print_callback=reports.append,
    )

    tracker.record(10.0, 20.0, 5.0, 1.0, 26.0, 36.0, "test")
    tracker.start_reporter()
    time.sleep(0.12)
    tracker.stop_reporter(print_final_summary=False)

    # At least one periodic summary table was emitted
    table_reports = [r for r in reports if "CAPTRAN LATENCY SUMMARY" in r]
    assert len(table_reports) >= 1
