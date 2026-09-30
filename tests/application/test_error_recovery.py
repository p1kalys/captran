"""Integration tests for error recovery, reconnection, graceful degradation, and thread-safe shutdown."""

import time
from typing import Iterator, List, Optional, Sequence

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    Language,
    PipelineStatus,
    TranscriptSegment,
)
from src.domain.ports import (
    AudioSource,
    CaptionPresenter,
    SpeechSegmenter,
    Transcriber,
    Translator,
)


class FlakyAudioSource(AudioSource):
    """Simulates an audio device that fails/disconnects on the first stream attempt,

    then successfully recovers on the second attempt after reconnecting.
    """

    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0
        self.stream_attempt = 0

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1

    def stream(self) -> Iterator[AudioChunk]:
        self.stream_attempt += 1
        if self.stream_attempt == 1:
            # Yield one chunk, then simulate unexpected device disconnect error
            yield AudioChunk(pcm_data=b"chunk1", sample_rate=16000)
            raise ConnectionResetError("WASAPI audio device was disconnected.")

        # Reconnected stream: yield recovered chunk
        yield AudioChunk(pcm_data=b"chunk_recovered", sample_rate=16000)


class PassThroughSegmenter(SpeechSegmenter):
    def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
        for chunk in chunk_stream:
            yield chunk


class FailingTranscriber(Transcriber):
    """Simulates an STT engine that crashes on the first utterance, but recovers on the second."""

    def __init__(self) -> None:
        self.call_count = 0

    def transcribe(
        self,
        audio: AudioChunk,
        source_language: Language = Language.JAPANESE,
    ) -> Sequence[TranscriptSegment]:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("CUDA Out of Memory or CTranslate2 execution failed.")
        return [
            TranscriptSegment(
                text="こんにちは",
                is_final=True,
                start_time=0.0,
                end_time=1.0,
                language=source_language,
            )
        ]


class FailingTranslator(Translator):
    """Simulates a translation engine that fails."""

    def translate(
        self,
        text: str,
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> str:
        raise ValueError("Argos model corrupted or pivot route unavailable.")


class RecordingPresenter(CaptionPresenter):
    def __init__(self) -> None:
        self.presented: List[CaptionSegment] = []

    def present(self, caption: CaptionSegment) -> None:
        self.presented.append(caption)


def test_audio_device_disconnect_and_automatic_reconnect() -> None:
    audio_source = FlakyAudioSource()
    segmenter = PassThroughSegmenter()
    transcriber = FailingTranscriber()  # call 1 fails, call 2 succeeds
    trans_mock = type(
        "MockTrans",
        (Translator,),
        {"translate": lambda s, t, src=Language.JAPANESE, tgt=Language.ENGLISH: "Hello"},
    )()
    presenter = RecordingPresenter()

    statuses: List[PipelineStatus] = []

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=trans_mock,
        presenter=presenter,
        status_callback=statuses.append,
        reconnect_interval_s=0.1,  # fast test backoff
    )

    use_case.start(run_in_background=True)

    # Wait for the reconnect loop to trigger and process chunk_recovered
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if audio_source.stream_attempt >= 2 and len(presenter.presented) >= 1:
            break
        time.sleep(0.05)

    use_case.stop(timeout=1.0)

    # Assert reconnect occurred
    assert audio_source.start_count >= 2
    assert audio_source.stream_attempt >= 2
    assert any(s.state == "reconnecting" for s in statuses)
    assert len(presenter.presented) == 1
    assert presenter.presented[0].text == "Hello"


def test_stt_failure_graceful_degradation() -> None:
    transcriber = FailingTranscriber()
    translator = type(
        "MockTrans",
        (Translator,),
        {"translate": lambda s, t, src=Language.JAPANESE, tgt=Language.ENGLISH: "Hello"},
    )()
    presenter = RecordingPresenter()
    statuses: List[PipelineStatus] = []

    use_case = LiveCaptionUseCase(
        audio_source=FlakyAudioSource(),
        segmenter=PassThroughSegmenter(),
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        status_callback=statuses.append,
    )

    # Utterance 1 fails STT
    chunk1 = AudioChunk(pcm_data=b"chunk1", sample_rate=16000)
    cues1 = use_case.process_utterance(chunk1)
    assert cues1 == []
    assert any(s.state == "degraded" and "STT transcription failed" in s.message for s in statuses)

    # Utterance 2 succeeds STT
    chunk2 = AudioChunk(pcm_data=b"chunk2", sample_rate=16000)
    cues2 = use_case.process_utterance(chunk2)
    assert len(cues2) == 1
    assert cues2[0].text == "Hello"


def test_translation_failure_graceful_fallback() -> None:
    transcriber = type(
        "MockSTT",
        (Transcriber,),
        {
            "transcribe": lambda s, a, src=Language.JAPANESE: [
                TranscriptSegment(text="ありがとう", is_final=True, language=src)
            ]
        },
    )()
    translator = FailingTranslator()
    presenter = RecordingPresenter()
    statuses: List[PipelineStatus] = []

    use_case = LiveCaptionUseCase(
        audio_source=FlakyAudioSource(),
        segmenter=PassThroughSegmenter(),
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        status_callback=statuses.append,
    )

    chunk = AudioChunk(pcm_data=b"audio", sample_rate=16000)
    cues = use_case.process_utterance(chunk)

    # Fallback to Japanese original text with status warning
    assert len(cues) == 1
    assert cues[0].text == "[JA] ありがとう"
    assert any(s.state == "degraded" and "Translation failed" in s.message for s in statuses)


def test_thread_safe_shutdown_from_any_state() -> None:
    audio_source = FlakyAudioSource()
    segmenter = PassThroughSegmenter()
    transcriber = FailingTranscriber()
    translator = type(
        "MockTrans",
        (Translator,),
        {"translate": lambda s, t, src=Language.JAPANESE, tgt=Language.ENGLISH: "Test"},
    )()
    presenter = RecordingPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        reconnect_interval_s=5.0,  # Long sleep to test interruptibility
    )

    # Start and immediately stop while it is initializing/reconnecting
    use_case.start(run_in_background=True)
    time.sleep(0.05)

    # Calling stop multiple times must be completely safe and instantaneous
    use_case.stop(timeout=1.0)
    use_case.stop(timeout=1.0)
    assert use_case.is_running is False
    assert use_case.current_status.state == "stopped"
