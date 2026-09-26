"""Unit tests for LiveCaptionUseCase using fake domain port implementations.

Verifies orchestration logic in complete isolation with zero real audio, models, or UI.
"""

import time
from typing import Iterator, List, Sequence

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    TranscriptSegment,
)
from src.domain.ports import (
    AudioSource,
    CaptionPresenter,
    SpeechSegmenter,
    Transcriber,
    Translator,
)


class FakeAudioSource(AudioSource):
    def __init__(self, chunks: List[AudioChunk]) -> None:
        self._chunks = list(chunks)
        self.is_started = False
        self.is_stopped = False

    def start(self) -> None:
        self.is_started = True

    def stop(self) -> None:
        self.is_stopped = True

    def stream(self) -> Iterator[AudioChunk]:
        for c in self._chunks:
            yield c


class FakeSpeechSegmenter(SpeechSegmenter):
    def __init__(self, utterance_chunks: List[AudioChunk]) -> None:
        self._utterances = list(utterance_chunks)

    def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
        # Consume incoming stream
        _ = list(chunk_stream)
        # Yield predefined utterances
        for u in self._utterances:
            yield u


class FakeTranscriber(Transcriber):
    def __init__(self, responses: List[Sequence[TranscriptSegment]]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def transcribe(self, audio: AudioChunk) -> Sequence[TranscriptSegment]:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return []


class FakeTranslator(Translator):
    def __init__(self) -> None:
        self.translated_queries: List[str] = []
        self._dictionary = {
            "こんにちは": "Hello",
            "こんにちは、世界": "Hello, world",
            "テスト": "Test",
        }

    def translate(self, japanese_text: str) -> str:
        self.translated_queries.append(japanese_text)
        return self._dictionary.get(japanese_text, f"[EN: {japanese_text}]")


class FakeCaptionPresenter(CaptionPresenter):
    def __init__(self) -> None:
        self.received_captions: List[CaptionSegment] = []

    def present(self, caption: CaptionSegment) -> None:
        self.received_captions.append(caption)


def test_pipeline_interim_and_final_captions() -> None:
    # Prepare dummy audio chunks
    chunk1 = AudioChunk(pcm_data=b"chunk1", sample_rate=16000, timestamp=1.0)
    utterance1 = AudioChunk(pcm_data=b"utterance1", sample_rate=16000, timestamp=1.0)

    # Transcriber returns an interim (partial) segment first, then a final segment
    stt_responses = [
        [
            TranscriptSegment(
                text="こんにちは",
                is_final=False,
                start_time=0.0,
                end_time=0.8,
            ),
            TranscriptSegment(
                text="こんにちは、世界",
                is_final=True,
                start_time=0.0,
                end_time=1.5,
            ),
        ]
    ]

    audio_source = FakeAudioSource([chunk1])
    segmenter = FakeSpeechSegmenter([utterance1])
    transcriber = FakeTranscriber(stt_responses)
    translator = FakeTranslator()
    presenter = FakeCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    # Execute synchronous processing
    use_case.start(run_in_background=False)

    assert audio_source.is_started is True
    assert transcriber.call_count == 1
    assert translator.translated_queries == ["こんにちは", "こんにちは、世界"]

    # Verify presenter received interim then final caption
    assert len(presenter.received_captions) == 2

    # Interim caption check
    interim_cap = presenter.received_captions[0]
    assert interim_cap.text == "Hello"
    assert interim_cap.is_final is False
    assert interim_cap.start_time == 0.0
    assert interim_cap.end_time == 0.8

    # Final caption check
    final_cap = presenter.received_captions[1]
    assert final_cap.text == "Hello, world"
    assert final_cap.is_final is True
    assert final_cap.start_time == 0.0
    assert final_cap.end_time == 1.5


def test_start_and_stop_lifecycle_clean_shutdown() -> None:
    audio_source = FakeAudioSource([])
    segmenter = FakeSpeechSegmenter([])
    transcriber = FakeTranscriber([])
    translator = FakeTranslator()
    presenter = FakeCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    assert use_case.is_running is False
    use_case.start(run_in_background=True)
    assert use_case.is_running is True
    assert audio_source.is_started is True

    use_case.stop(timeout=1.0)
    assert use_case.is_running is False
    assert audio_source.is_stopped is True


def test_whitespace_and_empty_transcriptions_skipped() -> None:
    utterance = AudioChunk(pcm_data=b"test", sample_rate=16000)
    stt_responses = [
        [
            TranscriptSegment(text="   \n", is_final=False),
            TranscriptSegment(text="", is_final=True),
        ]
    ]

    audio_source = FakeAudioSource([])
    segmenter = FakeSpeechSegmenter([utterance])
    transcriber = FakeTranscriber(stt_responses)
    translator = FakeTranslator()
    presenter = FakeCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    captions = use_case.process_utterance(utterance)
    assert len(captions) == 0
    assert len(translator.translated_queries) == 0
    assert len(presenter.received_captions) == 0


def test_direct_whisper_translation_bypasses_secondary_translator() -> None:
    utterance = AudioChunk(pcm_data=b"speech", sample_rate=16000)
    stt_responses = [
        [
            TranscriptSegment(
                text="Hello, thank you for joining.",
                is_final=True,
                language="en",
                start_time=0.0,
                end_time=2.0,
            ),
        ]
    ]

    audio_source = FakeAudioSource([])
    segmenter = FakeSpeechSegmenter([utterance])
    transcriber = FakeTranscriber(stt_responses)
    translator = FakeTranslator()
    presenter = FakeCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    captions = use_case.process_utterance(utterance)
    assert len(captions) == 1
    assert captions[0].text == "Hello, thank you for joining."
    assert captions[0].language == "en"
    # Translator was bypassed because segment was already in English
    assert len(translator.translated_queries) == 0
    assert len(presenter.received_captions) == 1
    assert presenter.received_captions[0].text == "Hello, thank you for joining."
