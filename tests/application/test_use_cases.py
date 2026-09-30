"""Unit tests for LiveCaptionUseCase application orchestrator."""

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


class MockAudioSource(AudioSource):
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def stream(self):
        for c in self.chunks:
            yield c


class MockSpeechSegmenter(SpeechSegmenter):
    def segment(self, chunk_stream):
        for chunk in chunk_stream:
            yield chunk


class MockTranscriber(Transcriber):
    def transcribe(self, audio, source_language=Language.JAPANESE):
        return [
            TranscriptSegment(
                text="こんにちは",
                is_final=True,
                start_time=0.0,
                end_time=1.0,
                language=source_language,
            )
        ]


class MockTranslator(Translator):
    def translate(self, text, source_language=Language.JAPANESE, target_language=Language.ENGLISH):
        return "Hello"


class MockCaptionPresenter(CaptionPresenter):
    def __init__(self):
        self.captions = []

    def present(self, caption):
        self.captions.append(caption)


def test_basic_orchestration_flow():
    chunk = AudioChunk(pcm_data=b"\x00\x00" * 16000, sample_rate=16000)
    source = MockAudioSource([chunk])
    segmenter = MockSpeechSegmenter()
    transcriber = MockTranscriber()
    translator = MockTranslator()
    presenter = MockCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
    )

    use_case.start(run_in_background=False)

    assert source.started is True
    assert len(presenter.captions) == 1
    assert presenter.captions[0].text == "Hello"
    assert presenter.captions[0].is_final is True

    use_case.stop()
    assert source.stopped is True
