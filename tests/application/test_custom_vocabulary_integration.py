"""Integration tests for LiveCaptionUseCase with CustomVocabulary."""

from typing import Sequence
import pytest

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    CustomVocabulary,
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


class FakeSTT(Transcriber):
    def __init__(self):
        self.initial_prompt = ""

    def transcribe(
        self,
        audio: AudioChunk,
        source_language: Language = Language.JAPANESE,
    ) -> Sequence[TranscriptSegment]:
        return [
            TranscriptSegment(
                text="キャプトランの最新機能です。",
                is_final=True,
                language=source_language,
            )
        ]


class FakeTranslator(Translator):
    def translate(
        self,
        text: str,
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> str:
        # If source substitution occurred, CapTran is in text
        if "CapTran" in text:
            return "This is the latest feature of CapTran."
        return "This is the latest feature of cap tran."


class FakePresenter(CaptionPresenter):
    def __init__(self):
        self.captions = []

    def present(self, caption: CaptionSegment) -> None:
        self.captions.append(caption)


def test_live_caption_use_case_with_custom_vocabulary() -> None:
    vocab = CustomVocabulary(
        prompt_terms=["CapTran", "Antigravity"],
        source_substitutions={"キャプトラン": "CapTran"},
        target_substitutions={"cap tran": "CapTran"},
    )

    stt = FakeSTT()
    translator = FakeTranslator()
    presenter = FakePresenter()

    use_case = LiveCaptionUseCase(
        transcriber=stt,
        translator=translator,
        presenter=presenter,
        vocabulary=vocab,
    )

    # 1. Verify prompt hint was passed to transcriber
    assert stt.initial_prompt == "CapTran, Antigravity"

    # 2. Process utterance
    chunk = AudioChunk(pcm_data=b"test_pcm", sample_rate=16000, is_final=True)
    captions = use_case.process_utterance(chunk)

    assert len(captions) == 1
    caption = captions[0]
    # Source text received pre-substitution
    assert "CapTran" in caption.original_text
    assert "キャプトラン" not in caption.original_text

    # Target text received translation and post-substitution
    assert "CapTran" in caption.text
    assert caption.text == "This is the latest feature of CapTran."
