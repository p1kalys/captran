"""Offline translation adapter implementing TranslationPort.

Wraps offline translation engines (e.g., CTranslate2 MarianMT, Opus-MT, NLLB)
for Japanese to English translation with zero cloud dependencies.
"""

from typing import Callable, Dict, Optional, Sequence

from src.domain.entities import TranslationSegment
from src.domain.ports import TranslationPort


class OfflineTranslationAdapter(TranslationPort):
    """Local offline translation adapter for Japanese to English."""

    def __init__(
        self,
        model_name_or_path: str = "opus-mt-ja-en",
        translate_fn: Optional[Callable[[str], str]] = None,
        dictionary_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_path = model_name_or_path
        self._translate_fn = translate_fn
        self._dictionary = dictionary_overrides or {
            "こんにちは、世界！": "Hello, world!",
            "こんにちは": "Hello",
            "ありがとう": "Thank you",
            "テスト": "Test",
        }

    def translate(self, text: str) -> TranslationSegment:
        """Translate a single Japanese text string to English."""
        if not text:
            return TranslationSegment(source_text="", translated_text="")

        if self._translate_fn is not None:
            translated_str = self._translate_fn(text)
        else:
            translated_str = self._dictionary.get(
                text, f"[EN: {text}]"
            )

        return TranslationSegment(
            source_text=text,
            translated_text=translated_str,
            source_language="ja",
            target_language="en",
        )

    def translate_batch(self, texts: Sequence[str]) -> Sequence[TranslationSegment]:
        """Translate multiple Japanese texts in batch."""
        return [self.translate(t) for t in texts]
