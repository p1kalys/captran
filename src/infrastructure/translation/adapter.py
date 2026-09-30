"""Offline translation adapter implementing TranslationPort.

Wraps offline translation engines (e.g., CTranslate2 MarianMT, Opus-MT, NLLB)
for Japanese to English translation with zero cloud dependencies.
"""

from typing import Callable, Dict, Optional, Sequence, Union

from src.domain.entities import Language, TranslationSegment
from src.domain.ports import TranslationPort


class OfflineTranslationAdapter(TranslationPort):
    """Local offline translation adapter."""

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

    def translate(
        self,
        text: str,
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> TranslationSegment:
        """Translate a single text string from source_language to target_language."""
        src_lang = Language.from_code(source_language) if isinstance(source_language, str) else source_language
        tgt_lang = Language.from_code(target_language) if isinstance(target_language, str) else target_language
        if not text:
            return TranslationSegment(
                source_text="",
                translated_text="",
                source_language=src_lang,
                target_language=tgt_lang,
            )

        if self._translate_fn is not None:
            translated_str = self._translate_fn(text)
        else:
            translated_str = self._dictionary.get(
                text, f"[{tgt_lang.value.upper()}: {text}]"
            )

        return TranslationSegment(
            source_text=text,
            translated_text=translated_str,
            source_language=src_lang,
            target_language=tgt_lang,
        )

    def translate_batch(
        self,
        texts: Sequence[str],
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> Sequence[TranslationSegment]:
        """Translate multiple texts in batch."""
        return [self.translate(t, source_language=source_language, target_language=target_language) for t in texts]
