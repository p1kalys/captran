"""Unit tests for offline translation adapter."""

from src.infrastructure.translation.adapter import OfflineTranslationAdapter


def test_offline_translation_known_text() -> None:
    adapter = OfflineTranslationAdapter()
    result = adapter.translate("こんにちは、世界！")
    assert result.source_text == "こんにちは、世界！"
    assert result.translated_text == "Hello, world!"
    assert result.source_language == "ja"
    assert result.target_language == "en"


def test_offline_translation_batch() -> None:
    adapter = OfflineTranslationAdapter()
    results = adapter.translate_batch(["こんにちは", "ありがとう"])
    assert len(results) == 2
    assert results[0].translated_text == "Hello"
    assert results[1].translated_text == "Thank you"


def test_offline_translation_empty() -> None:
    adapter = OfflineTranslationAdapter()
    result = adapter.translate("")
    assert result.translated_text == ""
