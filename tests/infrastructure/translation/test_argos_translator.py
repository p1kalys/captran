"""Unit and integration tests for ArgosTranslateTranslator."""

from unittest.mock import MagicMock
import pytest

from src.infrastructure.translation.argos_translator import ArgosTranslateTranslator


def test_argos_translator_initialization() -> None:
    translator = ArgosTranslateTranslator(from_code="ja", to_code="en")
    assert translator.from_code == "ja"
    assert translator.to_code == "en"


def test_argos_translator_empty_and_whitespace() -> None:
    translator = ArgosTranslateTranslator()
    assert translator.translate("") == ""
    assert translator.translate("   \n\t") == ""


def test_argos_translator_with_mock() -> None:
    translator = ArgosTranslateTranslator()

    mock_model = MagicMock()
    mock_model.translate.side_effect = lambda t: {
        "こんにちは": "Hello",
        "ありがとう": "Thank you",
        "これはテストです": "This is a test",
        "東京は日本の首都です": "Tokyo is the capital of Japan",
    }.get(t, f"Translated {t}")

    translator._translation_models[("ja", "en")] = mock_model

    test_cases = [
        ("こんにちは", ["hello"]),
        ("ありがとう", ["thank"]),
        ("これはテストです", ["test"]),
        ("東京は日本の首都です", ["tokyo", "japan"]),
    ]

    for ja_text, expected_keywords in test_cases:
        en_text = translator.translate(ja_text)
        assert en_text, f"Translation of '{ja_text}' should not be empty"
        lower_en = en_text.lower()
        for kw in expected_keywords:
            assert kw in lower_en, f"Keyword '{kw}' missing in translation '{en_text}' for '{ja_text}'"


@pytest.mark.integration
def test_argos_translator_live_sentences() -> None:
    """Live integration test asserting keywords from real Argos model output."""
    try:
        import argostranslate.translate
        installed = argostranslate.translate.get_installed_languages()
        has_ja = any(l.code == "ja" for l in installed)
        has_en = any(l.code == "en" for l in installed)
        if not (has_ja and has_en):
            pytest.skip("Argos ja->en language package not installed locally")
    except ImportError:
        pytest.skip("argostranslate is not installed")

    translator = ArgosTranslateTranslator(from_code="ja", to_code="en")

    test_cases = [
        ("こんにちは", ["hello", "hi", "good"]),
        ("ありがとうございます", ["thank"]),
        ("これはテストです", ["test"]),
        ("東京は日本の首都です", ["tokyo", "japan", "capital"]),
    ]

    for ja_text, expected_keywords in test_cases:
        en_text = translator.translate(ja_text)
        assert len(en_text.strip()) > 0, f"Expected non-empty translation for '{ja_text}'"
        lower_en = en_text.lower()
        # Ensure at least one expected keyword is found in MT output
        has_keyword = any(kw in lower_en for kw in expected_keywords)
        assert has_keyword, f"None of {expected_keywords} found in translated text '{en_text}'"
