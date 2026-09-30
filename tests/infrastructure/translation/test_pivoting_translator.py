"""Unit tests for PivotingTranslator."""

from unittest.mock import MagicMock, patch
import pytest

from src.domain.entities import Language
from src.infrastructure.translation.pivoting_translator import PivotingTranslator


def test_pivoting_translator_empty_and_same_language() -> None:
    translator = PivotingTranslator()
    assert translator.translate("", Language.JAPANESE, Language.ENGLISH) == ""
    assert translator.translate("   \n\t", Language.JAPANESE, Language.ENGLISH) == ""
    assert translator.translate("こんにちは", Language.JAPANESE, Language.JAPANESE) == "こんにちは"
    assert translator.translate("Hello world", "en", "en") == "Hello world"


def test_pivoting_translator_direct_route_with_mock() -> None:
    translator = PivotingTranslator()

    mock_ja_en = MagicMock()
    mock_ja_en.translate.side_effect = lambda t: f"Translated [en]: {t}"

    translator._translation_models[("ja", "en")] = mock_ja_en

    result = translator.translate("こんにちは", Language.JAPANESE, Language.ENGLISH)
    assert result == "Translated [en]: こんにちは"
    mock_ja_en.translate.assert_called_once_with("こんにちは")

    # Second call should use cached model without lookup
    assert ("ja", "en") in translator._translation_models


def test_pivoting_translator_pivot_route_with_mock() -> None:
    translator = PivotingTranslator()

    mock_ja_en = MagicMock()
    mock_ja_en.translate.side_effect = lambda t: "Hello" if t == "こんにちは" else "Unknown"

    mock_en_de = MagicMock()
    mock_en_de.translate.side_effect = lambda t: "Hallo" if t == "Hello" else "Unbekannt"

    translator._translation_models[("ja", "en")] = mock_ja_en
    translator._translation_models[("en", "de")] = mock_en_de

    # ja -> de pivots through en
    result = translator.translate("こんにちは", Language.JAPANESE, Language.GERMAN)
    assert result == "Hallo"

    mock_ja_en.translate.assert_called_once_with("こんにちは")
    mock_en_de.translate.assert_called_once_with("Hello")


def test_pivoting_translator_pivot_multi_pair_caching() -> None:
    translator = PivotingTranslator()

    # ja -> en -> es
    mock_ja_en = MagicMock()
    mock_ja_en.translate.return_value = "Hello"

    mock_en_es = MagicMock()
    mock_en_es.translate.return_value = "Hola"

    translator._translation_models[("ja", "en")] = mock_ja_en
    translator._translation_models[("en", "es")] = mock_en_es

    res1 = translator.translate("こんにちは", Language.JAPANESE, Language.SPANISH)
    assert res1 == "Hola"

    # ja -> en -> fr
    mock_en_fr = MagicMock()
    mock_en_fr.translate.return_value = "Bonjour"
    translator._translation_models[("en", "fr")] = mock_en_fr

    res2 = translator.translate("こんにちは", Language.JAPANESE, Language.FRENCH)
    assert res2 == "Bonjour"

    # Verify all models remain in memory cache
    assert ("ja", "en") in translator._translation_models
    assert ("en", "es") in translator._translation_models
    assert ("en", "fr") in translator._translation_models


def test_pivoting_translator_unsupported_route_raises_error() -> None:
    translator = PivotingTranslator()

    # Clear any models and mock _get_direct_translation_model to return None
    translator._translation_models.clear()

    with patch.object(translator, "_get_direct_translation_model", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            # ko -> de with no models installed
            translator.translate("안녕하세요", Language.KOREAN, Language.GERMAN)

        err_msg = str(exc_info.value)
        assert "No direct or pivot translation route available for 'ko->de'" in err_msg
        assert "Required packages" in err_msg


def test_pivoting_translator_invalid_language_codes() -> None:
    translator = PivotingTranslator()

    with pytest.raises(ValueError):
        translator.translate("test", "xx", "en")

    with pytest.raises(ValueError):
        translator.translate("test", "ja", "invalid_lang")
