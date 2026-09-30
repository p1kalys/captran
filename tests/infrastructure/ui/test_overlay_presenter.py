"""Unit tests for OverlayCaptionPresenter."""

import sys
from unittest.mock import MagicMock, patch
import pytest

from src.domain.entities import CaptionSegment
from src.infrastructure.ui.overlay_presenter import (
    CaptionOverlayWidget,
    OverlayCaptionPresenter,
)


def test_overlay_presenter_mocked_qt() -> None:
    # Test that presenting interim and final captions formats lines without error
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    widget = CaptionOverlayWidget(max_history_lines=3)

    # Interim caption
    interim = CaptionSegment(text="Hello interim", is_final=False)
    widget.update_caption(interim)
    assert widget._current_interim_text == "Hello interim"
    assert len(widget._final_lines) == 0

    # Final caption
    final = CaptionSegment(text="Hello final sentence", is_final=True)
    widget.update_caption(final)
    assert widget._current_interim_text == ""
    assert len(widget._final_lines) == 1
    assert widget._final_lines[0] == "Hello final sentence"

    # Add multiple final lines to verify max_history_lines rolling buffer
    widget.update_caption(CaptionSegment(text="Line 2", is_final=True))
    widget.update_caption(CaptionSegment(text="Line 3", is_final=True))
    widget.update_caption(CaptionSegment(text="Line 4", is_final=True))

    assert len(widget._final_lines) == 3
    assert list(widget._final_lines) == ["Line 2", "Line 3", "Line 4"]

    widget.clear_captions()
    assert len(widget._final_lines) == 0


def test_overlay_presenter_dual_subtitles_stacked_rendering() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    from src.domain.entities import Settings

    widget = CaptionOverlayWidget(max_history_lines=3, dual_subtitles=True)

    # Interim dual caption
    interim = CaptionSegment(
        text="Hello interim",
        original_text="こんにちは仮",
        is_final=False,
    )
    widget.update_caption(interim)
    assert "こんにちは仮" in widget.text_label.text()
    assert "Hello interim" in widget.text_label.text()

    # Final dual caption
    final = CaptionSegment(
        text="Hello final sentence",
        original_text="こんにちは世界",
        is_final=True,
    )
    widget.update_caption(final)
    assert "こんにちは世界" in widget.text_label.text()
    assert "Hello final sentence" in widget.text_label.text()

    # Toggle dual subtitles off via Settings
    widget.apply_settings(Settings(dual_subtitles=False))
    assert widget.dual_subtitles is False
    assert "Hello final sentence" in widget.text_label.text()
    assert "こんにちは世界" not in widget.text_label.text()


def test_overlay_presenter_html_escaping() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    widget = CaptionOverlayWidget(max_history_lines=3, dual_subtitles=True)

    # Caption with raw HTML characters
    cap = CaptionSegment(
        text="<script>alert('xss')</script> & 'quotes'",
        original_text="<タグ> & '引用'",
        is_final=True,
    )
    widget.update_caption(cap)
    rendered = widget.text_label.text()
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;タグ&gt;" in rendered


def test_overlay_presenter_multilingual_badges_and_no_detected_language() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    from src.domain.entities import Language

    widget = CaptionOverlayWidget(max_history_lines=3, dual_subtitles=True)

    # Verify header title and initial text do not contain "detected language" or static JA constraint
    assert "detected language" not in widget.title_label.text().lower()
    assert "Listening for audio..." in widget.text_label.text()

    # Spanish -> French caption
    cap = CaptionSegment(
        text="Bonjour le monde",
        original_text="Hola mundo",
        language=Language.FRENCH,
        source_language=Language.SPANISH,
        is_final=True,
    )
    widget.update_caption(cap)
    rendered = widget.text_label.text()

    assert "ES" in rendered
    assert "FR" in rendered
    assert "Hola mundo" in rendered
    assert "Bonjour le monde" in rendered
    assert "detected language" not in rendered.lower()


def test_font_stacks_coverage_for_all_7_languages() -> None:
    from src.domain.entities import Language
    from src.infrastructure.ui.overlay_presenter import (
        get_font_stack_for_language,
        LANGUAGE_FONT_FAMILIES,
    )

    for lang in Language:
        stack = get_font_stack_for_language(lang)
        assert stack is not None
        assert "sans-serif" in stack

    # Script-specific coverage
    hi_stack = get_font_stack_for_language(Language.HINDI)
    assert any(f in hi_stack for f in ["Nirmala UI", "Mangal", "Noto Sans Devanagari"])

    ko_stack = get_font_stack_for_language(Language.KOREAN)
    assert any(f in ko_stack for f in ["Malgun Gothic", "Noto Sans CJK KR", "Batang"])

    ja_stack = get_font_stack_for_language(Language.JAPANESE)
    assert any(f in ja_stack for f in ["Meiryo", "Yu Gothic", "MS PGothic"])


def test_overlay_hindi_and_korean_rendering() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["test"])
    except ImportError:
        pytest.skip("PySide6 not installed")

    from src.domain.entities import Language

    widget = CaptionOverlayWidget(max_history_lines=3, dual_subtitles=True)

    # Hindi (Devanagari) -> Korean (Hangul)
    cap = CaptionSegment(
        text="안녕하세요 세계",
        original_text="नमस्ते दुनिया",
        language=Language.KOREAN,
        source_language=Language.HINDI,
        is_final=True,
    )
    widget.update_caption(cap)
    rendered = widget.text_label.text()

    assert "HI" in rendered
    assert "KO" in rendered
    assert "नमस्ते दुनिया" in rendered
    assert "안녕하세요 세계" in rendered
    assert "Nirmala UI" in rendered or "Mangal" in rendered or "Devanagari" in rendered
    assert "Malgun Gothic" in rendered or "KR" in rendered or "Batang" in rendered


