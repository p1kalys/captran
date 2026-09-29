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
