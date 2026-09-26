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
