"""Unit tests for offline UI adapter."""

from src.domain.entities import CaptionCue
from src.infrastructure.ui.adapter import TerminalCaptionDisplayAdapter


def test_terminal_display_adapter() -> None:
    captured = []
    adapter = TerminalCaptionDisplayAdapter(on_cue_callback=captured.append)

    cue = CaptionCue(original_text="テスト", translated_text="Test")
    adapter.display_cue(cue)

    assert len(captured) == 1
    assert captured[0].original_text == "テスト"
    assert len(adapter.history) == 1

    adapter.clear()
    assert len(adapter.history) == 0

    adapter.close()
    adapter.display_cue(cue)
    assert len(adapter.history) == 0
