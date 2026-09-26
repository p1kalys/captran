"""Offline UI adapter implementing CaptionDisplayPort.

Handles presentation of live captions to terminal or GUI overlays.
"""

from typing import Callable, List, Optional

from src.domain.entities import CaptionCue
from src.domain.ports import CaptionDisplayPort


class TerminalCaptionDisplayAdapter(CaptionDisplayPort):
    """Terminal/Callback based subtitle presenter with zero external GUI dependencies."""

    def __init__(
        self,
        on_cue_callback: Optional[Callable[[CaptionCue], None]] = None,
        verbose: bool = False,
    ) -> None:
        self._on_cue = on_cue_callback
        self._verbose = verbose
        self._history: List[CaptionCue] = []
        self._is_closed = False

    @property
    def history(self) -> List[CaptionCue]:
        """Return history of displayed cues."""
        return list(self._history)

    def present(self, caption: CaptionCue) -> None:
        """Present live caption."""
        self.display_cue(caption)

    def display_cue(self, cue: CaptionCue) -> None:
        """Render the subtitle cue."""
        if self._is_closed:
            return

        self._history.append(cue)
        if self._on_cue is not None:
            self._on_cue(cue)

        if self._verbose:
            print(f"[JA] {cue.original_text} -> [EN] {cue.translated_text}")

    def clear(self) -> None:
        """Clear the history / active screen."""
        self._history.clear()

    def close(self) -> None:
        """Close presenter."""
        self._is_closed = True
