"""Console caption presenter implementing the CaptionPresenter domain port.

Renders live subtitles in the terminal:
- Interim (partial) captions overwrite the current line in real-time.
- Final captions commit as permanent new lines.
Zero external GUI dependencies.
"""

import shutil
import sys
import time
from typing import Optional

from src.domain.entities import CaptionSegment
from src.domain.ports import CaptionPresenter


class ConsoleCaptionPresenter(CaptionPresenter):
    """Terminal/CLI caption presenter supporting in-place interim updates."""

    def __init__(self, show_timestamps: bool = True, prefix: str = "[EN] ") -> None:
        self.show_timestamps = show_timestamps
        self.prefix = prefix
        self._last_line_len = 0

    def _get_terminal_width(self) -> int:
        """Get current terminal width safely."""
        try:
            return shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:
            return 80

    def present(self, caption: CaptionSegment) -> None:
        """Render interim or final caption segment to console."""
        text = caption.text.strip()
        if not text:
            return

        term_width = self._get_terminal_width()
        timestamp_str = ""
        if self.show_timestamps:
            timestamp_str = f"[{time.strftime('%H:%M:%S')}] "

        if caption.is_final:
            # Clear any remaining characters from previous interim overwrite
            clear_padding = " " * max(0, self._last_line_len - len(text) - len(timestamp_str) - len(self.prefix))
            output_line = f"\r{timestamp_str}{self.prefix}{text}{clear_padding}\n"
            sys.stdout.write(output_line)
            sys.stdout.flush()
            self._last_line_len = 0
        else:
            # Interim/partial update (overwrites active line)
            interim_tag = " [~] "
            interim_line = f"\r{timestamp_str}{self.prefix}{text}{interim_tag}"
            # Pad with spaces to wipe previous longer interim lines
            pad_len = max(0, self._last_line_len - len(interim_line))
            formatted_line = interim_line + (" " * pad_len)

            # Keep inside terminal width
            if len(formatted_line) > term_width:
                formatted_line = formatted_line[: term_width - 1]

            sys.stdout.write(formatted_line)
            sys.stdout.flush()
            self._last_line_len = len(formatted_line)

    def clear(self) -> None:
        """Clear active console line."""
        sys.stdout.write("\r" + " " * self._get_terminal_width() + "\r")
        sys.stdout.flush()
        self._last_line_len = 0
