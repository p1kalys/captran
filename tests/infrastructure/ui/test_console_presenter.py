"""Unit tests for ConsoleCaptionPresenter."""

import io
import sys
from src.domain.entities import CaptionSegment
from src.infrastructure.ui.console_presenter import ConsoleCaptionPresenter


def test_console_presenter_interim_and_final_output() -> None:
    captured_stdout = io.StringIO()
    presenter = ConsoleCaptionPresenter(show_timestamps=False, prefix="[EN] ")

    # Redirect stdout to StringIO
    old_stdout = sys.stdout
    sys.stdout = captured_stdout

    try:
        # Interim caption
        interim_cue = CaptionSegment(text="Hello", is_final=False)
        presenter.present(interim_cue)
        out1 = captured_stdout.getvalue()
        assert "[EN] Hello" in out1
        assert "[~]" in out1

        # Final caption
        final_cue = CaptionSegment(text="Hello world", is_final=True)
        presenter.present(final_cue)
        out2 = captured_stdout.getvalue()
        assert "[EN] Hello world" in out2
        assert "\n" in out2
    finally:
        sys.stdout = old_stdout


def test_console_presenter_dual_subtitles() -> None:
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout

    try:
        # 1. Dual subtitles enabled
        presenter_dual = ConsoleCaptionPresenter(show_timestamps=False, dual_subtitles=True)
        cap = CaptionSegment(text="Hello world", original_text="こんにちは世界", is_final=True)
        presenter_dual.present(cap)
        assert "[JA] こんにちは世界  ➔  [EN] Hello world" in captured_stdout.getvalue()

        # 2. Dual subtitles disabled
        captured_stdout.truncate(0)
        captured_stdout.seek(0)
        presenter_single = ConsoleCaptionPresenter(show_timestamps=False, dual_subtitles=False)
        presenter_single.present(cap)
        assert "[EN] Hello world" in captured_stdout.getvalue()
        assert "[JA]" not in captured_stdout.getvalue()
    finally:
        sys.stdout = old_stdout
