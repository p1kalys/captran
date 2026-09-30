"""PySide6 desktop overlay presenter implementing the CaptionPresenter domain port.

Provides a modern, draggable, translucent, frameless, always-on-top subtitle overlay:
- Commits final captions to solid high-contrast text, keeping last 2-3 lines visible.
- Displays interim captions in dimmed/italicized style in real time.
- Supports configurable font size, opacity, and window positioning from Settings.
- Fully thread-safe via Qt Queued Signals across worker threads.
"""

from collections import deque
import html
from pathlib import Path
import sys
from typing import Deque, Dict, List, Optional, Tuple, Union

from src.domain.entities import CaptionSegment, Language, Settings
from src.domain.ports import CaptionPresenter

try:
    from PySide6.QtCore import QObject, QPoint, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QMouseEvent, QPainter
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QObject = object  # type: ignore
    Signal = lambda *args: None  # type: ignore
    Slot = lambda *args: lambda fn: fn  # type: ignore
    QFontDatabase = None  # type: ignore


LANGUAGE_FONT_FAMILIES: Dict[Language, List[str]] = {
    Language.HINDI: ["Nirmala UI", "Mangal", "Aparajita", "Utsaah", "Noto Sans Devanagari", "Segoe UI"],
    Language.JAPANESE: ["Meiryo", "Yu Gothic", "MS PGothic", "MS Gothic", "Noto Sans CJK JP", "Segoe UI"],
    Language.KOREAN: ["Malgun Gothic", "Batang", "Dotum", "Gulim", "Noto Sans CJK KR", "Segoe UI"],
    Language.ENGLISH: ["Segoe UI", "Arial", "Helvetica Neue", "Noto Sans"],
    Language.SPANISH: ["Segoe UI", "Arial", "Helvetica Neue", "Noto Sans"],
    Language.FRENCH: ["Segoe UI", "Arial", "Helvetica Neue", "Noto Sans"],
    Language.GERMAN: ["Segoe UI", "Arial", "Helvetica Neue", "Noto Sans"],
}

UNIVERSAL_FONT_STACK = (
    '"Segoe UI", "Nirmala UI", "Malgun Gothic", "Yu Gothic", "Meiryo", '
    '"Noto Sans Devanagari", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans", Arial, sans-serif'
)


def load_bundled_fonts() -> List[str]:
    """Scan and register any bundled TTF/OTF fonts in assets/fonts or models/fonts directories."""
    loaded_families: List[str] = []
    if QFontDatabase is None:
        return loaded_families

    search_dirs = [
        Path(__file__).parent / "fonts",
        Path(__file__).parent.parent.parent.parent / "assets" / "fonts",
        Path(__file__).parent.parent.parent.parent / "models" / "fonts",
    ]
    for d in search_dirs:
        if d.is_dir():
            for font_file in d.glob("*.[to]tf"):
                try:
                    font_id = QFontDatabase.addApplicationFont(str(font_file.resolve()))
                    if font_id >= 0:
                        families = QFontDatabase.applicationFontFamilies(font_id)
                        loaded_families.extend(families)
                except Exception:
                    pass
    return loaded_families


def get_font_stack_for_language(lang: Optional[Union[str, Language]]) -> str:
    """Return an optimal CSS font-family string with glyph coverage for the specified language."""
    if lang is None:
        return UNIVERSAL_FONT_STACK

    try:
        lang_enum = Language.from_code(lang) if isinstance(lang, str) else lang
    except Exception:
        lang_enum = None

    preferred = LANGUAGE_FONT_FAMILIES.get(lang_enum, ["Segoe UI", "Arial", "sans-serif"])
    formatted = [f'"{f}"' if " " in f else f for f in preferred]
    if "sans-serif" not in formatted:
        formatted.append("sans-serif")
    return ", ".join(formatted)


class _CaptionBridge(QObject):
    """Thread-safe Qt signal bridge between application worker threads and GUI thread."""

    caption_received = Signal(object)


class CaptionOverlayWidget(QWidget):
    """Draggable, translucent, frameless, always-on-top subtitle overlay widget."""

    position_changed = Signal(int, int)

    def __init__(
        self,
        max_history_lines: int = 3,
        font_size: int = 16,
        opacity: float = 0.85,
        dual_subtitles: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.max_history_lines = max_history_lines
        self.font_size = font_size
        self.opacity = opacity
        self.dual_subtitles = dual_subtitles
        self._final_captions: Deque[CaptionSegment] = deque(maxlen=max_history_lines)
        self._final_lines: Deque[str] = deque(maxlen=max_history_lines)
        self._current_interim_caption: Optional[CaptionSegment] = None
        self._current_interim_text = ""
        self._drag_pos = QPoint()

        self._setup_window_attributes()
        self._init_ui()
        self._position_top_center()

    def _setup_window_attributes(self) -> None:
        """Configure frameless, translucent, always-on-top window flags."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMinimumSize(700, 110)
        self.resize(960, 140)

    def _init_ui(self) -> None:
        """Create dark glassmorphic styling and labels."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(4)

        # Content container with translucent background and border
        self.container = QWidget(self)
        self.container.setObjectName("CaptionContainer")
        self._update_container_style()

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(18, 10, 18, 10)
        container_layout.setSpacing(4)

        # Header / Drag Indicator Bar
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 2)

        self.title_label = QLabel("CAPTRAN | LIVE MULTILINGUAL SUBTITLES", self.container)
        self.title_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.45);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
        """)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Close button
        self.close_btn = QPushButton("✕", self.container)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet("""
            QPushButton {
                color: rgba(255, 255, 255, 0.4);
                background: transparent;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FF5252;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        header_layout.addWidget(self.close_btn)

        container_layout.addLayout(header_layout)

        load_bundled_fonts()

        # Subtitle Text Display Label
        self.text_label = QLabel("Listening for audio...", self.container)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setStyleSheet(f"color: #FFFFFF; font-family: {UNIVERSAL_FONT_STACK};")
        container_layout.addWidget(self.text_label)

        main_layout.addWidget(self.container)

    def _update_container_style(self) -> None:
        """Update container stylesheet based on current opacity."""
        alpha = int(max(0.1, min(1.0, self.opacity)) * 255)
        self.container.setStyleSheet(f"""
            QWidget#CaptionContainer {{
                background-color: rgba(15, 18, 26, {alpha});
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
            }}
        """)

    def _position_top_center(self) -> None:
        """Position window near the top center of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - self.width()) // 2
            y = geom.y() + 45  # 45px padding from top screen edge
            self.move(x, y)

    def apply_settings(self, settings: Settings) -> None:
        """Apply user configuration values to overlay."""
        self.font_size = settings.overlay_font_size
        self.opacity = settings.overlay_opacity
        self.max_history_lines = settings.max_history_lines
        self.dual_subtitles = getattr(settings, "dual_subtitles", True)
        self._final_captions = deque(self._final_captions, maxlen=self.max_history_lines)
        self._final_lines = deque(self._final_lines, maxlen=self.max_history_lines)
        self._update_container_style()
        self._render_text()

        if settings.overlay_position_x is not None and settings.overlay_position_y is not None:
            self.move(settings.overlay_position_x, settings.overlay_position_y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Enable dragging from anywhere on the overlay."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update window position while dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and not self._drag_pos.isNull():
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(new_pos)
            self.position_changed.emit(new_pos.x(), new_pos.y())
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Release drag lock."""
        self._drag_pos = QPoint()
        event.accept()

    @Slot(object)
    def update_caption(self, caption: CaptionSegment) -> None:
        """Update display with incoming caption segment."""
        if isinstance(caption, str):
            caption = CaptionSegment(text=caption, is_final=True)

        text = caption.text.strip()
        if not text:
            return

        if caption.is_final:
            self._final_captions.append(caption)
            self._final_lines.append(text)
            self._current_interim_caption = None
            self._current_interim_text = ""
        else:
            self._current_interim_caption = caption
            self._current_interim_text = text

        self._render_text()

    def _render_text(self) -> None:
        """Build HTML string with side-by-side (2-column) source and target subtitles."""
        sz = self.font_size
        badge_sz = max(10, sz - 4)
        dot_sz = max(10, sz - 4)
        rows_html = []

        def _fmt_badge(val, default=""):
            if val is None:
                return default
            if hasattr(val, "value"):
                return str(val.value).upper()
            return str(val).upper()

        # Committed final rows
        for cap in self._final_captions:
            tgt_text = html.escape(cap.text)
            src_text = html.escape(cap.original_text.strip()) if cap.original_text else ""
            src_badge = _fmt_badge(getattr(cap, "source_language", None), default="SRC")
            tgt_badge = _fmt_badge(getattr(cap, "language", None), default="TGT")
            src_font = get_font_stack_for_language(getattr(cap, "source_language", None))
            tgt_font = get_font_stack_for_language(getattr(cap, "language", None))

            if self.dual_subtitles and src_text:
                # Side-by-side 2-column view
                rows_html.append(
                    f"<tr>"
                    f"<td style='width: 48%; vertical-align: top; padding: 3px 10px 3px 0px; "
                    f"color: #CFD8DC; font-size: {sz}px; font-weight: 500; font-family: {src_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>"
                    f"<span style='color: #80DEEA; font-size: {badge_sz}px; font-weight: 700; background: rgba(0,188,212,0.15); padding: 1px 5px; border-radius: 4px; margin-right: 6px;'>{src_badge}</span>{src_text}"
                    f"</td>"
                    f"<td style='width: 4%; vertical-align: middle; text-align: center; color: rgba(255,255,255,0.3); font-size: 14px;'>➔</td>"
                    f"<td style='width: 48%; vertical-align: top; padding: 3px 0px 3px 10px; "
                    f"color: #FFFFFF; font-size: {sz}px; font-weight: 600; font-family: {tgt_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>"
                    f"<span style='color: #A5D6A7; font-size: {badge_sz}px; font-weight: 700; background: rgba(76,175,80,0.18); padding: 1px 5px; border-radius: 4px; margin-right: 6px;'>{tgt_badge}</span>{tgt_text}"
                    f"</td>"
                    f"</tr>"
                )
            else:
                rows_html.append(
                    f"<tr>"
                    f"<td colspan='3' style='padding: 3px 0px; color: #FFFFFF; font-size: {sz}px; font-weight: 600; font-family: {tgt_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>{tgt_text}</td>"
                    f"</tr>"
                )

        # Interim active line (updating live)
        if self._current_interim_caption is not None:
            interim_cap = self._current_interim_caption
            tgt_text = html.escape(interim_cap.text)
            src_text = html.escape(interim_cap.original_text.strip()) if interim_cap.original_text else ""
            src_badge = _fmt_badge(getattr(interim_cap, "source_language", None), default="SRC")
            tgt_badge = _fmt_badge(getattr(interim_cap, "language", None), default="TGT")
            src_font = get_font_stack_for_language(getattr(interim_cap, "source_language", None))
            tgt_font = get_font_stack_for_language(getattr(interim_cap, "language", None))

            if self.dual_subtitles and src_text:
                rows_html.append(
                    f"<tr>"
                    f"<td style='width: 48%; vertical-align: top; padding: 3px 10px 3px 0px; "
                    f"color: #90CAF9; font-size: {sz}px; font-style: italic; font-family: {src_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>"
                    f"<span style='color: #64B5F6; font-size: {badge_sz}px; font-weight: 700; font-style: normal; background: rgba(33,150,243,0.18); padding: 1px 5px; border-radius: 4px; margin-right: 6px;'>{src_badge}</span>{src_text}"
                    f"</td>"
                    f"<td style='width: 4%; vertical-align: middle; text-align: center; color: #64B5F6; font-size: 14px;'>➔</td>"
                    f"<td style='width: 48%; vertical-align: top; padding: 3px 0px 3px 10px; "
                    f"color: #64B5F6; font-size: {sz}px; font-style: italic; font-weight: 600; font-family: {tgt_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>"
                    f"<span style='color: #81D4FA; font-size: {badge_sz}px; font-weight: 700; font-style: normal; background: rgba(3,169,244,0.18); padding: 1px 5px; border-radius: 4px; margin-right: 6px;'>{tgt_badge}</span>{tgt_text} <span style='color: #42A5F5; font-size: {dot_sz}px;'>●</span>"
                    f"</td>"
                    f"</tr>"
                )
            else:
                rows_html.append(
                    f"<tr>"
                    f"<td colspan='3' style='padding: 3px 0px; color: #90CAF9; font-size: {sz}px; font-style: italic; font-family: {tgt_font}; "
                    f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>{tgt_text} <span style='color: #64B5F6; font-size: {dot_sz}px;'>●</span></td>"
                    f"</tr>"
                )
        elif self._current_interim_text:
            interim_text = html.escape(self._current_interim_text)
            rows_html.append(
                f"<tr>"
                f"<td colspan='3' style='padding: 3px 0px; color: #90CAF9; font-size: {sz}px; font-style: italic; "
                f"text-shadow: 0px 1px 3px rgba(0,0,0,0.9); line-height: 1.35;'>{interim_text} <span style='color: #64B5F6; font-size: {dot_sz}px;'>●</span></td>"
                f"</tr>"
            )

        if not rows_html:
            self.text_label.setText(f"<div style='color: rgba(255,255,255,0.4); font-size: {max(12, sz-2)}px;'>Listening for audio...</div>")
        else:
            self.text_label.setText(f"<table style='width: 100%; border-collapse: separate; border-spacing: 0px 4px;'>{''.join(rows_html)}</table>")

    def update_status_hint(self, status: object) -> None:
        """Update subtitle placeholder text with pipeline connection/loading status."""
        state = getattr(status, "state", "").lower()
        msg = getattr(status, "message", "")
        sz = self.font_size

        if not self._final_lines and not self._current_interim_text:
            if state in ("connecting", "loading", "downloading", "initializing"):
                self.text_label.setText(
                    f"<div style='color: #EBCB8B; font-size: {max(12, sz-2)}px; font-weight: 600;'>"
                    f"⏳ Connecting: {msg or 'Loading Whisper model...'}</div>"
                )
            elif state in ("error", "not connected"):
                self.text_label.setText(
                    f"<div style='color: #BF616A; font-size: {max(12, sz-2)}px; font-weight: 600;'>"
                    f"✕ Not Connected: {msg or 'Model/Device error'}</div>"
                )
            elif state == "running":
                self.text_label.setText(
                    f"<div style='color: rgba(255,255,255,0.4); font-size: {max(12, sz-2)}px;'>"
                    f"● Listening for audio...</div>"
                )

    def clear_captions(self) -> None:
        """Clear visible lines."""
        self._final_captions.clear()
        self._final_lines.clear()
        self._current_interim_caption = None
        self._current_interim_text = ""
        self._render_text()


class OverlayCaptionPresenter(CaptionPresenter):
    """Drop-in CaptionPresenter port implementation driving the PySide6 overlay."""

    def __init__(
        self,
        max_history_lines: int = 3,
        font_size: int = 16,
        opacity: float = 0.85,
        dual_subtitles: bool = True,
        show_window: bool = True,
    ) -> None:
        if "PySide6" not in sys.modules and QObject is object:
            raise ImportError("PySide6 is not installed. Please install PySide6.")

        # Ensure QApplication exists on GUI main thread
        self._app = QApplication.instance()
        self._owns_app = False
        if self._app is None:
            self._app = QApplication(sys.argv if sys.argv else ["captran"])
            self._owns_app = True

        self.widget = CaptionOverlayWidget(
            max_history_lines=max_history_lines,
            font_size=font_size,
            opacity=opacity,
            dual_subtitles=dual_subtitles,
        )
        self._bridge = _CaptionBridge()

        # Connect thread-safe signal to widget slot
        self._bridge.caption_received.connect(
            self.widget.update_caption,
            Qt.ConnectionType.QueuedConnection,
        )

        if show_window:
            self.widget.show()

    def update_status(self, status: object) -> None:
        """Update visible status hint on the overlay."""
        self.widget.update_status_hint(status)

    def present(self, caption: CaptionSegment) -> None:
        """Receive CaptionSegment from application worker thread and dispatch to Qt GUI thread."""
        self._bridge.caption_received.emit(caption)

    def apply_settings(self, settings: Settings) -> None:
        """Apply Settings object to overlay widget."""
        self.widget.apply_settings(settings)

    def get_position(self) -> Tuple[int, int]:
        """Get current overlay window position (x, y)."""
        return (self.widget.pos().x(), self.widget.pos().y())

    def clear(self) -> None:
        """Clear all active captions on overlay."""
        self.widget.clear_captions()

    def show(self) -> None:
        """Display the overlay widget."""
        self.widget.show()

    def hide(self) -> None:
        """Hide the overlay widget."""
        self.widget.hide()

    def close(self) -> None:
        """Close the overlay widget."""
        self.widget.close()
