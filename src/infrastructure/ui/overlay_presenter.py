"""PySide6 desktop overlay presenter implementing the CaptionPresenter domain port.

Provides a modern, draggable, translucent, frameless, always-on-top subtitle overlay:
- Commits final captions to solid high-contrast text, keeping last 2-3 lines visible.
- Displays interim captions in dimmed/italicized style in real time.
- Supports configurable font size, opacity, and window positioning from Settings.
- Fully thread-safe via Qt Queued Signals across worker threads.
"""

from collections import deque
import sys
from typing import Deque, Optional, Tuple

from src.domain.entities import CaptionSegment, Settings
from src.domain.ports import CaptionPresenter

try:
    from PySide6.QtCore import QObject, QPoint, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter
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
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.max_history_lines = max_history_lines
        self.font_size = font_size
        self.opacity = opacity
        self._final_lines: Deque[str] = deque(maxlen=max_history_lines)
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
        self.setMinimumSize(600, 110)
        self.resize(850, 130)

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

        self.title_label = QLabel("CAPTRAN | JAPANESE ➔ ENGLISH LIVE CAPTIONS (ja->en)", self.container)
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

        # Subtitle Text Display Label
        self.text_label = QLabel("Waiting for Japanese audio...", self.container)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setStyleSheet("color: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif;")
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
        text = caption.text.strip()
        if not text:
            return

        if caption.is_final:
            self._final_lines.append(text)
            self._current_interim_text = ""
        else:
            self._current_interim_text = text

        self._render_text()

    def _render_text(self) -> None:
        """Build HTML string with solid past lines and italicized/dimmed active interim line."""
        lines_html = []
        sz = self.font_size

        # Committed final lines (solid white with shadow)
        for line in self._final_lines:
            lines_html.append(
                f"<div style='color: #FFFFFF; font-size: {sz}px; font-weight: 600; line-height: 1.35; "
                f"text-shadow: 0px 1px 3px rgba(0,0,0,0.8);'>{line}</div>"
            )

        # Interim active line (dimmed & italicized)
        if self._current_interim_text:
            lines_html.append(
                f"<div style='color: #90CAF9; font-size: {sz}px; font-style: italic; line-height: 1.35; "
                f"text-shadow: 0px 1px 3px rgba(0,0,0,0.8);'>{self._current_interim_text} <span style='color: #64B5F6; font-size: {max(10, sz-4)}px;'>●</span></div>"
            )

        if not lines_html:
            self.text_label.setText(f"<div style='color: rgba(255,255,255,0.4); font-size: {max(12, sz-2)}px;'>Listening...</div>")
        else:
            self.text_label.setText("".join(lines_html))

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
                    f"● Listening for Japanese audio...</div>"
                )

    def clear_captions(self) -> None:
        """Clear visible lines."""
        self._final_lines.clear()
        self._current_interim_text = ""
        self._render_text()


class OverlayCaptionPresenter(CaptionPresenter):
    """Drop-in CaptionPresenter port implementation driving the PySide6 overlay."""

    def __init__(
        self,
        max_history_lines: int = 3,
        font_size: int = 16,
        opacity: float = 0.85,
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
