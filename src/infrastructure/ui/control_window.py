"""Control window GUI for configuring and controlling the live captioning pipeline.

Built with PySide6:
- Device selection dropdown (populated from AudioSourceFactory)
- Model size dropdown (tiny, base, small, medium, large-v3)
- Start / Stop button with real-time status feedback
- Real-time pipeline status indicator (Running, Reconnecting, Degraded, Error)
- Settings load/save integration
"""

import sys
from typing import Callable, Dict, List, Optional

try:
    from PySide6.QtCore import QPoint, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QWidget = object  # type: ignore
    Signal = lambda *args: None  # type: ignore
    Slot = lambda *args: lambda fn: fn  # type: ignore

from src.domain.entities import PipelineStatus, Settings


class ControlWindow(QWidget):
    """Small modern desktop control panel for JA->EN Live Captioner."""

    start_requested = Signal(dict)   # Emits config dict when starting
    stop_requested = Signal()         # Emits when stopping
    settings_changed = Signal(dict)   # Emits when user changes setting controls

    def __init__(
        self,
        devices: Optional[List[Dict]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._devices = devices or []
        self._is_running = False

        self.setWindowTitle("CapTran Control Panel")
        self.setFixedSize(440, 320)
        self._init_ui()

    def _init_ui(self) -> None:
        """Construct the control window layout and styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #12151E;
                color: #ECEFF4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                margin-top: 14px;
                padding: 12px 14px 14px 14px;
                font-weight: 600;
                color: #88C0D0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QComboBox {
                background-color: #1E222D;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 6px 10px;
                color: #FFFFFF;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E222D;
                selection-background-color: #3B4252;
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            QPushButton#ToggleBtn {
                background-color: #2E7D32;
                color: #FFFFFF;
                font-weight: 700;
                font-size: 14px;
                border-radius: 8px;
                padding: 10px;
                border: none;
            }
            QPushButton#ToggleBtn:hover {
                background-color: #388E3C;
            }
            QPushButton#ToggleBtn.running {
                background-color: #C62828;
            }
            QPushButton#ToggleBtn.running:hover {
                background-color: #D32F2F;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Header Title
        title_label = QLabel("🇯🇵 ➔ 🌐 CapTran Live Japanese ➔ English Captioner (ja->en)")
        title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #ECEFF4;")
        layout.addWidget(title_label)

        # Settings Group
        settings_group = QGroupBox("Audio & Model Configuration")
        form_layout = QFormLayout(settings_group)
        form_layout.setContentsMargins(8, 12, 8, 8)
        form_layout.setSpacing(10)

        # Device Selection
        self.device_combo = QComboBox()
        self.device_combo.addItem("Default Speaker Output (WASAPI Loopback)", None)
        for dev in self._devices:
            label = f"[{dev['index']}] {dev['name']}"
            self.device_combo.addItem(label, dev["index"])
        self.device_combo.currentIndexChanged.connect(self._on_setting_modified)
        form_layout.addRow("Audio Device:", self.device_combo)

        # Whisper Model Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small"])
        self.model_combo.setCurrentText("small")
        self.model_combo.currentIndexChanged.connect(self._on_setting_modified)
        form_layout.addRow("Whisper Model:", self.model_combo)

        # Dual Subtitle Mode Checkbox
        self.dual_subtitles_check = QCheckBox("Enable Dual Subtitles (Stacked JA + EN)")
        self.dual_subtitles_check.setChecked(True)
        self.dual_subtitles_check.stateChanged.connect(self._on_setting_modified)
        form_layout.addRow("Display Mode:", self.dual_subtitles_check)

        layout.addWidget(settings_group)

        # Status Label
        self.status_label = QLabel("● Ready to start")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #88C0D0; font-weight: 600;")
        layout.addWidget(self.status_label)

        # Start / Stop Toggle Button
        self.toggle_btn = QPushButton("Start Live Captioning")
        self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._on_toggle_clicked)
        layout.addWidget(self.toggle_btn)

    @Slot(object)
    def update_status(self, status: PipelineStatus) -> None:
        """Update visible status indicator from PipelineStatus updates."""
        state = status.state.lower()
        if state in ("connecting", "loading", "downloading", "initializing"):
            self.status_label.setText(f"⏳ Connecting / Preparing: {status.message}")
            self.status_label.setStyleSheet("color: #EBCB8B; font-weight: 600;")
        elif state == "running":
            self.status_label.setText(f"● Active ({status.message or 'Streaming'})")
            self.status_label.setStyleSheet("color: #A3BE8C; font-weight: 600;")
        elif state in ("not connected", "disconnected", "idle"):
            self.status_label.setText(f"○ Not Connected ({status.message or 'Ready to start'})")
            self.status_label.setStyleSheet("color: #D8DEE9; font-weight: 600;")
        elif state == "reconnecting":
            self.status_label.setText(f"▲ {status.message or 'Reconnecting audio device...'}")
            self.status_label.setStyleSheet("color: #EBCB8B; font-weight: 600;")
        elif state == "degraded":
            self.status_label.setText(f"⚠ Warning: {status.message}")
            self.status_label.setStyleSheet("color: #D08770; font-weight: 600;")
        elif state == "error":
            self.status_label.setText(f"✕ Not Connected / Error: {status.message}")
            self.status_label.setStyleSheet("color: #BF616A; font-weight: 600;")
            self.set_running_state(False)
        elif state == "stopped":
            self.status_label.setText("● Stopped (Not Connected)")
            self.status_label.setStyleSheet("color: #D8DEE9; font-weight: 600;")

    def apply_settings(self, settings: Settings) -> None:
        """Apply persisted settings values to the UI controls."""
        if settings.selected_audio_device is not None:
            idx = self.device_combo.findData(settings.selected_audio_device)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        idx = self.model_combo.findText(settings.whisper_model_size)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.dual_subtitles_check.setChecked(getattr(settings, "dual_subtitles", True))

    def _on_setting_modified(self) -> None:
        """Emit notification when dropdown options are changed."""
        self.settings_changed.emit({
            "selected_audio_device": self.device_combo.currentData(),
            "whisper_model_size": self.model_combo.currentText(),
            "dual_subtitles": self.dual_subtitles_check.isChecked(),
        })

    def update_devices(self, devices: List[Dict]) -> None:
        """Update available loopback device list."""
        self._devices = devices
        current_data = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("Default Speaker Output (WASAPI Loopback)", None)
        for dev in devices:
            label = f"[{dev['index']}] {dev['name']}"
            self.device_combo.addItem(label, dev["index"])

        idx = self.device_combo.findData(current_data)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def _on_toggle_clicked(self) -> None:
        """Handle start/stop button click."""
        if not self._is_running:
            config = {
                "device_index": self.device_combo.currentData(),
                "model_size": self.model_combo.currentText(),
            }
            self.set_running_state(True)
            self.start_requested.emit(config)
        else:
            self.set_running_state(False)
            self.stop_requested.emit()

    def set_running_state(self, running: bool) -> None:
        """Update UI elements to reflect running status."""
        self._is_running = running
        self.device_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)

        if running:
            self.toggle_btn.setText("Stop Live Captioning")
            self.toggle_btn.setProperty("class", "running")
            self.toggle_btn.setStyleSheet("background-color: #C62828;")
        else:
            self.toggle_btn.setText("Start Live Captioning")
            self.toggle_btn.setStyleSheet("background-color: #2E7D32;")
