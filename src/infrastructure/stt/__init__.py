"""Speech-to-text infrastructure package."""

from src.infrastructure.stt.adapter import OfflineSpeechToTextAdapter
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber

__all__ = ["OfflineSpeechToTextAdapter", "FasterWhisperTranscriber"]
