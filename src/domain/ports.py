"""Domain port interfaces.

Defines the abstract contracts for audio capture, segmentation (VAD),
transcription, translation, caption presentation, and configuration persistence.
Pure Python standard library with zero third-party dependencies.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Sequence

from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    Settings,
    TranscriptSegment,
)


class AudioSource(ABC):
    """Abstract port for capturing live audio streams."""

    @abstractmethod
    def start(self) -> None:
        """Start capturing audio from the input source."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio and release resources."""
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> Iterator[AudioChunk]:
        """Yield a stream of raw PCM AudioChunk objects."""
        raise NotImplementedError

    # Backward compatibility helpers
    def start_capture(self) -> None:
        self.start()

    def stop_capture(self) -> None:
        self.stop()


class SpeechSegmenter(ABC):
    """Abstract port for Voice Activity Detection (VAD) audio segmentation."""

    @abstractmethod
    def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
        """Take an AudioChunk stream and yield utterance-bounded audio segments."""
        raise NotImplementedError


class Transcriber(ABC):
    """Abstract port for speech-to-text transcription."""

    @abstractmethod
    def transcribe(self, audio: AudioChunk) -> Sequence[TranscriptSegment]:
        """Transcribe an audio segment into interim and final Japanese TranscriptSegments."""
        raise NotImplementedError

    def reset(self) -> None:
        pass


class Translator(ABC):
    """Abstract port for machine translation."""

    @abstractmethod
    def translate(self, japanese_text: str) -> str:
        """Translate Japanese text into English text."""
        raise NotImplementedError


class CaptionPresenter(ABC):
    """Abstract port for presenting/displaying live subtitles."""

    @abstractmethod
    def present(self, caption: CaptionSegment) -> None:
        """Receive a CaptionSegment and render or display it on screen."""
        raise NotImplementedError

    def display_cue(self, cue: CaptionSegment) -> None:
        self.present(cue)

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass


class SettingsRepository(ABC):
    """Abstract port for loading and saving application settings."""

    @abstractmethod
    def load(self) -> Settings:
        """Load settings from persistent storage, returning defaults if not found."""
        raise NotImplementedError

    @abstractmethod
    def save(self, settings: Settings) -> None:
        """Persist settings to storage."""
        raise NotImplementedError


# Backward compatibility aliases
AudioCapturePort = AudioSource
SpeechToTextPort = Transcriber
TranslationPort = Translator
CaptionDisplayPort = CaptionPresenter
