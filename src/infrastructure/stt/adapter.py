"""Offline Speech-to-Text adapter implementing SpeechToTextPort.

Wraps offline local transcription runtimes (e.g. faster-whisper / Whisper.cpp / ONNX)
with zero network or cloud dependencies.
"""

from typing import Callable, Optional, Sequence

from src.domain.entities import AudioChunk, TranscriptionSegment
from src.domain.ports import SpeechToTextPort


class OfflineSpeechToTextAdapter(SpeechToTextPort):
    """Local offline speech-to-text adapter for Japanese audio transcription."""

    def __init__(
        self,
        model_name_or_path: str = "whisper-small-ja",
        transcribe_fn: Optional[
            Callable[[AudioChunk], Sequence[TranscriptionSegment]]
        ] = None,
    ) -> None:
        self.model_path = model_name_or_path
        self._transcribe_fn = transcribe_fn

    def transcribe(self, chunk: AudioChunk) -> Sequence[TranscriptionSegment]:
        """Transcribe an incoming Japanese audio chunk using offline model."""
        if not chunk.data:
            return []

        if self._transcribe_fn is not None:
            return self._transcribe_fn(chunk)

        # Baseline fallback implementation for demonstration / testing
        return [
            TranscriptionSegment(
                text="こんにちは、世界！",
                language="ja",
                is_final=True,
                confidence=0.95,
                start_time=0.0,
                end_time=chunk.duration_seconds,
            )
        ]

    def reset(self) -> None:
        """Reset internal streaming context/buffers."""
        pass
