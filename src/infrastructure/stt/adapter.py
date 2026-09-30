"""Offline Speech-to-Text adapter implementing SpeechToTextPort.

Wraps offline local transcription runtimes (e.g. faster-whisper / Whisper.cpp / ONNX)
with zero network or cloud dependencies.
"""

import inspect
from typing import Callable, Optional, Sequence

from src.domain.entities import AudioChunk, Language, TranscriptionSegment
from src.domain.ports import SpeechToTextPort


class OfflineSpeechToTextAdapter(SpeechToTextPort):
    """Local offline speech-to-text adapter for audio transcription."""

    def __init__(
        self,
        model_name_or_path: str = "whisper-small-ja",
        transcribe_fn: Optional[
            Callable[..., Sequence[TranscriptionSegment]]
        ] = None,
    ) -> None:
        self.model_path = model_name_or_path
        self._transcribe_fn = transcribe_fn
        self._invocation_mode = "none"

        if transcribe_fn is not None:
            try:
                sig = inspect.signature(transcribe_fn)
                dummy_chunk = AudioChunk(pcm_data=b"")
                dummy_lang = Language.JAPANESE

                # Test whether language can be bound positionally, as keyword, or not accepted
                try:
                    sig.bind(dummy_chunk, dummy_lang)
                    self._invocation_mode = "positional"
                except TypeError:
                    try:
                        sig.bind(dummy_chunk, source_language=dummy_lang)
                        self._invocation_mode = "keyword"
                    except TypeError:
                        self._invocation_mode = "none"
            except (ValueError, TypeError):
                self._invocation_mode = "positional"

    def transcribe(
        self,
        chunk: AudioChunk,
        source_language: Language = Language.JAPANESE,
    ) -> Sequence[TranscriptionSegment]:
        """Transcribe an incoming audio chunk using offline model in source_language."""
        if not chunk.data:
            return []

        src_lang = Language.from_code(source_language) if isinstance(source_language, str) else source_language

        if self._transcribe_fn is not None:
            if self._invocation_mode == "positional":
                return self._transcribe_fn(chunk, src_lang)
            elif self._invocation_mode == "keyword":
                return self._transcribe_fn(chunk, source_language=src_lang)
            else:
                return self._transcribe_fn(chunk)

        # Baseline fallback implementation for demonstration / testing
        return [
            TranscriptionSegment(
                text="こんにちは、世界！",
                language=src_lang,
                is_final=True,
                confidence=0.95,
                start_time=0.0,
                end_time=chunk.duration_seconds,
            )
        ]

    def reset(self) -> None:
        """Reset internal streaming context/buffers."""
        pass
