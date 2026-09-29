"""Domain entities and value objects.

Pure Python standard library with zero third-party dependencies.
All entities and value objects are immutable dataclasses.
"""

from dataclasses import dataclass, field
import re
import time
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AudioChunk:
    """Raw PCM audio slice with sampling parameters and capture timestamp."""

    pcm_data: bytes = b""
    sample_rate: int = 16000
    timestamp: float = field(default_factory=time.time)
    channels: int = 1
    sample_width: int = 2
    is_final: bool = True

    def __init__(
        self,
        pcm_data: Optional[bytes] = None,
        sample_rate: int = 16000,
        timestamp: Optional[float] = None,
        channels: int = 1,
        sample_width: int = 2,
        data: Optional[bytes] = None,
        is_final: bool = True,
    ) -> None:
        raw = pcm_data if pcm_data is not None else (data if data is not None else b"")
        object.__setattr__(self, "pcm_data", raw)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "timestamp", timestamp if timestamp is not None else time.time())
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "sample_width", sample_width)
        object.__setattr__(self, "is_final", is_final)

    # Backward compatibility alias
    @property
    def data(self) -> bytes:
        return self.pcm_data

    @property
    def duration_seconds(self) -> float:
        """Calculate duration of audio chunk in seconds."""
        bytes_per_sample = self.sample_width * self.channels
        if bytes_per_sample == 0 or self.sample_rate == 0:
            return 0.0
        return len(self.pcm_data) / (bytes_per_sample * self.sample_rate)


@dataclass(frozen=True)
class TranscriptSegment:
    """Japanese transcript segment produced by speech-to-text transcription."""

    text: str = ""
    is_final: bool = True
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = "ja"
    confidence: float = 1.0

    def __init__(
        self,
        text: str = "",
        is_final: bool = True,
        start_time: float = 0.0,
        end_time: float = 0.0,
        language: str = "ja",
        confidence: float = 1.0,
        source_text: Optional[str] = None,
    ) -> None:
        txt = source_text if source_text is not None else text
        object.__setattr__(self, "text", txt)
        object.__setattr__(self, "is_final", is_final)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "confidence", confidence)

    @property
    def source_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class CaptionSegment:
    """English caption segment ready for presentation/display."""

    text: str = ""
    is_final: bool = True
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = "en"
    original_text: str = ""

    def __init__(
        self,
        text: str = "",
        is_final: bool = True,
        start_time: float = 0.0,
        end_time: float = 0.0,
        language: str = "en",
        original_text: str = "",
        source_text: Optional[str] = None,
        translated_text: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> None:
        final_text = translated_text if translated_text is not None else text
        final_orig = source_text if source_text is not None else original_text
        lang = target_language if target_language is not None else language
        object.__setattr__(self, "text", final_text)
        object.__setattr__(self, "is_final", is_final)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "language", lang)
        object.__setattr__(self, "original_text", final_orig)

    # Backward compatibility aliases
    @property
    def translated_text(self) -> str:
        return self.text

    @property
    def source_text(self) -> str:
        return self.original_text

    @property
    def source_language(self) -> str:
        return "ja"

    @property
    def target_language(self) -> str:
        return self.language


@dataclass(frozen=True)
class Settings:
    """Application user configuration and preferences."""

    selected_audio_device: Optional[int] = None
    whisper_model_size: str = "small"
    overlay_opacity: float = 0.85
    overlay_font_size: int = 16
    overlay_position_x: Optional[int] = None
    overlay_position_y: Optional[int] = None
    vad_sensitivity: float = 0.4
    vad_silence_timeout_ms: float = 500.0
    max_history_lines: int = 3
    fallback_to_base: bool = False
    dual_subtitles: bool = True


@dataclass(frozen=True)
class CustomVocabulary:
    """Domain-specific terms, acronyms, and substitution mappings."""

    prompt_terms: Tuple[str, ...] = field(default_factory=tuple)
    source_substitutions: Dict[str, str] = field(default_factory=dict)
    target_substitutions: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        prompt_terms: Optional[Sequence[str]] = None,
        source_substitutions: Optional[Dict[str, str]] = None,
        target_substitutions: Optional[Dict[str, str]] = None,
    ) -> None:
        object.__setattr__(self, "prompt_terms", tuple(prompt_terms) if prompt_terms else ())
        object.__setattr__(
            self,
            "source_substitutions",
            dict(source_substitutions) if source_substitutions else {},
        )
        object.__setattr__(
            self,
            "target_substitutions",
            dict(target_substitutions) if target_substitutions else {},
        )

    def get_initial_prompt(self) -> str:
        """Construct comma-separated prompt hint string for Whisper STT."""
        return ", ".join(self.prompt_terms)

    def apply_source_substitutions(self, text: str) -> str:
        """Pre-translation substitution pass on Japanese / source transcript."""
        if not text or not self.source_substitutions:
            return text
        res = text
        for pattern, repl in self.source_substitutions.items():
            if pattern in res:
                res = res.replace(pattern, repl)
        return res

    def apply_target_substitutions(self, text: str) -> str:
        """Post-translation substitution pass on English translated text."""
        if not text or not self.target_substitutions:
            return text
        res = text
        for pattern, repl in self.target_substitutions.items():
            if not pattern:
                continue
            # Build regex with word boundaries at alphanumeric edges
            left_b = r"\b" if pattern[0].isalnum() or pattern[0] == "_" else ""
            right_b = r"\b" if pattern[-1].isalnum() or pattern[-1] == "_" else ""
            regex_pat = f"{left_b}{re.escape(pattern)}{right_b}"
            try:
                res = re.sub(regex_pat, lambda _m, r=repl: r, res, flags=re.IGNORECASE)
            except Exception:
                res = res.replace(pattern, repl)
        return res


@dataclass(frozen=True)
class PipelineStatus:
    """Status state of the live captioning pipeline."""

    state: str  # "idle", "running", "reconnecting", "degraded", "stopped", "error"
    message: str = ""
    timestamp: float = field(default_factory=time.time)


# Backward-compatibility aliases
TranscriptionSegment = TranscriptSegment
TranslationSegment = CaptionSegment
CaptionCue = CaptionSegment
