"""Faster-Whisper STT implementation of the Transcriber domain port.

Runs offline CTranslate2-optimized Whisper models locally on CPU or GPU.
Automatically configures:
- CPU: compute_type="int8"
- CUDA (if GPU available): device="cuda", compute_type="float16"
- Model resolution: defaults to small or local distil-whisper (ja-tuned if available), with fallback to base for lower-end hardware.
Compatible with PyInstaller frozen bundles and source execution.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple
import numpy as np

from src.domain.entities import AudioChunk, TranscriptSegment
from src.domain.ports import Transcriber

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None  # type: ignore

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore


def _resolve_resource_path(relative_or_abs_path: str) -> Path:
    """Resolve path relative to sys._MEIPASS when frozen with PyInstaller, or cwd."""
    p = Path(relative_or_abs_path)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_candidate = Path(sys._MEIPASS) / relative_or_abs_path
        if meipass_candidate.exists():
            return meipass_candidate
    return p.resolve()


def is_cuda_available() -> bool:
    """Check if CUDA-compatible GPU is available via ctranslate2."""
    if ctranslate2 is not None:
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False
    return False


def resolve_device_and_compute_type(
    device: str = "auto",
    compute_type: Optional[str] = "auto",
) -> Tuple[str, str]:
    """Auto-detect CUDA and assign optimal compute_type (float16 for CUDA, int8 for CPU)."""
    if device == "auto":
        resolved_device = "cuda" if is_cuda_available() else "cpu"
    else:
        resolved_device = device

    if compute_type in (None, "auto", "default"):
        resolved_compute_type = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute_type = compute_type

    return resolved_device, resolved_compute_type


def resolve_model_name_or_path(
    model_size: str = "small",
    download_root: str = "models/whisper",
    fallback_to_base: bool = False,
) -> str:
    """Resolve model: distil-whisper (ja-tuned if local), small, or fallback to base."""
    if fallback_to_base:
        return "base"

    root_path = Path(download_root)
    # If user requests distil-whisper or auto, check for local Japanese-tuned distil models
    if model_size in ("distil-whisper", "distil", "auto", "default"):
        if root_path.exists():
            # Check for specific Japanese tuned distil models first
            for candidate in root_path.iterdir():
                if candidate.is_dir() and "distil" in candidate.name.lower() and "ja" in candidate.name.lower():
                    return str(candidate)
            # Check for general distil models next
            for candidate in root_path.iterdir():
                if candidate.is_dir() and "distil" in candidate.name.lower():
                    return str(candidate)
        return "small"

    return model_size


class FasterWhisperTranscriber(Transcriber):
    """Offline Japanese speech-to-text / direct translation using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: Optional[str] = "auto",
        download_root: str = "models/whisper",
        beam_size: int = 1,
        cpu_threads: Optional[int] = None,
        local_files_only: bool = False,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        fallback_to_base: bool = False,
    ) -> None:
        self.download_root = str(_resolve_resource_path(download_root))
        self.fallback_to_base = fallback_to_base
        self.model_size = resolve_model_name_or_path(
            model_size=model_size,
            download_root=self.download_root,
            fallback_to_base=fallback_to_base,
        )
        self.device, self.compute_type = resolve_device_and_compute_type(
            device=device,
            compute_type=compute_type,
        )
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads or min(8, os.cpu_count() or 4)
        self.task = task
        self.initial_prompt = initial_prompt
        self._model: Optional[WhisperModel] = None
        self._local_files_only = local_files_only

    def ensure_ready(self, status_callback: Optional[object] = None) -> None:
        """Pre-load and initialize the model, emitting status notifications."""
        if self._model is not None:
            return
        self._ensure_model_loaded(status_callback=status_callback)

    def _ensure_model_loaded(self, status_callback: Optional[object] = None) -> WhisperModel:
        """Lazy load the Whisper model into memory with automatic GPU/CPU fallback."""
        if self._model is None:
            if WhisperModel is None:
                raise ImportError(
                    "faster-whisper is not installed. Please install faster-whisper."
                )

            candidate_model_dir = Path(self.download_root) / self.model_size
            model_to_load = str(candidate_model_dir) if candidate_model_dir.exists() else self.model_size

            Path(self.download_root).mkdir(parents=True, exist_ok=True)
            if callable(status_callback):
                status_callback(
                    f"Connecting / loading Whisper model '{self.model_size}' "
                    f"({self.device}/{self.compute_type})..."
                )

            try:
                self._model = WhisperModel(
                    model_size_or_path=model_to_load,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    download_root=self.download_root,
                    local_files_only=self._local_files_only,
                )
            except Exception as e:
                # If CUDA load fails, fallback to CPU int8
                if self.device == "cuda":
                    if callable(status_callback):
                        status_callback(f"CUDA initialization failed ({e}), falling back to CPU int8...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        model_size_or_path=model_to_load,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cpu_threads,
                        download_root=self.download_root,
                        local_files_only=self._local_files_only,
                    )
                else:
                    raise

        return self._model

    def transcribe(self, audio: AudioChunk) -> Sequence[TranscriptSegment]:
        """Transcribe or directly translate an audio segment into TranscriptSegments."""
        if not audio.pcm_data:
            return []

        audio_int16 = np.frombuffer(audio.pcm_data, dtype=np.int16)
        if len(audio_int16) == 0:
            return []

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        model = self._ensure_model_loaded()

        segments_gen, info = model.transcribe(
            audio_float32,
            language="ja",
            task=self.task,
            beam_size=self.beam_size,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )

        segments = list(segments_gen)
        transcript_segments = []

        out_lang = "en" if self.task == "translate" else "ja"
        for seg in segments:
            text = seg.text.strip()
            if text:
                transcript_segments.append(
                    TranscriptSegment(
                        text=text,
                        is_final=True,
                        start_time=float(getattr(seg, "start", 0.0) or 0.0),
                        end_time=float(getattr(seg, "end", 0.0) or 0.0),
                        language=out_lang,
                    )
                )

        return transcript_segments
