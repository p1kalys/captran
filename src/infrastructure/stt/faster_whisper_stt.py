"""Faster-Whisper STT implementation of the Transcriber domain port.

Runs offline CTranslate2-optimized Whisper models locally on CPU or GPU.
Model weights are stored locally with zero per-request network calls.
Compatible with PyInstaller frozen bundles and source execution.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Sequence
import numpy as np

from src.domain.entities import AudioChunk, TranscriptSegment
from src.domain.ports import Transcriber

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


class FasterWhisperTranscriber(Transcriber):
    """Offline Japanese speech-to-text / direct translation using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        download_root: str = "models/whisper",
        beam_size: int = 1,
        cpu_threads: Optional[int] = None,
        local_files_only: bool = False,
        task: str = "translate",
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = str(_resolve_resource_path(download_root))
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads or min(4, os.cpu_count() or 4)
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
        """Lazy load the Whisper model into memory."""
        if self._model is None:
            if WhisperModel is None:
                raise ImportError(
                    "faster-whisper is not installed. Please install faster-whisper."
                )

            # Check if direct model directory exists inside download_root
            candidate_model_dir = Path(self.download_root) / self.model_size
            model_to_load = str(candidate_model_dir) if candidate_model_dir.exists() else self.model_size

            Path(self.download_root).mkdir(parents=True, exist_ok=True)
            if callable(status_callback):
                status_callback(f"Connecting / loading Whisper model '{self.model_size}' (downloading if not cached)...")

            self._model = WhisperModel(
                model_size_or_path=model_to_load,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=self.download_root,
                local_files_only=self._local_files_only,
            )
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
                        start_time=float(seg.start),
                        end_time=float(seg.end),
                        language=out_lang,
                    )
                )

        return transcript_segments
