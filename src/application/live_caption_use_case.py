"""LiveCaptionUseCase orchestrator with error handling and automatic recovery.

Coordinates:
AudioSource -> SpeechSegmenter (VAD) -> Transcriber (STT) -> Translator -> CaptionPresenter.

Features:
- Automatic reconnect with backoff on audio source failure / device disconnect.
- Graceful degradation on STT / Translation model errors.
- Thread-safe start, stop, and clean shutdown from any pipeline state.
- Real-time PipelineStatus reporting.
"""

from dataclasses import dataclass
import threading
from typing import Callable, Iterator, List, Optional

from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    PipelineStatus,
    TranscriptSegment,
)
from src.domain.ports import (
    AudioSource,
    CaptionPresenter,
    SpeechSegmenter,
    Transcriber,
    Translator,
)


class LiveCaptionUseCase:
    """Application orchestrator with built-in resilience, error recovery, and status tracking."""

    def __init__(
        self,
        audio_source: Optional[AudioSource] = None,
        segmenter: Optional[SpeechSegmenter] = None,
        transcriber: Optional[Transcriber] = None,
        translator: Optional[Translator] = None,
        presenter: Optional[CaptionPresenter] = None,
        status_callback: Optional[Callable[[PipelineStatus], None]] = None,
        reconnect_interval_s: float = 1.0,
        max_reconnect_attempts: Optional[int] = None,
        *,
        audio_capture: Optional[AudioSource] = None,
        speech_to_text: Optional[Transcriber] = None,
        translation: Optional[Translator] = None,
        caption_display: Optional[CaptionPresenter] = None,
    ) -> None:
        self._audio_source = audio_source or audio_capture  # type: ignore
        self._transcriber = transcriber or speech_to_text  # type: ignore
        self._translator = translator or translation  # type: ignore
        self._presenter = presenter or caption_display  # type: ignore

        if segmenter is not None:
            self._segmenter = segmenter
        else:
            class _PassThroughSegmenter(SpeechSegmenter):
                def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
                    yield from chunk_stream

            self._segmenter = _PassThroughSegmenter()

        self._status_callback = status_callback
        self._reconnect_interval_s = max(0.1, reconnect_interval_s)
        self._max_reconnect_attempts = max_reconnect_attempts

        self._is_running = False
        self._current_status = PipelineStatus(state="idle", message="Pipeline initialized")
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def process_next_chunk(self) -> List[CaptionSegment]:
        """Backward compatibility helper to process a single queued chunk."""
        if hasattr(self._audio_source, "read_chunk"):
            chunk = self._audio_source.read_chunk()
            if chunk:
                return self.process_utterance(chunk)
        return []

    @property
    def is_running(self) -> bool:
        """Check if the live captioning pipeline is active or reconnecting."""
        return self._is_running

    @property
    def current_status(self) -> PipelineStatus:
        """Get the current pipeline operational status."""
        return self._current_status

    def _report_status(self, state: str, message: str = "") -> None:
        """Update and emit pipeline status in a thread-safe manner."""
        status = PipelineStatus(state=state, message=message)
        self._current_status = status
        if self._status_callback is not None:
            try:
                self._status_callback(status)
            except Exception:
                pass

    def process_utterance(self, utterance: AudioChunk) -> List[CaptionSegment]:
        """Process an utterance segment with graceful error handling on STT/Translation failures."""
        if not utterance.pcm_data:
            return []

        # 1. Speech-to-Text Transcription with Graceful Degradation
        try:
            transcript_segments = self._transcriber.transcribe(utterance)
        except Exception as stt_err:
            self._report_status(
                state="degraded",
                message=f"STT transcription failed: {stt_err}",
            )
            return []

        generated_captions: List[CaptionSegment] = []

        for trans_seg in transcript_segments:
            seg_text = trans_seg.text.strip()
            if not seg_text:
                continue

            # 2. Machine Translation (or Direct Whisper Translation passthrough)
            if trans_seg.language == "en" or self._translator is None:
                en_text = seg_text
                ja_text = ""
            else:
                ja_text = seg_text
                try:
                    en_res = self._translator.translate(ja_text)
                    if hasattr(en_res, "translated_text"):
                        en_text = en_res.translated_text
                    else:
                        en_text = str(en_res)
                except Exception as trans_err:
                    self._report_status(
                        state="degraded",
                        message=f"Translation failed: {trans_err}. Displaying original text.",
                    )
                    en_text = f"[JA] {ja_text}"

            caption = CaptionSegment(
                text=en_text,
                is_final=trans_seg.is_final,
                start_time=trans_seg.start_time,
                end_time=trans_seg.end_time,
                language="en",
                original_text=ja_text,
            )

            try:
                self._presenter.present(caption)
            except Exception as ui_err:
                self._report_status(
                    state="degraded",
                    message=f"Caption presenter display error: {ui_err}",
                )

            generated_captions.append(caption)

        return generated_captions

    def process_stream(self, chunk_stream: Iterator[AudioChunk]) -> None:
        """Run segmentation, transcription, translation, and display on a chunk stream."""
        for utterance in self._segmenter.segment(chunk_stream):
            if self._stop_event.is_set():
                break
            self.process_utterance(utterance)

    def _run_loop(self) -> None:
        """Resilient background worker loop with automatic reconnect logic on audio disconnect."""
        reconnect_count = 0

        # Pre-load and verify transcriber model readiness before starting audio capture
        if hasattr(self._transcriber, "ensure_ready"):
            try:
                self._report_status("connecting", "Loading Whisper model...")
                self._transcriber.ensure_ready(
                    status_callback=lambda msg: self._report_status("connecting", msg)
                )
            except Exception as model_err:
                self._report_status("error", f"Model not ready / failed to load: {model_err}")
                with self._lock:
                    self._is_running = False
                return

        while not self._stop_event.is_set():
            try:
                self._report_status("running", "Audio capture and processing active")
                chunk_stream = self._audio_source.stream()
                self.process_stream(chunk_stream)

                # If stream completes cleanly (e.g. EOF or normal stop), break out
                break

            except Exception as e:
                if self._stop_event.is_set():
                    break

                reconnect_count += 1
                if self._max_reconnect_attempts and reconnect_count > self._max_reconnect_attempts:
                    self._report_status("error", f"Max reconnect attempts exceeded: {e}")
                    break

                self._report_status(
                    state="reconnecting",
                    message=f"Audio device disconnected ({e}). Reconnecting in {self._reconnect_interval_s}s...",
                )

                try:
                    self._audio_source.stop()
                except Exception:
                    pass

                # Interruptible wait for reconnect backoff
                if self._stop_event.wait(timeout=self._reconnect_interval_s):
                    break

                try:
                    self._audio_source.start()
                    self._report_status("running", "Audio device reconnected successfully")
                except Exception as start_err:
                    self._report_status(
                        state="reconnecting",
                        message=f"Reconnection attempt failed: {start_err}",
                    )

        with self._lock:
            self._is_running = False
            if not self._stop_event.is_set():
                self._report_status("stopped", "Pipeline stopped unexpectedly")

    def start(self, run_in_background: bool = True) -> None:
        """Start audio capture and processing pipeline in a thread-safe manner."""
        with self._lock:
            if self._is_running:
                return

            self._stop_event.clear()
            self._is_running = True

            try:
                self._audio_source.start()
                self._report_status("running", "Pipeline started")
            except Exception as e:
                self._report_status("error", f"Failed to start audio source: {e}")
                # Start reconnect loop if in background
                if not run_in_background:
                    self._is_running = False
                    raise

            if run_in_background:
                self._worker_thread = threading.Thread(
                    target=self._run_loop,
                    name="LiveCaptionPipelineWorker",
                    daemon=True,
                )
                self._worker_thread.start()
            else:
                self._run_loop()

    def stop(self, timeout: float = 2.0) -> None:
        """Thread-safe clean shutdown of all pipeline resources from any state."""
        with self._lock:
            if not self._is_running and self._stop_event.is_set():
                return

            self._stop_event.set()
            self._is_running = False
            self._report_status("stopped", "Pipeline stopped cleanly")

            try:
                self._audio_source.stop()
            except Exception:
                pass

            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                self._worker_thread = None
