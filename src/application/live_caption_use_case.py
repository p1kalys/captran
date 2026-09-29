"""LiveCaptionUseCase orchestrator with decoupled multi-threaded producer-consumer pipeline.

Architecture:
- Audio Capture Worker   -> Bounded _audio_queue
- VAD Segmentation Worker -> Bounded _stt_queue (with backpressure: drops/coalesces interims, preserves finals)
- Transcription Worker   -> Faster-Whisper STT (interims dispatched immediately, finals -> _translation_queue)
- Translation Worker     -> Argos Translate / Secondary MT (final subtitles committed to CaptionPresenter)

Features:
- Completely decoupled execution: slow translation never blocks audio capture or VAD.
- Backpressure handling: Drops/coalesces interim partials under heavy load, never drops final segments.
- Automatic reconnect with backoff on audio device disconnect.
- Graceful degradation on STT / Translation model errors.
- Thread-safe start, stop, and clean shutdown from any pipeline state.
- Real-time PipelineStatus reporting and high-resolution latency tracking.
"""

from queue import Empty, Full, Queue
import threading
import time
from typing import Callable, Iterator, List, Optional, Tuple

from src.application.latency_tracker import LatencyTracker
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
    """Decoupled multi-threaded live captioning use case with producer-consumer queues and backpressure."""

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
        debug_latency: bool = False,
        latency_report_interval_s: float = 30.0,
        latency_tracker: Optional[LatencyTracker] = None,
        audio_queue_maxsize: int = 50,
        stt_queue_maxsize: int = 8,
        translation_queue_maxsize: int = 10,
        vocabulary: Optional[object] = None,
    ) -> None:
        self._audio_source = audio_source or audio_capture  # type: ignore
        self._transcriber = transcriber or speech_to_text  # type: ignore
        self._translator = translator or translation  # type: ignore
        self._presenter = presenter or caption_display  # type: ignore
        self._vocabulary = vocabulary

        # Configure initial prompt hint on transcriber if available
        if self._vocabulary is not None and hasattr(self._transcriber, "initial_prompt"):
            prompt_hint = self._get_initial_prompt()
            if prompt_hint:
                self._transcriber.initial_prompt = prompt_hint

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

        if latency_tracker is not None:
            self._latency_tracker = latency_tracker
        else:
            self._latency_tracker = LatencyTracker(
                enabled=debug_latency,
                report_interval_s=latency_report_interval_s,
            )

        # Bounded Queues for decoupled producer-consumer pipeline
        self._audio_queue_maxsize = audio_queue_maxsize
        self._stt_queue_maxsize = stt_queue_maxsize
        self._translation_queue_maxsize = translation_queue_maxsize

        self._audio_queue: Queue[Optional[AudioChunk]] = Queue(maxsize=self._audio_queue_maxsize)
        self._stt_queue: Queue[Optional[Tuple[AudioChunk, float, float]]] = Queue(maxsize=self._stt_queue_maxsize)
        self._translation_queue: Queue[Optional[Tuple[TranscriptSegment, AudioChunk, float, float, float]]] = Queue(
            maxsize=self._translation_queue_maxsize
        )

        self._queue_lock = threading.Lock()
        self._is_running = False
        self._current_status = PipelineStatus(state="idle", message="Pipeline initialized")
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Thread references
        self._audio_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None
        self._translation_thread: Optional[threading.Thread] = None

    def _get_initial_prompt(self) -> str:
        """Get initial prompt hints from vocabulary."""
        if self._vocabulary is None:
            return ""
        if hasattr(self._vocabulary, "get_initial_prompt"):
            return self._vocabulary.get_initial_prompt()
        if hasattr(self._vocabulary, "load"):
            return self._vocabulary.load().get_initial_prompt()
        return ""

    def _apply_source_substitutions(self, text: str) -> str:
        """Apply pre-translation substitutions on Japanese text."""
        if self._vocabulary is None or not text:
            return text
        if hasattr(self._vocabulary, "apply_source_substitutions"):
            return self._vocabulary.apply_source_substitutions(text)
        if hasattr(self._vocabulary, "load"):
            return self._vocabulary.load().apply_source_substitutions(text)
        return text

    def _apply_target_substitutions(self, text: str) -> str:
        """Apply post-translation substitutions on English text."""
        if self._vocabulary is None or not text:
            return text
        if hasattr(self._vocabulary, "apply_target_substitutions"):
            return self._vocabulary.apply_target_substitutions(text)
        if hasattr(self._vocabulary, "load"):
            return self._vocabulary.load().apply_target_substitutions(text)
        return text

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

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Get the latency tracking instrumentation instance."""
        return self._latency_tracker

    def _report_status(self, state: str, message: str = "") -> None:
        """Update and emit pipeline status in a thread-safe manner."""
        status = PipelineStatus(state=state, message=message)
        self._current_status = status
        if self._status_callback is not None:
            try:
                self._status_callback(status)
            except Exception:
                pass

    # =========================================================================
    # Synchronous Direct Utterance Processing (for tests & compatibility)
    # =========================================================================

    def process_utterance(
        self,
        utterance: AudioChunk,
        vad_duration_ms: Optional[float] = None,
    ) -> List[CaptionSegment]:
        """Process an utterance segment directly with synchronous execution."""
        if not utterance.pcm_data:
            return []

        t_pipeline_start = time.perf_counter()

        # 1. Audio chunk received -> VAD segment complete duration
        if vad_duration_ms is not None:
            audio_to_vad_ms = max(0.0, vad_duration_ms)
        elif utterance.timestamp > 1e8 and (time.time() - utterance.timestamp) < 3600:
            audio_to_vad_ms = max(0.0, (time.time() - utterance.timestamp) * 1000.0)
        else:
            audio_to_vad_ms = 0.0

        # 2. Speech-to-Text Transcription with Graceful Degradation
        t_stt_start = time.perf_counter()
        try:
            transcript_segments = self._transcriber.transcribe(utterance)
        except Exception as stt_err:
            t_stt_end = time.perf_counter()
            self._report_status(
                state="degraded",
                message=f"STT transcription failed: {stt_err}",
            )
            stt_ms = (t_stt_end - t_stt_start) * 1000.0
            pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            self._latency_tracker.record(
                audio_to_vad_ms=audio_to_vad_ms,
                transcription_ms=stt_ms,
                translation_ms=0.0,
                display_ms=0.0,
                total_pipeline_ms=pipeline_ms,
                e2e_latency_ms=audio_to_vad_ms + pipeline_ms,
                text_preview="[STT Failed]",
            )
            return []

        t_stt_end = time.perf_counter()
        transcription_ms = (t_stt_end - t_stt_start) * 1000.0

        generated_captions: List[CaptionSegment] = []
        translation_duration_ms = 0.0
        display_duration_ms = 0.0
        is_utterance_final = getattr(utterance, "is_final", True)

        for trans_seg in transcript_segments:
            seg_text = trans_seg.text.strip()
            if not seg_text:
                continue

            is_segment_final = is_utterance_final and getattr(trans_seg, "is_final", True)

            # 3. Machine Translation (interim + final)
            t_trans_start = time.perf_counter()
            if trans_seg.language == "en" or self._translator is None:
                ja_text = ""
                en_text = self._apply_target_substitutions(seg_text)
            else:
                # Pre-translation source substitution pass
                ja_text = self._apply_source_substitutions(seg_text)
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

                # Post-translation target substitution pass
                en_text = self._apply_target_substitutions(en_text)

            t_trans_end = time.perf_counter()
            translation_duration_ms += (t_trans_end - t_trans_start) * 1000.0

            caption = CaptionSegment(
                text=en_text,
                is_final=is_segment_final,
                start_time=trans_seg.start_time,
                end_time=trans_seg.end_time,
                language="en" if (self._translator is not None or trans_seg.language == "en") else trans_seg.language,
                original_text=ja_text,
            )

            # 4. Caption Display / Presentation
            t_disp_start = time.perf_counter()
            try:
                if not is_segment_final and hasattr(self._presenter, "show_interim"):
                    self._presenter.show_interim(caption)
                elif is_segment_final and hasattr(self._presenter, "show_final"):
                    self._presenter.show_final(caption)
                else:
                    self._presenter.present(caption)
            except Exception as ui_err:
                self._report_status(
                    state="degraded",
                    message=f"Caption presenter display error: {ui_err}",
                )
            t_disp_end = time.perf_counter()
            display_duration_ms += (t_disp_end - t_disp_start) * 1000.0

            generated_captions.append(caption)

        total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        e2e_latency_ms = audio_to_vad_ms + total_pipeline_ms
        preview_text = generated_captions[0].text if generated_captions else ""

        self._latency_tracker.record(
            audio_to_vad_ms=audio_to_vad_ms,
            transcription_ms=transcription_ms,
            translation_ms=translation_duration_ms,
            display_ms=display_duration_ms,
            total_pipeline_ms=total_pipeline_ms,
            e2e_latency_ms=e2e_latency_ms,
            text_preview=preview_text,
        )

        return generated_captions

    # =========================================================================
    # Decoupled Multi-Threaded Pipeline Workers (Producer-Consumer)
    # =========================================================================

    def _push_stt_queue_with_backpressure(self, item: Tuple[AudioChunk, float, float]) -> None:
        """Push an utterance to the STT queue with backpressure handling.

        - If item is interim (is_final=False) and queue is full: drop/coalesce to relieve load.
        - If item is final (is_final=True): evict pending interim items to make room, but NEVER drop final.
        """
        utterance, _, _ = item
        is_final = getattr(utterance, "is_final", True)

        if not is_final:
            try:
                self._stt_queue.put_nowait(item)
            except Full:
                # Drop interim chunk under heavy STT load to keep pipeline responsive
                pass
            return

        # Final chunk: ensure it is enqueued without dropping
        while not self._stop_event.is_set():
            try:
                self._stt_queue.put(item, timeout=0.05)
                break
            except Full:
                # Evict any pending interim chunks to make room for final result
                with self._queue_lock:
                    surviving_items = []
                    while not self._stt_queue.empty():
                        try:
                            q_item = self._stt_queue.get_nowait()
                            self._stt_queue.task_done()
                            if q_item is not None and getattr(q_item[0], "is_final", True):
                                surviving_items.append(q_item)
                        except Empty:
                            break
                    for s_item in surviving_items:
                        while not self._stop_event.is_set():
                            try:
                                self._stt_queue.put(s_item, timeout=0.05)
                                break
                            except Full:
                                continue

    def _audio_capture_loop(self) -> None:
        """Worker Thread 1: Captures audio stream from AudioSource and queues to _audio_queue."""
        reconnect_count = 0

        while not self._stop_event.is_set():
            try:
                self._report_status("running", "Audio capture and processing active")
                for chunk in self._audio_source.stream():
                    if self._stop_event.is_set():
                        break
                    while not self._stop_event.is_set():
                        try:
                            self._audio_queue.put(chunk, timeout=0.1)
                            break
                        except Full:
                            continue

                # Stream ended cleanly (EOF)
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

        # Signal downstream that audio stream is finished
        while not self._stop_event.is_set():
            try:
                self._audio_queue.put(None, timeout=0.1)
                break
            except Full:
                continue

    def _vad_segmentation_loop(self) -> None:
        """Worker Thread 2: Reads raw chunks, runs VAD segmentation, and queues utterances to _stt_queue."""
        def _chunk_stream_gen() -> Iterator[AudioChunk]:
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                if chunk is None:
                    self._audio_queue.task_done()
                    break
                self._audio_queue.task_done()
                yield chunk

        try:
            for utterance in self._segmenter.segment(_chunk_stream_gen()):
                if self._stop_event.is_set():
                    break
                t_vad_complete = time.perf_counter()
                if utterance.timestamp > 1e8 and (time.time() - utterance.timestamp) < 3600:
                    audio_to_vad_ms = max(0.0, (time.time() - utterance.timestamp) * 1000.0)
                else:
                    audio_to_vad_ms = 0.0

                self._push_stt_queue_with_backpressure((utterance, t_vad_complete, audio_to_vad_ms))
        except Exception as vad_err:
            if not self._stop_event.is_set():
                self._report_status("degraded", f"VAD segmentation error: {vad_err}")

        # Signal STT worker that VAD stream finished
        while not self._stop_event.is_set():
            try:
                self._stt_queue.put(None, timeout=0.1)
                break
            except Full:
                continue

    def _stt_transcription_loop(self) -> None:
        """Worker Thread 3: Transcribes utterances, dispatches interims immediately, queues finals to MT."""
        while not self._stop_event.is_set():
            try:
                item = self._stt_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                self._stt_queue.task_done()
                break

            utterance, t_vad_complete, audio_to_vad_ms = item

            # If current item is interim and newer audio chunks have already arrived, coalesce directly to the latest
            if not getattr(utterance, "is_final", True):
                while not self._stt_queue.empty():
                    try:
                        newer_item = self._stt_queue.get_nowait()
                        self._stt_queue.task_done()
                        if newer_item is None:
                            # Re-enqueue None so the worker loop terminates after processing current utterance
                            try:
                                self._stt_queue.put_nowait(None)
                            except Full:
                                pass
                            break
                        item = newer_item
                        utterance, t_vad_complete, audio_to_vad_ms = item
                        # Stop coalescing once we reach a committed final segment
                        if getattr(utterance, "is_final", True):
                            break
                    except Empty:
                        break

            if item is None:
                break

            t_pipeline_start = time.perf_counter()

            try:
                t_stt_start = time.perf_counter()
                transcript_segments = self._transcriber.transcribe(utterance)
                t_stt_end = time.perf_counter()
                stt_ms = (t_stt_end - t_stt_start) * 1000.0
            except Exception as stt_err:
                self._report_status("degraded", f"STT failed: {stt_err}")
                self._stt_queue.task_done()
                continue

            is_utterance_final = getattr(utterance, "is_final", True)

            for trans_seg in transcript_segments:
                seg_text = trans_seg.text.strip()
                if not seg_text:
                    continue

                is_segment_final = is_utterance_final and getattr(trans_seg, "is_final", True)

                if not is_segment_final:
                    # Interim: Translate Japanese interim with secondary translator if present
                    t_trans_start = time.perf_counter()
                    if trans_seg.language != "en" and self._translator is not None:
                        ja_text = self._apply_source_substitutions(seg_text)
                        try:
                            en_res = self._translator.translate(ja_text)
                            if hasattr(en_res, "translated_text"):
                                en_text = en_res.translated_text
                            else:
                                en_text = str(en_res)
                        except Exception:
                            en_text = seg_text
                        en_text = self._apply_target_substitutions(en_text)
                    else:
                        en_text = self._apply_target_substitutions(seg_text)
                        ja_text = seg_text if trans_seg.language != "en" else ""
                    t_trans_end = time.perf_counter()
                    interim_trans_ms = (t_trans_end - t_trans_start) * 1000.0

                    interim_caption = CaptionSegment(
                        text=en_text,
                        is_final=False,
                        start_time=trans_seg.start_time,
                        end_time=trans_seg.end_time,
                        language="en" if (self._translator is not None or trans_seg.language == "en") else trans_seg.language,
                        original_text=ja_text,
                    )
                    t_disp_start = time.perf_counter()
                    try:
                        if hasattr(self._presenter, "show_interim"):
                            self._presenter.show_interim(interim_caption)
                        else:
                            self._presenter.present(interim_caption)
                    except Exception:
                        pass
                    t_disp_end = time.perf_counter()
                    disp_ms = (t_disp_end - t_disp_start) * 1000.0

                    total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
                    self._latency_tracker.record(
                        audio_to_vad_ms=audio_to_vad_ms,
                        transcription_ms=stt_ms,
                        translation_ms=interim_trans_ms,
                        display_ms=disp_ms,
                        total_pipeline_ms=total_pipeline_ms,
                        e2e_latency_ms=audio_to_vad_ms + total_pipeline_ms,
                        text_preview=en_text,
                    )
                else:
                    # Final segment
                    if trans_seg.language == "en" or self._translator is None:
                        # Direct translation / no secondary MT needed: commit immediately
                        en_text = self._apply_target_substitutions(seg_text)
                        final_caption = CaptionSegment(
                            text=en_text,
                            is_final=True,
                            start_time=trans_seg.start_time,
                            end_time=trans_seg.end_time,
                            language="en",
                            original_text="",
                        )
                        t_disp_start = time.perf_counter()
                        try:
                            if hasattr(self._presenter, "show_final"):
                                self._presenter.show_final(final_caption)
                            else:
                                self._presenter.present(final_caption)
                        except Exception:
                            pass
                        t_disp_end = time.perf_counter()
                        disp_ms = (t_disp_end - t_disp_start) * 1000.0

                        total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
                        self._latency_tracker.record(
                            audio_to_vad_ms=audio_to_vad_ms,
                            transcription_ms=stt_ms,
                            translation_ms=0.0,
                            display_ms=disp_ms,
                            total_pipeline_ms=total_pipeline_ms,
                            e2e_latency_ms=audio_to_vad_ms + total_pipeline_ms,
                            text_preview=en_text,
                        )
                    else:
                        # Queue to Translation Worker (never drop final segments)
                        while not self._stop_event.is_set():
                            try:
                                self._translation_queue.put(
                                    (trans_seg, utterance, t_pipeline_start, audio_to_vad_ms, stt_ms),
                                    timeout=0.1,
                                )
                                break
                            except Full:
                                continue

            self._stt_queue.task_done()

        # Signal Translation worker
        while not self._stop_event.is_set():
            try:
                self._translation_queue.put(None, timeout=0.1)
                break
            except Full:
                continue

    def _translation_loop(self) -> None:
        """Worker Thread 4: Translates final segments and commits final subtitles to CaptionPresenter."""
        while not self._stop_event.is_set():
            try:
                item = self._translation_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                self._translation_queue.task_done()
                break

            trans_seg, utterance, t_pipeline_start, audio_to_vad_ms, stt_ms = item
            raw_ja_text = trans_seg.text.strip()
            ja_text = self._apply_source_substitutions(raw_ja_text)

            t_trans_start = time.perf_counter()
            try:
                en_res = self._translator.translate(ja_text)
                en_text = en_res.translated_text if hasattr(en_res, "translated_text") else str(en_res)
            except Exception as trans_err:
                self._report_status("degraded", f"Translation error: {trans_err}")
                en_text = f"[JA] {ja_text}"

            en_text = self._apply_target_substitutions(en_text)
            t_trans_end = time.perf_counter()
            translation_ms = (t_trans_end - t_trans_start) * 1000.0

            caption = CaptionSegment(
                text=en_text,
                is_final=True,
                start_time=trans_seg.start_time,
                end_time=trans_seg.end_time,
                language="en",
                original_text=ja_text,
            )

            t_disp_start = time.perf_counter()
            try:
                if hasattr(self._presenter, "show_final"):
                    self._presenter.show_final(caption)
                else:
                    self._presenter.present(caption)
            except Exception as ui_err:
                self._report_status("degraded", f"Presenter error: {ui_err}")
            t_disp_end = time.perf_counter()
            disp_ms = (t_disp_end - t_disp_start) * 1000.0

            total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            e2e_latency_ms = audio_to_vad_ms + total_pipeline_ms

            self._latency_tracker.record(
                audio_to_vad_ms=audio_to_vad_ms,
                transcription_ms=stt_ms,
                translation_ms=translation_ms,
                display_ms=disp_ms,
                total_pipeline_ms=total_pipeline_ms,
                e2e_latency_ms=e2e_latency_ms,
                text_preview=en_text,
            )

            self._translation_queue.task_done()

    def process_stream(self, chunk_stream: Iterator[AudioChunk]) -> None:
        """Feed a chunk stream through the decoupled pipeline or process synchronously."""
        for chunk in chunk_stream:
            if self._stop_event.is_set():
                break
            # If pipeline threads are active, feed to queue
            if self._is_running and self._audio_thread and self._audio_thread.is_alive():
                self._audio_queue.put(chunk)
            else:
                # Direct utterance processing fallback
                self.process_utterance(chunk)

    def start(self, run_in_background: bool = True) -> None:
        """Start the decoupled multi-threaded pipeline."""
        with self._lock:
            if self._is_running:
                return

            self._stop_event.clear()
            self._is_running = True

            # Pre-load Whisper model if supported
            if hasattr(self._transcriber, "ensure_ready"):
                try:
                    self._report_status("connecting", "Loading Whisper model...")
                    self._transcriber.ensure_ready(
                        status_callback=lambda msg: self._report_status("connecting", msg)
                    )
                except Exception as model_err:
                    self._report_status("error", f"Model load failed: {model_err}")
                    self._is_running = False
                    raise

            try:
                self._audio_source.start()
                self._report_status("running", "Pipeline started")
            except Exception as e:
                self._report_status("error", f"Failed to start audio source: {e}")
                if not run_in_background:
                    self._is_running = False
                    raise

            if self._latency_tracker.is_enabled:
                self._latency_tracker.start_reporter()

            # Launch decoupled pipeline threads
            self._translation_thread = threading.Thread(
                target=self._translation_loop,
                name="LiveCaption-TranslationWorker",
                daemon=True,
            )
            self._stt_thread = threading.Thread(
                target=self._stt_loop if hasattr(self, "_stt_loop") else self._stt_transcription_loop,
                name="LiveCaption-STTWorker",
                daemon=True,
            )
            self._vad_thread = threading.Thread(
                target=self._vad_loop if hasattr(self, "_vad_loop") else self._vad_segmentation_loop,
                name="LiveCaption-VADWorker",
                daemon=True,
            )
            self._audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                name="LiveCaption-AudioWorker",
                daemon=True,
            )

            self._translation_thread.start()
            self._stt_thread.start()
            self._vad_thread.start()
            self._audio_thread.start()

            if not run_in_background:
                # Wait until audio stream completes and all pipeline queues are processed
                if self._audio_thread:
                    self._audio_thread.join()
                if self._vad_thread:
                    self._vad_thread.join()
                if self._stt_thread:
                    self._stt_thread.join()
                if self._translation_thread:
                    self._translation_thread.join()
                self.stop()

    def stop(self, timeout: float = 2.0) -> None:
        """Thread-safe clean shutdown of all decoupled pipeline threads and resources."""
        with self._lock:
            if not self._is_running and self._stop_event.is_set():
                return

            self._stop_event.set()
            self._is_running = False
            self._report_status("stopped", "Pipeline stopped cleanly")

            self._latency_tracker.stop_reporter()

            try:
                self._audio_source.stop()
            except Exception:
                pass

            # Push sentinels to wake up sleeping workers
            try:
                self._audio_queue.put_nowait(None)
            except Exception:
                pass
            try:
                self._stt_queue.put_nowait(None)
            except Exception:
                pass
            try:
                self._translation_queue.put_nowait(None)
            except Exception:
                pass

            # Join all worker threads
            threads = [self._audio_thread, self._vad_thread, self._stt_thread, self._translation_thread]
            for t in threads:
                if t is not None and t.is_alive():
                    t.join(timeout=max(0.1, timeout / 4.0))

            self._audio_thread = None
            self._vad_thread = None
            self._stt_thread = None
            self._translation_thread = None
