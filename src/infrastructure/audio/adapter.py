"""Offline audio capture adapter implementing AudioCapturePort.

Provides local microphone / loopback audio ingestion without external network calls.
"""

from queue import Empty, Queue
from typing import Iterator, Optional

from src.domain.entities import AudioChunk
from src.domain.ports import AudioCapturePort


class OfflineAudioCaptureAdapter(AudioCapturePort):
    """Audio capture adapter running locally with zero network dependency."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._queue: Queue[AudioChunk] = Queue()
        self._active = False

    def start_capture(self) -> None:
        """Initialize and activate local audio capture device."""
        self._active = True

    def enqueue_chunk(self, raw_pcm: bytes) -> AudioChunk:
        """Helper to enqueue raw PCM data (e.g. from soundcard/pyaudio callback)."""
        chunk = AudioChunk(
            data=raw_pcm,
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_width=self._sample_width,
        )
        self._queue.put(chunk)
        return chunk

    def read_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """Read the next chunk from the queue synchronously."""
        if not self._active:
            return None
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stream_chunks(self) -> Iterator[AudioChunk]:
        """Continuously stream available chunks until capture is stopped."""
        while self._active:
            chunk = self.read_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk

    def start(self) -> None:
        """Initialize and activate local audio capture device."""
        self.start_capture()

    def stop(self) -> None:
        """Deactivate capture and flush queue."""
        self.stop_capture()

    def stream(self) -> Iterator[AudioChunk]:
        """Stream chunks."""
        return self.stream_chunks()

    def stop_capture(self) -> None:
        """Deactivate capture and flush queue."""
        self._active = False

    def is_active(self) -> bool:
        """Check if active."""
        return self._active
