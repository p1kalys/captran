"""Decoupled Pipeline End-to-End Latency Benchmark."""

from pathlib import Path
import sys
import threading
import time
import wave

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.latency_tracker import LatencyTracker
from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import AudioChunk, CaptionSegment
from src.domain.ports import AudioSource, CaptionPresenter, SpeechSegmenter
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber


class MockAudioSource(AudioSource):
    def __init__(self, pcm_data: bytes, chunk_duration_ms: int = 200, count: int = 5):
        self._pcm = pcm_data
        self._bytes_per_sec = 16000 * 2
        self._chunk_size = int(self._bytes_per_sec * (chunk_duration_ms / 1000.0))
        self._running = False
        self._count = count

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def stream(self):
        for _ in range(self._count):
            if not self._running:
                break
            for offset in range(0, len(self._pcm), self._chunk_size):
                if not self._running:
                    break
                chunk = self._pcm[offset : offset + self._chunk_size]
                yield AudioChunk(pcm_data=chunk, sample_rate=16000, timestamp=time.time())
                time.sleep(0.05)


class MockSegmenter(SpeechSegmenter):
    def __init__(self, pcm_data: bytes):
        self._pcm = pcm_data

    def segment(self, chunk_stream):
        accumulated = b""
        t0 = time.time()
        for chunk in chunk_stream:
            accumulated += chunk.pcm_data
            # emit rolling interim every ~500ms
            if len(accumulated) >= 16000:
                yield AudioChunk(pcm_data=accumulated, sample_rate=16000, timestamp=t0, is_final=False)
            # emit final when buffer reaches full utterance
            if len(accumulated) >= len(self._pcm):
                yield AudioChunk(pcm_data=accumulated, sample_rate=16000, timestamp=t0, is_final=True)
                accumulated = b""
                t0 = time.time()


class BenchmarkPresenter(CaptionPresenter):
    def __init__(self):
        self.interims = []
        self.finals = []
        self._lock = threading.Lock()

    def present(self, caption: CaptionSegment) -> None:
        with self._lock:
            if caption.is_final:
                self.finals.append((time.time(), caption))
            else:
                self.interims.append((time.time(), caption))

    def show_interim(self, caption: CaptionSegment) -> None:
        with self._lock:
            self.interims.append((time.time(), caption))

    def show_final(self, caption: CaptionSegment) -> None:
        with self._lock:
            self.finals.append((time.time(), caption))


def run_decoupled_benchmark():
    sample_wav = Path("tests/data/sample_japanese.wav")
    with wave.open(str(sample_wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    print("\n" + "=" * 70)
    print("  DECOUPLED MULTI-THREADED PIPELINE LATENCY BENCHMARK")
    print("=" * 70, flush=True)

    print("\n[1/3] Loading Faster-Whisper (Small, INT8 CPU)...", flush=True)
    stt = FasterWhisperTranscriber(model_size="small", device="auto", compute_type="auto")
    stt.ensure_ready()

    tracker = LatencyTracker(enabled=True)
    presenter = BenchmarkPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=MockAudioSource(pcm, chunk_duration_ms=200, count=5),
        segmenter=MockSegmenter(pcm),
        transcriber=stt,
        presenter=presenter,
        latency_tracker=tracker,
    )

    print("[2/3] Warming up pipeline...", flush=True)
    warmup_chunk = AudioChunk(pcm_data=pcm, sample_rate=16000, timestamp=time.time(), is_final=True)
    use_case.process_utterance(warmup_chunk)

    print("[3/3] Running asynchronous multi-threaded pipeline with backpressure...", flush=True)
    # Start decoupled worker threads
    use_case.start()

    # Wait for the mock audio stream to be processed through the threaded pipeline
    time.sleep(12.0)

    # Stop pipeline
    use_case.stop()

    summary = tracker.format_summary_table()
    print("\n" + summary + "\n", flush=True)


if __name__ == "__main__":
    run_decoupled_benchmark()
