"""Latency benchmark comparing model configurations and decoding optimizations."""

from pathlib import Path
import sys
import time
import wave

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.latency_tracker import LatencyTracker
from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import AudioChunk
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber


def main() -> None:
    sample_wav = Path("tests/data/sample_japanese.wav")
    with wave.open(str(sample_wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    chunk = AudioChunk(pcm_data=pcm, sample_rate=16000, timestamp=time.time() - 1.5)

    print("\n" + "=" * 65)
    print("  WHISPER LATENCY BENCHMARK (p50 / p95 Comparison)")
    print("=" * 65)

    # 1. Faster-Whisper Small (Default)
    print("\nEvaluating: Faster-Whisper 'small' (INT8 CPU)...")
    t_small = FasterWhisperTranscriber(
        model_size="small",
        device="auto",
        compute_type="auto",
    )
    t_small.ensure_ready()
    tracker_small = LatencyTracker(enabled=False)
    uc_small = LiveCaptionUseCase(transcriber=t_small, latency_tracker=tracker_small)

    # Warmup
    uc_small.process_utterance(chunk)
    # Benchmark runs
    for _ in range(5):
        uc_small.process_utterance(chunk)

    stats_small = tracker_small.get_stage_stats()
    s_small = stats_small["STT (Faster-Whisper)"]
    print(f"  [Small] p50: {s_small.p50:.1f}ms | p95: {s_small.p95:.1f}ms | Min: {s_small.min:.1f}ms | Avg: {s_small.avg:.1f}ms")

    # 2. Faster-Whisper Base (Fallback option for lower-end hardware)
    print("\nEvaluating: Faster-Whisper 'base' (fallback_to_base=True, INT8 CPU)...")
    t_base = FasterWhisperTranscriber(
        model_size="small",
        fallback_to_base=True,
        device="auto",
        compute_type="auto",
    )
    t_base.ensure_ready()
    tracker_base = LatencyTracker(enabled=False)
    uc_base = LiveCaptionUseCase(transcriber=t_base, latency_tracker=tracker_base)

    # Warmup
    uc_base.process_utterance(chunk)
    # Benchmark runs
    for _ in range(5):
        uc_base.process_utterance(chunk)

    stats_base = tracker_base.get_stage_stats()
    s_base = stats_base["STT (Faster-Whisper)"]
    print(f"  [Base Fallback] p50: {s_base.p50:.1f}ms | p95: {s_base.p95:.1f}ms | Min: {s_base.min:.1f}ms | Avg: {s_base.avg:.1f}ms")

    # 3. Faster-Whisper Tiny (Ultra low-latency)
    print("\nEvaluating: Faster-Whisper 'tiny' (INT8 CPU)...")
    t_tiny = FasterWhisperTranscriber(
        model_size="tiny",
        device="auto",
        compute_type="auto",
    )
    t_tiny.ensure_ready()
    tracker_tiny = LatencyTracker(enabled=False)
    uc_tiny = LiveCaptionUseCase(transcriber=t_tiny, latency_tracker=tracker_tiny)

    # Warmup
    uc_tiny.process_utterance(chunk)
    # Benchmark runs
    for _ in range(5):
        uc_tiny.process_utterance(chunk)

    stats_tiny = tracker_tiny.get_stage_stats()
    s_tiny = stats_tiny["STT (Faster-Whisper)"]
    print(f"  [Tiny] p50: {s_tiny.p50:.1f}ms | p95: {s_tiny.p95:.1f}ms | Min: {s_tiny.min:.1f}ms | Avg: {s_tiny.avg:.1f}ms")

    print("\n" + "=" * 65)
    print("  SUMMARY LATENCY COMPARISON TABLE")
    print("=" * 65)
    print(f" {'Configuration':<30} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'Min (ms)':>9} | {'Avg (ms)':>9}")
    print("-" * 65)
    print(f" {'small (Default, INT8)':<30} | {s_small.p50:>9.1f} | {s_small.p95:>9.1f} | {s_small.min:>9.1f} | {s_small.avg:>9.1f}")
    print(f" {'base (Fallback, INT8)':<30} | {s_base.p50:>9.1f} | {s_base.p95:>9.1f} | {s_base.min:>9.1f} | {s_base.avg:>9.1f}")
    print(f" {'tiny (Ultra-low latency)':<30} | {s_tiny.p50:>9.1f} | {s_tiny.p95:>9.1f} | {s_tiny.min:>9.1f} | {s_tiny.avg:>9.1f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
