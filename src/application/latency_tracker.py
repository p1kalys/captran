"""Latency tracking and timing instrumentation for the LiveCaptionUseCase pipeline.

Captures stage durations:
1. Audio chunk received -> VAD segment complete
2. Transcription complete
3. Translation complete
4. Caption displayed
5. Total Pipeline & End-to-End latency

Supports:
- Per-utterance latency logging
- Thread-safe aggregation of metrics
- Calculation of p50 and p95 percentiles per stage
- Running summary reporting every 30 seconds
"""

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("captran.latency")


def calculate_percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile (0-100) of a list of floats using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


@dataclass
class UtteranceLatency:
    """Latency metrics recorded for a single utterance."""

    utterance_index: int
    audio_to_vad_ms: float
    transcription_ms: float
    translation_ms: float
    display_ms: float
    total_pipeline_ms: float
    e2e_latency_ms: float
    text_preview: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageStats:
    """Statistical summary for a single pipeline stage."""

    count: int
    p50: float
    p95: float
    min: float
    max: float
    avg: float


class LatencyTracker:
    """Collects per-utterance stage latencies and generates periodic p50/p95 reports."""

    def __init__(
        self,
        enabled: bool = False,
        report_interval_s: float = 30.0,
        print_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._enabled = enabled
        self._report_interval_s = max(0.01, report_interval_s)
        self._print_callback = print_callback or print
        self._lock = threading.Lock()
        self._records: List[UtteranceLatency] = []
        self._utterance_counter = 0

        self._reporter_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_report_time = time.perf_counter()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def record(
        self,
        audio_to_vad_ms: float,
        transcription_ms: float,
        translation_ms: float,
        display_ms: float,
        total_pipeline_ms: float,
        e2e_latency_ms: float,
        text_preview: str = "",
    ) -> UtteranceLatency:
        """Record timing metrics for an utterance and log the breakdown."""
        with self._lock:
            self._utterance_counter += 1
            record = UtteranceLatency(
                utterance_index=self._utterance_counter,
                audio_to_vad_ms=audio_to_vad_ms,
                transcription_ms=transcription_ms,
                translation_ms=translation_ms,
                display_ms=display_ms,
                total_pipeline_ms=total_pipeline_ms,
                e2e_latency_ms=e2e_latency_ms,
                text_preview=text_preview,
            )
            self._records.append(record)

        # Log per-stage duration for this utterance
        preview = f" ('{text_preview[:35]}...')" if text_preview else ""
        log_msg = (
            f"[LATENCY] Utterance #{record.utterance_index}{preview}: "
            f"Audio->VAD={audio_to_vad_ms:.1f}ms | "
            f"STT={transcription_ms:.1f}ms | "
            f"Translation={translation_ms:.1f}ms | "
            f"Display={display_ms:.1f}ms | "
            f"Pipeline={total_pipeline_ms:.1f}ms | "
            f"E2E={e2e_latency_ms:.1f}ms"
        )
        logger.info(log_msg)

        if self._enabled:
            # Also print to stdout if debug-latency is actively enabled
            self._print_callback(log_msg)

        return record

    def get_stage_stats(self) -> Dict[str, StageStats]:
        """Compute count, p50, p95, min, max, and avg for each stage."""
        with self._lock:
            records = list(self._records)

        stages: Dict[str, List[float]] = {
            "Audio -> VAD": [r.audio_to_vad_ms for r in records],
            "STT (Faster-Whisper)": [r.transcription_ms for r in records],
            "Translation": [r.translation_ms for r in records],
            "Caption Display": [r.display_ms for r in records],
            "Total Pipeline": [r.total_pipeline_ms for r in records],
            "End-to-End Latency": [r.e2e_latency_ms for r in records],
        }

        result: Dict[str, StageStats] = {}
        for name, values in stages.items():
            if not values:
                result[name] = StageStats(count=0, p50=0.0, p95=0.0, min=0.0, max=0.0, avg=0.0)
            else:
                result[name] = StageStats(
                    count=len(values),
                    p50=calculate_percentile(values, 50.0),
                    p95=calculate_percentile(values, 95.0),
                    min=min(values),
                    max=max(values),
                    avg=sum(values) / len(values),
                )
        return result

    def format_summary_table(self) -> str:
        """Generate a formatted ASCII table of stage latencies."""
        stats = self.get_stage_stats()
        sample_count = len(self._records)

        lines = [
            "",
            "=" * 82,
            f"  CAPTRAN LATENCY SUMMARY (Samples: {sample_count})",
            "=" * 82,
            f" {'Stage':<25} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'Min (ms)':>9} | {'Max (ms)':>9} | {'Avg (ms)':>9}",
            "-" * 82,
        ]

        for stage_name, s in stats.items():
            if stage_name == "Total Pipeline":
                lines.append("-" * 82)
            lines.append(
                f" {stage_name:<25} | {s.p50:>9.1f} | {s.p95:>9.1f} | {s.min:>9.1f} | {s.max:>9.1f} | {s.avg:>9.1f}"
            )

        lines.append("=" * 82)
        lines.append("")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the current latency summary table."""
        with self._lock:
            count = len(self._records)
        if count == 0:
            return
        table = self.format_summary_table()
        self._print_callback(table)

    def _reporter_loop(self) -> None:
        """Background thread loop emitting summary every report_interval_s."""
        while not self._stop_event.wait(timeout=self._report_interval_s):
            if self._stop_event.is_set():
                break
            if self._enabled:
                self.print_summary()

    def start_reporter(self) -> None:
        """Start the periodic summary reporter background thread."""
        if not self._enabled:
            return
        if self._reporter_thread is not None and self._reporter_thread.is_alive():
            return
        self._stop_event.clear()
        self._reporter_thread = threading.Thread(
            target=self._reporter_loop,
            name="LatencyTrackerReporter",
            daemon=True,
        )
        self._reporter_thread.start()

    def stop_reporter(self, print_final_summary: bool = True) -> None:
        """Stop the reporter thread and optionally print final summary."""
        self._stop_event.set()
        if self._reporter_thread is not None and self._reporter_thread.is_alive():
            self._reporter_thread.join(timeout=1.0)
            self._reporter_thread = None
        if self._enabled and print_final_summary:
            self.print_summary()

    def reset(self) -> None:
        """Reset all recorded metrics."""
        with self._lock:
            self._records.clear()
            self._utterance_counter = 0
            self._last_report_time = time.perf_counter()
