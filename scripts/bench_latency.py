"""Comprehensive Latency Benchmark across Multilingual Models, Direct Translation, and Pivot-Through-English Translation."""

from pathlib import Path
import sys
import time
import wave

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.application.latency_tracker import LatencyTracker
from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import AudioChunk, Language
from src.infrastructure.stt.faster_whisper_stt import FasterWhisperTranscriber
from src.infrastructure.translation.pivoting_translator import PivotingTranslator


def load_audio_chunk(wav_path: Path) -> AudioChunk:
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"Sample WAV must be 16kHz mono 16-bit (got {wf.getframerate()}Hz, "
                f"{wf.getnchannels()} channels, {wf.getsampwidth()} sample width)"
            )
        pcm = wf.readframes(wf.getnframes())
    return AudioChunk(pcm_data=pcm, sample_rate=16000, timestamp=time.time() - 1.5)


def main() -> None:
    ja_chunk = load_audio_chunk(Path("tests/data/sample_japanese.wav"))
    es_chunk = load_audio_chunk(Path("tests/data/sample_spanish.wav"))

    print("\n" + "=" * 80)
    print("  MULTILINGUAL LATENCY INSTRUMENTATION BENCHMARK (p50 / p95)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Faster-Whisper Model Latency (tiny, base, small, medium)
    # -------------------------------------------------------------------------
    print("\n>>> Phase 1: Faster-Whisper Transcription Latency Comparison")
    models_to_test = ["tiny", "base", "small", "medium"]
    model_stats = {}

    for m_size in models_to_test:
        print(f"  Evaluating Faster-Whisper '{m_size}' (INT8 CPU)...")
        try:
            t = FasterWhisperTranscriber(model_size=m_size, device="auto", compute_type="auto")
            t.ensure_ready()
            tracker = LatencyTracker(enabled=True, print_callback=lambda _: None)
            uc = LiveCaptionUseCase(
                transcriber=t,
                source_language=Language.JAPANESE,
                target_language=Language.JAPANESE,
                latency_tracker=tracker,
            )
            # Warmup
            uc.process_utterance(ja_chunk)
            tracker.reset()
            # 5 measurement runs
            for _ in range(5):
                uc.process_utterance(ja_chunk)
            stt_stat = tracker.get_stage_stats()["STT (Faster-Whisper)"]
            model_stats[m_size] = stt_stat
            print(f"    p50: {stt_stat.p50:.1f}ms | p95: {stt_stat.p95:.1f}ms | Avg: {stt_stat.avg:.1f}ms")
        except Exception as e:
            print(f"    [SKIP] Could not benchmark {m_size}: {e}")

    # -------------------------------------------------------------------------
    # 2. Translation Stage Latency: Direct Pair vs Pivot-Through-English
    # -------------------------------------------------------------------------
    print("\n>>> Phase 2: Translation Latency (Direct Pair vs Pivot-Through-English)")
    translator = PivotingTranslator()

    test_sentences = [
        ("Direct (JA -> EN)", "これはライブ字幕のテストです。", Language.JAPANESE, Language.ENGLISH),
        ("Pivot  (JA -> ES: 2-Hop)", "これはライブ字幕のテストです。", Language.JAPANESE, Language.SPANISH),
        ("Pivot  (HI -> FR: 2-Hop)", "यह लाइव कैप्शन का एक परीक्षण है।", Language.HINDI, Language.FRENCH),
    ]

    translation_stats = {}
    for label, text, src_lang, tgt_lang in test_sentences:
        times_ms = []
        failed_count = 0

        for _ in range(10):
            t0 = time.perf_counter()
            try:
                translator.translate(text, src_lang, tgt_lang)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
            except Exception:
                failed_count += 1

        if times_ms:
            times_ms.sort()
            p50 = times_ms[len(times_ms) // 2]
            p95 = times_ms[int(len(times_ms) * 0.95)]
            avg = sum(times_ms) / len(times_ms)
            translation_stats[label] = (p50, p95, avg)
            print(f"  {label:<30} | p50: {p50:6.1f}ms | p95: {p95:6.1f}ms | Avg: {avg:6.1f}ms (samples: {len(times_ms)})")
        else:
            translation_stats[label] = None
            print(f"  {label:<30} | [UNAVAILABLE] Model packages not installed locally ({failed_count} failed attempts)")

    # -------------------------------------------------------------------------
    # 3. End-to-End Pipeline Evaluation across Matrix
    # -------------------------------------------------------------------------
    print("\n>>> Phase 3: End-to-End Pipeline Latency (VAD + STT + MT + Display)")
    matrix_cases = [
        ("small + Direct (JA->EN)", "small", Language.JAPANESE, Language.ENGLISH, ja_chunk),
        ("small + Pivot  (JA->ES)", "small", Language.JAPANESE, Language.SPANISH, ja_chunk),
        ("medium + Direct (JA->EN)", "medium", Language.JAPANESE, Language.ENGLISH, ja_chunk),
        ("medium + Pivot  (JA->ES)", "medium", Language.JAPANESE, Language.SPANISH, ja_chunk),
    ]

    pipeline_results = []

    for name, model_size, src_l, tgt_l, chunk in matrix_cases:
        try:
            t = FasterWhisperTranscriber(model_size=model_size, device="auto", compute_type="auto")
            t.ensure_ready()
            tracker = LatencyTracker(enabled=True, print_callback=lambda _: None)
            uc = LiveCaptionUseCase(
                transcriber=t,
                translator=translator,
                source_language=src_l,
                target_language=tgt_l,
                latency_tracker=tracker,
            )
            # Warmup
            uc.process_utterance(chunk)
            tracker.reset()

            for _ in range(5):
                uc.process_utterance(chunk)

            stages = tracker.get_stage_stats()
            stt_p50 = stages.get("STT (Faster-Whisper)", stages.get("transcription", None))
            mt_p50 = stages.get("Translation", stages.get("translation", None))
            e2e = stages.get("End-to-End Latency", None)

            pipeline_results.append({
                "case": name,
                "stt_p50": stt_p50.p50 if stt_p50 else 0.0,
                "mt_p50": mt_p50.p50 if mt_p50 else None,
                "e2e_p50": e2e.p50 if e2e else 0.0,
                "e2e_p95": e2e.p95 if e2e else 0.0,
                "e2e_avg": e2e.avg if e2e else 0.0,
            })
        except Exception as err:
            print(f"  [SKIP] Could not run pipeline case '{name}': {err}")

    # -------------------------------------------------------------------------
    # Summary Tables & Findings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUMMARY: END-TO-END PIPELINE LATENCY MATRIX")
    print("=" * 80)
    print(f" {'Configuration Case':<28} | {'STT p50':>9} | {'MT p50':>9} | {'E2E p50':>9} | {'E2E p95':>9}")
    print("-" * 80)
    for res in pipeline_results:
        mt_str = f"{res['mt_p50']:>7.1f}ms" if res['mt_p50'] is not None else "     N/A "
        print(
            f" {res['case']:<28} | {res['stt_p50']:>7.1f}ms | {mt_str} | "
            f"{res['e2e_p50']:>7.1f}ms | {res['e2e_p95']:>7.1f}ms"
        )
    print("=" * 80)

    # UI Indicator Assessment
    print("\n>>> UI Indicator Assessment for Pivot-Through-English Paths:")
    pivot_extra_latencies = [
        res["mt_p50"] for res in pipeline_results if "Pivot" in res["case"] and res["mt_p50"] is not None
    ]
    if pivot_extra_latencies:
        avg_pivot_overhead = sum(pivot_extra_latencies) / len(pivot_extra_latencies)
        print(f"  - Measured average pivot translation stage latency: {avg_pivot_overhead:.1f}ms")
        if avg_pivot_overhead > 350.0:
            print("  - [FLAG]: Measured average pivot translation latency exceeds 350ms. A subtle 'translating...' UI indicator is RECOMMENDED.")
        else:
            print(f"  - [FLAG]: Measured average pivot translation latency ({avg_pivot_overhead:.1f}ms) does not exceed 350ms.")
            if avg_pivot_overhead < 100.0:
                print("    Measured translation latency is below 100ms. A dedicated 'translating...' spinner/state is NOT necessary.")
    else:
        print("  - [INFO]: Measured pivot translation latency statistics are unavailable.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

