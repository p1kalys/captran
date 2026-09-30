"""Main entry point for CapTran (Offline Japanese-to-English Live Captioner).

Defaults to launching the modern PySide6 GUI application (Control Panel + Draggable Overlay).
Supports --cli for headless terminal execution.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.infrastructure.audio.factory import AudioSourceFactory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CapTran: Offline Japanese-to-English Live Captioner (Hexagonal Architecture)"
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in terminal CLI mode instead of GUI mode",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available WASAPI loopback audio capture devices and exit",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Audio capture device index (auto-detects default speaker loopback if omitted)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "base", "small"],
        help="Whisper model size for offline Japanese STT (default: small, options: tiny, base, small)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for Whisper (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["default", "int8", "float16", "float32"],
        help="Quantization / compute type (default: int8 for fast low-latency CPU inference)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Whisper beam size (default: 1 for fast greedy real-time decoding)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.4,
        help="Silero VAD speech probability threshold (default: 0.4)",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=250.0,
        help="VAD silence timeout in milliseconds to close an utterance (default: 250.0)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper engine task: 'transcribe' for Japanese speech transcription + English translation (supports side-by-side dual subtitles), or 'translate' for direct English translation (default: transcribe)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Hide timestamps in CLI output",
    )
    parser.add_argument(
        "--debug-latency",
        action="store_true",
        help="Enable diagnostic latency timing instrumentation and print running p50/p95 stage summaries every 30s",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        default="ja",
        choices=["hi", "ja", "en", "es", "fr", "de", "ko"],
        help="Source spoken language (default: ja, options: hi, ja, en, es, fr, de, ko)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="en",
        choices=["hi", "ja", "en", "es", "fr", "de", "ko"],
        help="Target caption language (default: en, options: hi, ja, en, es, fr, de, ko)",
    )
    parser.add_argument(
        "--fallback-to-base",
        action="store_true",
        help="Fall back to the 'base' Whisper model for lower-end hardware",
    )

    args = parser.parse_args()

    # Handle --list-devices
    if args.list_devices:
        devices = AudioSourceFactory.list_available_devices()
        print(f"\nAvailable Loopback Devices ({len(devices)} found):")
        if not devices:
            print("  No loopback devices found. Ensure pyaudiowpatch is installed on Windows.")
        for dev in devices:
            print(f"  [{dev['index']}] {dev['name']} ({dev['channels']} ch, {dev['defaultSampleRate']} Hz)")
        return

    # CLI Terminal Mode
    if args.cli:
        from src.composition_root_cli import build_cli_application, run_cli_captioner

        try:
            app = build_cli_application(
                audio_device_index=args.device_index,
                whisper_model_size=args.model_size,
                whisper_device=args.device,
                whisper_compute_type=args.compute_type,
                whisper_task=args.task,
                whisper_beam_size=args.beam_size,
                source_language=args.source_language,
                target_language=args.target_language,
                caption_prefix=f"[{args.target_language.upper()}] ",
                vad_threshold=args.vad_threshold,
                vad_silence_timeout_ms=args.silence_timeout,
                show_timestamps=not args.no_timestamps,
                debug_latency=args.debug_latency,
                fallback_to_base=args.fallback_to_base,
            )
            run_cli_captioner(app)
        except Exception as e:
            print(f"\n[ERROR] Failed to start live captioner: {e}", file=sys.stderr)
            print("\nPlease ensure model prerequisites are met:", file=sys.stderr)
            print("  1. python scripts/setup_models.py", file=sys.stderr)
            print("  2. pip install -e .", file=sys.stderr)
            sys.exit(1)
        return

    # Default Mode: Launch PySide6 GUI (Control Window + Overlay)
    try:
        from src.composition_root_gui import run_gui_application

        sys.exit(
            run_gui_application(
                default_model_size=args.model_size,
                vad_threshold=args.vad_threshold,
                vad_silence_timeout_ms=args.silence_timeout,
                debug_latency=args.debug_latency,
            )
        )
    except ImportError as e:
        print(f"\n[WARNING] PySide6 GUI could not be initialized ({e}).")
        print("Falling back to CLI terminal mode...\n")
        from src.composition_root_cli import build_cli_application, run_cli_captioner

        app = build_cli_application(
            audio_device_index=args.device_index,
            whisper_model_size=args.model_size,
            whisper_device=args.device,
            whisper_compute_type=args.compute_type,
            vad_threshold=args.vad_threshold,
            vad_silence_timeout_ms=args.silence_timeout,
            show_timestamps=not args.no_timestamps,
            debug_latency=args.debug_latency,
            fallback_to_base=args.fallback_to_base,
        )
        run_cli_captioner(app)


if __name__ == "__main__":
    main()
