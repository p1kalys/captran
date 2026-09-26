"""Test script for Windows WASAPI Loopback Audio Capture.

Captures 10 seconds of system/meeting audio and saves to a 16kHz mono WAV file.
Displays real-time RMS volume meter to confirm it's capturing active speaker sound.
"""

import math
from pathlib import Path
import struct
import sys
import time
import wave
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infrastructure.audio.windows_loopback import WindowsLoopbackAudioSource


def calculate_rms_db(pcm_data: bytes) -> float:
    """Calculate RMS volume in dBFS for a 16-bit mono PCM chunk."""
    if not pcm_data:
        return -100.0
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    if rms <= 0:
        return -100.0
    return 20 * math.log10(rms / 32768.0)


def draw_volume_bar(db: float, width: int = 30) -> str:
    """Render a terminal volume meter bar."""
    if db < -60.0:
        filled = 0
    else:
        # Scale from -60 dB to 0 dB
        normalized = max(0.0, min(1.0, (db + 60.0) / 60.0))
        filled = int(normalized * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {db:6.1f} dB"


def main():
    print("=" * 60)
    print(" Windows WASAPI Loopback Capture Test (10 Seconds)")
    print("=" * 60)

    try:
        devices = WindowsLoopbackAudioSource.list_loopback_devices()
    except Exception as e:
        print(f"\n[ERROR] Failed to list loopback devices: {e}")
        print("\nPlease ensure pyaudiowpatch is installed:")
        print("    pip install pyaudiowpatch")
        sys.exit(1)

    print(f"\nFound {len(devices)} WASAPI loopback device(s):")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']} ch, {dev['defaultSampleRate']} Hz)")

    # Default to auto-detect default speaker loopback
    selected_idx = None
    if len(sys.argv) > 1:
        try:
            selected_idx = int(sys.argv[1])
            print(f"\nUsing specified device index: {selected_idx}")
        except ValueError:
            pass

    output_wav_path = Path("output_loopback_test.wav").resolve()
    print(f"\nInitializing WindowsLoopbackAudioSource (16kHz mono)...")
    source = WindowsLoopbackAudioSource(
        device_index=selected_idx,
        chunk_duration_ms=150,
        target_sample_rate=16000,
    )

    source.start()
    dev_info = source.selected_device_info
    if dev_info:
        print(f"Capturing from: {dev_info.get('name', 'Unknown')}")
        print(f"Native Source: {dev_info.get('maxInputChannels')} channels @ {dev_info.get('defaultSampleRate')} Hz")

    print("\n>>> RECORDING FOR 10 SECONDS... (Play some YouTube/video/meeting audio now!) <<<")
    print("-" * 60)

    captured_chunks = []
    start_time = time.time()
    last_print = 0.0

    try:
        for chunk in source.stream():
            captured_chunks.append(chunk)
            elapsed = time.time() - start_time

            # Update live volume meter every 100ms
            if time.time() - last_print > 0.1:
                db = calculate_rms_db(chunk.pcm_data)
                bar = draw_volume_bar(db)
                print(f"\rTime: {elapsed:4.1f}s / 10.0s | Vol: {bar}", end="", flush=True)
                last_print = time.time()

            if elapsed >= 10.0:
                break
    finally:
        source.stop()

    print("\n" + "-" * 60)
    print("Capture complete! Processing saved audio...")

    if not captured_chunks:
        print("[WARNING] No audio chunks received.")
        sys.exit(1)

    # Combine all PCM data
    all_pcm = b"".join(c.pcm_data for c in captured_chunks)
    total_duration = len(all_pcm) / (16000 * 2)

    # Save to WAV file
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(all_pcm)

    # Compute overall statistics
    all_samples = np.frombuffer(all_pcm, dtype=np.int16).astype(np.float32)
    overall_rms = np.sqrt(np.mean(all_samples**2))
    overall_db = 20 * math.log10(overall_rms / 32768.0) if overall_rms > 0 else -100.0
    max_peak = np.max(np.abs(all_samples)) / 32768.0

    print(f"\nSaved WAV File: {output_wav_path}")
    print(f"Duration:       {total_duration:.2f} seconds")
    print(f"Chunks Count:   {len(captured_chunks)}")
    print(f"Format:         16000 Hz, 16-bit, Mono PCM")
    print(f"Overall RMS:    {overall_db:.1f} dBFS")
    print(f"Peak Level:     {max_peak * 100:.1f}%")

    if overall_db < -55.0:
        print("\n[NOTE] Audio level was very low/silent. Make sure audio was playing through your speakers!")
    else:
        print("\n[SUCCESS] Audio detected! Open and play 'output_loopback_test.wav' to verify meeting audio.")


if __name__ == "__main__":
    main()
