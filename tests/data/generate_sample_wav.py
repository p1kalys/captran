"""Helper script to generate bundled synthetic speech wav file for VAD unit testing."""

import math
from pathlib import Path
import struct
import wave


def generate_synthetic_wav(output_path: str = "tests/data/sample_two_utterances.wav") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    # Structure:
    # 0.0 - 0.3s : Silence
    # 0.3 - 1.3s : Utterance 1 (Voice-like harmonic signal)
    # 1.3 - 2.3s : Silence (1.0s gap > silence_timeout)
    # 2.3 - 3.3s : Utterance 2 (Voice-like harmonic signal)
    # 3.3 - 4.0s : Silence

    total_duration = 4.0
    total_samples = int(total_duration * sample_rate)
    samples = []

    for i in range(total_samples):
        t = i / sample_rate
        val = 0.0

        # Utterance 1: 0.3s to 1.3s
        if 0.3 <= t < 1.3:
            # Multi-harmonic voice simulator
            f0 = 200.0  # Fundamental pitch
            val = (
                0.5 * math.sin(2 * math.pi * f0 * t)
                + 0.3 * math.sin(2 * math.pi * f0 * 3 * t)
                + 0.2 * math.sin(2 * math.pi * f0 * 5 * t)
            )
        # Utterance 2: 2.3s to 3.3s
        elif 2.3 <= t < 3.3:
            f0 = 250.0
            val = (
                0.5 * math.sin(2 * math.pi * f0 * t)
                + 0.3 * math.sin(2 * math.pi * f0 * 3 * t)
                + 0.2 * math.sin(2 * math.pi * f0 * 5 * t)
            )

        # Scale to 16-bit signed integer
        int16_val = int(max(-1.0, min(1.0, val)) * 30000)
        samples.append(int16_val)

    pcm_data = struct.pack(f"<{len(samples)}h", *samples)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return str(path)


if __name__ == "__main__":
    out = generate_synthetic_wav()
    print(f"Generated sample wav at: {out}")
