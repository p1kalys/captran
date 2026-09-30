"""Helper script to generate bundled synthetic speech wav file for VAD unit testing."""

import math
from pathlib import Path
import struct
import wave


def generate_synthetic_wav(
    output_path: str = "tests/data/sample_two_utterances.wav",
    duration_s: float = 2.0,
    pitch_f0: float = 200.0,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    total_samples = int(duration_s * sample_rate)
    samples = []

    for i in range(total_samples):
        t = i / sample_rate
        val = 0.0

        # Voice-like harmonic signal
        if 0.2 <= t < (duration_s - 0.2):
            val = (
                0.5 * math.sin(2 * math.pi * pitch_f0 * t)
                + 0.3 * math.sin(2 * math.pi * pitch_f0 * 2 * t)
                + 0.2 * math.sin(2 * math.pi * pitch_f0 * 3 * t)
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


def generate_all_language_samples() -> dict:
    """Generate bundled sample WAV files for all 7 supported languages."""
    languages_config = {
        "hindi": ("tests/data/sample_hindi.wav", 180.0),
        "japanese": ("tests/data/sample_japanese.wav", 220.0),
        "english": ("tests/data/sample_english.wav", 160.0),
        "spanish": ("tests/data/sample_spanish.wav", 190.0),
        "french": ("tests/data/sample_french.wav", 210.0),
        "german": ("tests/data/sample_german.wav", 150.0),
        "korean": ("tests/data/sample_korean.wav", 230.0),
    }

    generated = {}
    for lang, (wav_path, f0) in languages_config.items():
        out = generate_synthetic_wav(output_path=wav_path, duration_s=1.5, pitch_f0=f0)
        generated[lang] = out

    # Also ensure two_utterances sample is generated
    generated["two_utterances"] = generate_synthetic_wav(
        output_path="tests/data/sample_two_utterances.wav", duration_s=4.0, pitch_f0=200.0
    )
    return generated


if __name__ == "__main__":
    results = generate_all_language_samples()
    for name, p in results.items():
        print(f"Generated {name} sample wav at: {p}")

