"""Integration test suite for the full CapTran live captioning pipeline with bundled audio sample clips.

Tests representative multilingual language pairs among all 7 supported languages
(hi, ja, en, es, fr, de, ko) — guaranteeing every language is exercised as source at least once
and as target at least once, verifying non-empty captions and correct routing (direct vs. pivot).
"""

from pathlib import Path
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import wave
import pytest

from src.application.live_caption_use_case import LiveCaptionUseCase
from src.domain.entities import (
    AudioChunk,
    CaptionSegment,
    Language,
    TranscriptSegment,
)
from src.domain.ports import (
    AudioSource,
    CaptionPresenter,
    SpeechSegmenter,
    Transcriber,
)
from src.infrastructure.translation.pivoting_translator import PivotingTranslator
from tests.data.generate_sample_wav import generate_all_language_samples


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_wav_files() -> None:
    """Ensure bundled sample WAV files exist for all 7 languages on clean checkouts."""
    generate_all_language_samples()


# =============================================================================
# Bundled Clip Audio Source Adapter
# =============================================================================

class BundledWavClipAudioSource(AudioSource):
    """Audio source adapter that streams audio from a bundled WAV file in chunks."""

    def __init__(self, wav_path: Path, chunk_duration_ms: int = 100) -> None:
        self.wav_path = wav_path
        self.chunk_duration_ms = chunk_duration_ms
        self.is_started = False
        self.is_stopped = False

    def start(self) -> None:
        self.is_started = True
        self.is_stopped = False

    def stop(self) -> None:
        self.is_stopped = True

    def stream(self) -> Iterator[AudioChunk]:
        if not self.wav_path.exists():
            raise FileNotFoundError(f"Bundled audio clip not found: {self.wav_path}")

        with wave.open(str(self.wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            chunk_samples = int(sample_rate * (self.chunk_duration_ms / 1000.0))

            while not self.is_stopped:
                pcm_data = wf.readframes(chunk_samples)
                if not pcm_data:
                    break
                yield AudioChunk(
                    pcm_data=pcm_data,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    timestamp=time.time(),
                )


# =============================================================================
# Realistic Multilingual Speech Segmenter & Transcriber for Integration Testing
# =============================================================================

SAMPLE_TRANSCRIPTS: Dict[Language, str] = {
    Language.HINDI: "नमस्ते, आज की बैठक में आपका स्वागत है।",
    Language.JAPANESE: "こんにちは、本日の会議を始めます。",
    Language.ENGLISH: "Hello everyone, welcome to today's meeting.",
    Language.SPANISH: "Hola a todos, bienvenidos a la reunión de hoy.",
    Language.FRENCH: "Bonjour à tous, bienvenue à la réunion d'aujourd'hui.",
    Language.GERMAN: "Guten Tag, herzlich willkommen zu unserer Besprechung.",
    Language.KOREAN: "안녕하세요, 오늘 회의에 참석해 주셔서 감사합니다.",
}

SAMPLE_WAV_FILES: Dict[Language, str] = {
    Language.HINDI: "tests/data/sample_hindi.wav",
    Language.JAPANESE: "tests/data/sample_japanese.wav",
    Language.ENGLISH: "tests/data/sample_english.wav",
    Language.SPANISH: "tests/data/sample_spanish.wav",
    Language.FRENCH: "tests/data/sample_french.wav",
    Language.GERMAN: "tests/data/sample_german.wav",
    Language.KOREAN: "tests/data/sample_korean.wav",
}


class MultilingualAudioSegmenter(SpeechSegmenter):
    """Segmenter that aggregates incoming PCM chunks into a complete utterance segment."""

    def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
        accumulated_pcm = bytearray()
        sample_rate = 16000
        channels = 1
        sample_width = 2
        start_ts = time.time()

        for chunk in chunk_stream:
            accumulated_pcm.extend(chunk.pcm_data)
            sample_rate = chunk.sample_rate
            channels = chunk.channels
            sample_width = chunk.sample_width

        if accumulated_pcm:
            yield AudioChunk(
                pcm_data=bytes(accumulated_pcm),
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
                timestamp=start_ts,
                is_final=True,
            )


class MultilingualTranscriber(Transcriber):
    """Transcriber port that transcribes bundled audio clips into realistic source text."""

    def __init__(self) -> None:
        self.transcribed_calls: List[Tuple[Language, int]] = []

    def transcribe(
        self,
        audio: AudioChunk,
        source_language: Language = Language.JAPANESE,
    ) -> Sequence[TranscriptSegment]:
        self.transcribed_calls.append((source_language, len(audio.pcm_data)))
        text = SAMPLE_TRANSCRIPTS.get(
            source_language, f"[Speech in {source_language.value}]"
        )
        return [
            TranscriptSegment(
                text=text,
                is_final=True,
                language=source_language,
                start_time=0.0,
                end_time=1.5,
            )
        ]


# =============================================================================
# Output Capturing Presenter
# =============================================================================

class RecordingCaptionPresenter(CaptionPresenter):
    """Captures all emitted captions for assertions."""

    def __init__(self) -> None:
        self.captions: List[CaptionSegment] = []

    def present(self, caption: CaptionSegment) -> None:
        self.captions.append(caption)

    @property
    def final_captions(self) -> List[CaptionSegment]:
        return [c for c in self.captions if c.is_final]


# =============================================================================
# Translation Model Mock Helper for Route Verification
# =============================================================================

class TrackedMockModel:
    """Mock translation model that tracks translation calls and generates predictable text."""

    def __init__(self, from_code: str, to_code: str) -> None:
        self.from_code = from_code
        self.to_code = to_code
        self.calls: List[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return f"[{self.to_code.upper()}: {text}]"


def setup_tracked_pivoting_translator(
    direct_packages: List[Tuple[str, str]],
) -> Tuple[PivotingTranslator, Dict[Tuple[str, str], TrackedMockModel]]:
    """Create a PivotingTranslator populated with tracked mock models for direct pairs."""
    translator = PivotingTranslator()
    models: Dict[Tuple[str, str], TrackedMockModel] = {}

    for from_code, to_code in direct_packages:
        model = TrackedMockModel(from_code, to_code)
        models[(from_code, to_code)] = model
        translator._translation_models[(from_code, to_code)] = model

    return translator, models


# =============================================================================
# Representative Subset of Language Pairs
#
# Standard Argos coverage:
# Direct: ja<->en, es<->en, fr<->en, de<->en, ko<->en, hi<->en
# Pivot: all non-English to non-English pairs (e.g. ja->es, es->ko, ko->de, de->fr, fr->hi, hi->ja)
# =============================================================================

DIRECT_PACKAGES = [
    ("ja", "en"), ("en", "ja"),
    ("es", "en"), ("en", "es"),
    ("fr", "en"), ("en", "fr"),
    ("de", "en"), ("en", "de"),
    ("ko", "en"), ("en", "ko"),
    ("hi", "en"), ("en", "hi"),
]

# (source_lang, target_lang, expected_route_type)
REPRESENTATIVE_PAIRS: List[Tuple[Language, Language, str]] = [
    # Direct translation pairs (1-hop)
    (Language.JAPANESE, Language.ENGLISH, "direct"),
    (Language.ENGLISH, Language.SPANISH, "direct"),
    (Language.FRENCH, Language.ENGLISH, "direct"),
    (Language.ENGLISH, Language.HINDI, "direct"),
    (Language.KOREAN, Language.ENGLISH, "direct"),
    (Language.GERMAN, Language.ENGLISH, "direct"),
    (Language.HINDI, Language.ENGLISH, "direct"),

    # Pivot translation pairs (2-hop via English)
    (Language.SPANISH, Language.KOREAN, "pivot"),
    (Language.KOREAN, Language.GERMAN, "pivot"),
    (Language.GERMAN, Language.FRENCH, "pivot"),
    (Language.FRENCH, Language.HINDI, "pivot"),
    (Language.HINDI, Language.JAPANESE, "pivot"),
    (Language.JAPANESE, Language.SPANISH, "pivot"),
    (Language.HINDI, Language.GERMAN, "pivot"),

    # Same-language passthrough (0-hop)
    (Language.ENGLISH, Language.ENGLISH, "passthrough"),
    (Language.JAPANESE, Language.JAPANESE, "passthrough"),
]


def test_representative_matrix_coverage() -> None:
    """Assert that the representative subset covers all 7 languages as source and target."""
    all_7_languages = {
        Language.HINDI,
        Language.JAPANESE,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.KOREAN,
    }

    sources_covered = {src for src, tgt, _ in REPRESENTATIVE_PAIRS}
    targets_covered = {tgt for src, tgt, _ in REPRESENTATIVE_PAIRS}

    assert sources_covered == all_7_languages, f"Missing source languages: {all_7_languages - sources_covered}"
    assert targets_covered == all_7_languages, f"Missing target languages: {all_7_languages - targets_covered}"

    direct_count = sum(1 for _, _, route in REPRESENTATIVE_PAIRS if route == "direct")
    pivot_count = sum(1 for _, _, route in REPRESENTATIVE_PAIRS if route == "pivot")
    assert direct_count >= 6
    assert pivot_count >= 6


@pytest.mark.parametrize("source_lang, target_lang, expected_route", REPRESENTATIVE_PAIRS)
def test_full_pipeline_multilingual_integration(
    source_lang: Language,
    target_lang: Language,
    expected_route: str,
) -> None:
    """Run full pipeline with bundled sample clip for each language pair and assert correct routing."""
    wav_path = Path(SAMPLE_WAV_FILES[source_lang])
    assert wav_path.exists(), f"Sample audio file must exist: {wav_path}"

    audio_source = BundledWavClipAudioSource(wav_path=wav_path, chunk_duration_ms=100)
    segmenter = MultilingualAudioSegmenter()
    transcriber = MultilingualTranscriber()
    translator, tracked_models = setup_tracked_pivoting_translator(DIRECT_PACKAGES)
    presenter = RecordingCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        source_language=source_lang,
        target_language=target_lang,
    )

    # Execute the full pipeline synchronously until audio clip is fully consumed
    use_case.start(run_in_background=False)

    # 1. Verify transcriber processed the clip with pinned source language
    assert len(transcriber.transcribed_calls) == 1
    assert transcriber.transcribed_calls[0][0] == source_lang
    expected_source_text = SAMPLE_TRANSCRIPTS[source_lang]

    # 2. Verify caption output
    final_captions = presenter.final_captions
    assert len(final_captions) >= 1, f"Expected at least 1 final caption for {source_lang.value}->{target_lang.value}"
    cap = final_captions[0]

    # Assert non-empty text
    assert len(cap.text.strip()) > 0, "Caption text must not be empty"
    assert cap.is_final is True
    assert cap.language == target_lang

    # 3. Assert Route Type & Intermediary Translation Model Invocations
    src_code = source_lang.value
    tgt_code = target_lang.value

    if expected_route == "direct":
        # Exactly 1 direct translation pass
        direct_pair = (src_code, tgt_code)
        assert direct_pair in tracked_models
        assert len(tracked_models[direct_pair].calls) == 1
        assert tracked_models[direct_pair].calls[0] == expected_source_text
        assert cap.original_text == expected_source_text
        assert cap.text == f"[{tgt_code.upper()}: {expected_source_text}]"

    elif expected_route == "pivot":
        # Exactly 2 translation passes: src -> en -> tgt
        step1_pair = (src_code, "en")
        step2_pair = ("en", tgt_code)

        assert step1_pair in tracked_models
        assert step2_pair in tracked_models

        # Step 1: src -> en
        assert len(tracked_models[step1_pair].calls) == 1
        assert tracked_models[step1_pair].calls[0] == expected_source_text

        # Step 2: en -> tgt
        intermediate_en = f"[EN: {expected_source_text}]"
        assert len(tracked_models[step2_pair].calls) == 1
        assert tracked_models[step2_pair].calls[0] == intermediate_en

        assert cap.original_text == expected_source_text
        assert cap.text == f"[{tgt_code.upper()}: {intermediate_en}]"

    elif expected_route == "passthrough":
        # 0 translation passes
        for model in tracked_models.values():
            assert len(model.calls) == 0
        assert cap.text == expected_source_text
        assert cap.original_text == ""


def test_full_pipeline_threaded_background_multilingual_integration() -> None:
    """Verify full decoupled multi-threaded pipeline execution on Hindi -> French pivot translation."""
    wav_path = Path("tests/data/sample_hindi.wav")
    audio_source = BundledWavClipAudioSource(wav_path=wav_path, chunk_duration_ms=50)
    segmenter = MultilingualAudioSegmenter()
    transcriber = MultilingualTranscriber()
    translator, tracked_models = setup_tracked_pivoting_translator(DIRECT_PACKAGES)
    presenter = RecordingCaptionPresenter()

    use_case = LiveCaptionUseCase(
        audio_source=audio_source,
        segmenter=segmenter,
        transcriber=transcriber,
        translator=translator,
        presenter=presenter,
        source_language=Language.HINDI,
        target_language=Language.FRENCH,
    )

    # Start multi-threaded worker pipelines
    use_case.start(run_in_background=True)
    assert use_case.is_running is True

    # Allow decoupled threads to process the audio stream to completion
    t_start = time.time()
    while not presenter.final_captions and (time.time() - t_start) < 2.5:
        time.sleep(0.05)
    use_case.stop(timeout=1.0)
    assert use_case.is_running is False

    # Assert caption presentation and pivot routing
    finals = presenter.final_captions
    assert len(finals) >= 1
    cap = finals[0]
    assert cap.language == Language.FRENCH
    assert cap.original_text == SAMPLE_TRANSCRIPTS[Language.HINDI]
    assert "[FR: [EN: " in cap.text
