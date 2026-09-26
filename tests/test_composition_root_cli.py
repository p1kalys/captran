"""Unit test for CLI composition root assembly."""

from unittest.mock import MagicMock, patch
from src.composition_root_cli import build_cli_application


def test_build_cli_application_wires_all_adapters() -> None:
    with patch("src.infrastructure.audio.factory.AudioSourceFactory.create_audio_source") as mock_audio, \
         patch("src.infrastructure.audio.silero_vad.SileroVadSegmenter.__init__", return_value=None), \
         patch("src.infrastructure.stt.faster_whisper_stt.FasterWhisperTranscriber.__init__", return_value=None), \
         patch("src.infrastructure.translation.argos_translator.ArgosTranslateTranslator.__init__", return_value=None):

        app = build_cli_application(
            vad_model_path="models/silero_vad.onnx",
            whisper_model_size="tiny",
        )

        assert app.use_case is not None
        assert app.audio_source is not None
        assert app.segmenter is not None
        assert app.transcriber is not None
        assert app.translator is not None
        assert app.presenter is not None
