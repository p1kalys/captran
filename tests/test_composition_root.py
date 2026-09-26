"""Integration tests for composition root assembly."""

from src.composition_root import build_live_captioner


def test_build_live_captioner_and_run_pipeline() -> None:
    app = build_live_captioner(verbose_ui=False)

    assert app.use_case is not None
    assert app.audio_adapter is not None
    assert app.stt_adapter is not None
    assert app.translation_adapter is not None
    assert app.ui_adapter is not None

    # Test full end-to-end pipeline wiring
    app.use_case.start()
    assert app.use_case.is_running is True

    # Feed test audio chunk
    app.audio_adapter.enqueue_chunk(b"\x00\x00" * 16000)

    # Process chunk
    cues = app.use_case.process_next_chunk()
    assert len(cues) == 1
    assert cues[0].original_text == "こんにちは、世界！"
    assert cues[0].translated_text == "Hello, world!"
    assert len(app.ui_adapter.history) == 1

    app.use_case.stop()
    assert app.use_case.is_running is False
