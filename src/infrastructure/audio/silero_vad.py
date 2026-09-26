"""Silero VAD Speech Segmenter implementing the SpeechSegmenter domain port.

Loads local Silero VAD ONNX model using onnxruntime with zero network dependencies.
Compatible with PyInstaller frozen bundles and source execution.
"""

from pathlib import Path
import sys
from typing import Iterator, List, Optional
import numpy as np
import onnxruntime as ort

from src.domain.entities import AudioChunk
from src.domain.ports import SpeechSegmenter


def _resolve_resource_path(relative_or_abs_path: str) -> Path:
    """Resolve path relative to sys._MEIPASS when frozen with PyInstaller, or cwd."""
    p = Path(relative_or_abs_path)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_candidate = Path(sys._MEIPASS) / relative_or_abs_path
        if meipass_candidate.exists():
            return meipass_candidate
    return p.resolve()


class SileroVadSegmenter(SpeechSegmenter):
    """Utterance segmenter using offline Silero VAD ONNX model."""

    def __init__(
        self,
        model_path: str = "models/silero_vad.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 250.0,
        silence_timeout_ms: float = 250.0,
        max_speech_duration_ms: float = 3000.0,
        pre_speech_pad_ms: float = 64.0,
    ) -> None:
        resolved_path = _resolve_resource_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model file not found at: {resolved_path}"
            )

        self.model_path = str(resolved_path)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.pre_speech_pad_ms = pre_speech_pad_ms

        # Window size for Silero VAD (512 samples for 16kHz, 256 for 8kHz)
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        self.window_bytes = self.window_size_samples * 2  # 16-bit mono PCM

        # Initialize ONNX runtime session with minimal overhead
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.reset_state()

    def reset_state(self) -> None:
        """Reset internal recurrent state of Silero VAD."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((64,), dtype=np.float32)

    def _predict_prob(self, window_pcm: bytes) -> float:
        """Calculate speech probability for a single 512-sample audio frame."""
        audio_int16 = np.frombuffer(window_pcm, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        input_tensor = np.concatenate([self._context, audio_float32])[np.newaxis, :].astype(np.float32)
        self._context = input_tensor[0, -64:]

        input_names = [i.name for i in self._session.get_inputs()]
        inputs = {}

        if "input" in input_names:
            inputs["input"] = input_tensor
        elif "x" in input_names:
            inputs["x"] = input_tensor

        if "sr" in input_names:
            inputs["sr"] = np.array(self.sample_rate, dtype=np.int64)

        if "state" in input_names:
            inputs["state"] = self._state

        outputs = self._session.run(None, inputs)
        prob = float(outputs[0].squeeze())

        if len(outputs) > 1 and outputs[1] is not None:
            self._state = outputs[1]

        return prob

    def segment(self, chunk_stream: Iterator[AudioChunk]) -> Iterator[AudioChunk]:
        """Process AudioChunk stream and yield utterance-bounded AudioChunks upon silence or max duration."""
        self.reset_state()

        pcm_buffer = bytearray()
        pre_pad_chunks: List[bytes] = []
        max_pre_pad_bytes = int(
            (self.pre_speech_pad_ms / 1000.0) * self.sample_rate * 2
        )

        current_utterance_pcm = bytearray()
        triggered = False
        current_speech_ms = 0.0
        current_silence_ms = 0.0
        utterance_start_time: Optional[float] = None
        frame_duration_ms = (self.window_size_samples / self.sample_rate) * 1000.0

        for chunk in chunk_stream:
            pcm_buffer.extend(chunk.pcm_data)

            while len(pcm_buffer) >= self.window_bytes:
                window = bytes(pcm_buffer[: self.window_bytes])
                del pcm_buffer[: self.window_bytes]

                prob = self._predict_prob(window)

                if prob >= self.threshold:
                    if not triggered:
                        triggered = True
                        utterance_start_time = (
                            chunk.timestamp - (len(pre_pad_chunks) * frame_duration_ms / 1000.0)
                        )
                        for pad in pre_pad_chunks:
                            current_utterance_pcm.extend(pad)
                        pre_pad_chunks.clear()

                    current_utterance_pcm.extend(window)
                    current_speech_ms += frame_duration_ms
                    current_silence_ms = 0.0

                    # Stream slice if continuous speech exceeds max_speech_duration_ms
                    if current_speech_ms >= self.max_speech_duration_ms:
                        yield AudioChunk(
                            pcm_data=bytes(current_utterance_pcm),
                            sample_rate=self.sample_rate,
                            timestamp=utterance_start_time or chunk.timestamp,
                            channels=1,
                            sample_width=2,
                        )
                        current_utterance_pcm.clear()
                        current_speech_ms = 0.0
                        utterance_start_time = chunk.timestamp

                else:
                    if triggered:
                        current_utterance_pcm.extend(window)
                        current_silence_ms += frame_duration_ms

                        if current_silence_ms >= self.silence_timeout_ms:
                            if current_speech_ms >= self.min_speech_duration_ms:
                                yield AudioChunk(
                                    pcm_data=bytes(current_utterance_pcm),
                                    sample_rate=self.sample_rate,
                                    timestamp=utterance_start_time or chunk.timestamp,
                                    channels=1,
                                    sample_width=2,
                                )

                            # Reset utterance state
                            triggered = False
                            current_utterance_pcm.clear()
                            current_speech_ms = 0.0
                            current_silence_ms = 0.0
                            utterance_start_time = None
                            self.reset_state()
                    else:
                        pre_pad_chunks.append(window)
                        total_pre_bytes = sum(len(p) for p in pre_pad_chunks)
                        while total_pre_bytes > max_pre_pad_bytes and pre_pad_chunks:
                            total_pre_bytes -= len(pre_pad_chunks.pop(0))

        # Yield any trailing utterance at end of stream if it exceeds min_speech_duration
        if triggered and current_speech_ms >= self.min_speech_duration_ms and current_utterance_pcm:
            yield AudioChunk(
                pcm_data=bytes(current_utterance_pcm),
                sample_rate=self.sample_rate,
                timestamp=utterance_start_time or 0.0,
                channels=1,
                sample_width=2,
            )
