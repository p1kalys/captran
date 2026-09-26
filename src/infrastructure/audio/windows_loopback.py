"""Windows WASAPI Loopback AudioSource implementation.

Captures system speaker output (meetings, browser, media) using pyaudiowpatch.
Downmixes to mono and resamples to 16kHz for VAD and STT.
Zero cloud dependencies.
"""

from queue import Empty, Queue
import sys
import time
from typing import Any, Dict, Iterator, List, Optional
import numpy as np

from src.domain.entities import AudioChunk
from src.domain.ports import AudioSource

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None  # type: ignore


class WindowsLoopbackAudioSource(AudioSource):
    """Captures system/meeting speaker audio via Windows WASAPI loopback."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        chunk_duration_ms: int = 150,
        target_sample_rate: int = 16000,
    ) -> None:
        if sys.platform != "win32" and pyaudio is None:
            raise RuntimeError(
                "WindowsLoopbackAudioSource requires Windows and pyaudiowpatch."
            )

        self.target_sample_rate = target_sample_rate
        self.chunk_duration_ms = max(50, min(1000, chunk_duration_ms))
        self._device_index = device_index
        self._pyaudio: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._queue: Queue[AudioChunk] = Queue()
        self._is_active = False

        self._native_sample_rate: int = 48000
        self._native_channels: int = 2
        self._selected_device_info: Optional[Dict[str, Any]] = None

    @staticmethod
    def list_loopback_devices() -> List[Dict[str, Any]]:
        """Enumerate all available WASAPI loopback capture devices on the system."""
        if pyaudio is None:
            raise ImportError(
                "pyaudiowpatch is not installed. Please install pyaudiowpatch."
            )

        p = pyaudio.PyAudio()
        devices = []
        try:
            for dev in p.get_loopback_device_info_generator():
                devices.append(
                    {
                        "index": dev["index"],
                        "name": dev["name"],
                        "channels": dev["maxInputChannels"],
                        "defaultSampleRate": int(dev["defaultSampleRate"]),
                    }
                )
        finally:
            p.terminate()
        return devices

    @staticmethod
    def get_default_loopback_device(p_instance: Any) -> Dict[str, Any]:
        """Find the loopback device corresponding to the default WASAPI output speakers/headset."""
        # 1. First try pyaudiowpatch native default loopback helper
        try:
            if hasattr(p_instance, "get_default_wasapi_loopback"):
                dev = p_instance.get_default_wasapi_loopback()
                if dev:
                    return dev
        except Exception:
            pass

        # 2. Match loopback device corresponding to default WASAPI output device
        try:
            wasapi_info = p_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output_idx = wasapi_info["defaultOutputDevice"]
            default_speakers = p_instance.get_device_info_by_index(default_output_idx)
        except Exception:
            default_speakers = p_instance.get_default_output_device_info()

        if default_speakers.get("isLoopbackDevice", False):
            return default_speakers

        # Match loopback device by name
        for loopback in p_instance.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                return loopback

        # Fallback to the first available loopback device
        for loopback in p_instance.get_loopback_device_info_generator():
            return loopback

        raise RuntimeError("No WASAPI loopback capture device found on this system.")

    def _audio_callback(
        self, in_data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple[None, int]:
        """Callback from PyAudio receiving raw native PCM stream."""
        if not in_data or not self._is_active:
            return (None, pyaudio.paContinue)

        try:
            # Convert native int16 to numpy array
            native_samples = np.frombuffer(in_data, dtype=np.int16)
            if len(native_samples) == 0:
                return (None, pyaudio.paContinue)

            if self._native_channels > 1:
                rem = len(native_samples) % self._native_channels
                if rem != 0:
                    native_samples = native_samples[:-rem]
                # Reshape to (N, channels) and average to mono
                native_samples = native_samples.reshape(-1, self._native_channels)
                mono_samples = native_samples.mean(axis=1)
            else:
                mono_samples = native_samples.astype(np.float32)

            # Resample to target_sample_rate if different
            if self._native_sample_rate != self.target_sample_rate:
                duration_s = len(mono_samples) / self._native_sample_rate
                target_length = int(round(duration_s * self.target_sample_rate))
                if target_length > 0:
                    orig_indices = np.linspace(0, len(mono_samples), num=len(mono_samples), endpoint=False)
                    new_indices = np.linspace(0, len(mono_samples), num=target_length, endpoint=False)
                    resampled = np.interp(new_indices, orig_indices, mono_samples)
                else:
                    resampled = np.array([], dtype=np.float32)
            else:
                resampled = mono_samples

            # Clip and convert back to 16-bit signed integer PCM
            resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
            pcm_bytes = resampled_int16.tobytes()

            if pcm_bytes:
                chunk = AudioChunk(
                    pcm_data=pcm_bytes,
                    sample_rate=self.target_sample_rate,
                    timestamp=time.time(),
                    channels=1,
                    sample_width=2,
                )
                self._queue.put(chunk)

        except Exception:
            pass

        return (None, pyaudio.paContinue)

    def start(self) -> None:
        """Start capturing audio via WASAPI loopback."""
        if self._is_active:
            return

        if pyaudio is None:
            raise ImportError(
                "pyaudiowpatch is required for WindowsLoopbackAudioSource."
            )

        self._pyaudio = pyaudio.PyAudio()

        if self._device_index is not None:
            device_info = self._pyaudio.get_device_info_by_index(self._device_index)
        else:
            device_info = self.get_default_loopback_device(self._pyaudio)

        self._selected_device_info = device_info
        self._native_sample_rate = int(device_info["defaultSampleRate"])
        self._native_channels = int(device_info["maxInputChannels"])

        # Calculate buffer frames for requested interval (e.g. 150ms)
        frames_per_buffer = int(
            self._native_sample_rate * (self.chunk_duration_ms / 1000.0)
        )

        self._is_active = True
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self._native_channels,
            rate=self._native_sample_rate,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop capture and release WASAPI streams."""
        self._is_active = False

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

    def stream(self) -> Iterator[AudioChunk]:
        """Yield AudioChunk stream continuously while capture is active."""
        while self._is_active:
            try:
                chunk = self._queue.get(timeout=0.2)
                yield chunk
            except Empty:
                continue

    @property
    def selected_device_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata of the active loopback device."""
        return self._selected_device_info
