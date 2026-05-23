import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 1024
SILENCE_THRESHOLD = 0.01  # peak amplitude below this is treated as silence


class AudioRecorder:
    def __init__(
        self,
        device: Optional[int] = None,
        max_seconds: float = 30.0,
        min_seconds: float = 0.3,
        on_max_duration: Optional[Callable[[Optional[np.ndarray]], None]] = None,
    ):
        self._device = device
        self._max_seconds = max_seconds
        self._min_seconds = min_seconds
        self._on_max_duration = on_max_duration

        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._timer: Optional[threading.Timer] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return
        with self._lock:
            self._chunks = []
        self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=self._device,
                callback=self._callback,
                blocksize=BLOCK_SIZE,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._recording = False
            raise RuntimeError(
                f"Microphone error: {e}\n"
                "  → Check System Settings → Privacy & Security → Microphone."
            )

        if self._max_seconds > 0:
            self._timer = threading.Timer(self._max_seconds, self._hit_max)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> Optional[np.ndarray]:
        if not self._recording:
            return None
        return self._finalize()

    # ------------------------------------------------------------------ #

    def _hit_max(self) -> None:
        print(
            f"\n[whisperflow] Max recording duration ({self._max_seconds}s) reached.",
            flush=True,
        )
        audio = self._finalize()
        if self._on_max_duration:
            self._on_max_duration(audio)

    def _finalize(self) -> Optional[np.ndarray]:
        self._recording = False

        if self._timer:
            self._timer.cancel()
            self._timer = None

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            chunks = list(self._chunks)

        if not chunks:
            return None

        audio: np.ndarray = np.concatenate(chunks).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < self._min_seconds:
            print(
                f"[whisperflow] Recording too short ({duration:.2f}s < {self._min_seconds}s), skipping.",
                flush=True,
            )
            return None

        if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
            print("[whisperflow] Silence detected, skipping.", flush=True)
            return None

        return audio

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,  # noqa: ANN001
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        if self._recording:
            with self._lock:
                self._chunks.append(indata.copy())
