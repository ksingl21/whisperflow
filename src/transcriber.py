import gc
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

# 1 second of silence used for warm-up JIT inference
_WARMUP_AUDIO = np.zeros(16000, dtype="float32")


class Transcriber:
    def __init__(self, model_name: str = "base.en", compute_type: str = "int8"):
        self._model_name = model_name
        self._compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        print(
            f"[whisperflow] Loading Whisper ({self._model_name}, {self._compute_type})...",
            flush=True,
        )
        self._model = WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=self._compute_type,
        )
        # warm-up: forces CTranslate2 to JIT-compile; drops first-inference latency
        list(self._model.transcribe(_WARMUP_AUDIO, beam_size=1, language="en")[0])
        print("[whisperflow] Whisper ready.", flush=True)

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        gc.collect()
        print("[whisperflow] Whisper unloaded.", flush=True)

    def transcribe(
        self,
        audio: np.ndarray,
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Transcriber.load() must be called before transcribe().")
        segments, _ = self._model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language="en",
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
