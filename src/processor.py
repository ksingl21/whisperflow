import concurrent.futures
from typing import Optional

import httpx
import ollama

_SYSTEM_PROMPT = (
    "You are a transcription cleanup assistant. "
    "Fix transcription errors, remove filler words (um, uh, like, you know, so), "
    "and return only the cleaned text — no explanation, no quotes, no extra formatting. "
    "Preserve the speaker's meaning exactly. "
    "If the input is already clean, return it unchanged."
)

_WARMUP_PROMPT = "."


class Processor:
    def __init__(self, model: str = "llama3.2:1b", timeout: float = 15.0):
        self._model = model
        self._timeout = timeout
        self._client = ollama.Client(
            host="http://localhost:11434",
            timeout=httpx.Timeout(timeout),
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def health_check(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def activate(self) -> bool:
        """Pre-load and pin the model in Ollama memory."""
        try:
            self._client.generate(
                model=self._model,
                prompt=_WARMUP_PROMPT,
                options={"num_predict": 1},
                keep_alive=-1,
                stream=False,
            )
            print(f"[whisperflow] Ollama ({self._model}) pinned in memory.", flush=True)
            return True
        except Exception as e:
            print(f"[whisperflow] Ollama activate failed: {e}", flush=True)
            return False

    def deactivate(self) -> None:
        """Request Ollama to unload the model from memory."""
        try:
            self._client.generate(
                model=self._model,
                prompt=_WARMUP_PROMPT,
                options={"num_predict": 1},
                keep_alive=0,
                stream=False,
            )
            print("[whisperflow] Ollama model unloaded.", flush=True)
        except Exception:
            pass

    def process(self, text: str) -> str:
        """Clean up raw transcription text. Falls back to raw text on any failure."""
        if not text.strip():
            return text

        def _call() -> Optional[str]:
            response = self._client.generate(
                model=self._model,
                system=_SYSTEM_PROMPT,
                prompt=text,
                keep_alive=-1,
                stream=False,
                options={"temperature": 0.1, "num_predict": 512},
            )
            return response.response.strip()

        future = self._executor.submit(_call)
        try:
            result = future.result(timeout=self._timeout)
            return result if result else text
        except concurrent.futures.TimeoutError:
            print("[whisperflow] Ollama timed out — using raw transcript.", flush=True)
            future.cancel()
            return text
        except Exception as e:
            print(f"[whisperflow] Ollama error ({e}) — using raw transcript.", flush=True)
            return text
