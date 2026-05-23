"""
WhisperFlow — local push-to-talk voice-to-text daemon.

Hotkeys (configurable in config.toml):
  F7           : toggle voice mode on / off
  F8 (hold)    : record while held, transcribe + paste on release
  Ctrl+C       : quit
"""

import gc
import signal
import sys
import threading
import tomllib
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from pynput import keyboard as kb

from audio import AudioRecorder
from clipboard import ClipboardPaster
from processor import Processor
from transcriber import Transcriber

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE        = auto()
    ACTIVATING  = auto()
    READY       = auto()
    RECORDING   = auto()
    PROCESSING  = auto()
    DEACTIVATING = auto()
    ERROR       = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_key(s: str):
    """Convert config string ('f7', 'f8', 'a') to a pynput Key or KeyCode."""
    s = s.strip().lower()
    if hasattr(kb.Key, s):
        return getattr(kb.Key, s)
    if len(s) == 1:
        return kb.KeyCode.from_char(s)
    raise ValueError(f"Unknown hotkey in config: {s!r}")


def _load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    print(f"[whisperflow] config.toml not found at {path}, using defaults.", flush=True)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# VoiceSession
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSession:
    def __init__(self, config: dict):
        self._config = config
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._idle_timer: Optional[threading.Timer] = None

        # ── section references ──────────────────────────────────────────── #
        hk   = config.get("hotkeys",   {})
        wh   = config.get("whisper",   {})
        ol   = config.get("ollama",    {})
        rec  = config.get("recording", {})
        sess = config.get("session",   {})
        cl   = config.get("clipboard", {})

        self._activation_key = _parse_key(hk.get("activation",   "f7"))
        self._ptt_key        = _parse_key(hk.get("push_to_talk", "f8"))

        mic_raw = rec.get("mic_device", "")
        mic_device = int(mic_raw) if str(mic_raw).strip().isdigit() else None

        self._recorder = AudioRecorder(
            device=mic_device,
            max_seconds=float(rec.get("max_seconds", 30.0)),
            min_seconds=float(rec.get("min_seconds", 0.3)),
            on_max_duration=self._on_max_duration,
        )
        self._transcriber = Transcriber(
            model_name=wh.get("model", "base.en"),
            compute_type=wh.get("compute_type", "int8"),
        )
        self._processor = Processor(
            model=ol.get("model", "llama3.2:1b"),
            timeout=float(ol.get("timeout_seconds", 15.0)),
        )
        self._paster = ClipboardPaster()

        self._cleanup_enabled    = bool(ol.get("cleanup_enabled", True))
        self._beam_size          = int(wh.get("beam_size", 1))
        self._vad_filter         = bool(wh.get("vad_filter", True))
        self._idle_unload_sec    = float(sess.get("idle_unload_seconds", 300.0))
        self._restore_clipboard  = bool(cl.get("restore", True))
        self._restore_delay      = float(cl.get("restore_delay_seconds", 1.5))

    # ── state helpers ────────────────────────────────────────────────────── #

    def _get_state(self) -> State:
        with self._state_lock:
            return self._state

    def _set_state(self, new: State) -> None:
        with self._state_lock:
            self._state = new

    # ── idle timer ───────────────────────────────────────────────────────── #

    def _reset_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if self._idle_unload_sec > 0:
            self._idle_timer = threading.Timer(self._idle_unload_sec, self._on_idle_timeout)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self) -> None:
        if self._get_state() == State.READY:
            print(
                f"\n[whisperflow] Idle for {self._idle_unload_sec:.0f}s — unloading models.",
                flush=True,
            )
            self._spawn(_run_fn=self._run_deactivate)

    # ── public hotkey handlers ───────────────────────────────────────────── #

    def on_activation_key(self) -> None:
        state = self._get_state()
        if state in (State.IDLE, State.ERROR):
            self._set_state(State.ACTIVATING)
            self._spawn(self._run_activate)
        elif state == State.READY:
            self._set_state(State.DEACTIVATING)
            self._spawn(self._run_deactivate)
        # busy states (ACTIVATING, RECORDING, PROCESSING, DEACTIVATING) → ignore

    def on_ptt_press(self) -> None:
        if self._get_state() != State.READY:
            return
        self._cancel_idle_timer()
        self._set_state(State.RECORDING)
        try:
            self._recorder.start()
            print("[whisperflow] Recording… (release to stop)", flush=True)
        except RuntimeError as e:
            print(f"[whisperflow] {e}", flush=True)
            self._set_state(State.ERROR)

    def on_ptt_release(self) -> None:
        if self._get_state() != State.RECORDING:
            return
        audio = self._recorder.stop()
        self._set_state(State.PROCESSING)
        self._spawn(self._run_pipeline, audio)

    def shutdown(self) -> None:
        self._cancel_idle_timer()
        state = self._get_state()
        if state == State.RECORDING:
            self._recorder.stop()
        if state not in (State.IDLE, State.DEACTIVATING, State.ERROR):
            self._run_deactivate()

    # ── callback from AudioRecorder when max duration fires ─────────────── #

    def _on_max_duration(self, audio) -> None:
        if self._get_state() == State.RECORDING:
            self._set_state(State.PROCESSING)
            self._spawn(self._run_pipeline, audio)

    # ── worker dispatch ──────────────────────────────────────────────────── #

    def _spawn(self, _run_fn, *args) -> None:
        self._worker = threading.Thread(target=_run_fn, args=args, daemon=True)
        self._worker.start()

    # ── pipeline stages ──────────────────────────────────────────────────── #

    def _run_activate(self) -> None:
        print("[whisperflow] Activating voice mode…", flush=True)
        try:
            self._transcriber.load()
        except Exception as e:
            print(f"[whisperflow] Failed to load Whisper: {e}", flush=True)
            self._set_state(State.ERROR)
            return

        if self._cleanup_enabled:
            ok = self._processor.activate()
            if not ok:
                print(
                    "[whisperflow] Ollama unavailable — LLM cleanup disabled for this session.",
                    flush=True,
                )
                self._cleanup_enabled = False

        self._set_state(State.READY)
        self._reset_idle_timer()
        ptt = self._config.get("hotkeys", {}).get("push_to_talk", "F8").upper()
        print(f"[whisperflow] Ready. Hold {ptt} to record.", flush=True)

    def _run_deactivate(self) -> None:
        self._cancel_idle_timer()
        print("[whisperflow] Deactivating…", flush=True)
        self._transcriber.unload()
        if self._cleanup_enabled:
            self._processor.deactivate()
        gc.collect()
        self._set_state(State.IDLE)
        act = self._config.get("hotkeys", {}).get("activation", "F7").upper()
        print(f"[whisperflow] Voice mode off. Press {act} to activate.", flush=True)

    def _run_pipeline(self, audio) -> None:
        if audio is None:
            print("[whisperflow] No audio captured, skipping.", flush=True)
            self._set_state(State.READY)
            self._reset_idle_timer()
            return

        # ── transcribe ──────────────────────────────────────────────────── #
        try:
            print("[whisperflow] Transcribing…", flush=True)
            raw_text = self._transcriber.transcribe(
                audio,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
            )
        except Exception as e:
            print(f"[whisperflow] Transcription error: {e}", flush=True)
            self._set_state(State.READY)
            self._reset_idle_timer()
            return

        if not raw_text.strip():
            print("[whisperflow] Nothing heard, skipping.", flush=True)
            self._set_state(State.READY)
            self._reset_idle_timer()
            return

        print(f"[whisperflow] Raw: {raw_text}", flush=True)

        # ── optional LLM cleanup ────────────────────────────────────────── #
        if self._cleanup_enabled:
            print("[whisperflow] Cleaning up with LLM…", flush=True)
            final_text = self._processor.process(raw_text)
        else:
            final_text = raw_text

        print(f"[whisperflow] Pasting: {final_text}", flush=True)

        # ── paste ───────────────────────────────────────────────────────── #
        try:
            self._paster.paste(
                final_text,
                restore=self._restore_clipboard,
                restore_delay=self._restore_delay,
            )
        except Exception as e:
            print(f"[whisperflow] Paste failed: {e}", flush=True)
            print(f"[whisperflow] Your text: {final_text}", flush=True)

        self._set_state(State.READY)
        self._reset_idle_timer()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(config: dict) -> None:
    hk  = config.get("hotkeys", {})
    act = hk.get("activation",   "F7").upper()
    ptt = hk.get("push_to_talk", "F8").upper()
    ol  = config.get("ollama", {})
    wh  = config.get("whisper", {})
    print("=" * 52)
    print("  WhisperFlow — Local Voice to Text")
    print("=" * 52)
    print(f"  {act}          : Toggle voice mode on / off")
    print(f"  {ptt} (hold)  : Record; paste on release")
    print(f"  Ctrl+C       : Quit")
    print("─" * 52)
    print(f"  Whisper : {wh.get('model', 'base.en')}  |  Ollama : {ol.get('model', 'llama3.2:1b')}")
    print(f"  Cleanup : {'on' if ol.get('cleanup_enabled', True) else 'off'}")
    print("=" * 52, flush=True)


def main() -> None:
    config = _load_config(CONFIG_PATH)
    session = VoiceSession(config)

    # ── signal handler ───────────────────────────────────────────────────── #
    def _handle_sigint(sig, frame):
        print("\n[whisperflow] Shutting down…", flush=True)
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    _print_banner(config)

    hk_cfg = config.get("hotkeys", {})
    activation_key = _parse_key(hk_cfg.get("activation",   "f7"))
    ptt_key        = _parse_key(hk_cfg.get("push_to_talk", "f8"))

    print(
        f"[whisperflow] Listening. Press {hk_cfg.get('activation','F7').upper()} to activate.",
        flush=True,
    )

    def on_press(key):
        if key == activation_key:
            session.on_activation_key()
        elif key == ptt_key:
            session.on_ptt_press()

    def on_release(key):
        if key == ptt_key:
            session.on_ptt_release()

    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
