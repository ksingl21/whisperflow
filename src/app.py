"""
WhisperFlow — local push-to-talk voice-to-text daemon.

Hotkeys (configurable in config.toml):
  Shift+V      : toggle voice mode on / off
  Space (hold) : record while held, transcribe + paste on release
  Ctrl+C       : quit
"""

import gc
import signal
import subprocess
import sys
import threading
import tomllib
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from pynput import keyboard as kb


def _notify(title: str, body: str = "") -> None:
    try:
        script = f'display notification "{body}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], check=False, timeout=3,
                       capture_output=True)
    except Exception:
        pass

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

_MODIFIER_GROUPS = {
    "shift": frozenset({kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r}),
    "ctrl":  frozenset({kb.Key.ctrl,  kb.Key.ctrl_l,  kb.Key.ctrl_r}),
    "alt":   frozenset({kb.Key.alt,   kb.Key.alt_l,   kb.Key.alt_r}),
    "cmd":   frozenset({kb.Key.cmd,   kb.Key.cmd_l,   kb.Key.cmd_r}),
}


def _parse_key(s: str):
    """Convert a simple key name ('f7', 'space', 'a') to a pynput Key or KeyCode."""
    s = s.strip().lower()
    if hasattr(kb.Key, s):
        return getattr(kb.Key, s)
    if len(s) == 1:
        return kb.KeyCode.from_char(s)
    raise ValueError(f"Unknown hotkey in config: {s!r}")


def _parse_hotkey(s: str) -> tuple:
    """Parse 'shift+v', 'space', 'f7' etc.

    Returns (modifier_groups: list[frozenset], main_key).
    Each group requires at least one of its keys to be held simultaneously.
    """
    parts = [p.strip().lower() for p in s.split("+")]
    modifier_groups: list = []
    main_key = None
    for part in parts:
        if part in _MODIFIER_GROUPS:
            modifier_groups.append(_MODIFIER_GROUPS[part])
        else:
            main_key = _parse_key(part)
    if main_key is None:
        raise ValueError(f"No main key found in hotkey: {s!r}")
    return modifier_groups, main_key


def _mods_satisfied(modifier_groups: list, held: set) -> bool:
    """True when at least one key from each required modifier group is held."""
    return all(any(mod in held for mod in group) for group in modifier_groups)


def _key_matches(pressed: object, target: object) -> bool:
    """True if pressed matches target, case-insensitively for KeyCode chars."""
    if pressed == target:
        return True
    if (isinstance(pressed, kb.KeyCode) and isinstance(target, kb.KeyCode)
            and pressed.char and target.char):
        return pressed.char.lower() == target.char.lower()
    return False


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

        mic_raw = rec.get("mic_device", "")
        mic_device = int(mic_raw) if str(mic_raw).strip().isdigit() else None

        self._recorder = AudioRecorder(
            device=mic_device,
            max_seconds=float(rec.get("max_seconds", 0.0)),
            min_seconds=float(rec.get("min_seconds", 0.3)),
            silence_timeout=float(rec.get("silence_timeout_seconds", 2.0)),
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

    def on_ptt_toggle(self) -> None:
        state = self._get_state()
        if state == State.READY:
            self._cancel_idle_timer()
            self._set_state(State.RECORDING)
            try:
                self._recorder.start()
                print("[whisperflow] Recording… (press again to stop)", flush=True)
            except RuntimeError as e:
                print(f"[whisperflow] {e}", flush=True)
                self._set_state(State.ERROR)
        elif state == State.RECORDING:
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
            _notify("WhisperFlow Error", f"Failed to load Whisper: {e}")
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
        ptt = self._config.get("hotkeys", {}).get("push_to_talk", "Space").upper()
        print(f"[whisperflow] Ready. Press {ptt} to start recording.", flush=True)
        _notify("WhisperFlow Ready", f"Press {ptt} to start recording")

    def _run_deactivate(self) -> None:
        self._cancel_idle_timer()
        print("[whisperflow] Deactivating…", flush=True)
        self._transcriber.unload()
        if self._cleanup_enabled:
            self._processor.deactivate()
        gc.collect()
        self._set_state(State.IDLE)
        act = self._config.get("hotkeys", {}).get("activation", "Shift+V").upper()
        print(f"[whisperflow] Voice mode off. Press {act} to activate.", flush=True)
        _notify("WhisperFlow", f"Voice mode off — press {act} to reactivate")

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

        # ── optional LLM cleanup ────────────────────────────────────────── #
        if self._cleanup_enabled:
            final_text = self._processor.process(raw_text)
        else:
            final_text = raw_text

        print(f"[whisperflow] → {final_text}", flush=True)

        # ── paste ───────────────────────────────────────────────────────── #
        try:
            self._paster.paste(
                final_text,
                restore=self._restore_clipboard,
                restore_delay=self._restore_delay,
            )
            _notify("WhisperFlow", final_text[:100] + ("…" if len(final_text) > 100 else ""))
        except Exception as e:
            print(f"[whisperflow] Paste failed: {e}", flush=True)
            _notify("WhisperFlow — Paste Failed", final_text[:100])
            print(f"[whisperflow] Your text: {final_text}", flush=True)

        self._set_state(State.READY)
        self._reset_idle_timer()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(config: dict) -> None:
    hk  = config.get("hotkeys", {})
    act = hk.get("activation",   "Shift+V").upper()
    ptt = hk.get("push_to_talk", "Space").upper()
    ol  = config.get("ollama", {})
    wh  = config.get("whisper", {})
    print("=" * 52)
    print("  WhisperFlow — Local Voice to Text")
    print("=" * 52)
    print(f"  {act:<13}: Toggle voice mode on / off")
    print(f"  {ptt:<13}: Start / stop recording")
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
    act_mods, act_main   = _parse_hotkey(hk_cfg.get("activation",   "shift+v"))
    ptt_mods, ptt_main   = _parse_hotkey(hk_cfg.get("push_to_talk", "space"))

    act_label = hk_cfg.get("activation",   "Shift+V").upper()
    print(f"[whisperflow] Listening. Press {act_label} to activate.", flush=True)
    _notify("WhisperFlow Running", f"Press {act_label} to activate voice mode")

    _held: set = set()

    def on_press(key):
        if isinstance(key, kb.Key):
            _held.add(key)

        if _key_matches(key, act_main) and _mods_satisfied(act_mods, _held):
            session.on_activation_key()
        elif _key_matches(key, ptt_main) and _mods_satisfied(ptt_mods, _held):
            session.on_ptt_toggle()

    def on_release(key):
        if isinstance(key, kb.Key):
            _held.discard(key)

    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
