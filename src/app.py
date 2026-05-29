"""
WhisperFlow — local voice-to-text daemon with menu bar indicator.

Hotkeys (configurable in config.toml):
  Shift+V : toggle voice mode on / off
  Space   : start / stop recording
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

import rumps
from pynput import keyboard as kb

from audio import AudioRecorder
from clipboard import ClipboardPaster
from processor import Processor
from transcriber import Transcriber

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _notify(title: str, body: str = "") -> None:
    try:
        script = f'display notification "{body}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], check=False, timeout=3,
                       capture_output=True)
    except Exception:
        pass


def _load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    print(f"[whisperflow] config.toml not found at {path}, using defaults.", flush=True)
    return {}


_MODIFIER_GROUPS = {
    "shift": frozenset({kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r}),
    "ctrl":  frozenset({kb.Key.ctrl,  kb.Key.ctrl_l,  kb.Key.ctrl_r}),
    "alt":   frozenset({kb.Key.alt,   kb.Key.alt_l,   kb.Key.alt_r}),
    "cmd":   frozenset({kb.Key.cmd,   kb.Key.cmd_l,   kb.Key.cmd_r}),
}


def _parse_key(s: str):
    s = s.strip().lower()
    if hasattr(kb.Key, s):
        return getattr(kb.Key, s)
    if len(s) == 1:
        return kb.KeyCode.from_char(s)
    raise ValueError(f"Unknown hotkey in config: {s!r}")


def _parse_hotkey(s: str) -> tuple:
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
    return all(any(mod in held for mod in group) for group in modifier_groups)


def _key_matches(pressed: object, target: object) -> bool:
    if pressed == target:
        return True
    if (isinstance(pressed, kb.KeyCode) and isinstance(target, kb.KeyCode)
            and pressed.char and target.char):
        return pressed.char.lower() == target.char.lower()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE         = auto()
    ACTIVATING   = auto()
    READY        = auto()
    RECORDING    = auto()
    PROCESSING   = auto()
    DEACTIVATING = auto()
    ERROR        = auto()


_STATE_ICON = {
    State.IDLE:         "🎤",
    State.ACTIVATING:   "🎤…",
    State.READY:        "🎤 ✅",
    State.RECORDING:    "🔴 REC",
    State.PROCESSING:   "🎤 ⏳",
    State.DEACTIVATING: "🎤…",
    State.ERROR:        "⚠️",
}

_STATE_LABEL = {
    State.IDLE:         "Idle",
    State.ACTIVATING:   "Loading models…",
    State.READY:        "Ready — press Space to record",
    State.RECORDING:    "Recording…",
    State.PROCESSING:   "Transcribing…",
    State.DEACTIVATING: "Deactivating…",
    State.ERROR:        "Error",
}


# ─────────────────────────────────────────────────────────────────────────────
# VoiceSession
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSession:
    def __init__(self, config: dict, on_state_change=None):
        self._config = config
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._on_state_change = on_state_change  # callback(State)

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

        self._cleanup_enabled   = bool(ol.get("cleanup_enabled", True))
        self._beam_size         = int(wh.get("beam_size", 1))
        self._vad_filter        = bool(wh.get("vad_filter", True))
        self._idle_unload_sec   = float(sess.get("idle_unload_seconds", 300.0))
        self._restore_clipboard = bool(cl.get("restore", True))
        self._restore_delay     = float(cl.get("restore_delay_seconds", 1.5))

    def _get_state(self) -> State:
        with self._state_lock:
            return self._state

    def _set_state(self, new: State) -> None:
        with self._state_lock:
            self._state = new
        if self._on_state_change:
            self._on_state_change(new)

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
            print(f"\n[whisperflow] Idle for {self._idle_unload_sec:.0f}s — unloading models.", flush=True)
            self._spawn(self._run_deactivate)

    def on_activation_key(self) -> None:
        state = self._get_state()
        if state in (State.IDLE, State.ERROR):
            self._set_state(State.ACTIVATING)
            self._spawn(self._run_activate)
        elif state == State.READY:
            self._set_state(State.DEACTIVATING)
            self._spawn(self._run_deactivate)

    def on_ptt_toggle(self) -> None:
        state = self._get_state()
        if state == State.READY:
            self._cancel_idle_timer()
            self._set_state(State.RECORDING)
            try:
                self._recorder.start()
                print("[whisperflow] Recording… (press Space again to stop)", flush=True)
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

    def _on_max_duration(self, audio) -> None:
        if self._get_state() == State.RECORDING:
            self._set_state(State.PROCESSING)
            self._spawn(self._run_pipeline, audio)

    def _spawn(self, _run_fn, *args) -> None:
        self._worker = threading.Thread(target=_run_fn, args=args, daemon=True)
        self._worker.start()

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
                print("[whisperflow] Ollama unavailable — LLM cleanup disabled.", flush=True)
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

        try:
            print("[whisperflow] Transcribing…", flush=True)
            raw_text = self._transcriber.transcribe(
                audio, beam_size=self._beam_size, vad_filter=self._vad_filter)
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

        final_text = self._processor.process(raw_text) if self._cleanup_enabled else raw_text
        print(f"[whisperflow] → {final_text}", flush=True)

        try:
            self._paster.paste(final_text, restore=self._restore_clipboard,
                               restore_delay=self._restore_delay)
            _notify("WhisperFlow", final_text[:100] + ("…" if len(final_text) > 100 else ""))
        except Exception as e:
            print(f"[whisperflow] Paste failed: {e}", flush=True)
            _notify("WhisperFlow — Paste Failed", final_text[:100])

        self._set_state(State.READY)
        self._reset_idle_timer()


# ─────────────────────────────────────────────────────────────────────────────
# Menu bar app
# ─────────────────────────────────────────────────────────────────────────────

class WhisperFlowMenuBar(rumps.App):
    def __init__(self, config: dict):
        super().__init__("🎤", quit_button=None)
        self._config = config
        self._session: Optional[VoiceSession] = None

        hk = config.get("hotkeys", {})
        self._act_label = hk.get("activation",   "Shift+V").upper()
        self._ptt_label = hk.get("push_to_talk", "Space").upper()

        self._status_item   = rumps.MenuItem(f"Status: Idle")
        self._activate_item = rumps.MenuItem(
            f"Activate voice mode  ({self._act_label})",
            callback=self._on_activate_click,
        )

        self.menu = [
            self._status_item,
            None,
            self._activate_item,
            None,
            rumps.MenuItem("Quit WhisperFlow", callback=self._on_quit),
        ]

    def set_session(self, session: VoiceSession) -> None:
        self._session = session

    def update_state(self, state: State) -> None:
        self.title = _STATE_ICON.get(state, "🎤")
        self._status_item.title = f"Status: {_STATE_LABEL.get(state, str(state))}"
        if state in (State.IDLE, State.ERROR):
            self._activate_item.title = f"Activate voice mode  ({self._act_label})"
        elif state == State.READY:
            self._activate_item.title = f"Deactivate voice mode  ({self._act_label})"
        else:
            self._activate_item.title = f"Busy…"

    def _on_activate_click(self, _) -> None:
        if self._session:
            self._session.on_activation_key()

    def _on_quit(self, _) -> None:
        if self._session:
            self._session.shutdown()
        rumps.quit_application()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = _load_config(CONFIG_PATH)

    menu_bar = WhisperFlowMenuBar(config)

    session = VoiceSession(config, on_state_change=menu_bar.update_state)
    menu_bar.set_session(session)

    def _handle_sigint(sig, frame):
        print("\n[whisperflow] Shutting down…", flush=True)
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    hk_cfg = config.get("hotkeys", {})
    act_mods, act_main = _parse_hotkey(hk_cfg.get("activation",   "shift+v"))
    ptt_mods, ptt_main = _parse_hotkey(hk_cfg.get("push_to_talk", "space"))
    act_label = hk_cfg.get("activation", "Shift+V").upper()

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

    # Run keyboard listener in a background thread (rumps owns the main thread)
    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    menu_bar.run()


if __name__ == "__main__":
    main()
