# Personal CLI Voice-to-Text Implementation Plan (Ultra-Low Latency)

## Objective
Build a lightweight, purely local, Python-based CLI application that allows the user to record spoken English via a global hotkey. The app is optimized for low warm-path latency while still prioritizing reliability, predictable state transitions, and safe clipboard behavior. Models are loaded on-demand when voice mode is activated and unloaded when deactivated or after an idle timeout, keeping the background daemon lightweight when idle. It uses a memory-resident Whisper model for fast audio transcription, and optionally sends the output through a fast local micro-LLM (via Ollama) to clean up mistakes and structure sentences. The final text is pasted to the user's active cursor via the clipboard.

## Key Files & Context
- **Workspace:** `/Users/kapilsingla/Documents/whisperflow/`
- **Main Script:** `src/app.py` (Background listener, `VoiceSession` class, hotkey toggle, signal handling).
- **Audio Module:** `src/audio.py` (Microphone capture at 16kHz mono directly into memory).
- **Transcription Module:** `src/transcriber.py` (Loads/unloads the Whisper model; optimized inference settings).
- **Processor Module:** `src/processor.py` (Streaming connection to Ollama; model pin/unpin).
- **Dependencies:** `requirements.txt`

## Technical Stack
- **Audio Capture:** `sounddevice` and `numpy` — capturing at **16kHz mono** directly into memory (no disk I/O).
- **AI Transcription Model:** `faster-whisper` (`base.en`, `compute_type="int8"`, `beam_size=1`, `vad_filter=True`). Loaded into RAM only while voice mode is active.
- **AI Formatting Model:** `Ollama` running `llama3.2:1b`. Pinned in memory with `keep_alive=-1` during active session; force-unloaded with `keep_alive=0` on deactivation.
- **LLM Client:** `ollama` Python library (streaming responses).
- **Hotkey & Output:** `pynput` for hotkey listening. Output via **clipboard paste** (`pyperclip` + simulated `Cmd+V`) for instant, length-independent delivery.
- **Configuration:** simple local config file or CLI flags for hotkeys, model names, cleanup mode, timeout values, and clipboard behavior.

## Model Lifecycle (Lazy Loading)

| Event | faster-whisper | Ollama |
|-------|---------------|--------|
| App start | Not loaded | Not loaded |
| Voice mode ON | `WhisperModel(...)` instantiated + warm-up run | Request with `keep_alive=-1` to pin in memory |
| Recording/transcribing | Model in RAM, reused across recordings | Model pinned, streamed per request |
| Voice mode OFF / idle timeout | `del model` + `gc.collect()`; verify actual memory behavior empirically | Request with `keep_alive=0` to request unload |

**UX note:** First use after activating will take longer because models need to load. Print `"Voice mode activating... ready."` so the user knows to wait. Subsequent recordings in the same session are the optimized path.

**Important:** `del model`, `gc.collect()`, and Ollama `keep_alive=0` do not guarantee immediate OS-level memory release. Treat memory-release behavior as something to measure, not assume.

## Interaction Model

Use a clear state machine instead of overloading one hotkey with multiple meanings.

Recommended default:
- **Activation hotkey:** toggles voice mode on/off.
- **Push-to-talk hotkey:** press and hold to record; release to transcribe, optionally clean, and paste.
- **Idle timeout:** automatically unloads models after a configurable period without recording.

Alternative low-friction mode:
- One push-to-talk hotkey loads models on first use, records while held, processes on release, and unloads after idle timeout.

Avoid using the same key for both "toggle mode" and "hold to record" unless the state transitions are explicitly defined and tested.

## State Machine

Define explicit states in `VoiceSession`:
- `idle`: daemon is running, models are not loaded.
- `activating`: models are loading or warming up.
- `ready`: models are loaded and waiting for push-to-talk.
- `recording`: microphone stream is active.
- `processing`: transcription, optional cleanup, and paste are running.
- `deactivating`: cleanup and model unload are in progress.
- `error`: recoverable failure state with a clear message and reset path.

Rules:
- Ignore or queue hotkey input while `activating`, `processing`, or `deactivating`.
- Allow cancellation/deactivation during `ready`.
- Define behavior for deactivation during `recording` or `processing` before implementation.

## Implementation Steps

### Phase 1: Setup and Environment
1. Ensure Ollama is installed and the micro-model is pulled: `ollama pull llama3.2:1b`.
2. Create Python virtual environment and `requirements.txt`:
   - `faster-whisper`, `sounddevice`, `numpy`, `pynput`, `pyperclip`, `ollama`
3. **First-run model download:** `faster-whisper` downloads `base.en` (~150MB) on first instantiation. Run a one-time warm-up script during setup so the download happens explicitly, not mid-session.
4. Set up project structure.
5. Add a config file or CLI flags for:
   - activation hotkey
   - push-to-talk hotkey
   - Whisper model size
   - Ollama model
   - cleanup enabled/disabled
   - max recording duration
   - idle unload timeout
   - Ollama timeout
   - clipboard restore enabled/disabled

### Phase 2: Zero-Latency Audio Capture (`src/audio.py`)
1. Initialize microphone with `samplerate=16000, channels=1, dtype='float32'`.
2. Record raw audio stream directly into a numpy array in RAM (no `.wav` files).
3. On stop, return the flat float32 numpy array ready for Whisper.
4. Add audio guardrails:
   - configurable microphone device selection
   - minimum recording duration
   - maximum recording duration
   - silence-only detection
   - clipping or low-input warnings where feasible
   - friendly errors for missing microphone or denied permission

### Phase 3: Optimized Transcription (`src/transcriber.py`)
1. `load()`: Instantiate `WhisperModel("base.en", compute_type="int8")`. Run one silent warm-up inference to JIT-compile the model.
2. `transcribe(audio_np)`: Call with `beam_size=1, vad_filter=True` for minimum latency.
3. `unload()`: `del self.model`; call `gc.collect()`; measure whether memory is actually returned to the OS.
4. If transcription returns empty or near-empty output, skip LLM cleanup and paste nothing or show a short status message.

### Phase 4: Streaming LLM Post-Processing (`src/processor.py`)
1. `activate()`: Send a no-op Ollama request with `keep_alive=-1` to pre-load and pin the model.
2. `process(text)`: Stream response from Ollama. Accumulate the full cleaned text from the stream before paste.
3. `deactivate()`: Send a request with `keep_alive=0` to force Ollama to unload the model.
4. Prompt: instruct LLM to fix transcription errors, remove filler words, and return clean text only.
5. Add timeout and fallback behavior:
   - if Ollama is unavailable, paste raw Whisper output
   - if Ollama times out, paste raw Whisper output
   - if cleanup output is empty or invalid, paste raw Whisper output
6. Make cleanup optional via config or `--no-cleanup` for fastest possible output.

**Note:** Streaming the LLM response does not reduce end-to-end paste latency if the app waits for the full cleaned text before pasting. Streaming is still useful for responsiveness internally, but paste latency should be measured after the full cleanup step.

### Phase 5: Clipboard Output
1. After LLM returns cleaned text, write to clipboard with `pyperclip.copy(text)`.
2. Simulate `Cmd+V` with `pynput.keyboard.Controller` to paste instantly at active cursor.
3. This replaces character-by-character keystroke simulation — paste is instant regardless of text length.
4. Preserve the user's clipboard by default:
   - read existing clipboard before paste
   - copy dictated text
   - paste with `Cmd+V`
   - restore previous clipboard after a short configurable delay
5. If clipboard read/write fails, print a clear error and avoid destructive behavior where possible.
6. Consider a short confirmation or status print before paste to reduce risk of pasting into the wrong app.

### Phase 6: Main Controller (`src/app.py`)
1. Define `VoiceSession` class with `activate()` / `deactivate()` methods that manage model lifecycle.
2. On startup: verify Ollama is reachable (health check); print instructions; register `signal.SIGINT` handler for clean shutdown (calls `deactivate()` if active).
3. Implement the explicit state machine:
   - activation hotkey: `idle` -> `activating` -> `ready`; `ready` -> `deactivating` -> `idle`
   - push-to-talk down in `ready`: `ready` -> `recording`
   - push-to-talk up in `recording`: `recording` -> `processing` -> `ready`
4. Add an `is_processing` guard or state-based lock to prevent overlapping recordings.
5. Add timeout and cancellation behavior:
   - max recording duration stops recording automatically
   - Ollama timeout falls back to raw transcript
   - Ctrl+C during active session unloads models and exits
   - deactivation during processing either waits for completion or cancels cleanly, based on the chosen state rule
6. Add idle unload timer so the app can stay running without keeping models resident forever.

## Verification & Testing
- **Latency Check:** Measure from hotkey release to paste appearing on screen.
  - Cold start target: acceptable 2-8s depending on model load and hardware.
  - Warm raw transcription target: under ~1s for short clips where feasible.
  - Warm transcription + LLM cleanup target: ~1-3s depending on clip length and Ollama performance.
- **Load/Unload Check:** Monitor RAM before activation, during session, and after deactivation — confirm memory is released.
- **Clipboard Paste:** Verify text appears instantly in a text editor on hotkey release.
- **Clipboard Restore:** Verify the previous clipboard is restored after paste.
- **Overlap Guard:** Rapidly press/release hotkey twice — second press should be ignored while pipeline runs.
- **Ollama Down:** Start app without Ollama running — verify friendly error, not a crash.
- **Ollama Fallback:** Stop Ollama during a session — verify raw Whisper output is pasted.
- **Empty Audio:** Tap push-to-talk quickly or record silence — verify no confusing paste occurs.
- **Permission Failure:** Deny microphone or accessibility permissions — verify clear setup instructions.
- **Graceful Shutdown:** Press Ctrl+C during active session — verify Ollama model is unloaded cleanly.
- **State Tests:** Unit-test state transitions with mocked audio, transcription, processor, and paste layers.
- **Timeout Tests:** Unit-test recording timeout, Ollama timeout, and cleanup fallback behavior.

## Considerations
- **macOS Permissions:** Requires explicit **Accessibility** (for `pynput` keyboard simulation) and **Microphone** permissions. Grant these in System Settings → Privacy & Security before first run.
- **Clipboard Side Effect:** Clipboard preservation should be enabled by default because paste-based output otherwise overwrites user data.
- **Ollama Must Be Running:** The app checks Ollama health on startup. If Ollama is not running, print a clear error and exit rather than crashing mid-session.
- **Wrong-Focus Risk:** The app pastes into whatever field is focused. Keep status output clear and consider requiring push-to-talk release only after the user has selected the target field.
- **Platform Scope:** This plan is macOS-oriented because it uses `Cmd+V` and macOS privacy permissions. Document this explicitly or add platform-specific paste/hotkey handling later.
