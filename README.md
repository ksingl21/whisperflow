# WhisperFlow

WhisperFlow is a lightweight, purely local voice-to-text app for macOS. Speak, and your words are instantly transcribed, cleaned up by a local LLM, and pasted into your active application — no cloud, no latency.

## Features

- **Global hotkeys:** Activate and record from any app without switching windows.
- **Toggle recording:** Press Space to start, and it auto-stops after 2 seconds of silence.
- **Local transcription:** Uses `faster-whisper` for fast, private, on-device transcription.
- **LLM cleanup:** Optional Ollama integration removes filler words and fixes transcription errors.
- **Auto-paste:** Transcribed text is pasted directly into your active application.
- **macOS notifications:** Get notified when the app is ready, recording completes, and text is pasted.
- **Memory efficient:** Models load on demand and unload automatically after 5 minutes of inactivity.

## Prerequisites

- macOS
- Python 3.10+
- [Ollama](https://ollama.com/) (optional, for LLM cleanup)
- PortAudio — install via Homebrew: `brew install portaudio`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ksingl21/whisperflow.git
   cd whisperflow
   ```

2. Run the installer — this sets up the venv, installs dependencies, and copies `WhisperFlow.app` to `/Applications`:
   ```bash
   ./install.sh
   ```

   The installer takes a few minutes the first time (downloading Python packages and the Whisper model).

## Launching the App

Open `WhisperFlow` from Spotlight (`Cmd+Space`, type WhisperFlow) or from `/Applications`. No terminal needed.

## Permissions (first run only)

macOS requires two permissions before WhisperFlow can work:

1. **Accessibility** — needed to listen for global hotkeys.
   System Settings → Privacy & Security → Accessibility → add `WhisperFlow`

2. **Microphone** — needed to record audio.
   System Settings → Privacy & Security → Microphone → add `WhisperFlow`

After granting permissions, relaunch the app.

## Updating

To update after pulling new changes:
```bash
git pull
./install.sh
```

## Usage

| Action | Hotkey |
|--------|--------|
| Toggle voice mode on/off | **Shift + V** |
| Start recording | **Space** |
| Stop recording (manual) | **Space** again |
| Auto-stop | After **2 seconds of silence** |
| Quit | **Ctrl + C** (terminal) or quit the app |

**Workflow:**
1. Launch `WhisperFlow.app` — a notification confirms it's running.
2. Switch focus to any app where you want to type (browser, notes, editor, etc.).
3. Press **Shift + V** to activate voice mode — Whisper loads and a notification appears.
4. Press **Space** to start recording and speak naturally.
5. Stop speaking — after 2 seconds of silence the recording stops automatically, transcribes, and pastes the text into your active app.

## Configuration

Edit `config.toml` to customise behaviour:

```toml
[hotkeys]
activation    = "shift+v"   # toggle voice mode on/off
push_to_talk  = "space"     # start/stop recording

[whisper]
model        = "base.en"    # whisper model size (tiny.en, base.en, small.en, medium.en)
compute_type = "int8"       # int8 (CPU) or float16 (GPU)

[ollama]
model            = "llama3.2:1b"
cleanup_enabled  = true     # set false to skip LLM and paste raw Whisper output

[recording]
max_seconds             = 0.0   # hard time limit in seconds (0 = disabled)
min_seconds             = 0.3   # ignore recordings shorter than this
silence_timeout_seconds = 2.0   # auto-stop after this many seconds of silence

[session]
idle_unload_seconds = 300.0     # unload models after this many idle seconds (0 = never)
```

## License

MIT
