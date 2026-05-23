# WhisperFlow

WhisperFlow is a lightweight, purely local, push-to-talk voice-to-text daemon. It's designed for ultra-low latency, allowing you to record spoken English and have it instantly transcribed, cleaned up by a local LLM, and pasted into your active cursor.

## Features

- **Push-to-Talk:** Record audio only while holding a hotkey (default: `F8`).
- **Local Transcription:** Uses `faster-whisper` for fast, private, on-device transcription.
- **LLM Cleanup:** Optional integration with Ollama to remove filler words and fix transcription errors.
- **Auto-Paste:** Automatically pastes the final text into your active application.
- **Efficient:** Models are loaded on demand and can be unloaded after a period of inactivity.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (optional, for LLM cleanup)
- PortAudio (for `sounddevice`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/whisperflow.git
   cd whisperflow
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. (Optional) Pull the Ollama model:
   ```bash
   ollama pull llama3.2:1b
   ```

## Configuration

Edit `config.toml` to customize hotkeys, models, and other settings.

## Usage

Run the daemon:
```bash
python3 src/app.py
```

- Press `F7` to activate voice mode.
- Hold `F8` to record.
- Release `F8` to transcribe and paste.

## License

MIT
