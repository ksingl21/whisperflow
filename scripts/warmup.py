"""
First-run setup script.

Downloads the Whisper model and runs a warm-up inference so the model is
cached locally before your first real recording session.

Usage:
    python scripts/warmup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcriber import Transcriber


def main() -> None:
    print("WhisperFlow — first-run model setup")
    print("─" * 40)
    print("Downloading Whisper base.en model (~150 MB on first run)…")
    t = Transcriber(model_name="base.en", compute_type="int8")
    t.load()
    print("Warm-up complete.")
    t.unload()
    print()
    print("Setup done. Start the app with:")
    print("  python src/app.py")


if __name__ == "__main__":
    main()
