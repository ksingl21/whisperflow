#!/bin/bash
set -e

APPDATA="$HOME/Library/Application Support/WhisperFlow"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$SCRIPT_DIR/WhisperFlow.app"

echo "=== WhisperFlow Installer ==="

# ── 1. Copy runtime files ─────────────────────────────────────────────────── #
echo "→ Installing app data to $APPDATA"
mkdir -p "$APPDATA"
cp -r "$SCRIPT_DIR/src"              "$APPDATA/"
cp    "$SCRIPT_DIR/config.toml"      "$APPDATA/"
cp    "$SCRIPT_DIR/requirements.txt" "$APPDATA/"

# ── 2. Create venv and install dependencies ───────────────────────────────── #
if [ ! -d "$APPDATA/venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv "$APPDATA/venv"
    echo "→ Installing Python dependencies (this may take a few minutes)..."
    "$APPDATA/venv/bin/pip" install --quiet -r "$APPDATA/requirements.txt"
else
    echo "→ Virtual environment already exists, skipping."
fi

# ── 3. Pull Ollama model if Ollama is available ───────────────────────────── #
if command -v ollama &>/dev/null; then
    MODEL=$(awk '/^\[ollama\]/{f=1} f && /^model/{print; exit}' "$APPDATA/config.toml" | awk -F'"' '{print $2}')
    if [ -n "$MODEL" ]; then
        echo "→ Pulling Ollama model: $MODEL"
        ollama pull "$MODEL" || echo "  (Ollama pull failed — you can run it manually later)"
    fi
fi

# ── 4. Compile native binary launcher (no terminal window) ───────────────── #
echo "→ Compiling native launcher..."
LAUNCHER="$APP/Contents/MacOS/WhisperFlow"
cc -Os -o "$LAUNCHER" "$SCRIPT_DIR/launcher.c" 2>/dev/null \
    && echo "  Compiled OK" \
    || echo "  (cc not found — keeping shell script launcher)"

# ── 5. Generate app icon ──────────────────────────────────────────────────── #
echo "→ Generating app icon..."
ICON_OUT="$APP/Contents/Resources/AppIcon.icns"
mkdir -p "$APP/Contents/Resources"
"$APPDATA/venv/bin/python" "$SCRIPT_DIR/scripts/make_icon.py" "$ICON_OUT" \
    && echo "  Icon OK" \
    || echo "  (Icon generation failed — app will use default icon)"

# ── 6. Sign and install to /Applications ─────────────────────────────────── #
echo "→ Signing app bundle..."
xattr -cr "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP" 2>/dev/null

echo "→ Copying WhisperFlow.app to /Applications..."
rm -rf /Applications/WhisperFlow.app
cp -r "$APP" /Applications/WhisperFlow.app
xattr -cr /Applications/WhisperFlow.app 2>/dev/null || true
codesign --force --deep --sign - /Applications/WhisperFlow.app 2>/dev/null

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Next steps:"
echo "  1. Open /Applications/WhisperFlow.app"
echo "  2. System Settings → Privacy & Security → Accessibility → add WhisperFlow"
echo "  3. System Settings → Privacy & Security → Microphone → add WhisperFlow"
echo "  4. Press Shift+V to activate voice mode"
echo ""
echo "Config file: $APPDATA/config.toml"
