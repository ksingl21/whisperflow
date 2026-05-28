#!/usr/bin/env python3
"""Generate WhisperFlow.icns — run via install.sh."""
import subprocess, sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow"], check=True)
    from PIL import Image, ImageDraw  # type: ignore


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size

    # ── background rounded square (indigo) ────────────────────────────── #
    d.rounded_rectangle(
        [0, 0, s - 1, s - 1],
        radius=int(s * 0.22),
        fill=(79, 70, 229),   # indigo-600
    )

    cx = s / 2

    # ── mic capsule ───────────────────────────────────────────────────── #
    mw = s * 0.30
    mh = s * 0.36
    mt = s * 0.14
    d.rounded_rectangle(
        [cx - mw / 2, mt, cx + mw / 2, mt + mh],
        radius=int(mw / 2),
        fill="white",
    )

    # ── bracket arc around bottom of capsule ──────────────────────────── #
    ar = s * 0.26
    ac_y = mt + mh * 0.55          # arc center y (mid-lower capsule)
    lw = max(2, int(s * 0.045))
    d.arc(
        [cx - ar, ac_y - ar, cx + ar, ac_y + ar],
        start=0, end=180,          # bottom semicircle
        fill="white", width=lw,
    )

    # ── vertical stem ─────────────────────────────────────────────────── #
    stem_top    = ac_y + ar
    stem_bottom = s * 0.76
    sw = max(2, int(s * 0.04))
    d.rectangle(
        [cx - sw / 2, stem_top, cx + sw / 2, stem_bottom],
        fill="white",
    )

    # ── base ─────────────────────────────────────────────────────────── #
    bw = s * 0.40
    bh = max(2, int(s * 0.045))
    d.rounded_rectangle(
        [cx - bw / 2, stem_bottom, cx + bw / 2, stem_bottom + bh],
        radius=int(bh / 2),
        fill="white",
    )

    return img


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("WhisperFlow.icns")

    iconset = Path("/tmp/WhisperFlow.iconset")
    iconset.mkdir(exist_ok=True)

    for s in (16, 32, 64, 128, 256, 512, 1024):
        draw_icon(s).save(iconset / f"icon_{s}x{s}.png")
        if s <= 512:
            draw_icon(s * 2).save(iconset / f"icon_{s}x{s}@2x.png")

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out_path)],
        check=True,
    )
    print(f"Icon → {out_path}")


if __name__ == "__main__":
    main()
