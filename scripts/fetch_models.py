"""Download the speech models Aegrys needs.

faster-whisper fetches itself on first use, so only the two that don't are here:
Silero VAD (2 MB) and a Piper voice (63 MB). Everything lands under .cache/ so it
stays out of git and is trivial to relocate with AEGRYS_CACHE.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("AEGRYS_CACHE", Path(__file__).resolve().parents[1] / ".cache"))

PIPER_BASE = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
              "en/en_US/amy/low/en_US-amy-low")
FILES = [
    (CACHE / "vad" / "silero_vad.onnx",
     "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
     "Silero VAD"),
    (CACHE / "piper" / "en_US-amy-low.onnx", f"{PIPER_BASE}.onnx", "Piper voice"),
    (CACHE / "piper" / "en_US-amy-low.onnx.json", f"{PIPER_BASE}.onnx.json",
     "Piper voice config"),
]


def hook(name):
    def report(block, size, total):
        if total <= 0:
            return
        pct = min(100, block * size * 100 // total)
        sys.stdout.write(f"\r  {name}: {pct:3d}%  ({total/1e6:.0f} MB)")
        sys.stdout.flush()
    return report


def main() -> int:
    print(f"cache: {CACHE}\n")
    for dest, url, name in FILES:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {name}: already present ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, hook(name))
            tmp.replace(dest)
            print(f"\r  {name}: done ({dest.stat().st_size/1e6:.1f} MB)      ")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"\r  {name}: FAILED — {type(e).__name__}: {e}")
            return 1

    print("\nWhisper downloads itself on first run.")
    print("Next: ollama pull qwen2.5:3b-instruct-q4_K_M")
    print("      python scripts/seed_demo_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
