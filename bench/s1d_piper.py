"""S1d — Piper TTS as the R2 fallback after Kokoro failed S1.

Kokoro best case was RTF 1.055 (bar <0.6), leaving no headroom to run concurrently
with LLM decode. Piper is the designated fallback in docs/DESIGN.md R2.
Same bar, same text, same machine.
"""

import json
import time
import wave
from pathlib import Path

from piper import PiperVoice

CACHE = Path("D:/Aegrys/.cache/piper")
TEXT = "The timer is set for five minutes."
LONG = ("Good morning. You have three meetings today, starting with a design "
        "review at ten. The weather is sunny and twenty two degrees.")
BAR = 0.6


def bench(name, threads):
    voice = PiperVoice.load(str(CACHE / f"{name}.onnx"))
    try:
        voice.session.set_providers(["CPUExecutionProvider"])
    except Exception:
        pass

    def synth(text):
        t = time.perf_counter()
        chunks = list(voice.synthesize(text))
        ms = (time.perf_counter() - t) * 1000
        n = sum(len(c.audio_int16_bytes) for c in chunks)
        sr = chunks[0].sample_rate
        return ms, n / 2 / sr, chunks, sr

    synth(TEXT)  # warm
    out = {"voice": name, "threads": threads}
    for label, text in (("short", TEXT), ("long", LONG)):
        runs = [synth(text) for _ in range(3)]
        med = sorted(r[0] for r in runs)[1]
        dur = runs[0][1]
        rtf = (med / 1000) / dur
        flag = "PASS" if rtf < BAR else "FAIL"
        print(f"  {flag} {name:<22} {label:<6} {med:6.0f} ms -> {dur:5.2f}s "
              f"audio   RTF {rtf:.3f}", flush=True)
        out[label] = {"ms": round(med, 1), "audio_s": round(dur, 2),
                      "rtf": round(rtf, 4)}

    ms, dur, chunks, sr = synth(TEXT)
    with wave.open(f"piper_{name}.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for c in chunks:
            w.writeframes(c.audio_int16_bytes)
    out["sample_sr"] = sr
    return out


def main():
    print("=== Piper TTS (R2 fallback) ===", flush=True)
    rows = []
    for name in ("en_US-amy-low", "en_US-lessac-medium"):
        try:
            rows.append(bench(name, 4))
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__}: {e}", flush=True)

    Path("results/s1d_piper.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r["short"]["rtf"] < BAR]
    if ok:
        b = min(ok, key=lambda r: r["short"]["rtf"])
        speedup = 1.055 / b["short"]["rtf"]
        print(f"\nbest: {b['voice']} RTF {b['short']['rtf']:.3f} "
              f"({b['short']['ms']:.0f} ms for {b['short']['audio_s']:.2f}s audio)")
        print(f"=> {speedup:.1f}x faster than Kokoro fp32 best (RTF 1.055)")
    print(f"S1d: {len(ok)}/{len(rows)} voices under RTF {BAR}")


if __name__ == "__main__":
    main()
