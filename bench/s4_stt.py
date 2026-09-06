"""S4 — faster-whisper decode latency on CPU.

Pass criteria (docs/DESIGN.md §8): a 3s utterance decodes in < 400 ms.
Measures cold vs warm, across model sizes / beam widths / thread counts.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
SRC = HERE / "jfk.wav"
SR = 16000
DURATIONS = (3, 5, 10)
REPEATS = 3


def clips():
    data, sr = sf.read(SRC, dtype="float32")
    assert sr == SR, f"expected {SR}, got {sr}"
    out = {}
    for d in DURATIONS:
        n = min(int(d * SR), len(data))
        out[d] = data[:n]
    return out


def transcribe(model, audio, beam):
    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, beam_size=beam, language="en")
    text = " ".join(s.text for s in segs)  # generator: must drain to finish decode
    return (time.perf_counter() - t0) * 1000, text.strip()


def main():
    audio = clips()
    results = []
    configs = [
        ("distil-small.en", 1, 2),
        ("distil-small.en", 1, 4),
        ("distil-small.en", 3, 4),
        ("base.en", 1, 4),
        ("small.en", 1, 4),
    ]

    for name, beam, threads in configs:
        print(f"\n=== {name} | beam={beam} | cpu_threads={threads} ===", flush=True)
        try:
            t0 = time.perf_counter()
            model = WhisperModel(
                name, device="cpu", compute_type="int8",
                cpu_threads=threads, num_workers=1,
            )
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"  model load: {load_ms:.0f} ms", flush=True)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"model": name, "beam": beam, "threads": threads,
                            "error": f"{type(e).__name__}: {e}"})
            continue

        # warm up so the first timed run isn't paying lazy-init costs
        cold_ms, _ = transcribe(model, audio[3], beam)
        print(f"  cold (3s): {cold_ms:.0f} ms", flush=True)

        for d in DURATIONS:
            runs = [transcribe(model, audio[d], beam)[0] for _ in range(REPEATS)]
            med = statistics.median(runs)
            rtf = (med / 1000) / d
            print(f"  {d:>2}s: median {med:6.0f} ms  (min {min(runs):.0f})  RTF {rtf:.3f}",
                  flush=True)
            results.append({
                "model": name, "beam": beam, "threads": threads, "dur_s": d,
                "median_ms": round(med, 1), "min_ms": round(min(runs), 1),
                "rtf": round(rtf, 4), "load_ms": round(load_ms, 1),
                "cold_ms": round(cold_ms, 1),
            })

        _, text = transcribe(model, audio[10], beam)
        print(f"  text(10s): {text[:90]}", flush=True)
        del model

    out = HERE / "results" / "s4_stt.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    ok = [r for r in results if r.get("dur_s") == 3 and r.get("median_ms", 1e9) < 400]
    print(f"\nS4 PASS: {len(ok)} config(s) decode 3s in <400ms")
    for r in ok:
        print(f"  - {r['model']} beam={r['beam']} t={r['threads']}: {r['median_ms']:.0f} ms")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
