"""S4c — can we shrink Whisper's 30s encoder window, and what does accuracy cost?

S4b found tiny.en fast but inaccurate, base.en accurate but slow. Whisper pads
every input to a 30s mel window, so encoder cost is fixed. faster-whisper exposes
`chunk_length`; if a shorter window works, we get base.en accuracy at lower cost.

Reports latency AND word error rate, because S4b showed latency alone is a trap.
"""

import json
import re
import statistics
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
SR = 16000
REPEATS = 3

REF = ("and so my fellow americans ask not what your country can do for you "
       "ask what you can do for your country")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / max(len(r), 1)


def main():
    data, sr = sf.read(HERE / "jfk.wav", dtype="float32")
    assert sr == SR
    full = data  # ~11s, full reference sentence

    rows = []
    for name in ("tiny.en", "base.en", "small.en"):
        for cl in (None, 10, 15):
            try:
                model = WhisperModel(name, device="cpu", compute_type="int8",
                                     cpu_threads=6, num_workers=1)

                def run():
                    t0 = time.perf_counter()
                    kw = dict(beam_size=1, language="en", without_timestamps=True,
                              condition_on_previous_text=False)
                    if cl is not None:
                        kw["chunk_length"] = cl
                    segs, _ = model.transcribe(full, **kw)
                    txt = " ".join(s.text for s in segs)
                    return (time.perf_counter() - t0) * 1000, txt.strip()

                run()  # warm
                runs = [run() for _ in range(REPEATS)]
                med = statistics.median(r[0] for r in runs)
                txt = runs[-1][1]
                e = wer(REF, txt)
                print(f"{name:<9} chunk_length={str(cl):<4} "
                      f"{med:6.0f} ms  WER {e:5.1%}  | {txt[:60]}", flush=True)
                rows.append({"model": name, "chunk_length": cl,
                             "median_ms": round(med, 1), "wer": round(e, 4),
                             "text": txt})
                del model
            except Exception as ex:
                print(f"{name:<9} chunk_length={str(cl):<4} FAILED: "
                      f"{type(ex).__name__}: {str(ex)[:90]}", flush=True)
                rows.append({"model": name, "chunk_length": cl,
                             "error": f"{type(ex).__name__}: {ex}"})

    (HERE / "results" / "s4c_window_wer.json").write_text(json.dumps(rows, indent=2))
    good = [r for r in rows if r.get("wer", 1) == 0]
    if good:
        b = min(good, key=lambda r: r["median_ms"])
        print(f"\nfastest zero-WER config: {b['model']} "
              f"chunk_length={b['chunk_length']} -> {b['median_ms']:.0f} ms")


if __name__ == "__main__":
    main()
