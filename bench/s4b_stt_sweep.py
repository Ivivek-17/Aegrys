"""S4b — follow-up sweep after S4 failed the <400ms bar.

S4 showed decode time is ~flat across utterance duration (Whisper pads every
input to a 30s mel window), so cost is encoder-dominated. That means:
  - distil-* models don't help on CPU (they shrink the decoder, not the encoder)
  - smaller *encoders* (tiny/base) are the only real lever
This sweeps encoder size x thread count x decode options to find the frontier.
"""

import json
import statistics
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
SR = 16000
REPEATS = 3
BAR_MS = 400


def main():
    data, sr = sf.read(HERE / "jfk.wav", dtype="float32")
    assert sr == SR
    a3 = data[: 3 * SR]

    rows = []
    for name in ("tiny.en", "base.en"):
        for threads in (2, 4, 6, 8):
            for fast in (False, True):
                model = WhisperModel(name, device="cpu", compute_type="int8",
                                     cpu_threads=threads, num_workers=1)
                kw = dict(beam_size=1, language="en")
                if fast:
                    kw.update(without_timestamps=True,
                              condition_on_previous_text=False)

                def run():
                    t0 = time.perf_counter()
                    segs, _ = model.transcribe(a3, **kw)
                    txt = " ".join(s.text for s in segs)
                    return (time.perf_counter() - t0) * 1000, txt

                run()  # warm
                runs = [run()[0] for _ in range(REPEATS)]
                med = statistics.median(runs)
                _, txt = run()
                tag = "no-ts" if fast else "base "
                flag = "PASS" if med < BAR_MS else "    "
                print(f"{flag} {name:<8} t={threads} {tag}  median {med:6.0f} ms "
                      f"(min {min(runs):.0f})", flush=True)
                rows.append({"model": name, "threads": threads,
                             "without_timestamps": fast,
                             "median_ms": round(med, 1),
                             "min_ms": round(min(runs), 1),
                             "text": txt.strip()})
                del model

    (HERE / "results" / "s4b_stt_sweep.json").write_text(json.dumps(rows, indent=2))
    best = min(rows, key=lambda r: r["median_ms"])
    print(f"\nbest: {best['model']} t={best['threads']} "
          f"no-ts={best['without_timestamps']} -> {best['median_ms']:.0f} ms")
    print(f"text: {best['text'][:80]}")
    passing = [r for r in rows if r["median_ms"] < BAR_MS]
    print(f"S4b: {len(passing)}/{len(rows)} configs under {BAR_MS} ms")


if __name__ == "__main__":
    main()
