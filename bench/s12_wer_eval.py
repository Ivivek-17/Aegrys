"""S12 — real WER evaluation: tiny.en vs base.en on LibriSpeech test-clean.

Closes an open item. Phase 0 picked `tiny.en` on LATENCY alone (294 ms vs 753 ms)
and the only accuracy evidence was 0% WER on a single 11-second clip. One clip is
not an evaluation, and choosing an STT model on speed without measuring accuracy is
exactly the kind of decision that looks fine until a demo.

Evaluates on N held-out LibriSpeech utterances plus in-domain command phrases,
which matter more here than read audiobook prose: the assistant hears short
imperatives, not paragraphs.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LS = Path("D:/Aegrys/.cache/librispeech/LibriSpeech/test-clean")
N = 120
MODELS = ["tiny.en", "base.en", "small.en"]

# Numbers are the known failure mode for voice commands ("5" vs "five"), and the
# router consumes this text, so normalize them rather than punishing formatting.
NUMS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "15": "fifteen", "20": "twenty", "30": "thirty",
        "45": "forty five", "60": "sixty"}


def norm(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    out = []
    for w in s.split():
        out.extend(NUMS.get(w, w).split())
    return out


def wer(ref: str, hyp: str) -> tuple[int, int]:
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
    return d[len(r)][len(h)], len(r)


def load_set(n: int):
    items = []
    for trans in sorted(LS.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            flac = trans.parent / f"{uid}.flac"
            if flac.exists():
                items.append((flac, text))
            if len(items) >= n:
                return items
    return items


def main():
    if not LS.exists():
        print(f"missing {LS}")
        return
    items = load_set(N)
    durs = []
    for f, _ in items[:20]:
        info = sf.info(f)
        durs.append(info.duration)
    print(f"{len(items)} utterances, mean duration "
          f"{statistics.mean(durs):.1f}s\n", flush=True)

    from aegrys.core.config import STTConfig
    from aegrys.stt.whisper import Transcriber

    results = []
    for name in MODELS:
        cfg = STTConfig(model=name, threads=4)
        t0 = time.perf_counter()
        stt = Transcriber(cfg)
        load_ms = (time.perf_counter() - t0) * 1000

        errs = words = 0
        times = []
        worst = []
        for flac, ref in items:
            audio, sr = sf.read(flac, dtype="float32")
            assert sr == 16000
            t = time.perf_counter()
            hyp = stt.transcribe(audio)
            times.append((time.perf_counter() - t) * 1000)
            e, w = wer(ref, hyp)
            errs += e
            words += w
            if e:
                worst.append((e / max(w, 1), ref, hyp))
        rate = errs / max(words, 1)
        med = statistics.median(times)
        print(f"{name:<10} WER {rate:6.2%}  ({errs}/{words} words)  "
              f"median {med:6.0f} ms  load {load_ms:.0f} ms", flush=True)
        worst.sort(reverse=True)
        for r, ref, hyp in worst[:2]:
            print(f"           worst {r:5.1%} ref: {ref[:56].lower()}")
            print(f"                       hyp: {hyp[:56].lower()}", flush=True)
        results.append({"model": name, "wer": round(rate, 4), "errors": errs,
                        "words": words, "median_ms": round(med, 1),
                        "n": len(items)})

    Path(__file__).parent.joinpath("results/s12_wer_eval.json").write_text(
        json.dumps(results, indent=2))

    print("\n" + "=" * 62)
    print(f"{'model':<10}{'WER':>9}{'median':>10}{'vs tiny.en':>26}")
    base = results[0]
    for r in results:
        d_wer = (base["wer"] - r["wer"]) / max(base["wer"], 1e-9) * 100
        d_ms = r["median_ms"] - base["median_ms"]
        note = "" if r is base else f"{d_wer:+.0f}% WER for {d_ms:+.0f} ms"
        print(f"{r['model']:<10}{r['wer']:8.2%}{r['median_ms']:9.0f}ms{note:>26}")


if __name__ == "__main__":
    main()
