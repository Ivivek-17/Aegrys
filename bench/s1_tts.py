"""S1 — Kokoro-82M TTS on Windows CPU.

Pass criteria (docs/DESIGN.md §8): installs, and RTF < 0.6 on 10s of speech.

Also measures time-to-first-audio for a SHORT sentence, which is the number that
actually matters for TTFA (§7) -- we only synthesize sentence 1 before playback
starts, so whole-paragraph RTF is the wrong metric to optimize.

NOTE: set PYTHONIOENCODING=utf-8 on Windows. espeak returns IPA, and the default
cp1252 console encoding raises UnicodeEncodeError on characters like U+025B.
"""

import json
import os
import statistics
import time
from pathlib import Path

import espeakng_loader
import numpy as np
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

from kokoro_onnx import Kokoro  # noqa: E402  (must follow espeak wiring)

CACHE = Path("D:/Aegrys/.cache/kokoro")
HERE = Path(__file__).parent
VOICE = "af_heart"
SR = 24000

SHORT = "The timer is set for five minutes."
MEDIUM = ("I've added that reminder for tomorrow at six. "
          "You also have two meetings on your calendar.")
LONG = (
    "Good morning. You have three meetings today, starting with a design review "
    "at ten. Your first reminder is to call the dentist before noon. The weather "
    "is sunny and twenty two degrees, so it should be a pleasant afternoon. "
    "I've also summarized your unread email; there are four messages that look "
    "important, and the rest appear to be newsletters you can safely ignore.")


def main():
    results = {}
    onnx = CACHE / "kokoro-v1.0.int8.onnx"
    voices = CACHE / "voices-v1.0.bin"
    print(f"model: {onnx.name} ({onnx.stat().st_size/1e6:.0f} MB)", flush=True)

    for threads in (2, 4, 6):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        t0 = time.perf_counter()
        k = Kokoro(str(onnx), str(voices))
        load_ms = (time.perf_counter() - t0) * 1000

        def synth(text):
            t = time.perf_counter()
            samples, sr = k.create(text, voice=VOICE, speed=1.0, lang="en-us")
            ms = (time.perf_counter() - t) * 1000
            return ms, len(samples) / sr, samples, sr

        synth(SHORT)  # warm

        print(f"\n=== OMP_NUM_THREADS={threads} (load {load_ms:.0f} ms) ===",
              flush=True)
        row = {"threads": threads, "load_ms": round(load_ms, 1)}
        for label, text in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
            runs = [synth(text) for _ in range(3)]
            med = statistics.median(r[0] for r in runs)
            audio_s = runs[0][1]
            rtf = (med / 1000) / audio_s
            flag = "PASS" if rtf < 0.6 else "FAIL"
            print(f"  {flag} {label:<7} {med:6.0f} ms synth -> {audio_s:5.2f}s "
                  f"audio   RTF {rtf:.3f}", flush=True)
            row[label] = {"synth_ms": round(med, 1), "audio_s": round(audio_s, 2),
                          "rtf": round(rtf, 4)}
        results[f"threads_{threads}"] = row

    # Save a sample so S3 has real speech to play, and we can listen to quality.
    os.environ["OMP_NUM_THREADS"] = "4"
    k = Kokoro(str(onnx), str(voices))
    samples, sr = k.create(MEDIUM, voice=VOICE, speed=1.0, lang="en-us")
    import soundfile as sf
    out_wav = HERE / "tts_sample.wav"
    sf.write(out_wav, samples, sr)
    print(f"\nwrote {out_wav} ({len(samples)/sr:.2f}s @ {sr} Hz)")
    results["sample"] = {"path": str(out_wav), "sr": sr,
                         "dur_s": round(len(samples) / sr, 2)}

    (HERE / "results" / "s1_tts.json").write_text(json.dumps(results, indent=2))

    best = min((r for k_, r in results.items() if k_.startswith("threads_")),
               key=lambda r: r["short"]["synth_ms"])
    print(f"\nfastest short-sentence synth: {best['short']['synth_ms']:.0f} ms "
          f"@ {best['threads']} threads  (this is the TTFA-relevant number)")


if __name__ == "__main__":
    main()
