"""S1b — minimal Kokoro timing. Full sweep (s1_tts.py) was too slow to iterate on.

Isolates WHERE the time goes: model load vs phonemization vs ONNX inference,
for one short sentence at one thread count. Unbuffered, prints as it goes.
"""

import os
import sys
import time

t_start = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter()-t_start:7.2f}s] {msg}", flush=True)


os.environ.setdefault("OMP_NUM_THREADS", "4")

log("importing espeakng_loader + phonemizer")
import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

log("importing kokoro_onnx")
from kokoro_onnx import Kokoro  # noqa: E402

log("constructing Kokoro (loads 92MB int8 onnx + voices)")
k = Kokoro("D:/Aegrys/.cache/kokoro/kokoro-v1.0.int8.onnx",
           "D:/Aegrys/.cache/kokoro/voices-v1.0.bin")
log("Kokoro ready")

SHORT = "The timer is set for five minutes."

log("synth #1 (cold)")
t = time.perf_counter()
s, sr = k.create(SHORT, voice="af_heart", speed=1.0, lang="en-us")
cold = (time.perf_counter() - t) * 1000
log(f"cold: {cold:.0f} ms -> {len(s)/sr:.2f}s audio @ {sr}Hz  RTF {(cold/1000)/(len(s)/sr):.3f}")

for i in range(3):
    t = time.perf_counter()
    s, sr = k.create(SHORT, voice="af_heart", speed=1.0, lang="en-us")
    ms = (time.perf_counter() - t) * 1000
    log(f"warm #{i+1}: {ms:.0f} ms -> {len(s)/sr:.2f}s audio  "
        f"RTF {(ms/1000)/(len(s)/sr):.3f}")

import soundfile as sf  # noqa: E402
sf.write("tts_sample.wav", s, sr)
log(f"wrote tts_sample.wav ({len(s)/sr:.2f}s)")
sys.exit(0)
