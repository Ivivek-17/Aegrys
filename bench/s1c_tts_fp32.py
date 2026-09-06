"""S1c — is Kokoro's RTF 5.5 a model problem or a runtime-config problem?

S1b measured int8 at RTF ~5.5 (bar is <0.6), with cold FASTER than warm, which
suggests bad ONNX kernel selection rather than genuine model cost. Compares
fp32 vs int8 across explicit intra_op thread counts using Kokoro.from_session().
"""

import os
import time

os.environ["OMP_NUM_THREADS"] = "4"

import espeakng_loader
import onnxruntime as ort
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

from kokoro_onnx import Kokoro  # noqa: E402

CACHE = "D:/Aegrys/.cache/kokoro"
VOICES = f"{CACHE}/voices-v1.0.bin"
TEXT = "The timer is set for five minutes."
BAR = 0.6


def bench(model_file, intra, inter=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(f"{CACHE}/{model_file}", so,
                                providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, VOICES)
    load = (time.perf_counter() - t0) * 1000

    k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us")  # warm
    times = []
    for _ in range(3):
        t = time.perf_counter()
        s, sr = k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us")
        times.append((time.perf_counter() - t) * 1000)
    med = sorted(times)[1]
    dur = len(s) / sr
    rtf = (med / 1000) / dur
    flag = "PASS" if rtf < BAR else "FAIL"
    print(f"  {flag} {model_file:<24} intra={intra}  {med:7.0f} ms  "
          f"RTF {rtf:6.3f}  (load {load:.0f} ms)", flush=True)
    return {"model": model_file, "intra": intra, "ms": med, "rtf": rtf}


def main():
    rows = []
    print("=== fp32 vs int8 across intra_op threads ===", flush=True)
    for model in ("kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx"):
        for intra in (2, 4, 6):
            try:
                rows.append(bench(model, intra))
            except Exception as e:
                print(f"  ERR {model} intra={intra}: {type(e).__name__}: {e}",
                      flush=True)

    ok = [r for r in rows if r["rtf"] < BAR]
    best = min(rows, key=lambda r: r["rtf"])
    print(f"\nbest: {best['model']} intra={best['intra']} "
          f"RTF {best['rtf']:.3f} ({best['ms']:.0f} ms)")
    print(f"S1c: {len(ok)}/{len(rows)} configs under RTF {BAR}")

    import json
    with open("results/s1c_tts_fp32.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
