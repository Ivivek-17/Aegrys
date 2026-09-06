"""S5c — concurrency benchmark, second correction.

S5b was still confounded by two effects:
  1. ollama RELOADS the model whenever `num_thread` changes between requests
     (7/10 scenarios reloaded). Per-request LLM thread sweeping is impossible.
  2. the solo baseline drifted -20% (15.57 -> 12.46 tok/s) over a ~10 min run,
     which is either thermal throttling or reload contamination -- indistinguishable.

Design here:
  - num_thread is PINNED at 6 for every LLM call, so the model never reloads.
    (S2 already established 6 as optimal; the sweep is what caused the reloads.)
  - a solo baseline runs BEFORE AND AFTER every scenario, so each scenario is
    scored against the local interpolated baseline. Thermal drift cancels out,
    and is separately reported as a first-class result.
"""

import json
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

HOST = "http://127.0.0.1:11435"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
HERE = Path(__file__).parent
VAD_PATH = "D:/Aegrys/.cache/vad/silero_vad.onnx"
PIPER_PATH = "D:/Aegrys/.cache/piper/en_US-amy-low.onnx"
LLM_THREADS = 6          # PINNED - changing this mid-run forces a model reload
N_PREDICT = 120
KEEP_ALIVE = "30m"

SENTENCES = ["The timer is set for five minutes.",
             "You have two meetings on your calendar today.",
             "I've added that reminder for tomorrow morning."]

_pool = {}


def piper(threads):
    key = ("piper", threads)
    if key not in _pool:
        from piper import PiperVoice
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        v = PiperVoice.load(PIPER_PATH)
        v.session = ort.InferenceSession(PIPER_PATH, so,
                                         providers=["CPUExecutionProvider"])
        list(v.synthesize(SENTENCES[0]))
        _pool[key] = v
    return _pool[key]


def whisper(threads):
    key = ("whisper", threads)
    if key not in _pool:
        from faster_whisper import WhisperModel
        m = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                         cpu_threads=threads, num_workers=1)
        data, sr = sf.read(HERE / "jfk.wav", dtype="float32")
        audio = data[: 3 * sr]
        list(m.transcribe(audio, beam_size=1, language="en")[0])
        _pool[key] = (m, audio)
    return _pool[key]


def vad_sess():
    if "vad" not in _pool:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _pool["vad"] = ort.InferenceSession(VAD_PATH, so,
                                            providers=["CPUExecutionProvider"])
    return _pool["vad"]


class Loop(threading.Thread):
    def __init__(self, stop_evt, fn):
        super().__init__(daemon=True)
        self.stop_evt, self.fn, self.samples = stop_evt, fn, []

    def run(self):
        while not self.stop_evt.is_set():
            self.samples.append(self.fn())


def tts_step(threads):
    v, st = piper(threads), {"i": 0}

    def step():
        text = SENTENCES[st["i"] % len(SENTENCES)]
        st["i"] += 1
        t = time.perf_counter()
        ch = list(v.synthesize(text))
        ms = (time.perf_counter() - t) * 1000
        n = sum(len(c.audio_int16_bytes) for c in ch)
        return (ms / 1000) / (n / 2 / ch[0].sample_rate)
    return step


def stt_step(threads):
    m, audio = whisper(threads)

    def step():
        t = time.perf_counter()
        segs, _ = m.transcribe(audio, beam_size=1, language="en",
                               without_timestamps=True,
                               condition_on_previous_text=False)
        _ = " ".join(s.text for s in segs)
        return (time.perf_counter() - t) * 1000
    return step


class VadLoop(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.sess = vad_sess()
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)
        self.frame = np.zeros((1, 512), dtype=np.float32)
        self.samples, self.jitter = [], []

    def run(self):
        nxt, first = time.perf_counter(), True
        while not self.stop_evt.is_set():
            now = time.perf_counter()
            if not first:
                self.jitter.append((now - nxt) * 1000)
            first = False
            nxt = now + 0.032
            t = time.perf_counter()
            out = self.sess.run(None, {"input": self.frame, "state": self.state,
                                       "sr": self.sr})
            self.state = out[1]
            self.samples.append((time.perf_counter() - t) * 1000)
            s = nxt - time.perf_counter()
            if s > 0:
                time.sleep(s)


def llm(n_predict=N_PREDICT):
    body = {"model": MODEL, "stream": True, "keep_alive": KEEP_ALIVE,
            "prompt": "Describe the water cycle in detail.",
            "options": {"num_predict": n_predict, "temperature": 0,
                        "num_thread": LLM_THREADS}}
    req = urllib.request.Request(f"{HOST}/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n, rate, ld = None, 0, 0, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("response"):
                n += 1
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
            if d.get("done"):
                ev, evd = d.get("eval_count", 0), d.get("eval_duration", 1)
                rate = ev / (evd / 1e9) if evd else 0
                ld = d.get("load_duration", 0) / 1e6
                break
    return {"decode_tok_s": rate, "ttft_ms": ttft, "load_ms": ld}


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def run(label, tt=None, st=None, vad=False):
    stop = threading.Event()
    ws = {}
    if tt:
        ws["tts"] = Loop(stop, tts_step(tt))
    if st:
        ws["stt"] = Loop(stop, stt_step(st))
    if vad:
        ws["vad"] = VadLoop(stop)
    for w in ws.values():
        w.start()
    time.sleep(0.4)
    r = llm()
    stop.set()
    for w in ws.values():
        w.join(timeout=30)

    row = {"scenario": label, "decode_tok_s": round(r["decode_tok_s"], 2),
           "ttft_ms": round(r["ttft_ms"], 1), "load_ms": round(r["load_ms"], 1),
           "t": time.time()}
    out = f"  {label:<30} {r['decode_tok_s']:6.2f} tok/s"
    if r["load_ms"] > 50:
        out += f"  [!RELOAD {r['load_ms']:.0f}ms]"
    if "tts" in ws:
        s = ws["tts"].samples
        row["tts_rtf_med"] = round(statistics.median(s), 3)
        row["tts_rtf_p95"] = round(pct(s, 95), 3)
        row["tts_n"] = len(s)
        out += f" | TTS RTF {row['tts_rtf_med']:.3f}/p95 {row['tts_rtf_p95']:.3f}"
    if "stt" in ws:
        s = ws["stt"].samples
        row["stt_ms_med"] = round(statistics.median(s), 1)
        out += f" | STT {row['stt_ms_med']:.0f}ms"
    if "vad" in ws:
        row["vad_p99_ms"] = round(pct(ws["vad"].samples, 99), 3)
        row["vad_jit_p99"] = round(pct(ws["vad"].jitter, 99), 2)
        row["vad_jit_max"] = round(max(ws["vad"].jitter), 2)
        out += (f" | VAD p99 {row['vad_p99_ms']:.2f}ms "
                f"jit {row['vad_jit_p99']:+.1f}/{row['vad_jit_max']:+.1f}ms")
    print(out, flush=True)
    return row


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print("warming LLM + preloading sessions...", flush=True)
    llm(n_predict=16)
    piper(2), piper(4), whisper(2), vad_sess()
    llm(n_predict=16)
    print("  ready\n", flush=True)

    scenarios = [
        ("LLM + TTS t=2", dict(tt=2)),
        ("LLM + TTS t=4", dict(tt=4)),
        ("LLM + TTS t=2 + VAD", dict(tt=2, vad=True)),
        ("LLM + TTS t=2 + STT t=2 + VAD", dict(tt=2, st=2, vad=True)),
    ]

    rows, bases = [], []
    print("interleaved: baseline before AND after every scenario", flush=True)
    b = run("baseline solo")
    bases.append(b)
    rows.append(b)
    for label, kw in scenarios:
        r = run(label, **kw)
        rows.append(r)
        b = run("baseline solo")
        bases.append(b)
        rows.append(b)
        # score against the mean of the bracketing baselines
        local = (bases[-2]["decode_tok_s"] + bases[-1]["decode_tok_s"]) / 2
        r["local_baseline"] = round(local, 2)
        r["degradation_pct"] = round((r["decode_tok_s"] - local) / local * 100, 1)

    (HERE / "results" / "s5c_concurrency.json").write_text(json.dumps(rows, indent=2))

    bl = [b["decode_tok_s"] for b in bases]
    print("\n" + "=" * 78)
    print(f"baseline trace (thermal drift): "
          f"{' -> '.join(f'{x:.2f}' for x in bl)}")
    drift = (bl[-1] - bl[0]) / bl[0] * 100
    print(f"drift over run: {drift:+.1f}%  "
          f"({'THERMAL THROTTLING' if drift < -8 else 'stable'})")
    reloads = [r['scenario'] for r in rows if r['load_ms'] > 50]
    print(f"model reloads: {len(reloads)} "
          f"({'clean' if not reloads else reloads})")

    print(f"\n{'scenario':<32}{'tok/s':>8}{'base':>8}{'degrade':>9}"
          f"{'TTSrtf':>8}{'VADjit':>8}")
    for r in rows:
        if r["scenario"] == "baseline solo":
            continue
        print(f"{r['scenario']:<32}{r['decode_tok_s']:8.2f}"
              f"{r['local_baseline']:8.2f}{r['degradation_pct']:+8.1f}%"
              f"{r.get('tts_rtf_med', float('nan')):8.3f}"
              f"{r.get('vad_jit_p99', float('nan')):8.2f}")

    scen = [r for r in rows if r["scenario"] != "baseline solo"]
    rtf = max(r.get("tts_rtf_p95", 0) for r in scen)
    jit = max((r.get("vad_jit_max", 0) for r in scen), default=0)
    worst = min(r["degradation_pct"] for r in scen)
    print(f"\nworst TTS RTF p95    : {rtf:.3f}   "
          f"{'PASS' if rtf < 1.0 else 'FAIL'}  (must be < 1.0)")
    print(f"worst VAD jitter max : {jit:+.2f} ms   "
          f"{'PASS' if jit < 32 else 'FAIL'}  (must be << 32 ms)")
    print(f"worst LLM degradation: {worst:+.1f}%")


if __name__ == "__main__":
    main()
