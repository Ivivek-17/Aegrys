"""S5b — concurrency benchmark, corrected.

S5's first attempt was invalid: the "solo" baseline ran cold right after a server
restart (TTFT 7.8s = model load), so a contended run appeared 92% FASTER than solo.
Fixes here:
  - warm the LLM before any measurement, and pin keep_alive so it is never evicted
  - build every ONNX/CT2 session ONCE and reuse across scenarios (per-scenario
    construction was churning RAM and forcing ollama to reload the model)
  - report ollama's own eval timing (excludes load + prefill) as the primary metric
  - measure the solo baseline BEFORE and AFTER to detect drift/thermal effects

Pass criteria:
  1. Piper RTF < 1.0 under load, else the audio output underruns (stutter).
  2. VAD p99 frame latency and pacing jitter << 32 ms, else barge-in is late.
  3. LLM degradation bounded and explainable.
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
N_PREDICT = 120
KEEP_ALIVE = "30m"

SENTENCES = [
    "The timer is set for five minutes.",
    "You have two meetings on your calendar today.",
    "I've added that reminder for tomorrow morning.",
]

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
        list(v.synthesize(SENTENCES[0]))            # warm
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
        list(m.transcribe(audio, beam_size=1, language="en")[0])   # warm
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


# ---------------------------------------------------------------- workers

class Loop(threading.Thread):
    def __init__(self, stop_evt, fn):
        super().__init__(daemon=True)
        self.stop_evt, self.fn, self.samples = stop_evt, fn, []

    def run(self):
        while not self.stop_evt.is_set():
            self.samples.append(self.fn())


def tts_step(threads):
    v = piper(threads)
    state = {"i": 0}

    def step():
        text = SENTENCES[state["i"] % len(SENTENCES)]
        state["i"] += 1
        t = time.perf_counter()
        chunks = list(v.synthesize(text))
        ms = (time.perf_counter() - t) * 1000
        n = sum(len(c.audio_int16_bytes) for c in chunks)
        return (ms / 1000) / (n / 2 / chunks[0].sample_rate)
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
        nxt = time.perf_counter()
        first = True
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


# ---------------------------------------------------------------- llm

def llm(threads, n_predict=N_PREDICT):
    body = {"model": MODEL, "stream": True, "keep_alive": KEEP_ALIVE,
            "prompt": "Describe the water cycle in detail.",
            "options": {"num_predict": n_predict, "temperature": 0,
                        "num_thread": threads}}
    req = urllib.request.Request(f"{HOST}/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n = None, 0
    rate = 0
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
    return {"decode_tok_s": rate, "ttft_ms": ttft, "load_ms": ld,
            "tokens": n, "wall_s": time.perf_counter() - t0}


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def run(label, lt, tt=None, st=None, vad=False):
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
    r = llm(lt)
    stop.set()
    for w in ws.values():
        w.join(timeout=30)

    row = {"scenario": label, "llm_threads": lt, "tts_threads": tt,
           "stt_threads": st, "vad": vad,
           "decode_tok_s": round(r["decode_tok_s"], 2),
           "ttft_ms": round(r["ttft_ms"], 1), "load_ms": round(r["load_ms"], 1)}
    out = f"{label:<32} LLM {r['decode_tok_s']:5.2f} tok/s TTFT {r['ttft_ms']:5.0f}ms"
    if r["load_ms"] > 50:
        out += f" [!load {r['load_ms']:.0f}ms]"
    if "tts" in ws:
        s = ws["tts"].samples
        row["tts_rtf_med"] = round(statistics.median(s), 3)
        row["tts_rtf_p95"] = round(pct(s, 95), 3)
        out += f" | TTS RTF {row['tts_rtf_med']:.3f}/p95 {row['tts_rtf_p95']:.3f}"
    if "stt" in ws:
        s = ws["stt"].samples
        row["stt_ms_med"] = round(statistics.median(s), 1)
        out += f" | STT {row['stt_ms_med']:.0f}ms"
    if "vad" in ws:
        row["vad_p99_ms"] = round(pct(ws["vad"].samples, 99), 3)
        row["vad_jitter_p99_ms"] = round(pct(ws["vad"].jitter, 99), 2)
        row["vad_jitter_max_ms"] = round(max(ws["vad"].jitter), 2)
        out += (f" | VAD p99 {row['vad_p99_ms']:.2f}ms "
                f"jit {row['vad_jitter_p99_ms']:+.1f}/{row['vad_jitter_max_ms']:+.1f}ms")
    print(out, flush=True)
    return row


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    print("warming LLM (load into RAM, pin keep_alive)...", flush=True)
    w = llm(6, n_predict=16)
    print(f"  warm-up: load {w['load_ms']:.0f} ms, TTFT {w['ttft_ms']:.0f} ms",
          flush=True)
    print("preloading TTS/STT/VAD sessions...", flush=True)
    piper(2), piper(4), whisper(2), vad_sess()
    print("  done\n", flush=True)

    rows = []
    print("=== baseline (before) ===", flush=True)
    rows.append(run("LLM solo t=6 [pre]", 6))
    base = rows[-1]["decode_tok_s"]

    print("\n=== designed overlap: LLM + TTS ===", flush=True)
    for lt, tt in ((6, 2), (6, 4), (4, 2), (8, 2)):
        rows.append(run(f"LLM t={lt} + TTS t={tt}", lt, tt=tt))

    print("\n=== response phase: LLM + TTS + VAD ===", flush=True)
    rows.append(run("LLM t=6 + TTS t=2 + VAD", 6, tt=2, vad=True))
    rows.append(run("LLM t=4 + TTS t=2 + VAD", 4, tt=2, vad=True))

    print("\n=== worst case: + STT (barge-in mid-response) ===", flush=True)
    rows.append(run("all four (6/2/2)", 6, tt=2, st=2, vad=True))
    rows.append(run("all four (4/2/2)", 4, tt=2, st=2, vad=True))

    print("\n=== baseline (after, drift check) ===", flush=True)
    rows.append(run("LLM solo t=6 [post]", 6))
    post = rows[-1]["decode_tok_s"]

    (HERE / "results" / "s5b_concurrency.json").write_text(json.dumps(rows, indent=2))

    drift = (post - base) / base * 100
    print("\n" + "=" * 78)
    print(f"baseline pre {base:.2f} / post {post:.2f} tok/s ({drift:+.1f}% drift)")
    ref = (base + post) / 2
    print(f"using mean baseline {ref:.2f} tok/s\n")
    print(f"{'scenario':<32}{'tok/s':>7}{'vs base':>9}{'TTS RTF':>9}{'VADjit':>8}")
    for r in rows:
        d = (r["decode_tok_s"] - ref) / ref * 100
        print(f"{r['scenario']:<32}{r['decode_tok_s']:7.2f}{d:+8.1f}%"
              f"{r.get('tts_rtf_med', float('nan')):9.3f}"
              f"{r.get('vad_jitter_p99_ms', float('nan')):8.2f}")

    rtf = max((r.get("tts_rtf_p95", 0) for r in rows), default=0)
    jit = max((r.get("vad_jitter_p99_ms", 0) for r in rows), default=0)
    reload_hits = [r["scenario"] for r in rows if r["load_ms"] > 50]
    print(f"\nworst TTS RTF p95      : {rtf:.3f}  "
          f"{'PASS' if rtf < 1.0 else 'FAIL'} (must be < 1.0)")
    print(f"worst VAD jitter p99   : {jit:+.2f} ms  "
          f"{'PASS' if jit < 32 else 'FAIL'} (must be << 32 ms)")
    if reload_hits:
        print(f"WARNING: model reloaded during: {reload_hits} - results suspect")
    else:
        print("no model reloads: LLM stayed resident throughout")


if __name__ == "__main__":
    main()
