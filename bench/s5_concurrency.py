"""S5 — concurrency benchmark: what happens when the stages fight for 12 threads.

Every Phase 0 number was measured with ONE component running. The real pipeline
overlaps them by design (docs/DESIGN.md §3): TTS synthesizes sentence N while the
LLM still generates N+1, and during playback the VAD must keep listening for
barge-in. This measures the contention.

Pass criteria:
  1. Piper RTF stays < 1.0 under load  -- else the audio output underruns (stutter).
  2. VAD p99 frame latency << 32 ms and low pacing jitter -- else barge-in detection
     is late or misses, which breaks the single most important feature.
  3. LLM degradation is bounded and understood.

Note: ollama runs out-of-process, so its threads are genuinely outside this
process's GIL -- the contention measured here is real OS-level CPU contention.
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

SENTENCES = [
    "The timer is set for five minutes.",
    "You have two meetings on your calendar today.",
    "I've added that reminder for tomorrow morning.",
]


# ---------------------------------------------------------------- workers

class Worker(threading.Thread):
    """Loops its unit of work until stop_evt, recording per-iteration timings."""

    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.samples = []

    def run(self):
        while not self.stop_evt.is_set():
            self.step()


class TtsWorker(Worker):
    def __init__(self, stop_evt, threads):
        super().__init__(stop_evt)
        from piper import PiperVoice
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.voice = PiperVoice.load(PIPER_PATH)
        self.voice.session = ort.InferenceSession(
            PIPER_PATH, so, providers=["CPUExecutionProvider"])
        self.i = 0
        self.step()          # warm
        self.samples.clear()

    def step(self):
        text = SENTENCES[self.i % len(SENTENCES)]
        self.i += 1
        t = time.perf_counter()
        chunks = list(self.voice.synthesize(text))
        ms = (time.perf_counter() - t) * 1000
        n = sum(len(c.audio_int16_bytes) for c in chunks)
        dur = n / 2 / chunks[0].sample_rate
        self.samples.append((ms / 1000) / dur)   # RTF


class SttWorker(Worker):
    def __init__(self, stop_evt, threads):
        super().__init__(stop_evt)
        from faster_whisper import WhisperModel
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                                  cpu_threads=threads, num_workers=1)
        data, sr = sf.read(HERE / "jfk.wav", dtype="float32")
        self.audio = data[: 3 * sr]
        self.step()
        self.samples.clear()

    def step(self):
        t = time.perf_counter()
        segs, _ = self.model.transcribe(self.audio, beam_size=1, language="en",
                                        without_timestamps=True,
                                        condition_on_previous_text=False)
        _ = " ".join(s.text for s in segs)
        self.samples.append((time.perf_counter() - t) * 1000)


class VadWorker(Worker):
    """Paced 32ms frames. Records BOTH inference cost and real-time pacing jitter.

    Jitter is the number that matters: if this thread gets starved, barge-in
    detection is late even though the inference itself is cheap.
    """

    def __init__(self, stop_evt):
        super().__init__(stop_evt)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(VAD_PATH, so,
                                         providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)
        self.frame = np.zeros((1, 512), dtype=np.float32)
        self.jitter = []
        self._next = None

    def run(self):
        self._next = time.perf_counter()
        while not self.stop_evt.is_set():
            now = time.perf_counter()
            if self._next is not None:
                self.jitter.append((now - self._next) * 1000)
            self._next = now + 0.032
            t = time.perf_counter()
            out = self.sess.run(None, {"input": self.frame, "state": self.state,
                                       "sr": self.sr})
            self.state = out[1]
            self.samples.append((time.perf_counter() - t) * 1000)
            sleep = self._next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


# ---------------------------------------------------------------- llm

def llm_stream(threads, n_predict=N_PREDICT):
    body = {"model": MODEL, "stream": True,
            "prompt": "Describe the water cycle in detail.",
            "options": {"num_predict": n_predict, "temperature": 0,
                        "num_thread": threads}}
    req = urllib.request.Request(f"{HOST}/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
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
                model_rate = ev / (evd / 1e9) if evd else 0
                break
    wall = time.perf_counter() - t0
    return {"wall_tok_s": n / wall, "model_tok_s": model_rate,
            "ttft_ms": ttft, "tokens": n, "wall_s": wall}


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def scenario(label, llm_threads, tts_threads=None, stt_threads=None, vad=False):
    stop = threading.Event()
    workers = {}
    if tts_threads:
        workers["tts"] = TtsWorker(stop, tts_threads)
    if stt_threads:
        workers["stt"] = SttWorker(stop, stt_threads)
    if vad:
        workers["vad"] = VadWorker(stop)

    for w in workers.values():
        w.start()
    time.sleep(0.3)                      # let workers reach steady state
    llm = llm_stream(llm_threads)
    stop.set()
    for w in workers.values():
        w.join(timeout=30)

    row = {"scenario": label, "llm_threads": llm_threads,
           "tts_threads": tts_threads, "stt_threads": stt_threads, "vad": vad,
           "llm": {k: round(v, 2) for k, v in llm.items() if v is not None}}
    out = (f"{label:<34} LLM {llm['wall_tok_s']:5.2f} tok/s "
           f"(TTFT {llm['ttft_ms']:5.0f} ms)")
    if "tts" in workers:
        s = workers["tts"].samples
        row["tts_rtf_median"] = round(statistics.median(s), 3) if s else None
        row["tts_rtf_p95"] = round(pct(s, 95), 3)
        row["tts_n"] = len(s)
        out += f" | TTS RTF {row['tts_rtf_median']:.3f}"
    if "stt" in workers:
        s = workers["stt"].samples
        row["stt_ms_median"] = round(statistics.median(s), 1) if s else None
        row["stt_n"] = len(s)
        out += f" | STT {row['stt_ms_median']:.0f} ms"
    if "vad" in workers:
        s = workers["vad"].samples
        j = workers["vad"].jitter[1:]
        row["vad_ms_p50"] = round(statistics.median(s), 3)
        row["vad_ms_p99"] = round(pct(s, 99), 3)
        row["vad_jitter_p99_ms"] = round(pct(j, 99), 2)
        row["vad_jitter_max_ms"] = round(max(j), 2)
        out += (f" | VAD p99 {row['vad_ms_p99']:.2f} ms "
                f"jitter p99 {row['vad_jitter_p99_ms']:+.1f} ms")
    print(out, flush=True)
    return row


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []

    print("=== solo baselines ===", flush=True)
    rows.append(scenario("LLM solo (t=6)", 6))
    base_llm = rows[-1]["llm"]["wall_tok_s"]

    # TTS/STT/VAD solo, measured the same way but with a trivial LLM load
    print("\n=== designed overlap: LLM + TTS ===", flush=True)
    for lt, tt in ((6, 2), (6, 4), (4, 4), (8, 2)):
        rows.append(scenario(f"LLM t={lt} + TTS t={tt}", lt, tts_threads=tt))

    print("\n=== response phase: LLM + TTS + VAD (barge-in listening) ===",
          flush=True)
    rows.append(scenario("LLM t=6 + TTS t=2 + VAD", 6, tts_threads=2, vad=True))
    rows.append(scenario("LLM t=4 + TTS t=2 + VAD", 4, tts_threads=2, vad=True))

    print("\n=== worst case: LLM + TTS + STT + VAD ===", flush=True)
    rows.append(scenario("all four (6/2/2)", 6, tts_threads=2, stt_threads=2,
                         vad=True))
    rows.append(scenario("all four (4/2/2)", 4, tts_threads=2, stt_threads=2,
                         vad=True))

    (HERE / "results" / "s5_concurrency.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 74)
    print(f"solo LLM baseline: {base_llm:.2f} tok/s "
          f"(S2 measured 12.66 with num_thread=6)")
    print(f"{'scenario':<34} {'tok/s':>7} {'vs solo':>9} {'TTS RTF':>9} {'VAD p99':>9}")
    for r in rows:
        d = (r["llm"]["wall_tok_s"] - base_llm) / base_llm * 100
        print(f"{r['scenario']:<34} {r['llm']['wall_tok_s']:7.2f} {d:+8.1f}% "
              f"{r.get('tts_rtf_median', float('nan')):9.3f} "
              f"{r.get('vad_ms_p99', float('nan')):9.2f}")

    worst_rtf = max((r.get("tts_rtf_p95", 0) for r in rows), default=0)
    worst_vad = max((r.get("vad_jitter_p99_ms", 0) for r in rows), default=0)
    print(f"\nworst TTS RTF p95 under load : {worst_rtf:.3f} "
          f"({'PASS' if worst_rtf < 1.0 else 'FAIL'} - must stay < 1.0)")
    print(f"worst VAD pacing jitter p99  : {worst_vad:+.1f} ms "
          f"({'PASS' if worst_vad < 32 else 'FAIL'} - must stay << 32 ms)")


if __name__ == "__main__":
    main()
