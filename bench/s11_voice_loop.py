"""S11 — closed-loop voice test: TTS -> mic path -> VAD -> endpointer -> STT -> router.

Every latency number so far came from the TEXT path. The microphone path had never
been exercised end to end, which is the largest untested surface in the build.

This closes the loop without a human: Piper synthesizes an utterance, it is
upsampled to the 48 kHz device rate, and pushed through the REAL MicStream queue --
so the actual streaming decimator, 512-sample framing, Silero VAD, and adaptive
endpointer all run exactly as they do live. Only the acoustic path (speaker -> air
-> microphone) is substituted.

What this proves: the mic pipeline works.
What it does NOT prove: R1 (echo), microphone gain/noise handling, or STT accuracy
on real human speech. Synthetic speech is easier than the real thing.
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegrys.audio.resample import resample  # noqa: E402
from aegrys.core.config import VAD_FRAME, load  # noqa: E402
from aegrys.core.trace import Tracer  # noqa: E402

UTTERANCES = [
    "set a timer for five minutes",
    "what is on my calendar today",
    "remind me to call mom tomorrow",
    "what is the capital of France",
]


def feed(mic, audio48: np.ndarray, block: int) -> threading.Thread:
    """Push audio into the real MicStream queue, as the device callback does.

    Must run on its own thread: the queue is bounded (maxsize=64) and the consumer
    is listen(), so feeding synchronously deadlocks once the queue fills.
    """

    def run():
        for i in range(0, len(audio48), block):
            chunk = audio48[i:i + block]
            if len(chunk) < block:
                chunk = np.pad(chunk, (0, block - len(chunk)))
            mic._q.put(chunk.astype(np.float32))
        mic._q.put(None)          # end of stream

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main():
    cfg = load()
    cfg.trace = False
    from aegrys.core.tool_turn import ToolAssistant
    a = ToolAssistant(cfg, Tracer(False))

    sr_dev = cfg.audio.device_sr
    block = int(sr_dev * cfg.audio.in_block_ms / 1000)
    silence = np.zeros(int(sr_dev * 1.2), dtype=np.float32)   # > endpoint threshold

    rows = []
    print(f"device rate {sr_dev} Hz, block {block} samples "
          f"({cfg.audio.in_block_ms} ms)\n", flush=True)

    for text in UTTERANCES:
        # 1. synthesize, 2. convert to device rate, 3. bracket with silence
        speech16 = a.tts.synthesize(text)
        speech48 = resample(speech16, a.tts.sample_rate, sr_dev)
        # Slight attenuation: a real mic never receives full-scale audio.
        stream = np.concatenate([silence[:sr_dev // 2], speech48 * 0.4, silence])

        a.mic._dec.reset()
        while not a.mic._q.empty():
            a.mic._q.get_nowait()
        a.mic._buf = np.zeros(0, dtype=np.float32)
        a.vad.reset()
        a._running = True
        feeder = feed(a.mic, stream, block)

        t0 = time.perf_counter()
        captured = a.listen()
        listen_ms = (time.perf_counter() - t0) * 1000
        feeder.join(timeout=5)

        if captured.size == 0:
            print(f"  FAIL  no utterance detected | {text!r}", flush=True)
            rows.append({"text": text, "detected": False})
            continue

        t1 = time.perf_counter()
        heard = a.stt.transcribe(captured)
        stt_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        call = a.router.route(heard)
        route_ms = (time.perf_counter() - t2) * 1000

        norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").split()
        exact = norm(heard) == norm(text)
        print(f"  {'OK  ' if exact else 'DIFF'} captured {captured.size/16000:5.2f}s "
              f"| listen {listen_ms:5.0f} ms | stt {stt_ms:5.0f} ms "
              f"| route {route_ms:5.0f} ms -> {call.name if call else 'respond'}",
              flush=True)
        print(f"        said : {text}")
        print(f"        heard: {heard}", flush=True)
        rows.append({"text": text, "heard": heard, "exact": exact,
                     "detected": True,
                     "captured_s": round(captured.size / 16000, 2),
                     "listen_ms": round(listen_ms, 1),
                     "stt_ms": round(stt_ms, 1),
                     "route_ms": round(route_ms, 1),
                     "tool": call.name if call else None})

    a.shutdown()
    Path(__file__).parent.joinpath("results/s11_voice_loop.json").write_text(
        json.dumps(rows, indent=2))

    ok = [r for r in rows if r.get("detected")]
    exact = [r for r in ok if r.get("exact")]
    print(f"\ndetected      {len(ok)}/{len(rows)}")
    print(f"transcribed   {len(exact)}/{len(rows)} word-exact")
    if ok:
        print(f"STT median    {statistics.median(r['stt_ms'] for r in ok):.0f} ms")
        print(f"route median  {statistics.median(r['route_ms'] for r in ok):.0f} ms")
    print("\nNOTE: synthetic speech only. Does not validate echo (R1), mic gain,")
    print("      background noise, or accuracy on real human voices.")


if __name__ == "__main__":
    main()
