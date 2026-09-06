"""S6 — barge-in latency, measured end to end without a microphone.

Phase 2 exit criterion (DESIGN §9): interrupting mid-sentence stops audio in <100 ms.

Drives a real LLM + Piper turn, fires a barge-in partway through, and measures:
  1. how long the flush takes,
  2. the worst-case audible tail (bounded by the output chunk size by design),
  3. that the LLM stream actually STOPS rather than running to completion in the
     background -- the failure that makes an assistant feel like it ignored you.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegrys.core.config import load  # noqa: E402
from aegrys.core.trace import Tracer  # noqa: E402
from aegrys.core.barge import BargeInAssistant  # noqa: E402

LONG_PROMPT = "Explain in detail how the water cycle works, step by step."


def main():
    cfg = load()
    cfg.tools.enabled = False
    cfg.trace = False
    a = BargeInAssistant(cfg, Tracer(False))
    a.speaker.start()

    results = []
    for trial in range(3):
        epoch = a.epochs.begin()
        a.tracer.start_turn(epoch)
        a.history = a.history[:1]        # reset context between trials

        tokens = {"n": 0, "stopped_at": None}
        done = threading.Event()

        def run():
            # Count tokens the stream actually delivers, so we can prove it stopped.
            orig = a.llm.chat_stream

            def counting(*args, **kw):
                for t in orig(*args, **kw):
                    tokens["n"] += 1
                    yield t

            a.llm.chat_stream = counting
            try:
                a.respond(LONG_PROMPT, epoch)
            finally:
                a.llm.chat_stream = orig
                done.set()

        th = threading.Thread(target=run, daemon=True)
        th.start()

        # Wait until audio is actually queued, then interrupt.
        t_wait = time.perf_counter()
        while a.speaker.queued_ms() <= 0 and time.perf_counter() - t_wait < 60:
            time.sleep(0.005)
        queued_before = a.speaker.queued_ms()
        if queued_before <= 0:
            print(f"trial {trial+1}: no audio was queued; skipping")
            continue

        time.sleep(0.25)                 # let it speak a moment, like a real user
        n_at_barge = tokens["n"]

        t0 = time.perf_counter()
        a._trigger_barge_in()
        flush_ms = (time.perf_counter() - t0) * 1000
        queued_after = a.speaker.queued_ms()

        stopped = done.wait(timeout=20)
        stop_ms = (time.perf_counter() - t0) * 1000
        n_after = tokens["n"]

        tail_ms = cfg.audio.out_chunk_ms   # worst case: one callback already in flight
        ok = flush_ms < 100 and queued_after == 0
        print(f"trial {trial+1}: flush {flush_ms:6.2f} ms | queued "
              f"{queued_before:7.1f} -> {queued_after:.1f} ms | "
              f"worst tail {tail_ms} ms | tokens at barge {n_at_barge} -> "
              f"final {n_after} (+{n_after - n_at_barge}) | "
              f"respond() returned in {stop_ms:.0f} ms {'OK' if ok else 'FAIL'}",
              flush=True)
        results.append({
            "trial": trial + 1, "flush_ms": round(flush_ms, 3),
            "queued_before_ms": round(queued_before, 1),
            "queued_after_ms": round(queued_after, 1),
            "worst_tail_ms": tail_ms,
            "tokens_at_barge": n_at_barge, "tokens_final": n_after,
            "tokens_after_barge": n_after - n_at_barge,
            "respond_returned_ms": round(stop_ms, 1),
            "thread_stopped": stopped,
        })
        time.sleep(0.5)

    a.speaker.stop()
    Path(__file__).parent.joinpath("results/s6_bargein.json").write_text(
        json.dumps(results, indent=2))

    if not results:
        print("\nno trials completed")
        return
    worst_flush = max(r["flush_ms"] for r in results)
    worst_leak = max(r["tokens_after_barge"] for r in results)
    worst_total = max(r["worst_tail_ms"] + r["flush_ms"] for r in results)
    print(f"\nworst flush            : {worst_flush:.2f} ms")
    print(f"worst audible tail     : {worst_total:.1f} ms  "
          f"{'PASS' if worst_total < 100 else 'FAIL'} (Phase 2 bar: <100 ms)")
    print(f"tokens leaked after barge-in: {worst_leak} "
          f"({'PASS' if worst_leak <= 2 else 'FAIL'} - stream must stop promptly)")
    print(f"all respond() calls returned: "
          f"{all(r['thread_stopped'] for r in results)}")


if __name__ == "__main__":
    main()
