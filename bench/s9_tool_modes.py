"""S9 — router vs native tool binding, measured end to end.

Phase 3 shipped native tool binding (all 10 MCP schemas on one call) because S2b
said skipping the router saves 1096 ms. Live, that produced TTFA of 18-22 s.

S7 found why: the tool schema makes the prompt ~900 tokens, and a cold prefill of
900 tokens costs 15-20 s on this CPU (~50 tok/s). The router's prompt is ~270 tokens
across two calls, so even a cold prefill is cheap.

This benchmarks both on identical turns, warm, and reports TTFA + routing accuracy.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegrys.core.config import load  # noqa: E402
from aegrys.core.tool_turn import ToolAssistant  # noqa: E402
from aegrys.core.trace import Tracer  # noqa: E402

CASES = [
    ("set a timer for five minutes", "set_timer"),
    ("what is on my calendar today", "list_events"),
    ("remind me to call mom tomorrow at six", "add_reminder"),
    ("hello how are you", None),
    ("summarize my email", "summarize_email"),
    ("timer for thirty seconds", "set_timer"),
    ("what do I have scheduled tomorrow", "list_events"),
    ("tell me a joke", None),
]


def run_mode(mode: str) -> dict:
    cfg = load()
    cfg.tools.mode = mode
    cfg.trace = False
    a = ToolAssistant(cfg, Tracer(False))
    a.speaker.start()

    # Warm both code paths so we measure steady state, not first-call effects.
    a.handle_text("hello")
    a.speaker.flush()

    rows, correct = [], 0
    for text, expect in CASES:
        a.history = a.history[:1]
        t0 = time.perf_counter()
        a.tracer.start_turn(a.epochs.begin())
        a.tracer.mark("speech_end")
        reply = a.respond(text, a.epochs.current)
        wall = (time.perf_counter() - t0) * 1000
        tr = a.tracer.current
        got = tr.meta.get("tool")
        hit = (got == expect)
        correct += hit
        rows.append({"text": text, "expected": expect, "got": got, "hit": hit,
                     "ttfa_ms": round(tr.ttfa_ms or wall, 1),
                     "wall_ms": round(wall, 1), "reply": reply[:60]})
        print(f"  {'OK ' if hit else 'MISS'} TTFA {rows[-1]['ttfa_ms']:7.0f} ms "
              f"wall {wall:7.0f} ms | {str(got):<16} | {text[:34]}", flush=True)
        a.speaker.flush()
        time.sleep(0.2)

    a.speaker.stop()
    a.shutdown()
    ttfa = [r["ttfa_ms"] for r in rows]
    return {"mode": mode, "rows": rows, "correct": correct, "total": len(CASES),
            "ttfa_median": round(statistics.median(ttfa), 1),
            "ttfa_max": round(max(ttfa), 1),
            "wall_median": round(statistics.median(r["wall_ms"] for r in rows), 1)}


def main():
    out = []
    for mode in ("router", "native"):
        print(f"\n=== mode: {mode} ===", flush=True)
        try:
            out.append(run_mode(mode))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    Path(__file__).parent.joinpath("results/s9_tool_modes.json").write_text(
        json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print(f"{'mode':<10}{'TTFA median':>13}{'TTFA max':>11}{'wall median':>13}{'routing':>10}")
    for r in out:
        print(f"{r['mode']:<10}{r['ttfa_median']:12.0f}ms{r['ttfa_max']:10.0f}ms"
              f"{r['wall_median']:12.0f}ms{r['correct']:>7}/{r['total']}")
    if len(out) == 2:
        a, b = out
        print(f"\nrouter is {b['ttfa_median'] / a['ttfa_median']:.1f}x faster "
              f"on median TTFA than native on this hardware")


if __name__ == "__main__":
    main()
