"""S2 — local LLM throughput on i7-1355U, plus tool-routing viability.

Pass criteria (docs/DESIGN.md §8): >= 8 tok/s sustained (not burst) on 3B Q4_K_M.

Also validates the §6.1 claim that schema-constrained decoding makes a 3B model a
reliable tool router. Ollama's `format` field takes a JSON schema and constrains
sampling, which is the equivalent of llama.cpp's GBNF for our purposes.
"""

import json
import statistics
import time
import urllib.request

HOST = "http://127.0.0.1:11435"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
OUT = "results/s2_llm.json"

SYS = ("You are a voice assistant. Route the user's request to exactly one tool. "
       "Tools: set_timer(seconds), add_reminder(text,when), "
       "list_events(date), summarize_email(count), respond.")

SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string",
                 "enum": ["set_timer", "add_reminder", "list_events",
                          "summarize_email", "respond"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

ROUTING_CASES = [
    ("set a timer for five minutes", "set_timer"),
    ("remind me to call mom tomorrow at 6", "add_reminder"),
    ("what's on my calendar today", "list_events"),
    ("summarize my last three emails", "summarize_email"),
    ("hey how are you doing", "respond"),
    ("timer 30 seconds please", "set_timer"),
    ("what do I have scheduled for friday", "list_events"),
    ("tell me a joke", "respond"),
]


def call(prompt, system=None, threads=None, fmt=None, predict=128):
    body = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": predict, "temperature": 0},
    }
    if system:
        body["system"] = system
    if threads:
        body["options"]["num_thread"] = threads
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(
        f"{HOST}/api/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    wall = (time.perf_counter() - t0) * 1000
    ev, evd = d.get("eval_count", 0), d.get("eval_duration", 1)
    pe, ped = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1)
    return {
        "wall_ms": round(wall, 1),
        "tok_s": round(ev / (evd / 1e9), 2) if evd else 0,
        "eval_count": ev,
        "prefill_tok_s": round(pe / (ped / 1e9), 2) if ped else 0,
        "prefill_ms": round(ped / 1e6, 1),
        "text": d.get("response", ""),
    }


def main():
    results = {}

    print("=== warm-up (load model into RAM) ===", flush=True)
    w = call("hi", predict=8)
    print(f"  load+gen: {w['wall_ms']:.0f} ms", flush=True)

    print("\n=== thread sweep (decode tok/s) ===", flush=True)
    sweep = []
    prompt = "Explain in three sentences why the sky appears blue."
    for t in (2, 4, 6, 8, 12):
        runs = [call(prompt, threads=t, predict=100) for _ in range(2)]
        med = statistics.median(r["tok_s"] for r in runs)
        pf = statistics.median(r["prefill_tok_s"] for r in runs)
        flag = "PASS" if med >= 8 else "    "
        print(f"{flag} num_thread={t:<3} decode {med:6.2f} tok/s   "
              f"prefill {pf:7.1f} tok/s", flush=True)
        sweep.append({"threads": t, "decode_tok_s": med, "prefill_tok_s": pf})
    results["thread_sweep"] = sweep

    best = max(sweep, key=lambda r: r["decode_tok_s"])
    bt = best["threads"]
    print(f"\nbest: num_thread={bt} @ {best['decode_tok_s']:.2f} tok/s", flush=True)

    print("\n=== sustained throughput (thermal check, 6 consecutive) ===", flush=True)
    sust = []
    for i in range(6):
        r = call(prompt, threads=bt, predict=100)
        sust.append(r["tok_s"])
        print(f"  run {i+1}: {r['tok_s']:6.2f} tok/s", flush=True)
    drop = (sust[0] - sust[-1]) / sust[0] * 100 if sust[0] else 0
    print(f"  first {sust[0]:.2f} -> last {sust[-1]:.2f} tok/s "
          f"({drop:+.1f}% drift)", flush=True)
    results["sustained"] = {"runs": sust, "drift_pct": round(drop, 1),
                            "threads": bt}

    print("\n=== router latency: constrained vs free (first sentence proxy) ===",
          flush=True)
    lat = []
    for text, _ in ROUTING_CASES[:4]:
        r = call(text, system=SYS, threads=bt, fmt=SCHEMA, predict=64)
        lat.append(r["wall_ms"])
        print(f"  {r['wall_ms']:6.0f} ms  prefill {r['prefill_ms']:5.0f} ms  "
              f"{r['eval_count']:>3} tok  | {text[:34]}", flush=True)
    results["router_latency_ms"] = lat

    print("\n=== schema-constrained tool routing accuracy ===", flush=True)
    ok = 0
    routing = []
    for text, expect in ROUTING_CASES:
        r = call(text, system=SYS, threads=bt, fmt=SCHEMA, predict=64)
        try:
            got = json.loads(r["text"]).get("tool")
            valid = True
        except json.JSONDecodeError:
            got, valid = None, False
        hit = got == expect
        ok += hit
        print(f"  {'OK ' if hit else 'MISS'} {text[:38]:<40} "
              f"-> {got} (want {expect})", flush=True)
        routing.append({"text": text, "expected": expect, "got": got,
                        "valid_json": valid, "ms": r["wall_ms"]})
    results["routing"] = {"correct": ok, "total": len(ROUTING_CASES),
                          "cases": routing}
    print(f"\nrouting: {ok}/{len(ROUTING_CASES)} correct, "
          f"{sum(c['valid_json'] for c in routing)}/{len(routing)} valid JSON")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
