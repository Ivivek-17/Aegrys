"""S2b — minimize router tokens, and measure real TTFT / time-to-first-sentence.

S2 found decode at ~12.7 tok/s => ~79 ms per token. So router cost is dominated by
how many tokens it EMITS, not by the call count. Compares three router encodings,
then measures streaming time-to-first-sentence, which is the number that actually
drives TTFA (docs/DESIGN.md §7).
"""

import json
import statistics
import time
import urllib.request

HOST = "http://127.0.0.1:11435"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
THREADS = 6

TOOLS = ["set_timer", "add_reminder", "list_events", "summarize_email", "respond"]

SYS_FULL = ("You are a voice assistant. Route the user's request to exactly one "
            "tool. Tools: set_timer(seconds), add_reminder(text,when), "
            "list_events(date), summarize_email(count), respond.")

# Sharpened descriptions to fix the S2 miss (reminder-with-a-time -> set_timer).
SYS_SHARP = (
    "Route the user's request to exactly one tool.\n"
    "set_timer: a countdown for a DURATION (in N minutes/seconds).\n"
    "add_reminder: remember a TASK at a clock time or date (tomorrow, at 6, friday).\n"
    "list_events: read the calendar.\n"
    "summarize_email: read email.\n"
    "respond: anything else, chit-chat.")

SCHEMA_FULL = {"type": "object",
               "properties": {"tool": {"type": "string", "enum": TOOLS},
                              "args": {"type": "object"}},
               "required": ["tool", "args"]}
SCHEMA_TOOL = {"type": "object",
               "properties": {"tool": {"type": "string", "enum": TOOLS}},
               "required": ["tool"]}

CASES = [
    ("set a timer for five minutes", "set_timer"),
    ("remind me to call mom tomorrow at 6", "add_reminder"),
    ("what's on my calendar today", "list_events"),
    ("summarize my last three emails", "summarize_email"),
    ("hey how are you doing", "respond"),
    ("timer 30 seconds please", "set_timer"),
    ("what do I have scheduled for friday", "list_events"),
    ("tell me a joke", "respond"),
    ("remind me to take out the trash", "add_reminder"),
    ("set a timer for 2 minutes", "set_timer"),
]


def post(path, body, timeout=600):
    req = urllib.request.Request(f"{HOST}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def gen(prompt, system, fmt=None, predict=64):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"num_predict": predict, "temperature": 0,
                        "num_thread": THREADS}}
    if fmt:
        body["format"] = fmt
    t0 = time.perf_counter()
    d = json.loads(post("/api/generate", body).read())
    return (time.perf_counter() - t0) * 1000, d.get("response", ""), \
        d.get("eval_count", 0)


def stream_first_sentence(prompt, system, predict=160):
    """Measure TTFT and time until the first complete sentence is available."""
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": True,
            "options": {"num_predict": predict, "temperature": 0,
                        "num_thread": THREADS}}
    t0 = time.perf_counter()
    ttft = None
    first_sent = None
    buf = ""
    with post("/api/generate", body) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            tok = d.get("response", "")
            if tok and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            buf += tok
            if first_sent is None and any(c in buf for c in ".!?"):
                first_sent = (time.perf_counter() - t0) * 1000
            if d.get("done"):
                break
    total = (time.perf_counter() - t0) * 1000
    return ttft, first_sent, total, buf


def main():
    out = {}

    print("=== router encodings: latency vs tokens emitted ===", flush=True)
    variants = [
        ("full json {tool,args}", SYS_FULL, SCHEMA_FULL),
        ("tool only  {tool}", SYS_FULL, SCHEMA_TOOL),
        ("tool only + sharp sys", SYS_SHARP, SCHEMA_TOOL),
    ]
    enc = []
    for label, sysmsg, schema in variants:
        ms, toks, correct = [], [], 0
        for text, expect in CASES:
            w, resp, n = gen(text, sysmsg, fmt=schema)
            ms.append(w)
            toks.append(n)
            try:
                correct += json.loads(resp).get("tool") == expect
            except json.JSONDecodeError:
                pass
        print(f"  {label:<24} median {statistics.median(ms):6.0f} ms  "
              f"{statistics.median(toks):4.1f} tok  "
              f"acc {correct}/{len(CASES)}", flush=True)
        enc.append({"variant": label, "median_ms": statistics.median(ms),
                    "median_tokens": statistics.median(toks),
                    "correct": correct, "total": len(CASES)})
    out["router_encodings"] = enc

    print("\n=== streaming synthesis: TTFT and first sentence ===", flush=True)
    prompts = [
        "The timer is set. Confirm this to the user in one short sentence.",
        "Tell the user it is sunny and 22 degrees, in one short sentence.",
        "Greet the user briefly and ask how you can help.",
    ]
    rows = []
    for p in prompts:
        ttft, fs, total, txt = stream_first_sentence(
            p, "You are a concise voice assistant. Reply in one short sentence.")
        print(f"  TTFT {ttft:6.0f} ms | first sentence {fs:6.0f} ms | "
              f"full {total:6.0f} ms | {txt.strip()[:46]}", flush=True)
        rows.append({"prompt": p, "ttft_ms": round(ttft, 1),
                     "first_sentence_ms": round(fs, 1),
                     "total_ms": round(total, 1), "text": txt.strip()})
    out["streaming"] = rows

    med_fs = statistics.median(r["first_sentence_ms"] for r in rows)
    best_router = min(enc, key=lambda e: e["median_ms"])
    print(f"\nbest router: {best_router['variant']} "
          f"{best_router['median_ms']:.0f} ms "
          f"(acc {best_router['correct']}/{best_router['total']})")
    print(f"median time-to-first-sentence: {med_fs:.0f} ms")
    print(f"=> LLM contribution to TTFA (router + first sentence): "
          f"{best_router['median_ms'] + med_fs:.0f} ms")

    with open("results/s2b_router_ttft.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
