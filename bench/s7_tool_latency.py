"""S7 — why did tool turns cost 18-22 s?

Phase 3 routed correctly but TTFA on tool turns was 18240 ms and 21978 ms, growing
turn over turn, while a plain chat turn stayed at ~2000 ms. Something about binding
tools is expensive. This isolates the cause: schema size, history growth, or both.
"""

from __future__ import annotations

import json
import time
import urllib.request

HOST = "http://127.0.0.1:11435"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
THREADS = 6


def chat(messages, tools=None, predict=160):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": predict, "temperature": 0,
                        "num_thread": THREADS}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{HOST}/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    wall = (time.perf_counter() - t0) * 1000
    pe = d.get("prompt_eval_count", 0)
    ped = d.get("prompt_eval_duration", 1) / 1e6
    ev = d.get("eval_count", 0)
    evd = d.get("eval_duration", 1) / 1e6
    msg = d.get("message", {})
    return {"wall_ms": wall, "prompt_tokens": pe, "prefill_ms": ped,
            "eval_tokens": ev, "eval_ms": evd,
            "tool": (msg.get("tool_calls") or [{}])[0].get("function", {}).get("name"),
            "content": (msg.get("content") or "")[:60]}


def show(label, r):
    print(f"{label:<34} wall {r['wall_ms']:7.0f} ms | prompt {r['prompt_tokens']:5d} tok "
          f"prefill {r['prefill_ms']:7.0f} ms | gen {r['eval_tokens']:3d} tok "
          f"{r['eval_ms']:6.0f} ms | {r['tool'] or r['content'][:28]}", flush=True)


def main():
    from aegrys.tools.mcp_manager import MCPManager
    m = MCPManager()
    m.start()
    tools = m.openai_tools()
    schema_chars = len(json.dumps(tools))
    print(f"{len(tools)} tools, schema {schema_chars} chars "
          f"(~{schema_chars//4} tokens)\n")

    sys_msg = {"role": "system", "content": "You are a concise voice assistant."}
    user = {"role": "user", "content": "set a timer for five minutes"}

    print("=== effect of binding tools ===")
    show("no tools", chat([sys_msg, user]))
    show("no tools (repeat, warm cache)", chat([sys_msg, user]))
    show("with 10 tools", chat([sys_msg, user], tools))
    show("with 10 tools (repeat)", chat([sys_msg, user], tools))

    print("\n=== effect of tool count ===")
    for n in (1, 3, 5, 10):
        show(f"{n} tools", chat([sys_msg, user], tools[:n]))

    print("\n=== effect of growing history (10 tools bound) ===")
    hist = [sys_msg]
    for i in range(4):
        hist.append({"role": "user", "content": "set a timer for five minutes"})
        r = chat(hist, tools)
        show(f"history turn {i+1} ({len(hist)} msgs)", r)
        hist.append({"role": "assistant", "content": "Timer set for five minutes."})

    print("\n=== does num_predict matter for tool calls? ===")
    for p in (32, 160):
        show(f"num_predict={p}", chat([sys_msg, user], tools, predict=p))

    m.stop()


if __name__ == "__main__":
    main()
