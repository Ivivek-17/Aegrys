"""S8 — confirm the prompt-cache thrash, then find the fix.

Hypothesis from S7: a tool turn issues TWO calls with DIFFERENT prompt prefixes --
  call A: system + 10 tool schemas   (765 tokens)
  call B: system + tool_result       (no tools, different prefix)
They evict each other from ollama's single prompt-cache slot, so every call pays a
cold ~8-15 s prefill. That would explain live TTFA of 18-22 s while S7's repeated
same-prefix calls were 2.2 s.

Test: alternate A/B/A/B and watch prefill_ms. If prefill is cold every time, the
hypothesis holds. Then compare candidate fixes.

Run with OLLAMA_NUM_PARALLEL unset (default 1) and again with 2+ to compare.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

HOST = os.environ.get("AEGRYS_LLM_HOST", "http://127.0.0.1:11435")
MODEL = "qwen2.5:3b-instruct-q4_K_M"


def chat(messages, tools=None, predict=64):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": predict, "temperature": 0, "num_thread": 6}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{HOST}/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return {"wall": (time.perf_counter() - t0) * 1000,
            "prompt": d.get("prompt_eval_count", 0),
            "prefill": d.get("prompt_eval_duration", 1) / 1e6,
            "gen": d.get("eval_count", 0),
            "gen_ms": d.get("eval_duration", 1) / 1e6}


def main():
    from aegrys.tools.mcp_manager import MCPManager
    m = MCPManager()
    m.start()
    tools = m.openai_tools()

    SYS = {"role": "system", "content": "You are a concise voice assistant."}
    A = [SYS, {"role": "user", "content": "set a timer for five minutes"}]
    B = [SYS, {"role": "user",
               "content": "The tool returned: Timer 1 set for 5 minutes. "
                          "Tell the user in one short sentence."}]

    print(f"OLLAMA_NUM_PARALLEL={os.environ.get('OLLAMA_NUM_PARALLEL', '(default)')}")
    print("\n=== alternating prefixes (the live app's pattern) ===")
    chat(A, tools)          # prime
    chat(B)
    total = 0.0
    for i in range(3):
        ra = chat(A, tools)
        rb = chat(B)
        total += ra["wall"] + rb["wall"]
        print(f"  turn {i+1}: A(tools) prefill {ra['prefill']:7.0f} ms "
              f"wall {ra['wall']:7.0f} ms | B(synth) prefill {rb['prefill']:6.0f} ms "
              f"wall {rb['wall']:6.0f} ms", flush=True)
    print(f"  mean tool-turn cost: {total/3:.0f} ms")

    print("\n=== control: same prefix repeated (no alternation) ===")
    for i in range(3):
        ra = chat(A, tools)
        print(f"  A only {i+1}: prefill {ra['prefill']:7.0f} ms "
              f"wall {ra['wall']:7.0f} ms", flush=True)

    print("\n=== candidate fix: trim the tool schema ===")
    slim = []
    for t in tools:
        f = t["function"]
        slim.append({"type": "function", "function": {
            "name": f["name"],
            "description": f["description"].split(".")[0][:80],
            "parameters": f["parameters"]}})
    print(f"  schema chars {len(json.dumps(tools))} -> {len(json.dumps(slim))}")
    chat(A, slim)
    for i in range(2):
        r = chat(A, slim)
        print(f"  slim {i+1}: prompt {r['prompt']} tok prefill {r['prefill']:6.0f} ms "
              f"gen {r['gen']} tok wall {r['wall']:7.0f} ms", flush=True)

    m.stop()


if __name__ == "__main__":
    main()
