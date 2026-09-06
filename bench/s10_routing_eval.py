"""S10 — routing accuracy on an expanded set.

S9 reported 8/8, but its cases were all clean tool requests or obvious chit-chat.
A live run then routed "thanks that is all" to add_reminder. The eval set was too
easy: with only tools described in detail, the model treats every utterance as a
tool request and picks the nearest match.

This adds the cases that actually break routing -- closings, acknowledgements,
negations, and near-misses between timer (duration) and reminder (clock time).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegrys.core.config import load  # noqa: E402
from aegrys.llm.client import LLMClient  # noqa: E402
from aegrys.llm.router import CompactRouter  # noqa: E402
from aegrys.tools.mcp_manager import MCPManager  # noqa: E402

CASES = [
    # clear tool requests
    ("set a timer for five minutes", "set_timer"),
    ("timer for thirty seconds", "set_timer"),
    ("give me a ten minute timer", "set_timer"),
    ("remind me to call mom tomorrow at six", "add_reminder"),
    ("remind me to take out the bins", "add_reminder"),
    ("what is on my calendar today", "list_events"),
    ("what do I have scheduled tomorrow", "list_events"),
    ("what's my next meeting", "next_event"),
    ("summarize my email", "summarize_email"),
    ("read me my messages", "summarize_email"),
    ("cancel the timer", "cancel_timer"),
    ("what reminders do I have", "list_reminders"),
    # conversational -- the class that broke in the live run
    ("thanks that is all", None),
    ("thank you", None),
    ("never mind", None),
    ("ok cool", None),
    ("hello how are you", None),
    ("goodbye", None),
    ("tell me a joke", None),
    ("what is the capital of France", None),
    ("who won the world cup in 1998", None),
    ("that's wrong, try again", None),
]


def main():
    cfg = load()
    m = MCPManager()
    m.start()
    llm = LLMClient(cfg.llm)
    r = CompactRouter(llm, m.tools)
    print(f"router prompt: {len(r.system)} chars (~{len(r.system)//4} tokens)\n")

    rows, correct = [], 0
    tool_correct = tool_total = chat_correct = chat_total = 0
    for text, expect in CASES:
        t0 = time.perf_counter()
        got = r.pick(text)
        ms = (time.perf_counter() - t0) * 1000
        got_norm = None if got == "respond" else got
        hit = got_norm == expect
        correct += hit
        if expect is None:
            chat_total += 1
            chat_correct += hit
        else:
            tool_total += 1
            tool_correct += hit
        print(f"  {'OK  ' if hit else 'MISS'} {ms:6.0f} ms  "
              f"{str(got_norm):<17} (want {str(expect):<17}) | {text}", flush=True)
        rows.append({"text": text, "expected": expect, "got": got_norm,
                     "hit": hit, "ms": round(ms, 1)})

    Path(__file__).parent.joinpath("results/s10_routing_eval.json").write_text(
        json.dumps(rows, indent=2))
    print(f"\noverall       {correct}/{len(CASES)} "
          f"({correct/len(CASES)*100:.0f}%)")
    print(f"tool requests {tool_correct}/{tool_total}")
    print(f"conversational{chat_correct:>3}/{chat_total}  "
          f"<- the class S9 missed")
    misses = [r for r in rows if not r["hit"]]
    if misses:
        print("\nmisses:")
        for x in misses:
            print(f"  {x['text']!r} -> {x['got']} (want {x['expected']})")
    m.stop()


if __name__ == "__main__":
    main()
