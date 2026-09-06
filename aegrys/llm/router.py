"""Compact two-step tool routing.

Why this exists (S7/S8, Phase 4):

Binding all 10 MCP tools natively makes the prompt ~900 tokens. When ollama's
prefix cache misses -- which it does across turns once history grows and the two
calls of a tool turn alternate prefixes -- prefill of 900 tokens costs 15-20 SECONDS
on this CPU (~50 tok/s). Live TTFA was 18-22 s.

So instead of one call with a huge schema, use two calls with tiny ones:

  step 1  pick the tool     ~120 token prompt, enum-constrained, ~9 tokens out
  step 2  fill its args     ~150 token prompt, single-tool schema

Total prompt across both is ~270 tokens rather than 900, so even a cold prefill is
cheap. Both steps use JSON-schema-constrained sampling, which S2 measured at 8/8
valid JSON and (with the sharpened descriptions) 10/10 routing accuracy.

This is the design's original router (§6.2) reinstated -- but only because
measurement showed the alternative is worse ON THIS HARDWARE. On a GPU box, where
900 tokens of prefill is ~100 ms, native tool binding would win.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..tools.mcp_manager import ToolInfo
from .client import LLMClient, ToolCall

RESPOND = "respond"


@dataclass
class RouteResult:
    tool: str | None
    args: dict


def build_router_prompt(tools: dict[str, ToolInfo]) -> str:
    """One short line per tool. Keep it terse -- every token is ~79 ms of latency.

    The `respond` description is deliberately the most explicit one. A live run
    routed "thanks that is all" to add_reminder: with only tools described in
    detail, the model treats every utterance as a tool request and picks the
    nearest match. The default needs to be stated at least as forcefully as the
    alternatives, and the common conversational cases named outright.
    """
    lines = ["Pick exactly one tool for the user's request.",
             "Only pick a tool if the user is actually ASKING for that action."]
    for name, info in sorted(tools.items()):
        first = (info.description or "").strip().split("\n")[0]
        lines.append(f"{name}: {first[:110]}")
    lines.append(
        f"{RESPOND}: the DEFAULT. Use for greetings, thanks, acknowledgements "
        f"(\"thanks\", \"that's all\", \"never mind\", \"ok\", \"bye\"), general "
        f"knowledge questions, chit-chat, and anything not clearly one of the "
        f"actions above. When unsure, pick {RESPOND}.")
    return "\n".join(lines)


class CompactRouter:
    def __init__(self, llm: LLMClient, tools: dict[str, ToolInfo]):
        self.llm = llm
        self.tools = tools
        self.system = build_router_prompt(tools)
        self.names = sorted(tools) + [RESPOND]

    def pick(self, text: str) -> str:
        schema = {"type": "object",
                  "properties": {"tool": {"type": "string", "enum": self.names}},
                  "required": ["tool"]}
        raw = self.llm.generate_json(text, self.system, schema, num_predict=24)
        tool = (raw or {}).get("tool", RESPOND)
        return tool if tool in self.tools else RESPOND

    def fill_args(self, text: str, tool: str) -> dict:
        """Extract arguments using ONLY the chosen tool's schema (small prompt)."""
        info = self.tools[tool]
        schema = dict(info.schema or {"type": "object", "properties": {}})
        props = schema.get("properties") or {}
        if not props:
            return {}
        system = (f"Extract arguments for `{tool}`.\n"
                  f"{(info.description or '').strip()[:220]}\n"
                  f"Convert durations to seconds (five minutes = 300).")
        raw = self.llm.generate_json(text, system, schema, num_predict=64)
        return raw or {}

    def route(self, text: str) -> ToolCall | None:
        tool = self.pick(text)
        if tool == RESPOND:
            return None
        return ToolCall(name=tool, args=self.fill_args(text, tool))
