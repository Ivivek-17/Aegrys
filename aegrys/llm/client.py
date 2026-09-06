"""Streaming LLM client (ollama HTTP).

Architecture note (DESIGN §6.2, corrected by S2b):

The original design was two calls -- a constrained router, then synthesis. Measured,
that costs router 1096 ms + first sentence 1279 ms = 2375 ms of LLM before any audio,
which blows the whole budget. The cause is that at ~12.7 tok/s every emitted token
costs ~79 ms, so latency tracks TOKENS EMITTED, not the number of calls.

So the default here is a SINGLE streaming call with tools bound. Conversational
turns -- the common case -- skip routing entirely and pay only TTFT. Tool turns pay
a second synthesis call, which is masked by filler audio.

The constrained router is kept (`route()`) because it measured 10/10 on the routing
set and is a useful fallback if native tool-calling proves flaky on a 3B model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

from ..core.config import LLMConfig
from ..core.epoch import EpochController
from . import prompts


@dataclass
class ToolCall:
    name: str
    args: dict


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig, epochs: EpochController | None = None):
        self.cfg = cfg
        self.epochs = epochs
        self.last_stats: dict = {}

    # ---------------------------------------------------------------- http

    def _post(self, path: str, body: dict, timeout: float = 600):
        req = urllib.request.Request(
            f"{self.cfg.host}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach ollama at {self.cfg.host}: {e}. Start it with:\n"
                f"  OLLAMA_HOST=127.0.0.1:11435 "
                f"OLLAMA_MODELS=D:/Aegrys/.cache/ollama ollama serve") from e

    def _options(self, **over) -> dict:
        # num_thread must stay constant: S5b found ollama RELOADS the model whenever
        # it changes between requests (7/10 scenarios reloaded, ~7 s each).
        o = {"num_thread": self.cfg.threads, "temperature": self.cfg.temperature,
             "num_predict": self.cfg.num_predict}
        o.update(over)
        return o

    def health(self) -> bool:
        try:
            with self._post("/api/generate",
                            {"model": self.cfg.model, "prompt": "hi",
                             "stream": False, "keep_alive": self.cfg.keep_alive,
                             "options": self._options(num_predict=1)},
                            timeout=300):
                return True
        except LLMError:
            return False

    # ---------------------------------------------------------------- chat

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        epoch: int | None = None,
        on_tool_calls=None,
        on_stats=None,
    ) -> Iterator[str]:
        """Stream assistant tokens. Tool calls are surfaced via `on_tool_calls`.

        Stops immediately if `epoch` goes stale -- exiting the loop closes the HTTP
        response, which is what actually stops generation server-side.
        """
        body: dict[str, Any] = {
            "model": self.cfg.model, "messages": messages, "stream": True,
            "keep_alive": self.cfg.keep_alive, "options": self._options(),
        }
        if tools:
            body["tools"] = tools

        with self._post("/api/chat", body) as resp:
            for line in resp:
                if self.epochs and epoch is not None and not self.epochs.is_current(epoch):
                    return                      # barge-in: drop the connection
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = d.get("message") or {}
                calls = msg.get("tool_calls")
                if calls and on_tool_calls:
                    on_tool_calls([
                        ToolCall(name=c["function"]["name"],
                                 args=c["function"].get("arguments") or {})
                        for c in calls
                    ])
                tok = msg.get("content") or ""
                if tok:
                    yield tok
                if d.get("done"):
                    # Server-side timings: the only way to tell a slow prefill
                    # (cache miss / big tool schema) from slow decode.
                    if on_stats:
                        on_stats({
                            "prompt_tokens": d.get("prompt_eval_count", 0),
                            "prefill_ms": round(d.get("prompt_eval_duration", 0) / 1e6, 1),
                            "gen_tokens": d.get("eval_count", 0),
                            "gen_ms": round(d.get("eval_duration", 0) / 1e6, 1),
                            "load_ms": round(d.get("load_duration", 0) / 1e6, 1),
                        })
                    return

    def complete(self, messages: list[dict], epoch: int | None = None) -> str:
        return "".join(self.chat_stream(messages, epoch=epoch))

    def generate_json(self, prompt: str, system: str, schema: dict,
                      num_predict: int = 64) -> dict | None:
        """Schema-constrained single-shot generation.

        Constrained sampling makes malformed output structurally impossible --
        S2 measured 8/8 valid JSON where free-form prompting on a 3B model drifts.
        """
        body = {"model": self.cfg.model, "prompt": prompt, "system": system,
                "stream": False, "format": schema,
                "keep_alive": self.cfg.keep_alive,
                "options": self._options(num_predict=num_predict)}
        with self._post("/api/generate", body) as r:
            d = json.loads(r.read())
        self.last_stats = {
            "prompt_tokens": d.get("prompt_eval_count", 0),
            "prefill_ms": round(d.get("prompt_eval_duration", 0) / 1e6, 1),
            "gen_tokens": d.get("eval_count", 0),
            "gen_ms": round(d.get("eval_duration", 0) / 1e6, 1),
        }
        try:
            return json.loads(d.get("response", "") or "{}")
        except json.JSONDecodeError:
            return None

    # ---------------------------------------------------------------- router

    def route(self, text: str, tool_names: list[str]) -> str:
        """Schema-constrained single-tool router (fallback path).

        Measured 10/10 with the sharpened prompt in prompts.ROUTER_SYSTEM, and
        8/8 valid JSON -- constrained sampling makes malformed output impossible.
        Kept minimal: `args` is deliberately NOT in the schema because including it
        doubled emitted tokens and cost 47% more latency for no accuracy gain.
        """
        schema = {"type": "object",
                  "properties": {"tool": {"type": "string", "enum": tool_names}},
                  "required": ["tool"]}
        body = {"model": self.cfg.model, "prompt": text,
                "system": prompts.ROUTER_SYSTEM, "stream": False,
                "format": schema, "keep_alive": self.cfg.keep_alive,
                "options": self._options(num_predict=32)}
        with self._post("/api/generate", body) as r:
            d = json.loads(r.read())
        try:
            return json.loads(d.get("response", "{}")).get("tool", "respond")
        except json.JSONDecodeError:
            return "respond"
