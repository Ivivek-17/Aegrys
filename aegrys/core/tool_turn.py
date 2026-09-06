"""Phase 3 — MCP tools bound to the turn loop.

Turn shape (DESIGN §6.2 as corrected by S2b):

  ONE streaming call with tools bound.
    - conversational turn  -> tokens stream straight to TTS. No router tax.
    - tool turn            -> filler audio starts immediately, tools run, then a
                              SECOND synthesis call speaks the result.

The two-call router the design originally specified cost 1096 ms before synthesis
even started; skipping it on chat turns is where the latency win comes from.

Prompt-injection defence (DESIGN §6.5) is structural, not a plea in the prompt:
the synthesis call that SEES tool output has NO tools bound, so it cannot act on
anything the output tells it to do. The call that can act never sees the content.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..llm.prompts import SYSTEM, tool_result_prompt
from ..servers.store import connect, human_duration
from ..tools.mcp_manager import MCPManager
from ..tts.chunker import ClauseChunker
from .barge import BargeInAssistant
from .config import Config
from .trace import Tracer

FILLERS = ["Let me check.", "One moment.", "Checking that now."]


class ToolAssistant(BargeInAssistant):
    def __init__(self, cfg: Config, tracer: Tracer | None = None):
        super().__init__(cfg, tracer)
        self.mcp = MCPManager(timeout=cfg.tools.call_timeout_s)
        print("starting MCP servers...", flush=True)
        self.mcp.start()
        print(f"  {self.mcp.status()}", flush=True)
        self._tool_schema = self.mcp.openai_tools()
        from ..llm.router import CompactRouter
        self.router = CompactRouter(self.llm, self.mcp.tools)
        self.mode = cfg.tools.mode
        # Pre-synthesize filler audio once. It masks tool latency, and generating
        # it on demand would defeat the purpose.
        self._fillers = [self.tts.synthesize(f) for f in FILLERS]
        self._filler_i = 0
        self._timer_thread: threading.Thread | None = None

    def shutdown(self) -> None:
        try:
            self.mcp.stop()
        except Exception:
            pass
        super().shutdown()

    # ---------------------------------------------------------------- timers

    def start(self) -> None:
        super().start()
        self._reap_stale_timers()
        self._timer_thread = threading.Thread(target=self._watch_timers,
                                              daemon=True)
        self._timer_thread.start()

    def _reap_stale_timers(self) -> None:
        """Retire timers that expired while we weren't running -- silently.

        Without this, every timer left over from a previous session announces
        itself the moment the app starts. Observed live: ten stale timers from
        earlier benchmark runs all fired at once on startup. A timer that elapsed
        hours ago is not news.
        """
        try:
            con = connect()
            with con:
                cur = con.execute(
                    "UPDATE timers SET fired=1 WHERE fired=0 AND cancelled=0 "
                    "AND expires_at <= strftime('%s','now')")
                n = cur.rowcount
            con.close()
            if n:
                print(f"  \033[2mretired {n} timer(s) that expired while "
                      f"offline\033[0m", flush=True)
        except Exception:
            pass

    def _watch_timers(self) -> None:
        """Announce timers when they actually fire — otherwise they're just rows."""
        while self._running:
            time.sleep(1.0)
            try:
                con = connect()
                with con:
                    rows = con.execute(
                        "SELECT id,label FROM timers WHERE fired=0 AND cancelled=0 "
                        "AND expires_at <= strftime('%s','now')").fetchall()
                    for r in rows:
                        con.execute("UPDATE timers SET fired=1 WHERE id=?",
                                    (r["id"],))
                con.close()
            except Exception:
                continue
            for r in rows:
                if not self._running:
                    return
                self.speaker.write(
                    self.tts.synthesize(f"Your {r['label']} is done."))
                print(f"  \033[35m⏰ {r['label']} finished\033[0m", flush=True)

    # --------------------------------------------------------------- filler

    def _play_filler(self, epoch: int) -> None:
        if not self.epochs.is_current(epoch):
            return
        audio = self._fillers[self._filler_i % len(self._fillers)]
        self._filler_i += 1
        self.speaker.write(audio)
        self.tracer.mark("tts_first_chunk")   # perceived TTFA starts here

    # -------------------------------------------------------------- respond

    def respond(self, text: str, epoch: int) -> str:
        if self.mode == "router":
            return self._respond_router(text, epoch)
        return self._respond_native(text, epoch)

    def _respond_router(self, text: str, epoch: int) -> str:
        """Two small constrained calls instead of one 900-token schema (S7/S8)."""
        with self.tracer.span("route"):
            call = self.router.route(text)
        self.tracer.mark("route_done")
        if self.tracer.current:
            self.tracer.current.meta["route"] = self.llm.last_stats
        if call is not None:
            return self._run_tools([call], epoch)
        # Plain chat: no tool schema in the prompt at all, so the prefill is tiny.
        self.history.append({"role": "user", "content": text})
        return self._stream_to_speaker(self.history, epoch, remember=True)

    def _stream_to_speaker(self, messages, epoch: int, remember: bool) -> str:
        chunker = ClauseChunker(min_words=self.cfg.tts.min_chunk_words)
        spoken: list[str] = []

        def stats(d):
            if self.tracer.current:
                self.tracer.current.meta["llm"] = d

        first = True
        for tok in self.llm.chat_stream(messages, epoch=epoch, on_stats=stats):
            if first:
                self.tracer.mark("llm_first_token")
                first = False
            for chunk in chunker.push(tok):
                if not self.epochs.is_current(epoch):
                    return " ".join(spoken)
                self.tracer.mark("llm_first_sentence")
                with self.tracer.span("tts", chars=len(chunk)):
                    audio = self.tts.synthesize(chunk)
                self.tracer.mark("tts_first_chunk")
                self.speaker.write(audio)
                spoken.append(chunk)
        for chunk in chunker.flush():
            if not self.epochs.is_current(epoch):
                break
            self.tracer.mark("llm_first_sentence")
            audio = self.tts.synthesize(chunk)
            self.tracer.mark("tts_first_chunk")
            self.speaker.write(audio)
            spoken.append(chunk)
        reply = " ".join(spoken)
        if reply and remember:
            self.history.append({"role": "assistant", "content": reply})
        return reply

    def _respond_native(self, text: str, epoch: int) -> str:
        self.history.append({"role": "user", "content": text})
        chunker = ClauseChunker(min_words=self.cfg.tts.min_chunk_words)
        spoken: list[str] = []
        calls: list = []

        def speak(chunk: str) -> bool:
            if not self.epochs.is_current(epoch):
                return False
            with self.tracer.span("tts", chars=len(chunk)):
                audio = self.tts.synthesize(chunk)
            if not self.epochs.is_current(epoch):
                return False
            self.tracer.mark("tts_first_chunk")
            self.speaker.write(audio)
            spoken.append(chunk)
            return True

        def stats(d):
            if self.tracer.current:
                self.tracer.current.meta["llm1"] = d

        first = True
        for tok in self.llm.chat_stream(self.history, tools=self._tool_schema,
                                        epoch=epoch,
                                        on_tool_calls=calls.extend,
                                        on_stats=stats):
            if calls:
                break            # a tool call supersedes any partial chatter
            if first:
                self.tracer.mark("llm_first_token")
                first = False
            for chunk in chunker.push(tok):
                self.tracer.mark("llm_first_sentence")
                if not speak(chunk):
                    return " ".join(spoken)

        if calls:
            return self._run_tools(calls, epoch)

        for chunk in chunker.flush():
            self.tracer.mark("llm_first_sentence")
            if not speak(chunk):
                break
        reply = " ".join(spoken)
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        return reply

    def _run_tools(self, calls, epoch: int) -> str:
        call = calls[0]
        self.tracer.mark("tool_call")
        if self.tracer.current:
            self.tracer.current.meta["tool"] = call.name
        print(f"  \033[34m⚙ {call.name}({call.args})\033[0m", flush=True)

        # Filler covers the tool round-trip plus the synthesis call.
        self._play_filler(epoch)

        with self.tracer.span("tool", tool=call.name):
            result = self.mcp.call(call.name, call.args)
        self.tracer.mark("tool_done")
        if not self.epochs.is_current(epoch):
            return ""

        # NOTE: no `tools=` on this call. The model that sees untrusted tool output
        # is structurally unable to invoke anything (DESIGN §6.5).
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": tool_result_prompt(call.name, result)}]
        reply = self._stream_to_speaker(messages, epoch, remember=False)
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        return reply
