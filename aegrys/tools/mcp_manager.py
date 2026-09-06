"""MCP server supervisor.

MCP servers are stdio subprocesses and the SDK is asyncio; the orchestrator is
threaded and synchronous. So one asyncio loop runs on a dedicated thread and the
sync API here marshals onto it.

Two things the design insists on (DESIGN §6.4):
  - every call has a hard timeout, so one hung server cannot freeze the assistant;
  - subprocesses are supervised and reaped on exit, because they leak otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ServerSpec:
    name: str
    module: str          # python -m <module>
    enabled: bool = True


DEFAULT_SERVERS = [
    ServerSpec("timers", "aegrys.servers.timers_mcp"),
    ServerSpec("reminders", "aegrys.servers.reminders_mcp"),
    ServerSpec("calendar", "aegrys.servers.calendar_mcp"),
    ServerSpec("email", "aegrys.servers.email_mcp"),
]


@dataclass
class ToolInfo:
    name: str
    description: str
    schema: dict
    server: str

    def to_openai(self) -> dict:
        """The shape ollama/OpenAI-style tool calling expects."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.schema}}


class MCPManager:
    def __init__(self, servers: list[ServerSpec] | None = None,
                 timeout: float = 8.0):
        self.specs = [s for s in (servers or DEFAULT_SERVERS) if s.enabled]
        self.timeout = timeout
        self.tools: dict[str, ToolInfo] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = None
        self._sessions: dict[str, ClientSession] = {}
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("MCP servers failed to start within 60s")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        async with contextlib.AsyncExitStack() as stack:
            for spec in self.specs:
                try:
                    await self._connect(stack, spec)
                except Exception as e:      # one bad server must not sink the rest
                    self._errors[spec.name] = f"{type(e).__name__}: {e}"
            self._ready.set()
            await self._stop.wait()

    async def _connect(self, stack: contextlib.AsyncExitStack,
                       spec: ServerSpec) -> None:
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", spec.module])
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=30)
        listed = await asyncio.wait_for(session.list_tools(), timeout=30)
        self._sessions[spec.name] = session
        for t in listed.tools:
            # mcp 2.x renamed inputSchema -> input_schema; accept either so this
            # keeps working across SDK versions.
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
            self.tools[t.name] = ToolInfo(
                name=t.name,
                description=(t.description or "").strip(),
                schema=schema or {"type": "object", "properties": {}},
                server=spec.name)

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------ call

    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Invoke a tool synchronously. Never raises; returns an error string."""
        info = self.tools.get(name)
        if info is None:
            return f"There is no tool called {name}."
        session = self._sessions.get(info.server)
        if session is None or self._loop is None:
            return f"The {info.server} tool is unavailable."

        async def go():
            res = await asyncio.wait_for(
                session.call_tool(name, args or {}), timeout=self.timeout)
            parts = []
            for c in res.content:
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts) or "(no output)"

        fut = asyncio.run_coroutine_threadsafe(go(), self._loop)
        try:
            return fut.result(timeout=self.timeout + 2)
        except asyncio.TimeoutError:
            return f"The {name} tool timed out."
        except Exception as e:
            return f"The {name} tool failed: {type(e).__name__}."

    # ---------------------------------------------------------------- schema

    def openai_tools(self) -> list[dict]:
        return [t.to_openai() for t in self.tools.values()]

    @property
    def names(self) -> list[str]:
        return sorted(self.tools)

    def status(self) -> str:
        ok = f"{len(self.tools)} tools from {len(self._sessions)} servers"
        if self._errors:
            ok += " | failed: " + ", ".join(f"{k} ({v})"
                                            for k, v in self._errors.items())
        return ok
