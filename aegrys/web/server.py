"""Local control-plane API for the Aegrys dashboard.

Two audiences, one build:

  - Running LOCALLY (`aegrys-dash`), this serves the static UI *and* the control
    API, so the dashboard can start/stop the assistant and stream its logs.
  - Hosted STATICALLY on a domain, only the files in `static/` are deployed. The
    UI probes `/api/status`, gets nothing, and switches to OFFLINE mode: the docs,
    setup guide and benchmark screens all still work; the control screens explain
    that they need a local install.

That split is deliberate. Process control is inherently local — a website cannot
start a program on a visitor's machine, and should not pretend it can.

Binds to 127.0.0.1 only. These endpoints spawn processes; they are not
authenticated and must never be exposed to a network.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
LOG_LINES = 500

app = FastAPI(title="Aegrys Control", docs_url=None, redoc_url=None)


class Runner:
    """Supervises one child process and keeps a ring buffer of its output."""

    def __init__(self, name: str):
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=LOG_LINES)
        self.started_at: float | None = None
        self._pump: threading.Thread | None = None
        # Surfaced in /api/status so a browser reload can restore the input box
        # instead of silently leaving it disabled while the child waits on stdin.
        self.text_mode = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], env: dict | None = None) -> None:
        if self.running:
            return
        self.lines.clear()
        e = dict(os.environ)
        e["PYTHONIOENCODING"] = "utf-8"
        e["PYTHONUNBUFFERED"] = "1"
        if env:
            e.update(env)
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=e,
            stdin=subprocess.PIPE,      # text mode is driven from the MONITOR screen
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           if sys.platform == "win32" else 0),
        )
        self.started_at = time.time()
        # A plain thread, not asyncio.create_task: FastAPI runs sync endpoints in
        # a worker thread with no running event loop, so create_task raises there.
        # The SSE endpoint polls this buffer anyway, so nothing needs to be async.
        self._pump = threading.Thread(target=self._read, daemon=True)
        self._pump.start()

    def _read(self) -> None:
        stdout = self.proc.stdout if self.proc else None
        if stdout is None:
            return
        for line in iter(stdout.readline, ""):
            text = strip_ansi(line.rstrip("\n"))
            if is_shutdown_noise(text):
                continue
            self.lines.append(text)
        self.lines.append(f"[{self.name} exited]")

    def send(self, line: str) -> bool:
        """Write a line to the child's stdin (text mode)."""
        if not self.running or self.proc is None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(line.rstrip("\n") + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def stop(self) -> None:
        if not self.running or self.proc is None:
            return
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def status(self) -> dict:
        return {"name": self.name, "running": self.running,
                "pid": self.proc.pid if self.running and self.proc else None,
                "text_mode": self.text_mode and self.running,
                "uptime_s": round(time.time() - self.started_at, 1)
                if self.running and self.started_at else None}


def strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# Stopping on Windows sends CTRL_BREAK so the assistant can shut down cleanly
# (closing audio streams and reaping its four MCP subprocesses). The Intel Fortran
# runtime bundled with the numeric stack reacts by printing an alarming-looking
# stack dump. It is cosmetic — the process exits correctly — but it reads like a
# crash, so it is filtered out rather than shown to the user.
_NOISE = (
    "forrtl: error (200)",
    "Image              PC",
    "KERNELBASE.dll",
    "KERNEL32.DLL",
    "ntdll.dll",
    "libifcoremd.dll",
)


def is_shutdown_noise(line: str) -> bool:
    return any(marker in line for marker in _NOISE)


assistant = Runner("aegrys")
backend = Runner("ollama")


# ------------------------------------------------------------------ preflight

def cache_dir() -> Path:
    return Path(os.environ.get("AEGRYS_CACHE", ROOT / ".cache"))


def preflight() -> list[dict]:
    c = cache_dir()
    checks = []

    def add(key, label, ok, detail, fix=""):
        checks.append({"key": key, "label": label, "ok": bool(ok),
                       "detail": detail, "fix": fix})

    add("python", "Python 3.12+", sys.version_info >= (3, 12),
        ".".join(map(str, sys.version_info[:3])))
    add("ollama", "ollama installed", shutil.which("ollama") is not None,
        shutil.which("ollama") or "not on PATH",
        "Install from https://ollama.com")

    vad = c / "vad" / "silero_vad.onnx"
    add("vad", "Silero VAD model", vad.exists(),
        f"{vad.stat().st_size/1e6:.1f} MB" if vad.exists() else "missing",
        "python scripts/fetch_models.py")

    piper = list((c / "piper").glob("*.onnx")) if (c / "piper").exists() else []
    add("tts", "Piper voice", bool(piper),
        piper[0].name if piper else "missing",
        "python scripts/fetch_models.py")

    ics = Path(os.environ.get("AEGRYS_ICS", c / "calendar.ics"))
    add("fixtures", "Demo fixtures", ics.exists(),
        "calendar + mailbox" if ics.exists() else "not seeded",
        "python scripts/seed_demo_data.py")

    try:
        import sounddevice as sd
        di, do = sd.default.device
        din = sd.query_devices(di)["name"] if di is not None else "?"
        dout = sd.query_devices(do)["name"] if do is not None else "?"
        add("audio", "Audio devices", True, f"in: {din[:26]} / out: {dout[:26]}")
    except Exception as e:
        add("audio", "Audio devices", False, f"{type(e).__name__}", "Check drivers")

    add("llm", "LLM backend reachable", llm_alive(),
        os.environ.get("AEGRYS_LLM_HOST", "http://127.0.0.1:11435"),
        "Start it from the CONTROL screen")
    return checks


def llm_alive() -> bool:
    import urllib.error
    import urllib.request
    host = os.environ.get("AEGRYS_LLM_HOST", "http://127.0.0.1:11435")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5):
            return True
    except (urllib.error.URLError, OSError):
        return False


# ----------------------------------------------------------------------- api

@app.get("/api/status")
def api_status():
    return {
        "ok": True,
        "assistant": assistant.status(),
        "backend": backend.status(),
        "llm_alive": llm_alive(),
        "root": str(ROOT),
    }


@app.get("/api/preflight")
def api_preflight():
    checks = preflight()
    return {"checks": checks, "ready": all(c["ok"] for c in checks)}


@app.post("/api/backend/start")
def api_backend_start():
    if not shutil.which("ollama"):
        return JSONResponse({"error": "ollama is not installed"}, status_code=400)
    backend.start(
        ["ollama", "serve"],
        env={"OLLAMA_HOST": "127.0.0.1:11435",
             "OLLAMA_MODELS": str(cache_dir() / "ollama")})
    return {"ok": True}


@app.post("/api/backend/stop")
def api_backend_stop():
    backend.stop()
    return {"ok": True}


@app.post("/api/start")
async def api_start(opts: dict | None = None):
    opts = opts or {}
    cmd = [sys.executable, "-u", "-m", "aegrys.cli"]
    if opts.get("mode") == "text":
        cmd.append("--text")
    if opts.get("no_tools"):
        cmd.append("--no-tools")
    if opts.get("no_barge_in"):
        cmd.append("--no-barge-in")
    if opts.get("stt_model"):
        cmd += ["--stt-model", str(opts["stt_model"])]
    cmd += ["--trace-file", str(ROOT / "traces.jsonl")]
    assistant.text_mode = opts.get("mode") == "text"
    assistant.start(cmd)
    return {"ok": True, "cmd": " ".join(cmd), "text_mode": opts.get("mode") == "text"}


@app.post("/api/send")
async def api_send(payload: dict):
    """Type into the assistant when it is running in text mode."""
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    if not assistant.send(text):
        return JSONResponse({"error": "assistant is not accepting input"},
                            status_code=409)
    return {"ok": True}


@app.post("/api/stop")
def api_stop():
    assistant.stop()
    return {"ok": True}


@app.get("/api/logs")
async def api_logs(source: str = "assistant"):
    """Server-sent events: replay the buffer, then stream new lines."""
    runner = assistant if source == "assistant" else backend

    async def gen():
        sent = 0
        snapshot = list(runner.lines)
        for line in snapshot:
            yield f"data: {json.dumps(line)}\n\n"
        sent = len(snapshot)
        while True:
            await asyncio.sleep(0.35)
            cur = list(runner.lines)
            if len(cur) < sent:          # buffer was cleared on restart
                sent = 0
            for line in cur[sent:]:
                yield f"data: {json.dumps(line)}\n\n"
            sent = len(cur)
            yield ": ping\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/traces")
def api_traces(limit: int = 50):
    path = ROOT / "traces.jsonl"
    if not path.exists():
        return {"turns": []}
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"turns": rows[-limit:]}


@app.get("/api/devices")
def api_devices():
    try:
        import sounddevice as sd
        hostapis = [a["name"] for a in sd.query_hostapis()]
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] or d["max_output_channels"]:
                out.append({"index": i, "name": d["name"],
                            "hostapi": hostapis[d["hostapi"]],
                            "in": d["max_input_channels"],
                            "out": d["max_output_channels"]})
        return {"devices": out}
    except Exception as e:
        return {"devices": [], "error": str(e)}


# -------------------------------------------------------------------- static

@app.get("/")
def index():
    # No-store: this is a local tool people will edit. Serving a cached shell
    # after a UI change is a confusing first thing to debug.
    return FileResponse(STATIC / "index.html",
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.middleware("http")
async def no_cache(request, call_next):
    """Never cache the UI.

    Same reason as index(): this is a local tool people are expected to edit, and
    serving a stale shell after a change is a baffling first thing to debug.
    Applied as middleware rather than by subclassing StaticFiles, because the
    internal hook for that has moved between Starlette versions.
    """
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


app.mount("/", StaticFiles(directory=str(STATIC)), name="static")


def main() -> int:
    import uvicorn
    port = int(os.environ.get("AEGRYS_DASH_PORT", "7860"))
    print(f"\n  AEGRYS dashboard -> http://127.0.0.1:{port}\n", flush=True)
    # 127.0.0.1 only: these endpoints spawn processes and have no auth.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
