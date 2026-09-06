"""Per-turn latency tracing (DESIGN §10).

This is built in Phase 1 deliberately. Retrofitting instrumentation means
re-deriving every number by hand later, and the measured numbers ARE the
deliverable for this project.

Emits one line per turn:
  turn 7 | vad_end->stt 312ms | ->llm_tok1 640ms | ->tts_chunk1 380ms | TTFA 1332ms
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# The metric that matters is speech-end -> first audio out. Not time-to-first-token;
# the user cannot hear tokens.
TTFA_ANCHOR = "speech_end"
TTFA_TARGET = "tts_first_chunk"


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000


@dataclass
class TurnTrace:
    turn_id: int
    t0: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    cancelled: bool = False

    def mark(self, name: str) -> float:
        """Record an instant. First write wins (first token, first chunk)."""
        if name not in self.marks:
            self.marks[name] = time.perf_counter()
        return self.marks[name]

    def since(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return (self.marks[b] - self.marks[a]) * 1000
        return None

    @property
    def ttfa_ms(self) -> float | None:
        return self.since(TTFA_ANCHOR, TTFA_TARGET)

    def summary(self) -> str:
        parts = [f"turn {self.turn_id}"]
        stt = self.since("speech_end", "stt_done")
        tok1 = self.since("stt_done", "llm_first_token")
        tts1 = self.since("llm_first_sentence", "tts_first_chunk")
        if stt is not None:
            parts.append(f"stt {stt:.0f}ms")
        if tok1 is not None:
            parts.append(f"llm_tok1 {tok1:.0f}ms")
        # On tool turns the filler marks tts_first_chunk BEFORE the synthesis
        # call produces a sentence, so this interval is meaningless (negative).
        if tts1 is not None and tts1 >= 0:
            parts.append(f"tts1 {tts1:.0f}ms")
        if "tool" in self.meta:
            parts.append("filler-masked")
        if self.ttfa_ms is not None:
            parts.append(f"TTFA {self.ttfa_ms:.0f}ms")
        if self.cancelled:
            parts.append("CANCELLED")
        for k in ("tool", "llm1", "llm2", "text"):
            if k in self.meta:
                v = str(self.meta[k])
                parts.append(f"{k}={v[:110]}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        base = self.marks.get(TTFA_ANCHOR, self.t0)
        return {
            "turn_id": self.turn_id,
            "cancelled": self.cancelled,
            "ttfa_ms": round(self.ttfa_ms, 1) if self.ttfa_ms else None,
            "marks_ms": {k: round((v - base) * 1000, 1)
                         for k, v in sorted(self.marks.items(), key=lambda x: x[1])},
            "spans_ms": {s.name: round(s.ms, 1) for s in self.spans},
            "meta": self.meta,
        }


class Tracer:
    def __init__(self, enabled: bool = True, path: Path | None = None):
        self.enabled = enabled
        self.path = path
        self._lock = threading.Lock()
        self._current: TurnTrace | None = None
        self.history: list[TurnTrace] = []

    def start_turn(self, turn_id: int) -> TurnTrace:
        t = TurnTrace(turn_id=turn_id)
        with self._lock:
            self._current = t
            self.history.append(t)
        return t

    @property
    def current(self) -> TurnTrace | None:
        return self._current

    def mark(self, name: str) -> None:
        if self._current is not None:
            self._current.mark(name)

    @contextmanager
    def span(self, name: str, **meta):
        s = Span(name=name, start=time.perf_counter(), meta=meta)
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            if self._current is not None:
                self._current.spans.append(s)

    def end_turn(self) -> None:
        t = self._current
        if t is None or not self.enabled:
            return
        print(f"  \033[2m{t.summary()}\033[0m", flush=True)
        if self.path:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(t.to_dict()) + "\n")

    def table(self) -> str:
        """Aggregate view — the thing that goes in the README."""
        done = [t for t in self.history if t.ttfa_ms is not None]
        if not done:
            return "no completed turns"
        vals = sorted(t.ttfa_ms for t in done)
        med = vals[len(vals) // 2]
        rows = [f"{len(done)} turns | TTFA median {med:.0f}ms "
                f"min {vals[0]:.0f}ms max {vals[-1]:.0f}ms"]
        for stage, a, b in (("stt", "speech_end", "stt_done"),
                            ("llm_tok1", "stt_done", "llm_first_token"),
                            ("tts1", "llm_first_sentence", "tts_first_chunk")):
            xs = sorted(x for x in (t.since(a, b) for t in done)
                        if x is not None and x >= 0)
            if xs:
                rows.append(f"  {stage:<9} median {xs[len(xs)//2]:6.0f}ms")
        return "\n".join(rows)
