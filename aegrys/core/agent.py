"""Assistant factory.

Each phase adds capability without rewriting the one below it:
  Phase 1  Assistant          half-duplex loop
  Phase 2  BargeInAssistant   + epoch cancellation, full-duplex listening
  Phase 3  ToolAssistant      + MCP tools
"""

from __future__ import annotations

from .config import Config
from .trace import Tracer


def build_assistant(cfg: Config, tracer: Tracer | None = None):
    if cfg.tools.enabled:
        from .tool_turn import ToolAssistant
        return ToolAssistant(cfg, tracer)
    if cfg.barge_in:
        from .barge import BargeInAssistant
        return BargeInAssistant(cfg, tracer)
    from .turn import Assistant
    return Assistant(cfg, tracer)
