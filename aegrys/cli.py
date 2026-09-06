"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# espeak-ng emits IPA; the default Windows cp1252 console raises
# UnicodeEncodeError on characters like U+025B. Set before anything imports piper.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .core import config as config_mod  # noqa: E402
from .core.trace import Tracer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("aegrys", description="Local voice assistant")
    p.add_argument("--text", action="store_true",
                   help="type instead of speaking (no mic needed)")
    p.add_argument("--say", metavar="TEXT",
                   help="run one turn with this text and exit")
    p.add_argument("--devices", action="store_true", help="list audio devices")
    p.add_argument("--no-barge-in", action="store_true",
                   help="disable barge-in (half-duplex)")
    p.add_argument("--no-tools", action="store_true", help="disable MCP tools")
    p.add_argument("--in-device", type=int, default=None)
    p.add_argument("--out-device", type=int, default=None)
    p.add_argument("--stt-model", default=None,
                   help="override STT model (tiny.en, base.en, ...)")
    p.add_argument("--trace-file", type=Path, default=None,
                   help="append per-turn JSON traces here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.devices:
        from .audio.io import list_devices
        print(list_devices())
        return 0

    cfg = config_mod.load()
    if args.in_device is not None:
        cfg.audio.in_device = args.in_device
    if args.out_device is not None:
        cfg.audio.out_device = args.out_device
    if args.stt_model:
        cfg.stt.model = args.stt_model
    if args.no_barge_in:
        cfg.barge_in = False
    if args.no_tools:
        cfg.tools.enabled = False

    tracer = Tracer(cfg.trace, args.trace_file)

    from .core.agent import build_assistant
    assistant = build_assistant(cfg, tracer)

    if not assistant.llm.health():
        print("error: LLM backend unreachable.\nStart it with:\n"
              "  OLLAMA_HOST=127.0.0.1:11435 "
              "OLLAMA_MODELS=D:/Aegrys/.cache/ollama ollama serve",
              file=sys.stderr)
        return 2

    try:
        if args.say:
            assistant.speaker.start()
            assistant.handle_text(args.say)
            assistant.speaker.wait_drained()
            return 0
        if args.text:
            return _text_loop(assistant)
        assistant.run_forever()
    finally:
        assistant.shutdown()
    return 0


def _text_loop(assistant) -> int:
    print("\n\033[1mAegrys\033[0m (text mode). Ctrl-C or 'quit' to exit.\n")
    assistant.speaker.start()
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            assistant.handle_text(text)
            assistant.speaker.wait_drained()
    except KeyboardInterrupt:
        pass
    print("\n" + assistant.tracer.table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
