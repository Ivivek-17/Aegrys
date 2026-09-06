"""Shared SQLite store for the MCP servers.

Each server runs as its own stdio subprocess, so they cannot share memory -- but
they can share a database file. SQLite in WAL mode handles the concurrent access.

Fully local by design: no network, no accounts, nothing to authorize. This is what
makes timers and reminders the demo-safe tools to lead with.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(os.environ.get("AEGRYS_DB", "D:/Aegrys/.cache/aegrys.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS timers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL DEFAULT 'timer',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0,
    cancelled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    when_text TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def now() -> float:
    return time.time()


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        out = f"{mins} minute{'s' if mins != 1 else ''}"
        return f"{out} and {secs} seconds" if secs else out
    hours, mins = divmod(mins, 60)
    out = f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{out} and {mins} minutes" if mins else out
