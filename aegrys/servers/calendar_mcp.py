"""MCP server: calendar, backed by a local .ics file.

A local iCalendar file rather than Google Calendar on purpose (DESIGN §6.4): it
keeps the "fully local" claim honest and avoids an OAuth flow in a demo. Minimal
parser inline so there's no extra dependency for a handful of VEVENTs.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ICS_PATH = Path(os.environ.get("AEGRYS_ICS", "D:/Aegrys/.cache/calendar.ics"))

server = MCPServer(name="calendar", instructions="Read the local calendar.")


def _parse_dt(value: str) -> dt.datetime | None:
    value = value.strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M%SZ", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _events() -> list[dict]:
    if not ICS_PATH.exists():
        return []
    text = ICS_PATH.read_text(encoding="utf-8", errors="replace")
    # Unfold RFC 5545 continuation lines before parsing.
    text = re.sub(r"\r?\n[ \t]", "", text)
    out = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        ev = {}
        for line in block.strip().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.split(";")[0].upper()
            if key == "SUMMARY":
                ev["summary"] = val.strip()
            elif key == "DTSTART":
                ev["start"] = _parse_dt(val)
            elif key == "LOCATION":
                ev["location"] = val.strip()
        if ev.get("start") and ev.get("summary"):
            out.append(ev)
    return sorted(out, key=lambda e: e["start"])


def _match_day(when: str) -> dt.date | None:
    w = (when or "today").strip().lower()
    today = dt.date.today()
    if w in ("", "today"):
        return today
    if w == "tomorrow":
        return today + dt.timedelta(days=1)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]
    if w in days:
        delta = (days.index(w) - today.weekday()) % 7
        return today + dt.timedelta(days=delta or 7)
    return None


@server.tool()
def list_events(when: str = "today") -> str:
    """Read what is scheduled on the calendar.

    Args:
        when: "today", "tomorrow", a weekday name, or "week".
    """
    evs = _events()
    if not evs:
        return "The calendar is empty."
    if (when or "").strip().lower() == "week":
        end = dt.date.today() + dt.timedelta(days=7)
        sel = [e for e in evs if dt.date.today() <= e["start"].date() <= end]
        label = "in the next week"
    else:
        day = _match_day(when)
        if day is None:
            return f"I don't understand the date {when!r}."
        sel = [e for e in evs if e["start"].date() == day]
        label = when.strip().lower() or "today"
    if not sel:
        return f"Nothing scheduled {label}."
    parts = [f"{e['summary']} at {e['start'].strftime('%-I:%M %p') if os.name != 'nt' else e['start'].strftime('%I:%M %p').lstrip('0')}"
             for e in sel]
    return f"{len(sel)} event(s) {label}: " + "; ".join(parts) + "."


@server.tool()
def next_event() -> str:
    """The next upcoming event on the calendar."""
    now = dt.datetime.now()
    upcoming = [e for e in _events() if e["start"] >= now]
    if not upcoming:
        return "Nothing else is scheduled."
    e = upcoming[0]
    return (f"Next up is {e['summary']} at "
            f"{e['start'].strftime('%I:%M %p').lstrip('0')} on "
            f"{e['start'].strftime('%A')}.")


if __name__ == "__main__":
    server.run(transport="stdio")
