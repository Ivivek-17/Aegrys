"""MCP server: reminders. Fully local, SQLite-backed."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .store import connect, now

server = MCPServer(name="reminders", instructions="Personal reminders.")


@server.tool()
def add_reminder(text: str, when: str = "") -> str:
    """Remember a TASK, optionally at a clock time or date.

    Use this for "remind me to ...", including when a time is mentioned
    ("tomorrow at 6", "on Friday", "tonight"). Do NOT use set_timer for these --
    a clock time is not a countdown duration.

    Args:
        text: what to be reminded about, e.g. "call mom".
        when: when it should happen, in the user's own words, e.g. "tomorrow at 6".
    """
    text = text.strip()
    if not text:
        return "I need to know what to remind you about."
    con = connect()
    with con:
        con.execute(
            "INSERT INTO reminders (text, when_text, created_at) VALUES (?,?,?)",
            (text, when.strip(), now()))
    con.close()
    return f"Reminder saved: {text}" + (f" ({when})" if when.strip() else "") + "."


@server.tool()
def list_reminders() -> str:
    """List reminders that are not done yet."""
    con = connect()
    rows = con.execute(
        "SELECT id,text,when_text FROM reminders WHERE done=0 "
        "ORDER BY created_at DESC LIMIT 10").fetchall()
    con.close()
    if not rows:
        return "You have no reminders."
    parts = [r["text"] + (f" {r['when_text']}" if r["when_text"] else "")
             for r in rows]
    return f"You have {len(rows)} reminder(s): " + "; ".join(parts) + "."


@server.tool()
def complete_reminder(text: str) -> str:
    """Mark a reminder done by matching its text.

    Args:
        text: part of the reminder's wording, e.g. "call mom".
    """
    con = connect()
    with con:
        cur = con.execute(
            "UPDATE reminders SET done=1 WHERE done=0 AND id = ("
            "  SELECT id FROM reminders WHERE done=0 AND text LIKE ? "
            "  ORDER BY created_at DESC LIMIT 1)", (f"%{text.strip()}%",))
        n = cur.rowcount
    con.close()
    return "Marked done." if n else f"I couldn't find a reminder about {text}."


if __name__ == "__main__":
    server.run(transport="stdio")
