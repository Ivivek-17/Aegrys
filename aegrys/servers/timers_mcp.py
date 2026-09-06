"""MCP server: countdown timers. Fully local, SQLite-backed.

Tool descriptions matter more than usual here. S2b measured routing accuracy going
from 7/10 to 10/10 purely by contrasting a DURATION (timer) against a CLOCK TIME
(reminder) -- the 3B model was reading "at 6" as a duration. Keep that contrast
explicit in the docstrings; they become the schema the model actually sees.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .store import connect, human_duration, now

server = MCPServer(name="timers", instructions="Countdown timers.")


@server.tool()
def set_timer(seconds: int, label: str = "timer") -> str:
    """Start a countdown for a DURATION, e.g. "timer for 5 minutes", "3 minute timer".

    Use this ONLY for a length of time from now. For something at a clock time or
    date ("at 6", "tomorrow", "on Friday"), use add_reminder instead.

    Args:
        seconds: how long the countdown runs, in seconds.
        label: what the timer is for, e.g. "pasta".
    """
    if seconds <= 0:
        return "A timer needs a positive duration."
    if seconds > 24 * 3600:
        return "That's longer than a day; use a reminder instead."
    con = connect()
    with con:
        cur = con.execute(
            "INSERT INTO timers (label, created_at, expires_at) VALUES (?,?,?)",
            (label, now(), now() + seconds))
    con.close()
    return f"Timer {cur.lastrowid} set for {human_duration(seconds)}."


@server.tool()
def list_timers() -> str:
    """List timers that are still running."""
    con = connect()
    rows = con.execute(
        "SELECT id,label,expires_at FROM timers "
        "WHERE fired=0 AND cancelled=0 AND expires_at > ? ORDER BY expires_at",
        (now(),)).fetchall()
    con.close()
    if not rows:
        return "No timers are running."
    parts = [f"{r['label']} with {human_duration(r['expires_at'] - now())} left"
             for r in rows]
    return "Running timers: " + "; ".join(parts) + "."


@server.tool()
def cancel_timer(timer_id: int = 0) -> str:
    """Cancel a timer. With no id, cancels the one finishing soonest.

    Args:
        timer_id: which timer to cancel; 0 means the next one to finish.
    """
    con = connect()
    with con:
        if timer_id:
            cur = con.execute(
                "UPDATE timers SET cancelled=1 WHERE id=? AND fired=0", (timer_id,))
        else:
            cur = con.execute(
                "UPDATE timers SET cancelled=1 WHERE id = ("
                "  SELECT id FROM timers WHERE fired=0 AND cancelled=0 "
                "  ORDER BY expires_at LIMIT 1)")
        n = cur.rowcount
    con.close()
    return "Timer cancelled." if n else "There was no timer to cancel."


if __name__ == "__main__":
    server.run(transport="stdio")
