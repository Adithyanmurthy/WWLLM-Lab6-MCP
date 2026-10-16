"""
Experiment 1 — "Personal Assistant" Memory Server (MCP server)

Exposes CRUD tools over a local notes.json file:
    save_note(content, tags)      -> Create
    search_notes(query)           -> Read (keyword search over content + tags)
    list_notes()                  -> Read (everything)
    update_note(note_id, ...)     -> Update
    delete_note(note_id)          -> Delete

Run directly (stdio transport):   python server.py
Normally it is started by client.py, which talks to it over stdin/stdout.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

NOTES_FILE = Path(os.getenv("NOTES_FILE", Path(__file__).with_name("notes.json")))

mcp = FastMCP("personal-memory", log_level="WARNING")

STOPWORDS = {"what", "did", "i", "say", "about", "the", "a", "an", "is", "my", "me", "of",
             "on", "for", "to", "do", "you", "have", "any", "notes", "note", "and", "in"}


# ---------- storage helpers ----------
def load_notes():
    if not NOTES_FILE.exists():
        return []
    return json.loads(NOTES_FILE.read_text() or "[]")


def save_notes(notes):
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


def fmt(note):
    tags = ", ".join(note["tags"]) or "-"
    return f"[#{note['id']}] {note['content']}  (tags: {tags}; saved {note['created']})"


def clean_tags(tags):
    if isinstance(tags, str):  # LLMs sometimes send "a, b" instead of ["a", "b"]
        tags = tags.split(",")
    return [t.strip().lower().lstrip("#") for t in (tags or []) if t.strip()]


# ---------- MCP tools ----------
@mcp.tool()
def save_note(content: str, tags: list[str] = []) -> str:
    """Save a new note to memory. Use when the user tells you something to remember.
    content: the full text of the note. tags: short topic keywords, e.g. ["project", "deadline"]."""
    notes = load_notes()
    note = {
        "id": max((n["id"] for n in notes), default=0) + 1,
        "content": content.strip(),
        "tags": clean_tags(tags),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    notes.append(note)
    save_notes(notes)
    return f"Saved note {fmt(note)}"


@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes by keyword. Use when the user asks what they said / saved / noted
    about something. query: the topic words, e.g. "project deadline"."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOPWORDS]
    scored = []
    for n in load_notes():
        hay = (n["content"] + " " + " ".join(n["tags"])).lower()
        score = sum(hay.count(w) for w in words) + sum(3 for w in words if w in n["tags"])
        if score:
            scored.append((score, n))
    if not scored:
        return f"No notes found for '{query}'."
    scored.sort(key=lambda s: -s[0])
    return f"{len(scored)} matching note(s):\n" + "\n".join(fmt(n) for _, n in scored)


@mcp.tool()
def list_notes() -> str:
    """List every saved note."""
    notes = load_notes()
    return "\n".join(fmt(n) for n in notes) if notes else "No notes saved yet."


@mcp.tool()
def update_note(note_id: int, content: str = "", tags: list[str] = []) -> str:
    """Change the content and/or tags of an existing note, by its id number."""
    notes = load_notes()
    for n in notes:
        if n["id"] == int(note_id):
            if content:
                n["content"] = content.strip()
            if tags:
                n["tags"] = clean_tags(tags)
            save_notes(notes)
            return f"Updated note {fmt(n)}"
    return f"No note with id {note_id}."


@mcp.tool()
def delete_note(note_id: int) -> str:
    """Delete a note by its id number."""
    notes = load_notes()
    kept = [n for n in notes if n["id"] != int(note_id)]
    if len(kept) == len(notes):
        return f"No note with id {note_id}."
    save_notes(kept)
    return f"Deleted note #{note_id}."


if __name__ == "__main__":
    mcp.run()  # stdio transport
