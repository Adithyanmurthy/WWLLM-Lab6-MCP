"""
Tests for both MCP servers.
  * unit tests call the tool functions directly
  * protocol tests start the real server over stdio and talk MCP to it (no LLM needed)

    python -m pytest -q tests
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "expt1_memory_server"))
import server as memory  # noqa: E402

sys.path.pop(0)
sys.modules.pop("server")
sys.path.insert(0, str(ROOT / "expt2_weather_dashboard"))
import server as weather  # noqa: E402

sys.path.insert(0, str(ROOT))
from common.agent import _text_tool_call, mcp_tools_to_openai  # noqa: E402


@pytest.fixture
def notes_file(tmp_path, monkeypatch):
    f = tmp_path / "notes.json"
    monkeypatch.setattr(memory, "NOTES_FILE", f)
    return f


# ---------- Expt 1: memory server (CRUD) ----------
def test_save_and_search(notes_file):
    memory.save_note("Project deadline is Oct 30", ["project", "deadline"])
    memory.save_note("Buy milk", ["groceries"])
    out = memory.search_notes("What did I say about the project deadline?")
    assert "Oct 30" in out and "milk" not in out
    assert json.loads(notes_file.read_text())[0]["tags"] == ["project", "deadline"]


def test_tags_given_as_string_are_split(notes_file):
    memory.save_note("x", "Work, #Urgent")
    assert memory.load_notes()[0]["tags"] == ["work", "urgent"]


def test_update_and_delete(notes_file):
    memory.save_note("old text", ["a"])
    assert "new text" in memory.update_note(1, content="new text")
    assert "Deleted" in memory.delete_note(1)
    assert memory.list_notes() == "No notes saved yet."
    assert "No note" in memory.delete_note(99)


def test_search_no_match(notes_file):
    memory.save_note("hello", [])
    assert memory.search_notes("deadline").startswith("No notes found")


# ---------- Expt 2: weather server ----------
FAKE = {
    "nearest_area": [{"areaName": [{"value": "Shikinejima"}], "country": [{"value": "Japan"}]}],
    "current_condition": [{"observation_time": "07:43 AM", "weatherDesc": [{"value": "Overcast"}],
                           "temp_C": "22", "FeelsLikeC": "24", "humidity": "69",
                           "windspeedKmph": "4", "winddir16Point": "NNW", "precipMM": "0.0",
                           "cloudcover": "100", "uvIndex": "0", "visibility": "10"}],
    "weather": [{"date": "2026-10-16", "mintempC": "18", "maxtempC": "23",
                 "hourly": [{"weatherDesc": [{"value": "Cloudy"}], "chanceofrain": "40"}] * 8}],
}


def test_current_weather_structured(monkeypatch):
    monkeypatch.setattr(weather, "fetch", lambda loc: FAKE)
    out = weather.get_current_weather("Tokyo")
    assert "temperature_c: 22" in out
    assert "Tokyo, Japan (nearest station: Shikinejima)" in out


def test_forecast(monkeypatch):
    monkeypatch.setattr(weather, "fetch", lambda loc: FAKE)
    assert "today (2026-10-16): Cloudy" in weather.get_forecast("Tokyo")


def test_weather_error_is_reported(monkeypatch):
    def boom(loc):
        raise RuntimeError("network down")
    monkeypatch.setattr(weather, "fetch", boom)
    assert weather.get_current_weather("Nowhere").startswith("ERROR")


# ---------- agent helpers ----------
def test_text_tool_call_recovery():
    call = _text_tool_call('{"name": "search_notes", "arguments": {"query": "x"}}', {"search_notes"})
    assert call["function"]["name"] == "search_notes"
    assert _text_tool_call("just text", {"search_notes"}) is None


# ---------- real MCP protocol over stdio ----------
async def _session_roundtrip(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "expt1_memory_server" / "server.py")],
        env={**os.environ, "NOTES_FILE": str(tmp_path / "n.json")})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            await s.call_tool("save_note", {"content": "MCP works", "tags": ["lab"]})
            res = await s.call_tool("search_notes", {"query": "mcp"})
            return tools, res.content[0].text


def test_mcp_protocol_roundtrip(tmp_path):
    tools, text = asyncio.run(_session_roundtrip(tmp_path))
    names = {t.name for t in tools}
    assert {"save_note", "search_notes", "list_notes", "update_note", "delete_note"} <= names
    schema = mcp_tools_to_openai(tools)[0]["function"]
    assert schema["parameters"]["type"] == "object"
    assert "MCP works" in text


# ---------- grounding guard (fake LLM + fake MCP session) ----------
def test_guard_forces_a_tool_call(monkeypatch):
    import common.agent as agent

    replies = iter([
        {"content": "I've saved your note!"},  # lie: no tool call
        {"content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "save_note", "arguments": '{"content": "buy milk", "tags": ["groceries"]}'}}]},
        {"content": "Saved note #1."},
    ])
    monkeypatch.setattr(agent, "chat", lambda messages, tools: next(replies))

    class Tool:
        name, description, inputSchema = "save_note", "", {"type": "object", "properties": {}}

    class Text:
        text = "Saved note [#1] buy milk"

    class Session:
        calls = []

        async def list_tools(self):
            return type("R", (), {"tools": [Tool()]})()

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return type("R", (), {"content": [Text()], "isError": False})()

    s = Session()
    answer, trace, _ = asyncio.run(agent.run_agent(s, "sys", "Note that I need to buy milk"))
    assert [t["step"] for t in trace] == ["user", "guard", "decide", "result", "answer"]
    assert s.calls == [("save_note", {"content": "buy milk", "tags": ["groceries"]})]
    assert answer == "Saved note #1."
