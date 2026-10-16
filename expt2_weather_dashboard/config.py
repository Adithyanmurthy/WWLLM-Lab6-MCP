"""Shared settings for the weather app (app.py) and terminal client (cli.py)."""

import sys
from pathlib import Path

from mcp import StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable,
                               args=[str(Path(__file__).with_name("server.py"))])
SYSTEM = (
    "You are a weather assistant connected to a live weather MCP server.\n"
    "For any question about current weather call get_current_weather; for tomorrow / "
    "the next days call get_forecast and read ONLY the line labelled with the day asked "
    "about. Pass only the place name as location.\n"
    "Then give a short, friendly summary using the real numbers from the tool result "
    "(temperature, feels-like, condition, humidity, wind) and one practical tip. "
    "Never guess weather without calling a tool."
)
