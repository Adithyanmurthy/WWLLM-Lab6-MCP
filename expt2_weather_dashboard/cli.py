"""
Experiment 2 — terminal version of the weather client (same MCP server, same agent loop).

    python cli.py "What's the weather in Tokyo?"
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SERVER, SYSTEM  # noqa: E402  (same server + prompt as the web app)
from common.agent import run_agent  # noqa: E402


async def main(question):
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            answer, trace, _ = await run_agent(session, SYSTEM, question)
    for t in trace:
        if t["step"] == "user":
            print(f"🧑 {t['text']}")
        elif t["step"] == "decide":
            print(f"🤔 LLM decided → call {t['tool']}({t['args']})")
        elif t["step"] == "result":
            print("🛠  MCP server returned:\n    " + t["text"].replace("\n", "\n    "))
        else:
            print(f"🤖 {t['text']}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What's the weather in Tokyo?"))
