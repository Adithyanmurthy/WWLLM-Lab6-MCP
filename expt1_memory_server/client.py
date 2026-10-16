"""
Experiment 1 — MCP Client for the Memory Server

Starts server.py as a subprocess, connects to it over MCP (stdio), and lets you chat.
For every message the LLM decides whether to call save_note, search_notes (or another
CRUD tool), the tool runs on the server, and the LLM writes the final reply.

    python client.py                 # interactive chat
    python client.py "remember ..."  # one-shot query(ies) from the command line
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.agent import LLM_MODEL, run_agent  # noqa: E402

SYSTEM = (
    "You are a personal assistant with a long-term memory stored on an MCP server.\n"
    "- If the user tells you something to remember / note / save, call save_note with the "
    "fact itself as content (drop phrases like 'remember that' or 'tag it X') and 1-3 "
    "short tags.\n"
    "- If the user asks what they said, saved or noted about something, call search_notes "
    "with the key topic words (not the whole question).\n"
    "- Use list_notes, update_note or delete_note when asked to list, change or remove notes.\n"
    "Answer only from tool results; never invent notes. Keep replies short."
)

C = {"dim": "\033[2m", "cyan": "\033[36m", "yellow": "\033[33m", "green": "\033[32m",
     "bold": "\033[1m", "end": "\033[0m"}


def show_trace(trace):
    for t in trace:
        if t["step"] == "guard":
            print(f"{C['dim']}  ⚠️  LLM answered without a tool → guard asks it to call one{C['end']}")
        elif t["step"] == "decide":
            print(f"{C['yellow']}  🤔 LLM decided → call {t['tool']}({t['args']}){C['end']}")
        elif t["step"] == "result":
            print(f"{C['dim']}  🛠  server returned: {t['text']}{C['end']}")
        elif t["step"] == "answer":
            print(f"{C['green']}{C['bold']}🤖 {t['text']}{C['end']}\n")


async def main(queries):
    server = StdioServerParameters(command=sys.executable,
                                   args=[str(Path(__file__).with_name("server.py"))])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print(f"{C['cyan']}Connected to MCP server 'personal-memory' | tools: {tools} "
                  f"| LLM: {LLM_MODEL}{C['end']}\n")

            history = []
            interactive = not queries
            while True:
                if interactive:
                    try:
                        q = input(f"{C['bold']}🧑 You: {C['end']}").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if q.lower() in {"exit", "quit", "q"}:
                        break
                    if not q:
                        continue
                else:
                    if not queries:
                        break
                    q = queries.pop(0)
                    print(f"{C['bold']}🧑 You: {C['end']}{q}")
                _, trace, msgs = await run_agent(session, SYSTEM, q, history[-6:])
                history = msgs
                show_trace(trace)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
