# Lab 06 — LLMs for Code Generation & Software Development: MCP Servers and Clients

**Course:** Working with Large Language Models · **USN:** 1RVU23CSE024 · **Name:** Adithya N Murthy

This lab builds two **Model Context Protocol (MCP)** applications. In each one, an LLM decides which tool to call, the MCP server runs the tool, and the result is fed back to the LLM to write the answer.

| | Experiment | MCP server tools | Client |
|---|---|---|---|
| 1 | **"Personal Assistant" Memory Server** | `save_note`, `search_notes`, `list_notes`, `update_note`, `delete_note` (full CRUD on `notes.json`) | Terminal chat (`client.py`) |
| 2 | **"Data Dashboard" Connector** | `get_current_weather`, `get_forecast` (live data from **wttr.in**) | **Streamlit web app** (`app.py`) that shows the LLM's thought process, plus a terminal version (`cli.py`) |

The LLM runs **locally** with Ollama (`qwen2.5:3b`), so no API key is needed. Any OpenAI-compatible provider also works (see [Using a different LLM](#using-a-different-llm)).

![Weather dashboard](screenshots/expt2_01_tokyo_weather.png)

---

## Architecture

```
 ┌──────────────┐  question   ┌────────────────────────────┐   tools/list, tools/call   ┌──────────────────────┐
 │    User      │ ──────────▶ │  MCP CLIENT  (client.py /   │ ◀────── JSON-RPC ───────▶  │  MCP SERVER          │
 │ (terminal or │ ◀────────── │  app.py)                    │      over stdio            │  (server.py)         │
 │  browser)    │   answer    │   └ agent loop (agent.py)   │                            │  ├ notes.json  (E1)  │
 └──────────────┘             └─────────────┬──────────────┘                            │  └ wttr.in API (E2)  │
                                            │ messages + tool schemas                    └──────────────────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │ LLM (Ollama,     │  decides: answer directly,
                                   │ qwen2.5:3b)      │  or call tool X with args Y
                                   └──────────────────┘
```

**The agent loop** (`common/agent.py`) handles one question like this:

1. `session.list_tools()` gets the tool names, descriptions and JSON schemas from the server.
2. The question and the tool schemas are sent to the LLM (`/chat/completions` with `tools`).
3. If the LLM returns a `tool_call`, the client runs `session.call_tool(name, args)` and sends the result back to the LLM. This repeats if needed.
4. When the LLM replies with plain text, that is the final answer.
5. **Grounding guard:** if the model's first reply calls no tool (for example "I've saved your note!" when nothing was saved), the client tells it to call a tool and retries once.

## Folder layout

```
L6/
├── common/agent.py                 # shared MCP-client ↔ LLM tool-calling loop
├── expt1_memory_server/
│   ├── server.py                   # MCP server: CRUD tools over notes.json
│   ├── client.py                   # MCP client: terminal chat, LLM picks the tool
│   └── notes.json                  # the memory store (created on first save)
├── expt2_weather_dashboard/
│   ├── server.py                   # MCP server: get_current_weather, get_forecast (wttr.in)
│   ├── app.py                      # Streamlit web app: shows the thought process
│   ├── cli.py                      # same flow in the terminal
│   └── config.py                   # server launch settings + system prompt
├── tests/test_servers.py           # 10 tests: tools, MCP over stdio, guard
├── screenshots/                    # screenshots for submission
├── docs/                           # lab report (md / html / pdf)
└── requirements.txt
```

## Setup

```bash
# 1. Python packages (Python 3.10+)
pip install -r requirements.txt

# 2. Local LLM (free, offline): install Ollama from https://ollama.com, then
ollama pull qwen2.5:3b
```

## Run

**Experiment 1: Memory Server**
```bash
python expt1_memory_server/client.py            # interactive chat ("exit" to quit)
python expt1_memory_server/client.py "Remember that the project deadline is Oct 30" \
                                     "What did I say about the project deadline?"
```

**Experiment 2: Weather Dashboard**
```bash
cd expt2_weather_dashboard
streamlit run app.py                            # opens http://localhost:8501
python cli.py "What's the weather in Tokyo?"    # terminal version
```

**Tests**
```bash
python -m pytest -v tests
```

You can also open either server in the official MCP Inspector: `npx @modelcontextprotocol/inspector python expt1_memory_server/server.py`

## Using a different LLM

The client works with any OpenAI-compatible endpoint. Set these environment variables:

```bash
export LLM_BASE_URL=https://api.groq.com/openai/v1   LLM_API_KEY=...  LLM_MODEL=llama-3.3-70b-versatile
# or OpenAI:  LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
```

## Screenshots

| | |
|---|---|
| ![](screenshots/expt1_01_save_and_search.png) Expt 1: save and search | ![](screenshots/expt1_02_crud_and_notes_json.png) Expt 1: list, update, delete, and `notes.json` |
| ![](screenshots/expt2_01_tokyo_weather.png) Expt 2: "What's the weather in Tokyo?" | ![](screenshots/expt2_02_london_forecast.png) Expt 2: forecast tool |
| ![](screenshots/expt2_03_cli.png) Expt 2: terminal client | ![](screenshots/tests_passing.png) Test suite |

The full lab report is in [`docs/`](docs/).
