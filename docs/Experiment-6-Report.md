---
title: "Lab 06 — LLMs for Code Generation & Software Development"
subtitle: "Building MCP Servers and LLM-driven MCP Clients"
---

| | |
|---|---|
| **Aim** | To implement Model Context Protocol (MCP) servers that expose tools, and LLM-driven MCP clients that decide which tool to call: (1) a CRUD "Personal Assistant" memory server, and (2) a real-time "Data Dashboard" weather connector. |
| **Name** | Adithya N Murthy |
| **USN** | 1RVU23CSE024 |
| **Date** | 16-10-2026 |
| **GitHub repository** | [https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP](https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP) |
| **Clone** | `git clone https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP.git` |

---

# 1. Objectives

1. Understand the **Model Context Protocol**: hosts, clients, servers, tools, and the JSON-RPC messages `tools/list` and `tools/call`.
2. Build an **MCP server** in Python that exposes CRUD tools over a local JSON file (Expt 1).
3. Build an **MCP client** in which an **LLM decides** whether to call `save_note` or `search_notes` (Expt 1).
4. Connect an LLM to a **live external data source** (wttr.in) through an MCP tool, for *context enhancement* (Expt 2).
5. Build a **web app** that shows the LLM's thought process: tool discovery, its decision to call a tool, the raw tool result, and the final answer (Expt 2).
6. Use an AI coding assistant to generate, test, debug, refactor and document the code (the lab theme).

# 2. Theory

## 2.1 What MCP is

The **Model Context Protocol** is an open standard, introduced by Anthropic in 2024, for connecting LLM applications to external tools and data. Before MCP, each application wrote its own integration for each tool (an M × N problem). With MCP, a tool is written once as a **server**, and any MCP-capable **client** can use it (M + N).

| Role | What it is | In this lab |
|---|---|---|
| **Host** | The application the user talks to | Terminal chat (E1), Streamlit app (E2) |
| **Client** | Keeps a 1-to-1 session with one server | `mcp.ClientSession` inside `client.py` / `app.py` |
| **Server** | Exposes *tools*, *resources* and *prompts* | `server.py` in each experiment (built with `FastMCP`) |
| **Transport** | How messages travel | **stdio**: the client starts the server as a subprocess |

The messages are JSON-RPC 2.0:

* `initialize`: the client and server exchange protocol version and capabilities.
* `tools/list`: the server returns each tool's **name, description and JSON input schema**. `FastMCP` builds the schema automatically from the Python type hints and docstring.
* `tools/call`: the client sends `{name, arguments}`, and the server runs the function and returns `content` (text) plus an `isError` flag.

## 2.2 Where the LLM fits: tool calling

MCP itself does not decide anything. The **LLM** does. The client converts the MCP tool list into the LLM's *function-calling* format and sends it with the user's message. The model then either:

* answers in text, or
* returns a **`tool_call`** such as `search_notes({"query": "project deadline"})`.

The client runs that call on the MCP server, adds the result to the conversation as a `tool` message, and asks the LLM again. This **observe → decide → act → observe** cycle is the basic *agent loop* (ReAct pattern). The model's "thought process" is the sequence of tool decisions it makes.

## 2.3 Context enhancement

An LLM's knowledge is frozen at its training cutoff, so it cannot know today's weather. Experiment 2 fetches **real-time** data through a tool and places it in the model's context. The model then *summarises facts it was given* rather than *recalling facts it might hallucinate*. This is the same idea as retrieval-augmented generation (RAG), applied to live API data.

# 3. System Design

## 3.1 Architecture

```text
            User  (terminal  /  browser)
               │ question            ▲ final answer
               ▼                     │
 ┌──────────────────────────────────────────┐
 │ MCP CLIENT / HOST   client.py · app.py   │
 │ agent loop          common/agent.py      │
 └──────────────────────────────────────────┘
       │      ▲                   │      ▲
  tools│      │results   messages │      │ "call tool X(args)"
  list/│      │          + tool   │      │  or final text
  call ▼      │          schemas  ▼      │
 ┌───────────────────┐       ┌───────────────────┐
 │ MCP SERVER        │       │ LLM               │
 │ server.py         │       │ Ollama            │
 │ (FastMCP, stdio)  │       │ qwen2.5:3b        │
 │ E1 → notes.json   │       └───────────────────┘
 │ E2 → wttr.in API  │
 └───────────────────┘
```

## 3.2 Shared agent loop: `common/agent.py`

Both experiments use the same loop, so the MCP plumbing is written once:

1. `session.list_tools()` returns the tools, which `mcp_tools_to_openai()` converts to the OpenAI `tools` schema.
2. `chat(messages, tools)` sends one request to any OpenAI-compatible endpoint. The default is local **Ollama**, so no API key is needed.
3. For each `tool_call` the client runs `session.call_tool(name, args)` and appends the result.
4. The loop ends when the model replies in plain text (at most 5 steps).
5. Every step is recorded in a **trace** (`user → decide → result → answer`), which the terminal client prints and the web app renders.

Two robustness features were added after testing:

* **Text tool-call recovery.** Small local models sometimes write the call as JSON in the message text instead of a real `tool_call`. `_text_tool_call()` detects this and runs it anyway.
* **Grounding guard.** During testing, `qwen2.5:3b` once answered *"I've added a note with ID #3"* **without calling `save_note`**, so nothing was saved (§6.1). Now, if the model's first reply contains no tool call, the client tells it *"You did not call any tool, so nothing was saved or looked up. Call the correct tool now"* and retries once. The trace shows a `guard` step when this happens.

## 3.3 Experiment 1: Memory server tools

| Tool | CRUD | Arguments | Behaviour |
|---|---|---|---|
| `save_note` | Create | `content: str`, `tags: list[str]` | Appends `{id, content, tags, created}` to `notes.json` |
| `search_notes` | Read | `query: str` | Keyword search over content and tags. Stop-words are removed, tag matches are weighted ×3, results are ranked |
| `list_notes` | Read | — | Returns all notes |
| `update_note` | Update | `note_id: int`, `content`, `tags` | Edits a note in place |
| `delete_note` | Delete | `note_id: int` | Removes a note |

`notes.json` is the only storage. Its path can be overridden with the `NOTES_FILE` environment variable, which the tests use so they never touch the real file. Tags sent as `"a, b"` instead of a list are split and normalised.

## 3.4 Experiment 2: Weather connector tools

| Tool | Data source | Returns (structured text) |
|---|---|---|
| `get_current_weather(location)` | `https://wttr.in/<location>?format=j1` | location, observation time, condition, temperature, feels-like, humidity, wind, precipitation, cloud cover, UV, visibility |
| `get_forecast(location)` | same | Three lines labelled **today / tomorrow / day after tomorrow**: condition, min/max °C, maximum chance of rain |

The output is one `key: value` per line, so the LLM can read it reliably and the web app can turn it into metric cards. Network errors return an `ERROR:` string instead of crashing the server. wttr.in reports the *nearest weather station*, which can have a different name (Tokyo → Shikinejima), so the tool shows both: `Tokyo, Japan (nearest station: Shikinejima)`.

The **Streamlit app** shows, for every question:

1. **Tools discovered** on the MCP server
2. **LLM decided to call a tool**, with the tool name and JSON arguments
3. **MCP server returned**, the raw live data
4. **Final answer**, plus metric cards (temperature, condition, humidity, wind)

# 4. Procedure

1. Created the `L6/` folder with `common/`, `expt1_memory_server/`, `expt2_weather_dashboard/`, `tests/`, `screenshots/` and `docs/`.
2. Installed `mcp` (Python SDK 1.27), `httpx`, `streamlit` and `pytest`. Installed Ollama and pulled `qwen2.5:3b` (1.9 GB), a small model with native tool-calling support.
3. Wrote the memory server with `FastMCP` and decorated each function with `@mcp.tool()`. The type hints and docstrings become the tool schemas.
4. Wrote the shared agent loop and the terminal client. The client starts the server with `StdioServerParameters` and opens a `ClientSession`.
5. Wrote the weather server, the Streamlit app and a terminal client.
6. Wrote 10 automated tests: direct tool tests, a **real MCP round-trip over stdio**, text-tool-call recovery, and the grounding guard with a fake LLM.
7. Ran both experiments with real queries, found and fixed three LLM-behaviour bugs (§6), and captured screenshots.
8. Committed the `L6/` folder as a git repository and pushed it to GitHub: <https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP>.

**Commands**

```bash
pip install -r requirements.txt && ollama pull qwen2.5:3b
python expt1_memory_server/client.py                     # Expt 1
cd expt2_weather_dashboard && streamlit run app.py       # Expt 2 (web)
python expt2_weather_dashboard/cli.py "What's the weather in Tokyo?"
python -m pytest -v tests
```

# 5. Output

## 5.1 Experiment 1: Personal Assistant Memory Server

**Run 1: create and search** (actual terminal output)

```text
$ python expt1_memory_server/client.py
Connected to MCP server 'personal-memory' | tools: ['save_note', 'search_notes', 'list_notes', 'update_note', 'delete_note'] | LLM: qwen2.5:3b

🧑 You: Remember that the project deadline is October 30th and the demo is on the 28th
  🤔 LLM decided → call save_note({'content': 'The project deadline is October 30th and the demo is on the 28th.', 'tags': ['project', 'deadline']})
  🛠  server returned: Saved note [#1] The project deadline is October 30th and the demo is on the 28th.  (tags: project, deadline; saved 2026-10-16 14:08)
  🤔 LLM decided → call list_notes({})
  🛠  server returned: [#1] The project deadline is October 30th and the demo is on the 28th.  (tags: project, deadline; saved 2026-10-16 14:08)
🤖 I have saved your note about the project deadline and demo dates. You can review it using the list_notes function if needed.

🧑 You: Save a note: my MCP lab viva is on Friday at 10am, tag it lab
  🤔 LLM decided → call save_note({'content': 'my MCP lab viva is on Friday at 10am', 'tags': ['lab']})
  🛠  server returned: Saved note [#2] my MCP lab viva is on Friday at 10am  (tags: lab; saved 2026-10-16 14:09)
🤖 I have saved your note about the lab viva. You can review it using the list_notes function if needed.

🧑 You: Note that I need to buy milk and eggs
  🤔 LLM decided → call save_note({'tags': ['groceries'], 'content': 'I need to buy milk and eggs.'})
  🛠  server returned: Saved note [#3] I need to buy milk and eggs.  (tags: groceries; saved 2026-10-16 14:09)
🤖 Your note has been saved with the ID #3. You can list your notes using the `list_notes` function if you need to review them.

🧑 You: What did I say about the project deadline?
  ⚠️  LLM answered without a tool → guard asks it to call one
  🤔 LLM decided → call search_notes({'query': 'project deadline'})
  🛠  server returned: 1 matching note(s):
[#1] The project deadline is October 30th and the demo is on the 28th.  (tags: project, deadline; saved 2026-10-16 14:08)
🤖 Your note about the project deadline was found. The details are as follows: The project deadline is October 30th and the demo is on the 28th.
```

![Figure 1: Expt 1, the LLM chooses save_note for statements and search_notes for questions](../screenshots/expt1_01_save_and_search.png)

**Run 2: list, update, delete, then search** (same `notes.json`, new process, so the memory persists)

```text
$ python expt1_memory_server/client.py
Connected to MCP server 'personal-memory' | tools: ['save_note', 'search_notes', 'list_notes', 'update_note', 'delete_note'] | LLM: qwen2.5:3b

🧑 You: Show all my notes
  🤔 LLM decided → call list_notes({})
  🛠  server returned: [#1] The project deadline is October 30th and the demo is on the 28th.  (tags: project, deadline; saved 2026-10-16 14:08)
[#2] my MCP lab viva is on Friday at 10am  (tags: lab; saved 2026-10-16 14:09)
[#3] I need to buy milk and eggs.  (tags: groceries; saved 2026-10-16 14:09)
🤖 Here are your notes:
- Project deadline is on October 30th, demo on the 28th.
- Your lab viva is scheduled for Friday at 10 AM.
- You need to buy milk and eggs.

🧑 You: Update note 3 to say buy milk, eggs and bread
  🤔 LLM decided → call update_note({'note_id': 3, 'content': 'buy milk, eggs and bread', 'tags': ['shopping']})
  🛠  server returned: Updated note [#3] buy milk, eggs and bread  (tags: shopping; saved 2026-10-16 14:09)
🤖 Your note has been updated. It now says:
- Buy milk, eggs and bread.

🧑 You: Delete note 3
  ⚠️  LLM answered without a tool → guard asks it to call one
  🤔 LLM decided → call delete_note({'note_id': 3})
  🛠  server returned: Deleted note #3.
🤖 Here are your remaining notes:
- Project deadline is on October 30th, demo on the 28th.
- Your lab viva is scheduled for Friday at 10 AM.

🧑 You: Do I have anything saved about groceries?
  🤔 LLM decided → call search_notes({'query': 'groceries'})
  🛠  server returned: No notes found for 'groceries'.
🤖 There are no saved notes related to groceries.

🧑 You: When is my viva?
  🤔 LLM decided → call search_notes({'query': 'viva'})
  🛠  server returned: 1 matching note(s):
[#2] my MCP lab viva is on Friday at 10am  (tags: lab; saved 2026-10-16 14:09)
🤖 My lab viva is scheduled for Friday at 10 AM.


$ cat expt1_memory_server/notes.json
[
  {
    "id": 1,
    "content": "The project deadline is October 30th and the demo is on the 28th.",
    "tags": [
      "project",
      "deadline"
    ],
    "created": "2026-10-16 14:08"
  },
  {
    "id": 2,
    "content": "my MCP lab viva is on Friday at 10am",
    "tags": [
      "lab"
    ],
    "created": "2026-10-16 14:09"
  }
]
```

![Figure 2: Expt 1, list, update, delete and search, then the resulting notes.json](../screenshots/expt1_02_crud_and_notes_json.png)

## 5.2 Experiment 2: Data Dashboard Connector

![Figure 3: Expt 2, "What's the weather in Tokyo?" The thought process shows tool discovery, the decision to call get_current_weather, the live wttr.in data, the final answer and metric cards](../screenshots/expt2_01_tokyo_weather.png)

![Figure 4: Expt 2, "Should I carry an umbrella in London tomorrow?" The LLM chooses get_forecast and reads the "tomorrow" line](../screenshots/expt2_02_london_forecast.png)

**Terminal client** (`cli.py`)

```text
$ python expt2_weather_dashboard/cli.py "What's the weather in Bengaluru?"
🧑 What's the weather in Bengaluru?
🤔 LLM decided → call get_current_weather({'location': 'Bengaluru'})
🛠  MCP server returned:
    location: Bengaluru, India (nearest station: Bangalore)
    observed_at_utc: 08:10 AM
    condition: Light rain shower
    temperature_c: 26
    feels_like_c: 28
    humidity_pct: 67
    wind_kmph: 10 NNE
    precipitation_mm: 1.1
    cloud_cover_pct: 100
    uv_index: 8
    visibility_km: 10
🤖 The current weather in Bengaluru is light rain shower with a temperature of 26°C and feels like 28°C. The humidity is at 67%, wind is blowing from the NNE direction at 10 km/h, visibility is good at 10 kilometers, and there's no clear sky as it's completely covered by clouds (100%). UV index is very high at level 8. It feels quite humid so make sure to carry an umbrella for protection against rain.
```

![Figure 5: Expt 2, terminal client using the same MCP server](../screenshots/expt2_03_cli.png)

## 5.3 Tests

![Figure 6: All 10 automated tests passing](../screenshots/tests_passing.png)

# 6. Observations

## 6.1 LLM decision quality (qwen2.5:3b, local)

Across the 12 recorded queries (9 for Expt 1, 3 for Expt 2):

| Measure | Result |
|---|---|
| Correct tool eventually called | **12 / 12** |
| Correct tool on the **first** try | 10 / 12. Twice ("What did I say about…", "Delete note 3") the model first replied without a tool, and the **grounding guard** corrected it |
| `save_note` chosen for statements | 4 / 4 |
| `search_notes` chosen for "what did I say / when is my…" questions | 3 / 3 |
| Search query extracted correctly (topic words, not the whole sentence) | 3 / 3 (`project deadline`, `groceries`, `viva`) |
| Instruction phrasing removed from content ("tag it lab" → tag, not text) | Yes, after the prompt fix |
| Unnecessary extra calls | 1 (a `list_notes` after a save; harmless) |
| Answers grounded in tool output | Yes. All note contents and weather numbers match the server output exactly |
| Interpretation errors | 2 small ones: the London answer called a **14 %** chance of rain "high", and `update_note` replaced the tag `groceries` with `shopping` without being asked |

**Persistence:** Run 2 is a *separate process*, yet `list_notes` returned the three notes saved in Run 1. The memory lives in the server's `notes.json`, not in the LLM.

**Semantic gap:** after note #3 (milk, eggs, bread) was deleted, "anything about groceries?" correctly returned nothing. Before deletion, a keyword search for "groceries" only matched because of the *tag*. A note saved without that tag would be missed. Embedding-based search would fix this (Viva Q13).

## 6.2 Bugs found while testing and how they were fixed

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | *"I've added a note with ID #3"*, but `notes.json` had only 2 notes | The 3B model **claimed** to call a tool without calling it | **Grounding guard** in the agent loop, plus a unit test (`test_guard_forces_a_tool_call`) |
| 2 | "Umbrella in London **tomorrow**?" was answered using **today's** forecast | The model passed `days=1`, then described that day as "tomorrow" | Removed the `days` argument. The tool always returns 3 labelled days and the prompt says to read the matching line |
| 3 | "Weather in Tokyo" showed *Shikinejima, Japan* | wttr.in returns the nearest station's name | Location is shown as `Tokyo, Japan (nearest station: Shikinejima)` |
| 4 | Server INFO logs mixed into the chat output | FastMCP's default log level | `FastMCP(..., log_level="WARNING")` |
| 5 | Tags sometimes arrived as `"work, urgent"` | The model ignored the `list[str]` schema | `clean_tags()` accepts both forms |

## 6.3 Response times (M-series Mac, CPU/GPU shared with other apps)

| Step | Time |
|---|---|
| First query (model loads into memory) | ≈ 40 s |
| Later queries, end to end | ≈ 15–25 s |
| wttr.in API call | ≈ 1–2 s |
| MCP `tools/list` / `tools/call` overhead (stdio) | < 50 ms |

Almost all of the time is spent in local LLM inference. The MCP protocol overhead is negligible. A hosted model (Groq or OpenAI) selected through `LLM_BASE_URL` answers in about 1–2 s.

# 7. Result

Both experiments were implemented and run successfully:

* **Expt 1:** an MCP server with five CRUD tools over `notes.json`, and an MCP client in which the LLM chose `save_note` for statements and `search_notes` for questions such as *"What did I say about the project deadline?"*, answering from the stored notes.
* **Expt 2:** an MCP server that fetches **live** weather from wttr.in, and a Streamlit web app that makes the client-server interaction visible: tool discovery → LLM decision → live data → final answer.
* 10/10 automated tests pass, including a real MCP round-trip over stdio.
* Testing showed that **a small LLM can claim a tool action it never performed**. A client-side grounding guard fixed this, which shows that MCP clients need to check tool use, not just trust the model's text.

# 8. Viva Questions

**1. What is MCP and what problem does it solve?**
An open protocol that standardises how LLM applications connect to tools and data. It turns the M × N integration problem (every app × every tool) into M + N: a tool is written once as a server and works with any MCP client.

**2. Host vs. client vs. server?**
The host is the user-facing app. A client is a connection inside the host with exactly one server. A server exposes tools, resources and prompts.

**3. Tools vs. resources vs. prompts?**
*Tools* are functions the **model** chooses to call, and they can have side effects (`save_note`). *Resources* are read-only data the **application** attaches as context (for example a file). *Prompts* are reusable templates the **user** picks.

**4. Which transports does MCP support?**
**stdio** for local servers started as subprocesses (used here), and **Streamable HTTP** (earlier HTTP + SSE) for remote servers.

**5. Who decides which tool to call: MCP or the LLM?**
The LLM. MCP only describes tools (`tools/list`) and runs them (`tools/call`). The client passes the schemas to the LLM, and the model returns a `tool_call`.

**6. How does FastMCP know a tool's parameters?**
From the Python **type hints** and **docstring**: `content: str, tags: list[str]` becomes a JSON Schema, and the docstring becomes the description the LLM reads.

**7. Why are tool descriptions so important?**
They are the only thing the LLM sees when choosing a tool. "Use when the user asks what they said about something" is what makes the model pick `search_notes` for a question.

**8. What is "context enhancement" in Expt 2?**
Adding fresh external data (live weather) to the model's context at query time, so the answer is grounded in real data instead of training memory. It is RAG with an API as the source.

**9. What did the grounding guard fix?**
The small model said it saved a note without calling `save_note`. If the first reply has no tool call, the client asks the model to call a tool and retries once, so a claimed action becomes a real one.

**10. Why run the LLM locally with Ollama?**
No API key or cost, the data stays on the machine, and it works offline. The trade-off is slower, weaker reasoning than large hosted models. The code accepts any OpenAI-compatible endpoint through `LLM_BASE_URL`.

**11. How did you test without an LLM?**
Unit tests call the tool functions directly, a protocol test starts the real server over stdio and runs `save_note` then `search_notes` through a `ClientSession`, and the guard test uses a fake LLM that returns scripted replies.

**12. What are the security concerns with MCP tools?**
Tools act with the user's permissions. Risks include prompt injection through tool results (for example a malicious web page telling the model to delete notes), over-broad tools, and untrusted third-party servers. Mitigations include least privilege, confirmation for destructive actions, input validation, and treating tool output as data rather than instructions.

**13. How would you scale the memory server?**
Replace `notes.json` with SQLite or Postgres to get concurrency and transactions, use embeddings for semantic search (so "groceries" also finds "milk"), and add per-user namespaces.

**14. Why return structured `key: value` text instead of prose?**
The LLM reads exact numbers reliably, the UI can parse the same output into metric cards, and the format is easy to test.

**15. What was the role of the AI coding assistant in this lab?**
It generated the servers, client and app; wrote the tests; debugged the three LLM-behaviour issues from real transcripts; refactored shared code into `common/agent.py` and `config.py`; and wrote this documentation. That covers generation, testing, debugging, refactoring and documentation.

# 9. Conclusion

MCP separates *what tools exist* (servers) from *who decides to use them* (the LLM in the client). This lab built a stateful CRUD memory server and a stateless real-time data connector on the same protocol and the same agent loop. The main practical lesson came from testing: tool-calling with small models is unreliable at the edges, because the model may skip a tool, under-fetch data, or claim an action it never took. The client must therefore make tool use visible (the thought-process trace) and enforce it (the grounding guard).

# Appendix A: GitHub Repository

**Repository URL:** <https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP>

The complete source code, screenshots and this report are in the repository. To run it:

```bash
git clone https://github.com/Adithyanmurthy/WWLLM-Lab6-MCP.git
cd WWLLM-Lab6-MCP
pip install -r requirements.txt
ollama pull qwen2.5:3b
python expt1_memory_server/client.py                       # Experiment 1
cd expt2_weather_dashboard && streamlit run app.py         # Experiment 2 (web dashboard)
```

**Repository structure**

```text
WWLLM-Lab6-MCP/
├── README.md                     # GitHub front page: setup, run steps, screenshots
├── requirements.txt              # mcp, httpx, streamlit, pytest
├── common/
│   └── agent.py                  # shared MCP client + LLM agent loop (+ grounding guard)
├── expt1_memory_server/
│   ├── server.py                 # MCP server: save/search/list/update/delete notes
│   ├── client.py                 # MCP client: chat, LLM picks the tool
│   └── notes.json                # local note store (created by the server)
├── expt2_weather_dashboard/
│   ├── server.py                 # MCP server: get_current_weather, get_forecast (wttr.in)
│   ├── app.py                    # Streamlit dashboard showing the thought process
│   ├── cli.py                    # terminal version of the weather client
│   └── config.py                 # shared settings for app.py and cli.py
├── tests/
│   └── test_servers.py           # 10 tests (unit + real MCP-over-stdio round trips)
├── screenshots/                  # 6 screenshots used in this report
└── docs/                         # this report (.md / .html / .pdf) and the question
```

# Appendix B: Screenshot Index

All screenshots are embedded in §5 and are also stored in the repository under `screenshots/`.

| Figure | File | Shows |
|---|---|---|
| Figure 1 | `screenshots/expt1_01_save_and_search.png` | Expt 1: save_note vs search_notes chosen by the LLM |
| Figure 2 | `screenshots/expt1_02_crud_and_notes_json.png` | Expt 1: list / update / delete and the resulting notes.json |
| Figure 3 | `screenshots/expt2_01_tokyo_weather.png` | Expt 2: dashboard, "What's the weather in Tokyo?" |
| Figure 4 | `screenshots/expt2_02_london_forecast.png` | Expt 2: dashboard, umbrella in London tomorrow (get_forecast) |
| Figure 5 | `screenshots/expt2_03_cli.png` | Expt 2: terminal client |
| Figure 6 | `screenshots/tests_passing.png` | All 10 automated tests passing |

# Appendix C: Source Code

Full source of every program file in the repository.

## C.1 `requirements.txt`

```text
mcp>=1.27
httpx>=0.28
streamlit>=1.40
pytest>=8
```

## C.2 `common/agent.py`

```python
"""
Shared MCP-client + LLM "agent loop" used by both experiments.

Flow for one user message:
    1. Ask the MCP server which tools it has            (session.list_tools)
    2. Send the user message + tool schemas to the LLM   (chat completions with `tools`)
    3. If the LLM asks for a tool -> call it on the MCP server (session.call_tool)
       and feed the result back to the LLM; repeat.
    4. When the LLM answers in plain text -> that is the final answer.

Every step is recorded in a `trace` list so the client/UI can show the
LLM's "thought process" (which tool it picked, with what arguments, and what came back).

LLM backend: any OpenAI-compatible /chat/completions endpoint.
    default  -> local Ollama  (http://localhost:11434/v1, model qwen2.5:3b, no key needed)
    override -> set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL  (e.g. Groq / OpenAI / Gemini)
"""

import json
import os
import re

import httpx

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
MAX_STEPS = 5
GUARD = ("You did not call any tool, so nothing was saved or looked up. "
         "Call the correct tool now instead of answering from memory.")


def mcp_tools_to_openai(tools):
    """Convert MCP tool descriptions into the OpenAI `tools` format the LLM understands."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


def chat(messages, tools):
    """One call to the LLM. Returns the assistant message dict."""
    r = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "messages": messages, "tools": tools, "temperature": 0},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _text_tool_call(content, tool_names):
    """Small local models sometimes write the tool call as JSON text instead of a real
    tool_call. Recover it so the loop still works."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if name in tool_names and isinstance(args, dict):
        return {"id": "call_text_0", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
    return None


async def run_agent(session, system_prompt, user_text, history=None, require_tool=True):
    """Run the tool-calling loop against an open MCP ClientSession.

    Returns (final_answer, trace, new_history)."""
    tools = mcp_tools_to_openai((await session.list_tools()).tools)
    tool_names = {t["function"]["name"] for t in tools}
    messages = [{"role": "system", "content": system_prompt}] + (history or [])
    messages.append({"role": "user", "content": user_text})
    trace = [{"step": "user", "text": user_text}]

    for step in range(MAX_STEPS):
        msg = chat(messages, tools)
        calls = msg.get("tool_calls") or []
        if not calls:
            recovered = _text_tool_call(msg.get("content"), tool_names)
            calls = [recovered] if recovered else []

        if not calls and step == 0 and require_tool:
            # Grounding guard: small models sometimes *claim* "I've saved it" without
            # calling any tool. Push back once and make it actually use a tool.
            trace.append({"step": "guard", "text": msg.get("content") or ""})
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            messages.append({"role": "user", "content": GUARD})
            continue

        if not calls:  # plain answer -> done
            answer = (msg.get("content") or "").strip()
            trace.append({"step": "answer", "text": answer})
            return answer, trace, _compact(history, user_text, answer)

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) and args else (args or {})
            trace.append({"step": "decide", "tool": name, "args": args})

            result = await session.call_tool(name, args)  # <-- the MCP request
            text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
            trace.append({"step": "result", "tool": name, "text": text, "error": result.isError})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": text})

    answer = "Stopped: too many tool steps."
    trace.append({"step": "answer", "text": answer})
    return answer, trace, _compact(history, user_text, answer)


def _compact(history, user_text, answer):
    """Conversation memory for the next turn: only user/assistant text, no tool plumbing."""
    return (history or []) + [{"role": "user", "content": user_text},
                              {"role": "assistant", "content": answer}]
```

## C.3 `expt1_memory_server/server.py`

```python
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
```

## C.4 `expt1_memory_server/client.py`

```python
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
```

## C.5 `expt1_memory_server/notes.json`

```json
[
  {
    "id": 1,
    "content": "The project deadline is October 30th and the demo is on the 28th.",
    "tags": [
      "project",
      "deadline"
    ],
    "created": "2026-10-16 14:08"
  },
  {
    "id": 2,
    "content": "my MCP lab viva is on Friday at 10am",
    "tags": [
      "lab"
    ],
    "created": "2026-10-16 14:09"
  }
]
```

## C.6 `expt2_weather_dashboard/server.py`

```python
"""
Experiment 2 — "Data Dashboard" Connector (MCP server)

Connects an AI to a real-time data source: the free wttr.in weather API (no key needed).
Tools:
    get_current_weather(location)   -> live conditions as structured text
    get_forecast(location)           -> 3-day forecast (today / tomorrow / day after)

Run directly (stdio transport):   python server.py
Normally it is started by app.py (Streamlit) or cli.py.
"""

from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-connector", log_level="WARNING")


def fetch(location: str) -> dict:
    """Call wttr.in's JSON API (format=j1)."""
    r = httpx.get(f"https://wttr.in/{quote(location.strip())}", params={"format": "j1"},
                  headers={"User-Agent": "mcp-lab6"}, timeout=20)
    r.raise_for_status()
    return r.json()


def place_name(data: dict, requested: str) -> str:
    """wttr.in reports the nearest weather station, which can differ from the city name
    (e.g. Tokyo -> Shikinejima), so show both."""
    try:
        a = data["nearest_area"][0]
        station, country = a["areaName"][0]["value"], a["country"][0]["value"]
    except (KeyError, IndexError):
        return requested
    if station.lower() == requested.strip().lower():
        return f"{station}, {country}"
    return f"{requested.strip().title()}, {country} (nearest station: {station})"


@mcp.tool()
def get_current_weather(location: str) -> str:
    """Get the current, real-time weather for a city or place, e.g. "Tokyo" or "Bengaluru"."""
    try:
        data = fetch(location)
        c = data["current_condition"][0]
    except Exception as e:  # network error, unknown place, API change
        return f"ERROR: could not fetch weather for '{location}': {e}"
    return "\n".join([
        f"location: {place_name(data, location)}",
        f"observed_at_utc: {c['observation_time']}",
        f"condition: {c['weatherDesc'][0]['value'].strip()}",
        f"temperature_c: {c['temp_C']}",
        f"feels_like_c: {c['FeelsLikeC']}",
        f"humidity_pct: {c['humidity']}",
        f"wind_kmph: {c['windspeedKmph']} {c['winddir16Point']}",
        f"precipitation_mm: {c['precipMM']}",
        f"cloud_cover_pct: {c['cloudcover']}",
        f"uv_index: {c['uvIndex']}",
        f"visibility_km: {c['visibility']}",
    ])


@mcp.tool()
def get_forecast(location: str) -> str:
    """Get the 3-day weather forecast for a city or place, labelled today / tomorrow /
    day after tomorrow. Use for questions about tomorrow or the coming days."""
    try:
        data = fetch(location)
    except Exception as e:
        return f"ERROR: could not fetch forecast for '{location}': {e}"
    lines = [f"location: {place_name(data, location)}"]
    labels = ["today", "tomorrow", "day after tomorrow"]
    for label, d in zip(labels, data["weather"][:3]):
        midday = d["hourly"][len(d["hourly"]) // 2]
        rain = max(int(h["chanceofrain"]) for h in d["hourly"])
        lines.append(f"{label} ({d['date']}): {midday['weatherDesc'][0]['value'].strip()}, "
                     f"min {d['mintempC']}°C / max {d['maxtempC']}°C, "
                     f"max chance of rain {rain}%")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
```

## C.7 `expt2_weather_dashboard/config.py`

```python
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
```

## C.8 `expt2_weather_dashboard/app.py`

```python
"""
Experiment 2 — "Data Dashboard" web app (MCP client + Streamlit UI)

    streamlit run app.py

Type "What's the weather in Tokyo?". The page shows each step of the LLM's
"thought process": the tools it discovered on the MCP server, its decision to call a
tool (with arguments), the raw data the server fetched from wttr.in, and the final answer.
"""

import asyncio
import sys
import time
from pathlib import Path

import streamlit as st
from mcp import ClientSession
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.agent import LLM_MODEL, run_agent  # noqa: E402

from config import SERVER, SYSTEM  # noqa: E402


async def ask(question):
    """Open an MCP session, run the agent loop, return (tools, answer, trace, seconds)."""
    t0 = time.time()
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            answer, trace, _ = await run_agent(session, SYSTEM, question)
    return tools, answer, trace, time.time() - t0


st.set_page_config(page_title="Weather MCP Dashboard", page_icon="🌦️", layout="wide")
st.title("🌦️ Data Dashboard Connector — MCP + LLM")
st.caption(f"MCP server: `weather-connector` (wttr.in live data) · LLM: `{LLM_MODEL}` · "
           "transport: stdio")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Client** starts the MCP server and lists its tools\n"
        "2. **LLM** reads your question + tool schemas and *decides* which tool to call\n"
        "3. **MCP server** fetches real-time data from `wttr.in`\n"
        "4. **LLM** summarises the structured data into an answer")
    st.divider()
    st.markdown("**Try:**")
    for ex in ["What's the weather in Tokyo?", "Is it raining in London right now?",
               "Should I carry an umbrella in Bengaluru tomorrow?"]:
        if st.button(ex, use_container_width=True):
            st.session_state.q = ex

q = st.chat_input("Ask about the weather anywhere…") or st.session_state.pop("q", None)

if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Thinking…", expanded=True) as status:
            tools, answer, trace, secs = asyncio.run(ask(q))
            st.markdown("**① Tools discovered on MCP server:** " +
                        ", ".join(f"`{t.name}`" for t in tools))
            n = 2
            for t in trace:
                if t["step"] == "guard":
                    st.warning("LLM tried to answer without a tool → grounding guard asked "
                               "it to call one")
                elif t["step"] == "decide":
                    st.markdown(f"**{'②③④⑤⑥'[n-2]} LLM decided to call a tool →** "
                                f"`{t['tool']}`")
                    st.json(t["args"])
                    n += 1
                elif t["step"] == "result":
                    st.markdown(f"**{'②③④⑤⑥'[min(n-2, 4)]} MCP server returned "
                                f"(live from wttr.in):**")
                    st.code(t["text"], language="yaml")
                    n += 1
            if not any(t["step"] == "decide" for t in trace):
                st.markdown("**LLM answered directly — no tool needed.**")
            status.update(label=f"Thought process — {secs:.1f}s", state="complete",
                          expanded=True)
        st.markdown("### Final answer")
        st.success(answer)

        # small metric strip from the structured tool output
        data = {}
        for t in trace:
            if t["step"] == "result" and t["tool"] == "get_current_weather":
                data = dict(l.split(": ", 1) for l in t["text"].splitlines() if ": " in l)
        if "temperature_c" in data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{data['temperature_c']} °C",
                      f"feels {data['feels_like_c']} °C", delta_color="off")
            c2.metric("Condition", data["condition"])
            c3.metric("Humidity", f"{data['humidity_pct']} %")
            c4.metric("Wind", f"{data['wind_kmph']}")
```

## C.9 `expt2_weather_dashboard/cli.py`

```python
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
```

## C.10 `tests/test_servers.py`

```python
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
```
