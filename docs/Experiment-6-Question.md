# Lab_06: LLMs for Code Generation & Software Development

*Abdul Muqtadeer Ahmed · 100 points · Due Sep 24*

Use AI coding assistants for code generation, testing, debugging, refactoring, documentation, and tool comparison.

## Expt 1: The "Personal Assistant" Memory Server
**Objective:** Implement a basic MCP server with tools that perform CRUD operations.

- **MCP Server:** Exposes tools for `save_note` (takes content and tags) and `search_notes` (takes a query). The server reads from and writes to a local `notes.json` file.
- **MCP Client:** A Python script that connects to this server, lets the user type a query (e.g., "What did I say about the project deadline?"), and uses an LLM to decide whether to call `save_note` or `search_notes`.

## Expt 2: The "Data Dashboard" Connector
**Objective:** Connect an AI to a real-time data source (like a weather service or a public database) to fetch real-time data and summarize it. This is a classic "context enhancement" use case.

- **MCP Server:** Exposes a tool like `get_current_weather(location)`. It fetches data from a free API (e.g., wttr.in) and returns the result as structured text.
- **MCP Client:** A simple web app. The user types "What's the weather in Tokyo?"; the app shows the LLM's "thought process" (deciding to call the tool) and then displays the final answer. This makes the client-server interaction visible and tangible.

**Submission:** Upload GitHub URL & screenshots.
