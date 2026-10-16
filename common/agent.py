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
