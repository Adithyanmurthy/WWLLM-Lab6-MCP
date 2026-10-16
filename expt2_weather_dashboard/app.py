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
