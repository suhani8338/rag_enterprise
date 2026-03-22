"""
src/serving/streamlit_app.py
─────────────────────────────
Streamlit chat UI for the Enterprise RAG Multi-Agent System.

Features:
  - Chat interface with streaming responses (one token per SSE event)
  - Agent trace panel: shows intent, route, SQL query, chunk count per turn
  - Persona selector sidebar
  - Session history with clear button
  - Index status panel (sources, chunk counts)
  - Connects to the FastAPI server at localhost:8000

Run:
  # Start the API first:
  uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

  # Then in a second terminal:
  streamlit run src/serving/streamlit_app.py
"""

import json
import time
from typing import Optional

import requests
import sseclient
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Enterprise RAG System",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_BASE = "http://localhost:8000"

# ── Session state defaults ────────────────────────────────────────────────────

if "messages"   not in st.session_state: st.session_state.messages   = []
if "session_id" not in st.session_state: st.session_state.session_id = "streamlit-default"
if "trace_log"  not in st.session_state: st.session_state.trace_log  = []
if "persona"    not in st.session_state: st.session_state.persona    = "analyst"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Persona
    persona = st.selectbox(
        "Persona",
        ["analyst", "executive", "engineer", "hr"],
        index=["analyst", "executive", "engineer", "hr"].index(
            st.session_state.persona
        ),
        help="Controls the tone and format of answers",
    )
    st.session_state.persona = persona

    # API health
    st.divider()
    st.subheader("API Status")
    if st.button("🔄 Check Health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            data = r.json()
            st.success(f"✅ Online · {data['chroma_chunks']} chunks indexed")
        except Exception as e:
            st.error(f"❌ API offline: {e}")

    # Index status
    if st.button("📊 Index Stats"):
        try:
            r = requests.get(f"{API_BASE}/status", timeout=5)
            data = r.json()
            st.metric("ChromaDB Chunks", data["chroma_chunks"])
            if data["sources"]:
                st.write("**Ingested sources:**")
                for src in data["sources"]:
                    st.write(f"  • {src.get('file_name','')} ({src.get('file_type','')})")
        except Exception as e:
            st.error(f"Failed to fetch status: {e}")

    # Clear session
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages  = []
        st.session_state.trace_log = []
        try:
            requests.post(
                f"{API_BASE}/history/reset",
                headers={"X-Session-Id": st.session_state.session_id},
                timeout=3,
            )
        except Exception:
            pass
        st.rerun()

    # Example questions
    st.divider()
    st.subheader("💡 Try these")
    examples = [
        "What is our parental leave policy?",
        "How many cloud products do we have?",
        "What does AcmeMesh do and what does it cost?",
        "Summarise Q4 revenue highlights",
        "Which products launched after 2022?",
        "Compare cloud vs software products",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state._pending_question = ex
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────

col_chat, col_trace = st.columns([2, 1])

with col_chat:
    st.title("🔍 Enterprise RAG System")
    st.caption("Multi-agent · Hybrid search · Local LLM")

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")
            if msg.get("meta"):
                m = msg["meta"]
                st.caption(
                    f"intent=`{m.get('intent','?')}` · "
                    f"route=`{m.get('agent_route',[])}` · "
                    f"{m.get('latency_ms',0):.0f}ms"
                )

    # Chat input
    question = st.chat_input("Ask anything about your documents or data...")

    # Handle example button clicks
    if hasattr(st.session_state, "_pending_question"):
        question = st.session_state._pending_question
        del st.session_state._pending_question

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder   = st.empty()
            sources_ph    = st.empty()
            meta_ph       = st.empty()

            full_answer   = ""
            final_sources = []
            meta          = {}

            try:
                # Call streaming endpoint
                response = requests.post(
                    f"{API_BASE}/ask/stream",
                    json    = {"question": question, "persona": persona},
                    headers = {"X-Session-Id": st.session_state.session_id},
                    stream  = True,
                    timeout = 120,
                )
                client = sseclient.SSEClient(response)

                for event in client.events():
                    if not event.data:
                        continue
                    try:
                        data = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue

                    node = data.get("node", "")

                    if node == "supervisor":
                        meta["intent"]      = data.get("intent", "")
                        meta["agent_route"] = data.get("agent_route", [])
                        placeholder.markdown(
                            f"*Routing → {meta['agent_route']}…*"
                        )
                        # Log to trace
                        st.session_state.trace_log.append({
                            "turn":   len(st.session_state.messages),
                            "node":   "supervisor",
                            "intent": meta["intent"],
                            "route":  str(meta["agent_route"]),
                        })

                    elif node == "retriever":
                        placeholder.markdown(
                            f"*Retrieving {data.get('rag_chunks', '?')} chunks…*"
                        )
                        st.session_state.trace_log.append({
                            "turn":   len(st.session_state.messages),
                            "node":   "retriever",
                            "chunks": data.get("rag_chunks", 0),
                        })

                    elif node == "sql":
                        placeholder.markdown("*Running SQL query…*")
                        st.session_state.trace_log.append({
                            "turn":      len(st.session_state.messages),
                            "node":      "sql",
                            "sql_query": data.get("sql_query", ""),
                        })

                    elif node == "synthesizer":
                        full_answer   = data.get("final_answer", "")
                        final_sources = data.get("final_sources", [])
                        placeholder.markdown(full_answer)

                    elif node == "done":
                        t = data.get("latency_ms", 0)
                        meta["latency_ms"] = t

                    elif node == "error":
                        full_answer = f"⚠️ Error: {data.get('detail', 'Unknown error')}"
                        placeholder.markdown(full_answer)

                # Show sources and meta after streaming
                if final_sources:
                    with sources_ph.expander("📎 Sources", expanded=False):
                        for src in final_sources:
                            st.caption(f"• {src}")
                if meta:
                    meta_ph.caption(
                        f"intent=`{meta.get('intent','?')}` · "
                        f"route=`{meta.get('agent_route',[])}` · "
                        f"{meta.get('latency_ms',0):.0f}ms"
                    )

            except requests.exceptions.ConnectionError:
                full_answer = (
                    "⚠️ Cannot connect to API. "
                    "Start it with: `uvicorn src.serving.api:app --port 8000`"
                )
                placeholder.markdown(full_answer)
            except Exception as e:
                full_answer = f"⚠️ Unexpected error: {e}"
                placeholder.markdown(full_answer)

        # Save assistant message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": final_sources,
            "meta":    meta,
        })

# ── Trace panel ───────────────────────────────────────────────────────────────

with col_trace:
    st.subheader("🔬 Agent Trace")
    if not st.session_state.trace_log:
        st.caption("Agent activity will appear here as you chat.")
    else:
        for entry in reversed(st.session_state.trace_log[-20:]):
            node = entry.get("node", "")
            if node == "supervisor":
                st.markdown(
                    f"**🧠 Supervisor**  \n"
                    f"intent: `{entry.get('intent')}`  \n"
                    f"route: `{entry.get('route')}`"
                )
            elif node == "retriever":
                st.markdown(
                    f"**📄 Retriever**  \n"
                    f"chunks: `{entry.get('chunks', 0)}`"
                )
            elif node == "sql":
                sql = entry.get("sql_query", "")
                st.markdown(f"**🗄️ SQL Agent**")
                if sql:
                    st.code(sql[:200], language="sql")
            st.divider()