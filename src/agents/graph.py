"""
src/agents/graph.py
────────────────────
LangGraph state machine — wires all four agents into a directed graph.

Graph structure:

  [START]
     │
     ▼
  supervisor          ← classifies intent, sets agent_route
     │
     ├─── intent="rag"      ──► retriever ──► synthesizer ──► [END]
     │
     ├─── intent="sql"      ──► sql       ──► synthesizer ──► [END]
     │
     ├─── intent="both"     ──► retriever ─┐
     │                                     ├──► synthesizer ──► [END]
     │                      ──► sql      ──┘
     │
     └─── intent="chitchat" ──────────────► synthesizer ──► [END]

Conditional routing:
  After the supervisor node, a routing function reads agent_route from
  state and returns the next node name(s). LangGraph supports returning
  a list of next nodes for parallel execution ("both" case).

Usage:
    graph = build_graph()
    result = graph.invoke({
        "question": "How many cloud products cost under $500?",
        "persona":  "analyst",
        "chat_history": [],
    })
    print(result["final_answer"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from langgraph.graph import END, START, StateGraph

from src.agents.retriever_agent import RetrieverAgent
from src.agents.sql_agent import SQLAgent
from src.agents.state import AgentState
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.embedding.embedder import LocalEmbedder
from src.rag.llm_factory import get_llm
from src.rag.rag_chain import RAGChain
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Routing function ───────────────────────────────────────────────────────────

def _route_after_supervisor(state: AgentState) -> List[str]:
    """
    Called by LangGraph after the supervisor node.
    Returns the list of next node(s) to execute.

    LangGraph treats a list return as parallel fan-out — both retriever
    and sql will run concurrently when intent is "both".
    """
    route = state.get("agent_route", [])

    if not route:                        # chitchat — skip to synthesizer
        return ["synthesizer"]
    if route == ["retriever"]:
        return ["retriever"]
    if route == ["sql"]:
        return ["sql"]
    if "retriever" in route and "sql" in route:
        return ["retriever", "sql"]      # parallel execution
    return ["synthesizer"]               # fallback


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(
    llm=None,
    rag_chain: RAGChain = None,
) -> StateGraph:
    """
    Instantiate all agents and wire them into a compiled LangGraph graph.

    Args:
        llm:       Pre-built LLM instance (optional — built from settings if None).
        rag_chain: Pre-built RAGChain (optional — built from settings if None).

    Returns:
        Compiled LangGraph StateGraph ready to .invoke() or .stream().
    """
    # Build shared components if not supplied
    if llm is None:
        llm = get_llm()

    if rag_chain is None:
        embedder  = LocalEmbedder()
        from src.vectorstore.chroma_store import ChromaVectorStore
        store     = ChromaVectorStore(embedder.as_langchain_embeddings())
        if store.collection_count() > 0:
            store.load_bm25_from_existing()
        rag_chain = RAGChain()

    # Instantiate agents
    supervisor  = SupervisorAgent(llm)
    retriever   = RetrieverAgent(rag_chain)
    sql_agent   = SQLAgent(llm)
    synthesizer = SynthesizerAgent(llm)

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",  supervisor)
    graph.add_node("retriever",   retriever)
    graph.add_node("sql",         sql_agent)
    graph.add_node("synthesizer", synthesizer)

    # Entry edge
    graph.add_edge(START, "supervisor")

    # Conditional fan-out from supervisor
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "retriever":   "retriever",
            "sql":         "sql",
            "synthesizer": "synthesizer",
        },
    )

    # Both retriever and sql feed into synthesizer
    graph.add_edge("retriever",   "synthesizer")
    graph.add_edge("sql",         "synthesizer")

    # Synthesizer is always the last node
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    logger.info("LangGraph compiled | nodes=supervisor→[retriever,sql]→synthesizer")
    return compiled


# ── AgentSystem wrapper ────────────────────────────────────────────────────────

class AgentSystem:
    """
    Thin wrapper around the compiled LangGraph graph.
    Provides a clean .ask() interface and manages chat_history across turns.

    Usage:
        system = AgentSystem()
        result = system.ask("How many cloud products do we have?")
        print(result["final_answer"])

        # Follow-up (memory preserved automatically)
        result = system.ask("Which one has the highest margin?")
    """

    def __init__(self):
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(
            f"[bold cyan]{settings.project_name}[/] — Phase 3: Multi-Agent System",
            border_style="cyan",
        ))

        self._graph   = build_graph()
        self._history: List[Dict[str, str]] = []
        self._persona = (
            settings.rag.default_persona if settings.rag else "analyst"
        )
        logger.info("AgentSystem ready")

    def ask(
        self,
        question: str,
        persona:  str = None,
    ) -> Dict[str, Any]:
        """
        Run the full multi-agent pipeline for one question.

        Args:
            question: User's natural language question.
            persona:  Override default persona.

        Returns:
            Final AgentState dict with final_answer, final_sources,
            intent, agent_route, and all intermediate agent outputs.
        """
        persona = persona or self._persona

        initial_state: AgentState = {
            "question":     question,
            "persona":      persona,
            "chat_history": self._history,
        }

        result = self._graph.invoke(initial_state)

        # Persist updated history for the next turn
        self._history = result.get("chat_history", self._history)

        return result

    def stream(self, question: str, persona: str = None):
        """
        Stream the graph execution step-by-step.
        Yields (node_name, partial_state) tuples as each agent completes.
        Useful for showing real-time progress in a UI.
        """
        persona = persona or self._persona
        initial_state: AgentState = {
            "question":     question,
            "persona":      persona,
            "chat_history": self._history,
        }
        for node_name, partial_state in self._graph.stream(initial_state):
            yield node_name, partial_state

        self._history = self._graph.invoke(initial_state).get(
            "chat_history", self._history
        )

    def reset_memory(self) -> None:
        """Clear chat history — start a fresh conversation."""
        self._history = []
        logger.info("Agent memory cleared")

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)