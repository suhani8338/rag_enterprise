"""
src/agent_pipeline.py
──────────────────────
Phase 3 CLI entry point — multi-agent orchestration.

Run with:
    python -m src.agent_pipeline --question "How many cloud products do we have?"
    python -m src.agent_pipeline --question "What is our remote work policy?" --persona hr
    python -m src.agent_pipeline --question "What does AcmeMesh cost and how does it work?"
    python -m src.agent_pipeline --interactive
    python -m src.agent_pipeline --trace "Which products launched in 2024?"
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from src.agents.graph import AgentSystem
from src.rag.prompt_templates import AVAILABLE_PERSONAS
from src.utils.logger import get_logger

logger  = get_logger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Phase 3 multi-agent pipeline")
    parser.add_argument("--question",    type=str, help="Ask a single question")
    parser.add_argument("--persona",     type=str, default="analyst",
                        choices=AVAILABLE_PERSONAS)
    parser.add_argument("--trace",       type=str,
                        help="Ask a question and show full agent trace")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive multi-turn chat")
    args = parser.parse_args()

    system = AgentSystem()

    if args.trace:
        _run_with_trace(system, args.trace, args.persona)
        return

    if args.question:
        _run_single(system, args.question, args.persona)
        return

    if args.interactive:
        _interactive_loop(system, args.persona)
        return

    parser.print_help()


# ── Single question ────────────────────────────────────────────────────────────

def _run_single(system: AgentSystem, question: str, persona: str) -> None:
    console.print(f"\n[dim]Question:[/] [italic]{question}[/]")
    console.print(f"[dim]Persona: {persona}[/]\n")

    t0     = time.perf_counter()
    result = system.ask(question, persona=persona)
    elapsed = (time.perf_counter() - t0) * 1000

    _print_result(result, elapsed)


# ── Traced run ─────────────────────────────────────────────────────────────────

def _run_with_trace(system: AgentSystem, question: str, persona: str) -> None:
    """Show which agents ran and what each produced."""
    console.print(f"\n[dim]Tracing:[/] [italic]{question}[/]\n")

    t0     = time.perf_counter()
    result = system.ask(question, persona=persona)
    elapsed = (time.perf_counter() - t0) * 1000

    # Trace table
    table = Table(title="Agent Trace", show_lines=True, expand=False)
    table.add_column("Agent",  style="cyan",  no_wrap=True)
    table.add_column("Output", style="white")

    table.add_row("Intent",    result.get("intent", "—"))
    table.add_row("Route",     str(result.get("agent_route", [])))
    table.add_row("Plan",      result.get("plan", "—")[:120])

    rag_ans = result.get("rag_answer", "")
    if rag_ans:
        table.add_row("Retriever", rag_ans[:200] + ("…" if len(rag_ans) > 200 else ""))

    sql_q = result.get("sql_query", "")
    sql_a = result.get("sql_answer", "")
    if sql_q:
        table.add_row("SQL query", sql_q[:120])
    if sql_a:
        table.add_row("SQL answer", sql_a[:200] + ("…" if len(sql_a) > 200 else ""))

    console.print(table)
    console.print()
    _print_result(result, elapsed)


# ── Interactive loop ───────────────────────────────────────────────────────────

def _interactive_loop(system: AgentSystem, default_persona: str) -> None:
    console.print(
        "\n[bold cyan]Enterprise Multi-Agent System — Interactive Mode[/]\n"
        "[dim]Commands: /persona <n> | /trace | /memory | /reset | /quit[/]\n"
    )
    persona = default_persona
    trace   = False

    while True:
        try:
            raw = console.input("[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if raw == "/quit":
            console.print("[dim]Goodbye.[/]")
            break
        if raw.startswith("/persona "):
            p = raw.split(" ", 1)[1].strip()
            if p in AVAILABLE_PERSONAS:
                persona = p
                console.print(f"[dim]Persona → {persona}[/]")
            else:
                console.print(f"[yellow]Options: {AVAILABLE_PERSONAS}[/]")
            continue
        if raw == "/trace":
            trace = not trace
            console.print(f"[dim]Trace mode → {'on' if trace else 'off'}[/]")
            continue
        if raw == "/memory":
            history = system.history
            if not history:
                console.print("[dim]No conversation history yet.[/]")
            for turn in history[-6:]:
                role = turn.get("role", "?").capitalize()
                console.print(f"[dim]{role}:[/] {turn.get('content','')[:150]}")
            continue
        if raw == "/reset":
            system.reset_memory()
            console.print("[dim]Memory cleared.[/]")
            continue

        # ── Question ──────────────────────────────────────────────────────────
        t0     = time.perf_counter()
        result = system.ask(raw, persona=persona)
        elapsed = (time.perf_counter() - t0) * 1000

        if trace:
            console.print(
                f"[dim]intent={result.get('intent')} "
                f"route={result.get('agent_route')} "
                f"chunks={result.get('rag_chunks', 0)}[/]"
            )

        _print_result(result, elapsed)
        console.print(Rule(style="dim"))


# ── Pretty printer ─────────────────────────────────────────────────────────────

def _print_result(result: dict, elapsed_ms: float) -> None:
    answer  = result.get("final_answer", "[No answer]")
    sources = result.get("final_sources", [])
    intent  = result.get("intent", "?")
    route   = result.get("agent_route", [])

    console.print(
        Panel(
            f"[bold]{answer}[/]",
            title=(
                f"[cyan]Answer[/] "
                f"[dim](intent={intent} · route={route} · {elapsed_ms:.0f}ms)[/]"
            ),
            border_style="cyan",
        )
    )
    if sources:
        console.print("[dim]Sources:[/]")
        for src in sources:
            console.print(f"  [dim]•[/] {src}")
    console.print()


if __name__ == "__main__":
    main()