"""
src/rag_pipeline.py
────────────────────
Phase 2 CLI entry point.

Run with:
    python -m src.rag_pipeline --question "What is our parental leave policy?"
    python -m src.rag_pipeline --question "Explain AcmeMesh" --persona engineer
    python -m src.rag_pipeline --question "Q4 highlights" --persona executive
    python -m src.rag_pipeline --summarise
    python -m src.rag_pipeline --compare "cloud" "software"
    python -m src.rag_pipeline --interactive          # chat loop
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from src.rag.prompt_templates import AVAILABLE_PERSONAS
from src.rag.rag_chain import RAGChain
from src.utils.logger import get_logger

logger  = get_logger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Phase 2 RAG pipeline")
    parser.add_argument("--question",    type=str,  help="Ask a single question")
    parser.add_argument("--persona",     type=str,  default="analyst",
                        choices=AVAILABLE_PERSONAS,
                        help="Response persona (default: analyst)")
    parser.add_argument("--mode",        type=str,  default="hybrid",
                        choices=["dense", "hybrid", "mmr"],
                        help="Retrieval mode (default: hybrid)")
    parser.add_argument("--no-rewrite",  action="store_true",
                        help="Skip query rewriting (faster)")
    parser.add_argument("--summarise",   action="store_true",
                        help="Summarise all indexed documents")
    parser.add_argument("--compare",     nargs=2,   metavar=("TOPIC_A", "TOPIC_B"),
                        help="Compare two topics")
    parser.add_argument("--interactive", action="store_true",
                        help="Start an interactive chat loop")
    args = parser.parse_args()

    # Initialise the chain (loads models — takes ~5s on first run)
    chain = RAGChain()

    if args.summarise:
        result = chain.summarise(persona=args.persona)
        result.pretty_print()
        return

    if args.compare:
        result = chain.compare(args.compare[0], args.compare[1], persona=args.persona)
        result.pretty_print()
        return

    if args.question:
        result = chain.ask(
            args.question,
            persona      = args.persona,
            mode         = args.mode,
            skip_rewrite = args.no_rewrite,
        )
        result.pretty_print()
        return

    if args.interactive:
        _interactive_loop(chain, args.persona)
        return

    # No args — print help
    parser.print_help()


def _interactive_loop(chain: RAGChain, default_persona: str) -> None:
    """Simple REPL for chatting with the RAG system."""
    console.print(
        "\n[bold cyan]Enterprise RAG — Interactive Mode[/]\n"
        "[dim]Commands: /persona <name> | /mode <dense|hybrid|mmr> | /summarise | /quit[/]\n"
    )

    persona = default_persona
    mode    = "hybrid"

    while True:
        try:
            raw = console.input("[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if raw.startswith("/quit"):
            console.print("[dim]Goodbye.[/]")
            break

        if raw.startswith("/persona "):
            new_persona = raw.split(" ", 1)[1].strip()
            if new_persona in AVAILABLE_PERSONAS:
                persona = new_persona
                console.print(f"[dim]Persona set to: {persona}[/]")
            else:
                console.print(f"[yellow]Unknown persona. Options: {AVAILABLE_PERSONAS}[/]")
            continue

        if raw.startswith("/mode "):
            new_mode = raw.split(" ", 1)[1].strip()
            if new_mode in ("dense", "hybrid", "mmr"):
                mode = new_mode
                console.print(f"[dim]Mode set to: {mode}[/]")
            else:
                console.print("[yellow]Unknown mode. Options: dense, hybrid, mmr[/]")
            continue

        if raw == "/summarise":
            result = chain.summarise(persona=persona)
            result.pretty_print()
            continue

        # ── Regular question ──────────────────────────────────────────────────
        console.print(f"[dim]({persona} · {mode})[/]")
        try:
            result = chain.ask(raw, persona=persona, mode=mode)
            result.pretty_print()
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

        console.print(Rule(style="dim"))


if __name__ == "__main__":
    main()