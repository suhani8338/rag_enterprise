"""
src/evaluation/ragas_eval.py
─────────────────────────────
RAGAS evaluation pipeline — measures RAG quality across three metrics:

  faithfulness       — Does the answer contain only facts from the context?
                       Catches hallucinations. Score: 0–1.
  answer_relevancy   — Does the answer actually address the question asked?
                       Catches verbose or off-topic answers. Score: 0–1.
  context_precision  — Are the retrieved chunks actually useful for the answer?
                       Catches over-retrieval of irrelevant chunks. Score: 0–1.

How it works:
  1. Load (or auto-generate) a test set of (question, ground_truth) pairs.
  2. Run each question through the RAGChain to get answer + contexts.
  3. Pass everything to RAGAS which uses an LLM judge (Ollama) to score.
  4. Log all scores to MLflow with the experiment name from settings.yaml.
  5. Print a summary table and save results to data/eval_results.json.

Test set format (data/eval_test_set.json):
  [
    {
      "question": "What is our parental leave policy?",
      "ground_truth": "Employees receive 16 weeks primary / 8 weeks secondary parental leave."
    },
    ...
  ]

Usage:
  python -m src.evaluation.ragas_eval               # run full eval
  python -m src.evaluation.ragas_eval --generate    # auto-generate test set first
  python -m src.evaluation.ragas_eval --questions 5 # limit to 5 questions
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.rag.rag_chain import RAGChain
from src.utils.config import settings
from src.utils.logger import get_logger

logger  = get_logger(__name__)
console = Console()

# ── Default test questions (used when no test set file exists) ────────────────

DEFAULT_TEST_QUESTIONS = [
    {
        "question": "What is the parental leave policy?",
        "ground_truth": "Employees get 16 weeks primary and 8 weeks secondary parental leave.",
    },
    {
        "question": "What is the remote work policy?",
        "ground_truth": "Employees must be in office a minimum of 2 days per week.",
    },
    {
        "question": "What is the annual learning budget per employee?",
        "ground_truth": "Each employee receives $2,000 per year for courses, conferences, and books.",
    },
    {
        "question": "What cloud products does Acme offer?",
        "ground_truth": "AcmeCloud Compute, AcmeDB Managed, AcmeMesh, and AcmeAI Studio.",
    },
    {
        "question": "What was Acme's total revenue in 2024?",
        "ground_truth": "Total revenue was $4.2 billion, up 18% year-over-year.",
    },
    {
        "question": "What compliance certifications does AcmeCloud hold?",
        "ground_truth": "SOC 2 Type II, ISO 27001, PCI-DSS Level 1, HIPAA, and FedRAMP Moderate.",
    },
    {
        "question": "What is the company 401k match policy?",
        "ground_truth": "Acme matches 4% of salary with immediate vesting.",
    },
    {
        "question": "What is the net revenue retention for the cloud division?",
        "ground_truth": "Net Revenue Retention is 128%, up from 118% the prior year.",
    },
]


class RAGASEvaluator:
    """
    Runs RAGAS evaluation on the RAGChain and logs results to MLflow.
    """

    def __init__(self, rag_chain: Optional[RAGChain] = None):
        self._chain   = rag_chain or RAGChain()
        cfg           = settings.evaluation
        self._metrics         = cfg.metrics        if cfg else ["faithfulness", "answer_relevancy"]
        self._experiment_name = cfg.experiment_name if cfg else "ragas_evaluation"
        self._test_set_path   = (
            settings.project_root / cfg.test_set_path
            if cfg else settings.project_root / "data" / "eval_test_set.json"
        )
        logger.info(f"RAGASEvaluator ready | metrics={self._metrics}")

    # ── Load / generate test set ──────────────────────────────────────────────

    def load_test_set(self) -> List[Dict[str, str]]:
        """Load test set from JSON file, or return defaults if file doesn't exist."""
        if self._test_set_path.exists():
            with open(self._test_set_path) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} test questions from {self._test_set_path}")
            return data
        logger.info("No test set file found — using built-in default questions")
        return DEFAULT_TEST_QUESTIONS

    def save_test_set(self, questions: List[Dict[str, str]]) -> None:
        """Save a test set to the configured path."""
        self._test_set_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._test_set_path, "w") as f:
            json.dump(questions, f, indent=2)
        console.print(f"[green]✓[/] Test set saved to {self._test_set_path}")

    # ── Run evaluation ────────────────────────────────────────────────────────

    def run(
        self,
        max_questions: Optional[int] = None,
        persona:       str = "analyst",
    ) -> pd.DataFrame:
        """
        Run full RAGAS evaluation.

        Args:
            max_questions: Limit evaluation to first N questions (cheaper for testing).
            persona:       Persona to use for RAG generation.

        Returns:
            DataFrame with one row per question and columns for each metric.
        """
        test_set = self.load_test_set()
        if max_questions:
            test_set = test_set[:max_questions]

        console.rule(f"[bold]RAGAS Evaluation — {len(test_set)} questions")

        # ── Step 1: Run RAG chain on every question ────────────────────────────
        records = []
        for i, item in enumerate(test_set, 1):
            q  = item["question"]
            gt = item.get("ground_truth", "")
            console.print(f"[dim]({i}/{len(test_set)})[/] {q[:70]}...")

            try:
                t0       = time.perf_counter()
                response = self._chain.ask(q, persona=persona, skip_rewrite=True)
                elapsed  = (time.perf_counter() - t0) * 1000

                records.append({
                    "question":     q,
                    "answer":       response.answer,
                    "ground_truth": gt,
                    "contexts":     _get_contexts(response),
                    "latency_ms":   elapsed,
                })
            except Exception as e:
                logger.warning(f"Question {i} failed: {e}")
                records.append({
                    "question":     q,
                    "answer":       f"[ERROR: {e}]",
                    "ground_truth": gt,
                    "contexts":     [],
                    "latency_ms":   0.0,
                })

        # ── Step 2: Score with RAGAS ───────────────────────────────────────────
        results_df = self._score_with_ragas(records)

        # ── Step 3: Log to MLflow ──────────────────────────────────────────────
        self._log_to_mlflow(results_df)

        # ── Step 4: Save results ───────────────────────────────────────────────
        out_path = settings.project_root / "data" / "eval_results.json"
        results_df.to_json(out_path, orient="records", indent=2)
        console.print(f"[green]✓[/] Results saved to {out_path}")

        # ── Step 5: Print summary table ───────────────────────────────────────
        self._print_summary(results_df)

        return results_df

    # ── RAGAS scoring ─────────────────────────────────────────────────────────

    def _score_with_ragas(self, records: List[Dict]) -> pd.DataFrame:
        """Run RAGAS metrics on the collected records."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
            from langchain_ollama import OllamaLLM
            from langchain_huggingface import HuggingFaceEmbeddings

            metric_map = {
                "faithfulness":      faithfulness,
                "answer_relevancy":  answer_relevancy,
                "context_precision": context_precision,
            }
            selected_metrics = [
                metric_map[m] for m in self._metrics if m in metric_map
            ]

            # Build HuggingFace dataset expected by RAGAS
            ragas_data = {
                "question":     [r["question"]     for r in records],
                "answer":       [r["answer"]        for r in records],
                "contexts":     [r["contexts"]      for r in records],
                "ground_truth": [r["ground_truth"]  for r in records],
            }
            dataset = Dataset.from_dict(ragas_data)

            # Use local Ollama as the RAGAS judge LLM
            cfg = settings.llm
            llm_judge = OllamaLLM(
                model    = cfg.model    if cfg else "mistral",
                base_url = cfg.base_url if cfg else "http://localhost:11434",
            )
            embed_model = HuggingFaceEmbeddings(
                model_name = settings.embedding.model_name,
            )

            console.print("[dim]Running RAGAS scoring (this takes a few minutes)…[/]")
            result = evaluate(
                dataset,
                metrics    = selected_metrics,
                llm        = llm_judge,
                embeddings = embed_model,
            )
            scores_df = result.to_pandas()

            # Merge with latency
            latency_series = pd.Series([r["latency_ms"] for r in records], name="latency_ms")
            scores_df["latency_ms"] = latency_series.values
            return scores_df

        except Exception as e:
            logger.warning(f"RAGAS scoring failed ({e}) — returning raw records without scores")
            df = pd.DataFrame(records)
            for m in self._metrics:
                df[m] = None
            return df

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_to_mlflow(self, df: pd.DataFrame) -> None:
        mlflow.set_tracking_uri(f"file://{settings.paths.mlflow_uri}")
        mlflow.set_experiment(self._experiment_name)

        with mlflow.start_run(run_name="ragas_eval"):
            mlflow.log_params({
                "num_questions":   len(df),
                "metrics":         ",".join(self._metrics),
                "embedding_model": settings.embedding.model_name,
                "llm_model":       settings.llm.model if settings.llm else "unknown",
                "persona":         "analyst",
            })

            # Log mean score per metric
            for metric in self._metrics:
                if metric in df.columns and df[metric].notna().any():
                    mean_val = df[metric].mean()
                    mlflow.log_metric(f"mean_{metric}", round(float(mean_val), 4))
                    logger.info(f"  {metric}: {mean_val:.4f}")

            if "latency_ms" in df.columns:
                mlflow.log_metric("mean_latency_ms", round(df["latency_ms"].mean(), 1))

            # Save results as MLflow artifact
            out_path = settings.project_root / "data" / "eval_results.json"
            if out_path.exists():
                mlflow.log_artifact(str(out_path))

        console.print(f"[green]✓[/] Results logged to MLflow experiment '{self._experiment_name}'")

    # ── Summary table ─────────────────────────────────────────────────────────

    def _print_summary(self, df: pd.DataFrame) -> None:
        table = Table(title="RAGAS Evaluation Summary", show_lines=True)
        table.add_column("Metric",       style="cyan")
        table.add_column("Mean Score",   justify="right")
        table.add_column("Min",          justify="right")
        table.add_column("Max",          justify="right")

        for metric in self._metrics:
            if metric in df.columns and df[metric].notna().any():
                col = df[metric].dropna()
                table.add_row(
                    metric,
                    f"{col.mean():.4f}",
                    f"{col.min():.4f}",
                    f"{col.max():.4f}",
                )
            else:
                table.add_row(metric, "N/A", "N/A", "N/A")

        if "latency_ms" in df.columns:
            table.add_row(
                "latency_ms",
                f"{df['latency_ms'].mean():.0f}",
                f"{df['latency_ms'].min():.0f}",
                f"{df['latency_ms'].max():.0f}",
            )

        console.print(table)
        console.print(f"\n[dim]Full results: data/eval_results.json | MLflow: mlflow ui[/]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_contexts(response) -> List[str]:
    """
    Extract the retrieved context strings from a RAGResponse.
    RAGAS expects a list of strings (one per retrieved chunk).
    We reconstruct them from the sources list since RAGResponse doesn't
    store raw chunk text — sources are descriptive strings.
    """
    # RAGResponse.sources contains citation strings like "annual_report.txt, page 3"
    # For RAGAS context_precision we need the actual chunk text — use sources as proxy
    return response.sources if response.sources else ["[no context retrieved]"]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--generate",  action="store_true",
                        help="Save the default test set to data/eval_test_set.json")
    parser.add_argument("--questions", type=int, default=None,
                        help="Limit to N questions (default: all)")
    parser.add_argument("--persona",   type=str, default="analyst")
    args = parser.parse_args()

    evaluator = RAGASEvaluator()

    if args.generate:
        evaluator.save_test_set(DEFAULT_TEST_QUESTIONS)
        console.print(
            f"[green]✓[/] Test set written to {evaluator._test_set_path}\n"
            "Edit it to add your own questions and ground truths."
        )
        return

    evaluator.run(max_questions=args.questions, persona=args.persona)


if __name__ == "__main__":
    main()