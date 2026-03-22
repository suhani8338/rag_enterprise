"""
src/utils/logger.py
───────────────────
Centralised logger + MLflow helpers.
Every module does:  from src.utils.logger import get_logger
                    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import mlflow

from src.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (inherits root config set in config.py)."""
    return logging.getLogger(name)


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def init_mlflow(experiment_name: str = "phase1_ingestion") -> None:
    """Set the local MLflow tracking URI and create/select the experiment."""
    mlflow.set_tracking_uri(f"file://{settings.paths.mlflow_uri}")
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager that wraps a block in an MLflow run."""
    init_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags or {}):
        yield


def log_params(params: Dict[str, Any]) -> None:
    """Safe wrapper — only logs if an active MLflow run exists."""
    try:
        mlflow.log_params(params)
    except Exception:
        pass


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Safe wrapper for MLflow metric logging."""
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


# ── Timing decorator ───────────────────────────────────────────────────────────

def timed(func: Callable) -> Callable:
    """Decorator: logs execution time and records it as an MLflow metric."""
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        log_metrics({f"{func.__name__}_seconds": elapsed})
        return result

    return wrapper