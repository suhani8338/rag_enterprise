"""
src/rag/llm_factory.py
───────────────────────
Creates and returns the configured LLM instance.

Primary:  Ollama (local, free, requires `ollama serve` running)
Fallback: A stub LLM that returns a helpful error message if Ollama is down,
          so the rest of the pipeline doesn't crash during development.

Ollama setup (one-time):
  1. Install from https://ollama.com
  2. Run: ollama pull mistral
  3. Ollama starts automatically on system boot (or run: ollama serve)

Usage:
    llm = get_llm()
    response = llm.invoke("Hello, what is RAG?")
"""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> BaseLanguageModel:
    """
    Return a configured LLM based on settings.yaml [llm] section.
    Currently supports: ollama.

    Falls back to a stub if Ollama is unreachable, so the rest of Phase 2
    code can be imported and tested even without Ollama running.
    """
    cfg = settings.llm
    if cfg is None:
        raise RuntimeError(
            "No [llm] section in settings.yaml. "
            "Add the Phase 2 config block (see README)."
        )

    if cfg.provider == "ollama":
        return _build_ollama(cfg)

    raise ValueError(f"Unsupported LLM provider: '{cfg.provider}'. Use 'ollama'.")


def _build_ollama(cfg) -> BaseLanguageModel:
    """Build an Ollama LLM, with a connectivity check and clear error message."""
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        raise ImportError(
            "Install langchain-ollama: pip install langchain-ollama"
        )

    llm = OllamaLLM(
        model       = cfg.model,
        base_url    = cfg.base_url,
        temperature = cfg.temperature,
        num_predict = cfg.max_tokens,
    )

    # Lightweight connectivity check
    try:
        llm.invoke("ping")
        logger.info(
            f"Ollama connected | model={cfg.model} "
            f"url={cfg.base_url} temp={cfg.temperature}"
        )
    except Exception as e:
        logger.warning(
            f"\n{'='*60}\n"
            f"⚠  Ollama is not running or model '{cfg.model}' is not pulled.\n"
            f"   To fix:\n"
            f"   1. Install Ollama: https://ollama.com\n"
            f"   2. Run: ollama pull {cfg.model}\n"
            f"   3. Ollama starts automatically, or run: ollama serve\n"
            f"   Error: {e}\n"
            f"{'='*60}\n"
        )
        return _StubLLM(cfg.model)

    return llm


class _StubLLM:
    """
    Drop-in stub used when Ollama is unavailable.
    Returns a helpful message instead of crashing — useful for testing
    the rest of the pipeline without a running LLM.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt, **kwargs) -> str:
        return (
            f"[LLM unavailable — Ollama with model '{self.model_name}' is not running. "
            f"Install from https://ollama.com and run: ollama pull {self.model_name}]"
        )

    def __or__(self, other):
        """Allow stub to participate in LangChain chains without crashing."""
        class _StubChain:
            def __init__(self, stub):
                self._stub = stub
            def invoke(self, inputs, **kwargs):
                return self._stub.invoke("")
        return _StubChain(self)