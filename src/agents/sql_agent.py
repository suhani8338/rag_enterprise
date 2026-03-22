"""
src/agents/sql_agent.py
────────────────────────
SQL Agent — converts natural language questions into SQL queries,
runs them against SQLite, and returns a natural-language interpretation.

Two-step process:
  Step 1 — Text-to-SQL:
    LLM is given the database schema + question and generates a SQL query.
  Step 2 — SQL-to-Answer:
    The SQL result (as a table) is fed back to the LLM which explains it
    in plain English with the user's persona in mind.

Database:
  The SQLite database at data/metadata.db already has:
    dim_source  — one row per ingested document (name, type, size, date)
    fact_chunk  — one row per chunk (char count, tokens, page number)

  For richer product/financial queries, we also create a products table
  seeded from products.csv the first time the SQL agent runs.

Safety:
  Only SELECT statements are allowed. Any attempt to run INSERT/UPDATE/
  DELETE/DROP is blocked and returns an error — the LLM is occasionally
  tempted to "fix" data it finds during text-to-SQL generation.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.agents.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_TEXT2SQL_TEMPLATE = """\
You are an expert SQL writer for SQLite. Given the database schema and a question,
write a single SQL SELECT query that answers the question accurately.

Rules:
- Write ONLY the SQL query — no explanation, no markdown, no backticks
- Use only SELECT statements — never INSERT, UPDATE, DELETE, DROP, or CREATE
- Limit results to {max_rows} rows unless the question asks for all
- Use proper SQLite syntax (no ILIKE, use LOWER() for case-insensitive matching)
- If the question cannot be answered from the schema, write: SELECT 'No relevant data' AS message

Database schema:
{schema}

Question: {question}

SQL query:"""

_SQL2ANSWER_TEMPLATE = """\
{persona_preamble}

A SQL query returned the following result. Interpret it in plain language
to answer the user's original question. Be concise and specific.
If the result is empty or says 'No relevant data', say so clearly.

Original question: {question}

SQL query used:
{sql_query}

Query result (as table):
{sql_result}

Answer:"""

_TEXT2SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "max_rows"],
    template=_TEXT2SQL_TEMPLATE,
)
_SQL2ANSWER_PROMPT = PromptTemplate(
    input_variables=["persona_preamble", "question", "sql_query", "sql_result"],
    template=_SQL2ANSWER_TEMPLATE,
)

# ── SQLAgent ───────────────────────────────────────────────────────────────────

class SQLAgent:
    """
    Natural-language → SQL → natural-language answer agent.
    Operates on the project SQLite database.
    """

    def __init__(self, llm, db_path: Optional[Path] = None):
        self.llm        = llm
        self.db_path    = db_path or settings.paths.sqlite_db
        cfg             = settings.agents
        self._max_rows  = cfg.sql_max_rows if cfg else 20
        self._conn      = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._schema    = self._introspect_schema()
        self._t2s_chain = _TEXT2SQL_PROMPT  | llm | StrOutputParser()
        self._s2a_chain = _SQL2ANSWER_PROMPT | llm | StrOutputParser()

        # Seed a products table from CSV if it doesn't exist yet
        self._seed_products_table()
        # Refresh schema after seeding
        self._schema = self._introspect_schema()

        logger.info(f"SQLAgent ready | db={self.db_path} | tables={self._list_tables()}")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node — returns partial state update."""
        question = state.get("question", "")
        persona  = state.get("persona", "analyst")
        route    = state.get("agent_route", [])

        if "sql" not in route:
            logger.debug("SQLAgent skipped (not in route)")
            return {}

        logger.info(f"SQLAgent answering: '{question[:80]}'")

        try:
            # Step 1: Text → SQL
            sql = self._generate_sql(question)
            logger.info(f"Generated SQL: {sql[:120]}")

            # Step 2: Execute
            df = self._execute(sql)
            result_dict = df.to_dict("records") if df is not None else []

            # Step 3: SQL result → natural language answer
            answer = self._interpret(question, sql, df, persona)

            return {
                "sql_query":  sql,
                "sql_result": result_dict,
                "sql_answer": answer,
            }
        except Exception as e:
            logger.error(f"SQLAgent failed: {e}")
            return {
                "sql_query":  "",
                "sql_result": None,
                "sql_answer": f"[SQL Agent error: {e}]",
                "error":      str(e),
            }

    # ── SQL generation ─────────────────────────────────────────────────────────

    def _generate_sql(self, question: str) -> str:
        raw = self._t2s_chain.invoke({
            "schema":   self._schema,
            "question": question,
            "max_rows": self._max_rows,
        })
        return _clean_sql(raw)

    # ── Execution ──────────────────────────────────────────────────────────────

    def _execute(self, sql: str) -> Optional[pd.DataFrame]:
        """Run SQL — SELECT only. Returns DataFrame or None."""
        _guard_select_only(sql)
        try:
            df = pd.read_sql_query(sql, self._conn)
            logger.info(f"SQL returned {len(df)} row(s)")
            return df
        except Exception as e:
            raise RuntimeError(f"SQL execution failed: {e}\nQuery: {sql}")

    # ── Interpretation ─────────────────────────────────────────────────────────

    def _interpret(
        self,
        question: str,
        sql:      str,
        df:       Optional[pd.DataFrame],
        persona:  str,
    ) -> str:
        from src.rag.prompt_templates import PERSONA_PREAMBLES
        preamble   = PERSONA_PREAMBLES.get(persona, PERSONA_PREAMBLES["analyst"])
        result_str = df.to_string(index=False) if df is not None and not df.empty \
                     else "No rows returned."

        return self._s2a_chain.invoke({
            "persona_preamble": preamble,
            "question":         question,
            "sql_query":        sql,
            "sql_result":       result_str[:3000],  # hard cap
        })

    # ── Schema introspection ───────────────────────────────────────────────────

    def _introspect_schema(self) -> str:
        """Return a compact schema string: table(col type, ...) per line."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        lines  = []
        for tbl in tables:
            cols = self._conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"{tbl}({col_defs})")
        return "\n".join(lines)

    def _list_tables(self) -> List[str]:
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [r[0] for r in cursor.fetchall()]

    # ── Seed products table ────────────────────────────────────────────────────

    def _seed_products_table(self) -> None:
        """
        Create and populate a `products` table from products.csv if it doesn't
        already exist. This gives the SQL Agent richer data to query.
        """
        tables = self._list_tables()
        if "products" in tables:
            return

        csv_path = settings.paths.raw_data / "products.csv"
        if not csv_path.exists():
            logger.debug("products.csv not found — skipping products table seed")
            return

        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            df.to_sql("products", self._conn, if_exists="replace", index=False)
            self._conn.commit()
            logger.info(f"Seeded products table with {len(df)} rows from {csv_path.name}")
        except Exception as e:
            logger.warning(f"Could not seed products table: {e}")

    # ── Direct query (used in tests / notebooks) ───────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Run raw SQL directly — useful for debugging."""
        return self._execute(sql)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM SQL output."""
    sql = raw.strip()
    # Remove ```sql ... ``` or ``` ... ```
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|ATTACH)\b",
    re.IGNORECASE,
)

def _guard_select_only(sql: str) -> None:
    """Raise ValueError if the SQL contains any mutation keywords."""
    match = _FORBIDDEN.search(sql)
    if match:
        raise ValueError(
            f"SQL Agent only allows SELECT statements. "
            f"Blocked keyword: '{match.group()}'"
        )