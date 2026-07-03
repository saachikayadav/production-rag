"""Orchestration layer joining retrieval, governed SQL, and cited answers."""

from __future__ import annotations

import sqlite3
from typing import Any

from .database import initialize, schema_catalog
from .retrieval import HybridRetriever
from .sql_engine import SafeSQLExecutor, UnsupportedQuestion, compile_question


class DemandLensService:
    def __init__(self, connection: sqlite3.Connection):
        initialize(connection)
        self.connection = connection
        policy_docs = [
            {
                "id": row["document_id"],
                "title": row["title"],
                "section": row["section"],
                "content": row["content"],
                "source_type": "policy",
            }
            for row in connection.execute("SELECT * FROM planning_documents")
        ]
        self.retriever = HybridRetriever(policy_docs + schema_catalog(connection))
        self.executor = SafeSQLExecutor(connection)

    @staticmethod
    def _needs_data(question: str) -> bool:
        terms = (
            "which", "how many", "compare", "trend", "demand", "forecast", "wape",
            "bias", "inventory", "shortage", "region", "product", "units", "risk",
        )
        return any(term in question.lower() for term in terms)

    def ask(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, limit=4)
        citations = [
            {
                "id": result.document["id"],
                "title": result.document["title"],
                "section": result.document["section"],
                "excerpt": result.document["content"],
                "source_type": result.document["source_type"],
                "fused_score": round(result.fused_score, 6),
                "bm25_rank": result.bm25_rank,
                "semantic_rank": result.semantic_rank,
            }
            for result in retrieved
        ]

        sql_payload = None
        warning = None
        if self._needs_data(question):
            try:
                plan = compile_question(question)
                execution = self.executor.execute(plan)
                sql_payload = {
                    "sql": " ".join(plan.sql.split()),
                    "parameters": list(plan.parameters),
                    "description": plan.description,
                    **execution,
                }
            except UnsupportedQuestion as exc:
                warning = str(exc)

        answer = self._summarize(question, sql_payload, citations, warning)
        route = "hybrid" if sql_payload and citations else "sql" if sql_payload else "rag"
        return {
            "question": question,
            "answer": answer,
            "route": route,
            "sql": sql_payload,
            "citations": citations,
            "warning": warning,
        }

    @staticmethod
    def _summarize(
        question: str,
        sql_payload: dict[str, Any] | None,
        citations: list[dict[str, Any]],
        warning: str | None,
    ) -> str:
        if sql_payload and sql_payload["rows"]:
            first = sql_payload["rows"][0]
            if "wape_pct" in first:
                return (
                    f"The largest forecast miss in this result is {first['product_name']} in "
                    f"{first['region_name']}, with WAPE of {first['wape_pct']}% and "
                    f"{first['missed_units']} net missed units. WAPE is aggregated using absolute "
                    "error divided by actual demand, as defined in the cited metrics handbook."
                )
            if "forecast_bias_pct" in first:
                direction = "over-forecasting" if first["forecast_bias_pct"] > 0 else "under-forecasting"
                return (
                    f"The strongest bias is {first['product_name']} in {first['region_name']}: "
                    f"{first['forecast_bias_pct']}%, indicating {direction}. The policy threshold "
                    "recommends review when absolute bias exceeds 10%."
                )
            if "projected_surplus" in first:
                state = "at shortage risk" if first["projected_surplus"] < 0 else "above the risk threshold"
                return (
                    f"{first['product_name']} in {first['region_name']} is {state}, with a projected "
                    f"surplus of {first['projected_surplus']} units after inbound inventory, safety "
                    "stock, and expected four-week demand."
                )
            return f"The query returned {sql_payload['row_count']} rows. The most recent result is {first}."
        if citations:
            return citations[0]["excerpt"]
        return warning or f"I could not find governed evidence for: {question}"
