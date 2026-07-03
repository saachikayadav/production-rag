"""Small, deterministic benchmark for retrieval and text-to-SQL regression testing."""

from __future__ import annotations

import json

from demand_lens.database import connect, initialize
from demand_lens.service import DemandLensService


CASES = [
    ("How is forecast bias calculated?", "forecast-metrics-1", None),
    ("What does WAPE mean?", "forecast-metrics-2", None),
    ("What makes inventory a shortage risk?", "inventory-policy-1", "projected_surplus"),
    ("What controls are required for AI-generated SQL?", "governance-1", None),
    ("Show forecast accuracy by region.", "forecast-metrics-2", "wape_pct"),
    ("Show forecast bias for Model Alpha.", "forecast-metrics-1", "forecast_bias_pct"),
    ("Which products are at inventory risk?", "inventory-policy-1", "projected_surplus"),
]


def run() -> dict:
    connection = connect()
    initialize(connection)
    service = DemandLensService(connection)
    retrieval_hits = 0
    sql_hits = 0
    sql_cases = 0
    details = []
    for question, expected_doc, expected_column in CASES:
        response = service.ask(question)
        retrieved_ids = [citation["id"] for citation in response["citations"][:3]]
        retrieval_ok = expected_doc in retrieved_ids
        retrieval_hits += retrieval_ok
        sql_ok = None
        if expected_column:
            sql_cases += 1
            sql_ok = bool(response["sql"] and expected_column in response["sql"]["columns"])
            sql_hits += sql_ok
        details.append({"question": question, "retrieval_at_3": retrieval_ok, "sql_result": sql_ok})
    return {
        "cases": len(CASES),
        "retrieval_recall_at_3": round(retrieval_hits / len(CASES), 3),
        "sql_execution_accuracy": round(sql_hits / sql_cases, 3),
        "details": details,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
