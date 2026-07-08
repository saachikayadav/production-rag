import sqlite3

import pytest

from demand_lens.database import connect, initialize, schema_catalog
from demand_lens.retrieval import HybridRetriever
from demand_lens.service import DemandLensService
from demand_lens.sql_engine import QueryPlan, SafeSQLExecutor, compile_question


@pytest.fixture()
def connection():
    db = connect()
    initialize(db)
    yield db
    db.close()


def test_seeded_warehouse_has_demand_data(connection):
    assert connection.execute("SELECT COUNT(*) FROM weekly_demand").fetchone()[0] == 108
    assert len(schema_catalog(connection)) == 5


def test_hybrid_retrieval_exposes_rrf_provenance(connection):
    documents = [
        {
            "id": row["document_id"],
            "title": row["title"],
            "section": row["section"],
            "content": row["content"],
            "source_type": "policy",
        }
        for row in connection.execute("SELECT * FROM planning_documents")
    ]
    results = HybridRetriever(documents).search("How is forecast bias calculated?")
    assert results[0].document["id"] == "forecast-metrics-1"
    assert results[0].bm25_rank == 1
    assert results[0].fused_score > 0


def test_hybrid_retriever_builds_embeddings_lazily(connection):
    documents = [
        {
            "id": row["document_id"],
            "title": row["title"],
            "section": row["section"],
            "content": row["content"],
            "source_type": "policy",
        }
        for row in connection.execute("SELECT * FROM planning_documents")
    ]
    retriever = HybridRetriever(documents)
    retriever._bm25("How is forecast bias calculated?")
    assert retriever.embeddings is None
    retriever._semantic("How is forecast bias calculated?")
    assert retriever.embeddings is not None


def test_text_to_sql_finds_deliberate_forecast_anomaly(connection):
    plan = compile_question("Which region has the largest forecast miss for Model Beta?")
    result = SafeSQLExecutor(connection).execute(plan)
    assert result["rows"][0]["region_name"] == "West"
    assert result["rows"][0]["product_name"] == "Model Beta"


@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DROP TABLE products",
        "SELECT * FROM products; DELETE FROM products",
        "PRAGMA table_info(products)",
        "UPDATE inventory SET on_hand_units = 0",
    ],
)
def test_sql_guard_rejects_unsafe_statements(connection, unsafe_sql):
    executor = SafeSQLExecutor(connection)
    with pytest.raises(ValueError):
        executor.execute(QueryPlan(unsafe_sql, (), "unsafe"))


def test_sqlite_authorizer_blocks_mutation_hidden_in_cte(connection):
    executor = SafeSQLExecutor(connection)
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        executor.execute(QueryPlan("WITH x AS (SELECT 1) DELETE FROM inventory", (), "unsafe"))


def test_mixed_question_returns_sql_and_policy_citations(connection):
    response = DemandLensService(connection).ask(
        "Which region has the largest forecast miss for Model Beta, and how should WAPE be interpreted?"
    )
    assert response["route"] == "hybrid"
    assert response["sql"]["rows"][0]["region_name"] == "West"
    assert any(citation["id"] == "forecast-metrics-2" for citation in response["citations"])
    assert "WAPE" in response["answer"]


def test_inventory_risk_question(connection):
    response = DemandLensService(connection).ask("Which products are at inventory shortage risk?")
    assert response["sql"]["rows"]
    assert "projected_surplus" in response["sql"]["rows"][0]
