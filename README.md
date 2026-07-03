# DemandLens

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_DemandLens-e82127?style=for-the-badge)](https://production-rag-1.onrender.com/)

**Live application:** [production-rag-1.onrender.com](https://production-rag-1.onrender.com/)

DemandLens is a deployed-style AI copilot for demand planners. It answers mixed questions across a relational planning warehouse and internal planning documentation, exposes generated SQL, cites evidence, and records retrieval provenance.

> All products, regions, policies, and demand records are synthetic. The project is an engineering demonstration and is not affiliated with Tesla.

## Why this project exists

Demand planning questions rarely live in one data source. “Which region has the largest forecast miss, and how should the metric be interpreted?” requires both a database calculation and the current business definition. DemandLens routes a question through both evidence channels and returns one inspectable answer.

## Architecture

```text
Question
  ├── Hybrid knowledge retrieval
  │     ├── BM25 keyword ranking
  │     ├── deterministic semantic baseline
  │     └── explicit weighted reciprocal-rank fusion (RRF)
  ├── Text-to-SQL planner
  │     ├── schema-derived context
  │     ├── SELECT/WITH allowlist
  │     ├── SQLite read-only authorizer
  │     ├── EXPLAIN, timeout, and row limit
  │     └── parameterized execution
  └── Evidence-grounded synthesis
        ├── SQL and result preview
        ├── policy citations
        └── BM25, semantic, and fused ranks
```

The default semantic path uses deterministic local feature hashing so the demo and benchmark work without API keys. It is deliberately an offline baseline: a production deployment can replace `hashed_embedding` with a hosted embedding model or pgvector without changing the fusion interface.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000`. API documentation is at `/docs`.

Useful endpoints:

- `POST /api/ask` — demand-planning RAG + SQL
- `GET /api/examples` — representative questions
- `GET /health` — component health
- `GET /metrics` — API metrics
- `POST /chat` — original general LangGraph endpoint

## Evaluation

```bash
python evaluation.py
pytest -q
```

`evaluation.py` is a deterministic regression benchmark covering retrieval Recall@3 and SQL execution accuracy. The next production step is a larger, versioned dataset comparing BM25, dense retrieval, RRF, and RRF plus reranking on Recall@K, MRR, nDCG, latency, and cost.

## Security model

Generated queries are not trusted. DemandLens accepts one `SELECT` or `WITH` statement, rejects mutation and DDL tokens, uses database-level authorization, executes `EXPLAIN QUERY PLAN`, interrupts long queries, and truncates result sets. These controls are defense in depth; string filtering alone is not treated as a security boundary.

## Repository map

```text
demand_lens/database.py   synthetic warehouse + schema-to-document catalog
demand_lens/retrieval.py  BM25, semantic baseline, and explicit weighted RRF
demand_lens/sql_engine.py text-to-SQL baseline and governed execution
demand_lens/service.py    retrieval/SQL orchestration and synthesis
evaluation.py             reproducible retrieval and SQL benchmark
tests/                    retrieval, SQL safety, and end-to-end regression tests
```

## Known limitations

- The included text-to-SQL planner is deterministic for reproducibility and supports a bounded analytics vocabulary. A hosted LLM planner can be added behind the same `QueryPlan` boundary, but its output must pass identical controls.
- SQLite is appropriate for the self-contained demo. A multi-user deployment should use PostgreSQL, a read-only database role, statement timeouts, and pgvector.
- The local semantic baseline captures lexical-semantic features but is not a substitute for a benchmarked dense embedding model.
