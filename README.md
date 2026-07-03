# Groundwire

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_Groundwire-e35335?style=for-the-badge)](https://production-rag-1.onrender.com/)

Groundwire is a RAG operations studio for product managers, knowledge owners, and junior AI teams. It turns knowledge ingestion, retrieval tuning, guardrails, evaluations, and production failures into visible workflows instead of configuration hidden in code.

The included **DemandLens** demand-planning copilot is the first published assistant built through the workspace. All data is synthetic; this project is not affiliated with Tesla.

## Product workflow

```text
Connect sources → inspect chunks → compare retrieval → configure guardrails
       → run a release evaluation → publish assistant → review incidents
```

### Knowledge sources

- Paste Markdown or text into a workspace
- SHA-256 content deduplication
- Paragraph-aware chunking with bounded overlap
- Source status, size, and chunk inspection
- Stable source and chunk identifiers for provenance

### Retrieval lab

Every test question displays BM25, semantic-baseline, and hybrid results side by side. Hybrid retrieval uses explicit weighted reciprocal-rank fusion:

```text
RRF(document) = Σ weightᵢ / (k + rankᵢ(document))
```

The UI exposes keyword rank, semantic rank, fused score, source, chunk, and per-channel latency. The deterministic local semantic baseline makes the product usable without an API key; the retrieval interface can be replaced with a dense embedding provider or pgvector.

### Guardrail studio

Policies are versionable structured configuration enforced in Python, not suggestions embedded in a system prompt:

- Prompt-injection blocking
- Email, phone, and SSN masking
- Citation requirements
- Retrieval-confidence abstention
- Visible policy test traces

Blocked and abstained requests produce operational incidents explaining which policy acted and why.

### Evaluation studio

The seeded release gate covers expected sources and expected safety behavior. Runs report retrieval Recall@3 and guardrail behavior accuracy with case-level results and stable run identifiers.

### Published assistant

DemandLens demonstrates mixed structured and unstructured evidence:

- Governed text-to-SQL over synthetic demand, forecast, and inventory data
- Schema-derived knowledge documents
- Read-only database authorization
- SQL allowlisting, `EXPLAIN`, timeout, and row limits
- Policy citations and retrieval provenance

## Architecture

```text
Browser control plane
        │
FastAPI application
        ├── Knowledge source registry + chunker
        ├── BM25 / semantic baseline / weighted RRF
        ├── Structured guardrail policy engine
        ├── Evaluation runner
        ├── Incident inbox
        └── DemandLens published assistant
                ├── planning-document retrieval
                └── governed SQL executor
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000`; API documentation is available at `/docs`.

## Control-plane API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/studio/overview` | Workspace health |
| `GET/POST` | `/api/studio/sources` | List or ingest sources |
| `GET` | `/api/studio/sources/{id}/chunks` | Inspect chunk boundaries |
| `POST` | `/api/studio/retrieval/compare` | Compare three retrievers |
| `GET/PUT` | `/api/studio/guardrails/{key}` | Read or update policies |
| `POST` | `/api/studio/test` | Test the active guardrails |
| `GET/POST` | `/api/studio/evaluations` | Manage evaluation cases |
| `POST` | `/api/studio/evaluations/run` | Run a release evaluation |
| `GET` | `/api/studio/incidents` | Review operational failures |
| `POST` | `/api/ask` | Query the published DemandLens assistant |

## Verification

```bash
pytest -q
python evaluation.py
```

The tests cover ingestion, deduplication, chunking, retrieval provenance, guardrail enforcement and toggles, PII masking, incidents, evaluation metrics, SQL safety, and mixed RAG/SQL answers.

## Current MVP boundaries

- The hosted demo uses an in-memory SQLite workspace, so operator changes reset when the service restarts. Production persistence should use PostgreSQL and object storage.
- Text ingestion is implemented end to end. URL and PDF ingestion need background workers, content extraction, malware checks, and object storage before being exposed to users.
- LangSmith tracing remains available in the original model path. The incident schema includes a trace URL extension point; syncing LangSmith feedback and runs is the next integration.
- This is currently a single-workspace product without authentication. Multi-user use requires identity, tenant isolation, encrypted secrets, audit logs, and authorization tests.

These limitations are explicit because a trustworthy production tool should distinguish completed controls from roadmap claims.
