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

- Upload PDF, DOCX, Markdown, TXT, or HTML files up to 10 MB
- Paste Markdown or text directly into a workspace
- SHA-256 content deduplication
- File-signature, extension, encoding, and extraction validation
- Structure-aware extraction preserving headings, pages, tables, and section paths
- Parent–child chunking, immediate-neighbor expansion, and ordered context packing
- Source status, size, extractor, version, and chunk inspection
- Stable source and chunk identifiers for provenance

### Retrieval lab

Every test question displays BM25, semantic-baseline, and hybrid results side by side. Hybrid retrieval uses explicit weighted reciprocal-rank fusion:

```text
RRF(document) = Σ weightᵢ / (k + rankᵢ(document))
```

The UI exposes keyword rank, semantic rank, fused score, source, chunk, and per-channel latency. Production uses Pinecone's integrated `llama-text-embed-v2` index; local development falls back to deterministic feature hashing without an API key.

### Agentic retrieval

The bounded retrieval orchestrator plans intent and modalities, generates query variants, runs hybrid retrieval, fuses results across queries, grades evidence, expands neighboring chunks, and either returns grounded context or abstains. Every decision is returned as an inspectable trace.

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
        ├── PostgreSQL knowledge registry + chunker
        ├── BM25 / Pinecone dense retrieval / weighted RRF
        ├── Multi-query planner + evidence grader
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
| `POST` | `/api/studio/sources/upload` | Upload and extract a knowledge file |
| `GET` | `/api/studio/sources/{id}/chunks` | Inspect chunk boundaries |
| `POST` | `/api/studio/retrieval/compare` | Compare three retrievers |
| `POST` | `/api/retrieve` | Run the configured hybrid retriever for production latency tests |
| `POST` | `/api/studio/agentic-query` | Run bounded multi-query RAG orchestration |
| `POST` | `/api/studio/vector/sync` | Start batched background reconciliation with Pinecone |
| `GET` | `/api/studio/vector/sync/status` | Inspect vector sync progress and failures |
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

The tests cover file validation, Markdown/HTML/DOCX extraction, ingestion, deduplication, parent and neighbor context, chunk provenance, retrieval ranking, guardrail enforcement and toggles, PII masking, incidents, evaluation metrics, SQL safety, and mixed RAG/SQL answers.

## Production retrieval benchmark

Use `/api/retrieve` for retrieval-only performance claims. Do not benchmark `/api/studio/retrieval/compare` or the assistant-generation endpoints and describe the result as retrieval latency.

Warm the deployed service first:

```bash
curl https://production-rag-1.onrender.com/ready
```

Then run k6 against the deployed URL:

```bash
k6 run \
  --summary-export=results-1-user-paced-run-1.json \
  -e BASE_URL=https://production-rag-1.onrender.com \
  -e VUS=1 \
  -e DURATION=2m \
  -e SLEEP_SECONDS=5 \
  load-tests/retrieval.js
```

To create a public-demo-safe synthetic corpus before testing:

```bash
python scripts/seed_synthetic_corpus.py \
  --base-url https://production-rag-1.onrender.com \
  --documents 60 \
  --sections-per-document 170 \
  --sync-poll-seconds 5
```

Vector sync runs as a bounded background job and can be inspected separately:

```bash
curl https://production-rag-1.onrender.com/api/studio/vector/sync/status
```

On constrained free-tier infrastructure, report only successful low-concurrency runs. Do not claim production-scale throughput from rate-limited infrastructure. Pair latency with retrieval-quality metrics such as Recall@K, MRR, and citation precision. A benchmark should disclose the corpus and infrastructure:

- Number of documents
- Number of chunks
- Average chunk length
- Pinecone index/model
- Render instance type

Example claim format:

> Measured X% Recall@3 and Y ms p95 retrieval latency at low concurrency over a 1,800-record synthetic demand-planning corpus, with vector sync redesigned from a monolithic request into batched background jobs.

## Current MVP boundaries

- Production uses PostgreSQL when `DATABASE_URL` is configured; local development uses a persistent SQLite file. Original file bytes still require object storage before uploaded originals can be downloaded later.
- File ingestion and extraction are implemented synchronously for the MVP. Production should move extraction to SQS workers, scan originals for malware, and store them in S3.
- LangSmith tracing remains available in the original model path. The incident schema includes a trace URL extension point; syncing LangSmith feedback and runs is the next integration.
- This is currently a single-workspace product without authentication. Multi-user use requires identity, tenant isolation, encrypted secrets, audit logs, and authorization tests.

These limitations are explicit because a trustworthy production tool should distinguish completed controls from roadmap claims.
