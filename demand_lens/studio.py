"""No-code KnowledgeOps control plane for retrieval, policies, and evaluations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from typing import Any

from .ingestion import ExtractedDocument, ExtractedSection
from .retrieval import HybridRetriever, tokenize
from .vector_store import LocalVectorStore, VectorStore


INJECTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"dump\s+(all\s+)?(secrets|instructions|records)",
        r"bypass\s+(all\s+)?(rules|restrictions|guardrails)",
    )
]
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d[\s.-]?)?(?:\d[\s.-]?){9,12}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    """Paragraph-aware deterministic chunking with bounded overlap."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


class KnowledgeOpsStudio:
    def __init__(self, connection: Any, vector_store: VectorStore | None = None, namespace: str = "workspace-default"):
        self.connection = connection
        self.vector_store = vector_store or LocalVectorStore()
        self.namespace = namespace

    def _scalar(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        row = self.connection.execute(sql, params).fetchone()
        return next(iter(row.values())) if isinstance(row, dict) else row[0]

    def _persist_sections(self, source_id: str, sections: list[ExtractedSection]) -> int:
        position = 0
        for section_index, section in enumerate(sections, 1):
            parent_id = f"{source_id}::parent-{section_index}"
            for content in chunk_text(section.content):
                position += 1
                chunk_id = f"{source_id}::chunk-{position}"
                self.connection.execute(
                    """INSERT INTO knowledge_chunks
                    (chunk_id, source_id, parent_id, position, page_number, section_path, content, character_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                    parent_id=excluded.parent_id, position=excluded.position,
                    page_number=excluded.page_number, section_path=excluded.section_path,
                    content=excluded.content, character_count=excluded.character_count""",
                    (
                        chunk_id, source_id, parent_id, position, section.page_number,
                        section.section_path, content, len(content),
                    ),
                )
        return position

    def _ensure_chunks(self, source: sqlite3.Row) -> None:
        existing = self.connection.execute(
            "SELECT 1 FROM knowledge_chunks WHERE source_id = ? LIMIT 1", (source["source_id"],)
        ).fetchone()
        if not existing:
            self._persist_sections(
                source["source_id"],
                [ExtractedSection(source["content"], source["name"])],
            )
            self.connection.commit()

    def _documents(self) -> list[dict[str, str]]:
        documents = []
        for source in self.connection.execute(
            "SELECT source_id, name, source_type, content FROM knowledge_sources WHERE status = 'indexed' ORDER BY created_at"
        ):
            self._ensure_chunks(source)
            for chunk in self.connection.execute(
                "SELECT * FROM knowledge_chunks WHERE source_id = ? ORDER BY position", (source["source_id"],)
            ):
                header = f"Document: {source['name']}\nSection: {chunk['section_path']}"
                if chunk["page_number"]:
                    header += f"\nPage: {chunk['page_number']}"
                documents.append(
                    {
                        "id": chunk["chunk_id"],
                        "source_id": source["source_id"],
                        "title": source["name"],
                        "section": chunk["section_path"],
                        "content": f"{header}\n\n{chunk['content']}",
                        "source_type": source["source_type"],
                    }
                )
        return documents

    def overview(self) -> dict[str, Any]:
        source_count = self._scalar("SELECT COUNT(*) FROM knowledge_sources")
        incident_count = self._scalar("SELECT COUNT(*) FROM incidents WHERE status = 'open'")
        enabled = self._scalar("SELECT COUNT(*) FROM guardrail_policies WHERE enabled = 1")
        cases = self._scalar("SELECT COUNT(*) FROM evaluation_cases")
        return {
            "workspace": "DemandLens Production",
            "sources": source_count,
            "chunks": len(self._documents()),
            "enabled_guardrails": enabled,
            "evaluation_cases": cases,
            "open_incidents": incident_count,
            "dense_retrieval_provider": self.vector_store.provider,
            "persistence_provider": getattr(self.connection, "dialect", "sqlite"),
            "published_assistant": {"name": "DemandLens", "status": "live", "endpoint": "/api/ask"},
        }

    def list_sources(self) -> list[dict[str, Any]]:
        items = []
        for row in self.connection.execute("SELECT * FROM knowledge_sources ORDER BY created_at DESC"):
            self._ensure_chunks(row)
            item = dict(row)
            item["character_count"] = len(item.pop("content"))
            item["chunk_count"] = self._scalar(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE source_id = ?", (row["source_id"],)
            )
            file_row = self.connection.execute(
                "SELECT original_filename, mime_type, byte_size, version, extraction_method FROM source_files WHERE source_id = ?",
                (row["source_id"],),
            ).fetchone()
            item["file"] = dict(file_row) if file_row else None
            items.append(item)
        return items

    def add_source(self, name: str, content: str, source_type: str = "text") -> dict[str, Any]:
        normalized = content.strip()
        if len(normalized) < 20:
            raise ValueError("Source content must contain at least 20 characters")
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        duplicate = self.connection.execute(
            "SELECT source_id FROM knowledge_sources WHERE content_hash = ?", (digest,)
        ).fetchone()
        if duplicate:
            raise ValueError(f"Duplicate content already exists as {duplicate['source_id']}")
        source_id = f"src-{uuid.uuid4().hex[:10]}"
        self.connection.execute(
            "INSERT INTO knowledge_sources(source_id, name, source_type, content, content_hash, status) VALUES (?, ?, ?, ?, ?, 'indexing')",
            (source_id, name.strip(), source_type, normalized, digest),
        )
        chunk_count = self._persist_sections(source_id, [ExtractedSection(normalized, name.strip())])
        self.connection.commit()
        self._finish_indexing(source_id)
        return {"source_id": source_id, "name": name.strip(), "status": "indexed", "chunk_count": chunk_count}

    def add_uploaded_source(self, document: ExtractedDocument) -> dict[str, Any]:
        normalized = document.text.strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        duplicate = self.connection.execute(
            "SELECT source_id FROM knowledge_sources WHERE content_hash = ?", (digest,)
        ).fetchone()
        if duplicate:
            raise ValueError(f"Duplicate content already exists as {duplicate['source_id']}")
        source_id = f"src-{uuid.uuid4().hex[:10]}"
        self.connection.execute(
            "INSERT INTO knowledge_sources(source_id, name, source_type, content, content_hash, status) VALUES (?, ?, ?, ?, ?, 'indexing')",
            (source_id, document.filename, "file", normalized, digest),
        )
        self.connection.execute(
            """INSERT INTO source_files
            (source_id, original_filename, mime_type, byte_size, version, extraction_method)
            VALUES (?, ?, ?, ?, 1, ?)""",
            (source_id, document.filename, document.mime_type, document.byte_size, document.extraction_method),
        )
        chunk_count = self._persist_sections(source_id, document.sections)
        self.connection.commit()
        self._finish_indexing(source_id)
        return {
            "source_id": source_id, "name": document.filename, "status": "indexed",
            "chunk_count": chunk_count, "mime_type": document.mime_type,
            "byte_size": document.byte_size, "extraction_method": document.extraction_method,
        }

    def source_chunks(self, source_id: str) -> list[dict[str, Any]]:
        row = self.connection.execute(
            "SELECT source_id, name, content, source_type FROM knowledge_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if not row:
            raise KeyError(source_id)
        self._ensure_chunks(row)
        return [dict(chunk) for chunk in self.connection.execute(
            """SELECT chunk_id, parent_id, position, page_number, section_path,
            character_count AS characters, content FROM knowledge_chunks
            WHERE source_id = ? ORDER BY position""",
            (source_id,),
        )]

    def compare_retrieval(self, query: str, limit: int = 5) -> dict[str, Any]:
        documents = self._documents()
        retriever = HybridRetriever(documents)
        started = time.perf_counter()
        bm25_raw = retriever._bm25(query)[:limit]
        bm25_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        dense_matches = self.vector_store.search(self.namespace, query, documents, limit)
        by_id = {document["id"]: index for index, document in enumerate(documents)}
        semantic_raw = [(by_id[match.chunk_id], match.score) for match in dense_matches if match.chunk_id in by_id]
        semantic_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        bm25_ranks = {index: rank for rank, (index, _score) in enumerate(bm25_raw, 1)}
        semantic_ranks = {index: rank for rank, (index, _score) in enumerate(semantic_raw, 1)}
        candidates = set(bm25_ranks) | set(semantic_ranks)
        fused = sorted(
            candidates,
            key=lambda index: 0.45 / (60 + bm25_ranks.get(index, 10_000)) + 0.55 / (60 + semantic_ranks.get(index, 10_000)),
            reverse=True,
        )[:limit]
        hybrid_ms = (time.perf_counter() - started) * 1000

        def basic(items: list[tuple[int, float]]) -> list[dict[str, Any]]:
            return [
                {
                    "rank": rank,
                    "chunk_id": documents[index]["id"],
                    "source_id": documents[index]["source_id"],
                    "title": documents[index]["title"],
                    "content": documents[index]["content"],
                    "score": round(score, 6),
                }
                for rank, (index, score) in enumerate(items, 1)
            ]

        return {
            "query": query,
            "query_tokens": tokenize(query),
            "bm25": {"latency_ms": round(bm25_ms, 2), "results": basic(bm25_raw)},
            "semantic": {"latency_ms": round(semantic_ms, 2), "results": basic(semantic_raw)},
            "hybrid": {
                "latency_ms": round(hybrid_ms, 2),
                "weights": {"bm25": 0.45, "semantic": 0.55, "rrf_k": 60},
                "provider": self.vector_store.provider,
                "results": [
                    {
                        "rank": rank,
                        "chunk_id": documents[index]["id"],
                        "source_id": documents[index]["source_id"],
                        "title": documents[index]["title"],
                        "content": documents[index]["content"],
                        "score": round(0.45 / (60 + bm25_ranks.get(index, 10_000)) + 0.55 / (60 + semantic_ranks.get(index, 10_000)), 6),
                        "bm25_rank": bm25_ranks.get(index),
                        "semantic_rank": semantic_ranks.get(index),
                    }
                    for rank, index in enumerate(fused, 1)
                ],
            },
        }

    def retrieve(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Run the configured hybrid retriever only, for production latency tests."""
        started = time.perf_counter()
        comparison = self.compare_retrieval(query, limit)
        retrieval_ms = (time.perf_counter() - started) * 1000
        hybrid = comparison["hybrid"]
        return {
            "results": hybrid["results"],
            "retrieval_ms": round(retrieval_ms, 2),
            "provider": hybrid["provider"],
        }

    def _finish_indexing(self, source_id: str) -> None:
        try:
            self.connection.execute("UPDATE knowledge_sources SET status = 'indexed' WHERE source_id = ?", (source_id,))
            self.connection.commit()
            documents = [doc for doc in self._documents() if doc["source_id"] == source_id]
            self.vector_store.upsert(self.namespace, documents)
        except Exception:
            self.connection.rollback()
            self.connection.execute("UPDATE knowledge_sources SET status = 'failed' WHERE source_id = ?", (source_id,))
            self.connection.commit()
            raise

    def sync_vector_store(self) -> int:
        documents = self._documents()
        self.vector_store.upsert(self.namespace, documents)
        return len(documents)

    def list_guardrails(self) -> list[dict[str, Any]]:
        return [
            {**dict(row), "enabled": bool(row["enabled"]), "config": json.loads(row["config_json"])}
            for row in self.connection.execute("SELECT * FROM guardrail_policies ORDER BY policy_key")
        ]

    def update_guardrail(self, key: str, enabled: bool, config: dict[str, Any]) -> dict[str, Any]:
        found = self.connection.execute("SELECT 1 FROM guardrail_policies WHERE policy_key = ?", (key,)).fetchone()
        if not found:
            raise KeyError(key)
        self.connection.execute(
            "UPDATE guardrail_policies SET enabled = ?, config_json = ? WHERE policy_key = ?",
            (int(enabled), json.dumps(config), key),
        )
        self.connection.commit()
        return next(policy for policy in self.list_guardrails() if policy["policy_key"] == key)

    def _policy(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT enabled, config_json FROM guardrail_policies WHERE policy_key = ?", (key,)
        ).fetchone()
        return json.loads(row["config_json"]) if row and row["enabled"] else None

    def guarded_query(self, question: str, record_incident: bool = True) -> dict[str, Any]:
        events = []
        processed = question
        injection = self._policy("prompt_injection")
        if injection and any(pattern.search(question) for pattern in INJECTION_PATTERNS):
            events.append({"policy": "prompt_injection", "action": "block", "reason": "Instruction override pattern detected"})
            if record_incident:
                self._incident("high", "guardrail", question, "Prompt-injection policy blocked the request")
            return {"status": "blocked", "question": question, "processed_question": None, "events": events, "results": [], "context": []}

        pii = self._policy("pii_masking")
        if pii:
            for kind in pii.get("types", []):
                pattern = PII_PATTERNS.get(kind)
                if pattern and pattern.search(processed):
                    processed = pattern.sub(f"[{kind.upper()} REDACTED]", processed)
                    events.append({"policy": "pii_masking", "action": "mask", "reason": f"{kind} detected"})

        comparison = self.compare_retrieval(processed)
        results = comparison["hybrid"]["results"]
        threshold = self._policy("retrieval_threshold")
        if threshold and (not results or results[0]["score"] < threshold.get("minimum_rrf_score", 0)):
            events.append({"policy": "retrieval_threshold", "action": "abstain", "reason": "Evidence score below configured threshold"})
            if record_incident:
                self._incident("medium", "retrieval", question, "The assistant abstained because retrieval evidence was weak")
            return {"status": "abstained", "question": question, "processed_question": processed, "events": events, "results": results, "context": []}
        events.append({"policy": "citations_required", "action": "allow", "reason": f"{len(results)} cited chunks available"})
        context = self._pack_context(results)
        events.append({"policy": "context_packing", "action": "expand", "reason": f"Packed {len(context)} ordered chunks with neighbors"})
        return {"status": "answered", "question": question, "processed_question": processed, "events": events, "results": results, "context": context}

    def _pack_context(self, results: list[dict[str, Any]], character_budget: int = 6000) -> list[dict[str, Any]]:
        """Expand matched chunks with immediate neighbors, deduplicate, and restore source order."""
        selected: dict[str, dict[str, Any]] = {}
        matched_ids = {result["chunk_id"] for result in results[:3]}
        for result in results[:3]:
            row = self.connection.execute(
                "SELECT source_id, position FROM knowledge_chunks WHERE chunk_id = ?", (result["chunk_id"],)
            ).fetchone()
            if not row:
                continue
            neighbors = self.connection.execute(
                """SELECT c.*, s.name AS source_name FROM knowledge_chunks c
                JOIN knowledge_sources s ON s.source_id = c.source_id
                WHERE c.source_id = ? AND c.position BETWEEN ? AND ? ORDER BY c.position""",
                (row["source_id"], max(1, row["position"] - 1), row["position"] + 1),
            )
            for chunk in neighbors:
                item = dict(chunk)
                item["matched"] = chunk["chunk_id"] in matched_ids
                selected[chunk["chunk_id"]] = item
        ordered = sorted(selected.values(), key=lambda item: (item["source_id"], item["position"]))
        packed = []
        used = 0
        for item in ordered:
            if used + item["character_count"] > character_budget and packed:
                break
            packed.append(item)
            used += item["character_count"]
        return packed

    def _incident(self, severity: str, category: str, question: str, explanation: str) -> None:
        self.connection.execute(
            "INSERT INTO incidents(severity, category, question, explanation) VALUES (?, ?, ?, ?)",
            (severity, category, question, explanation),
        )
        self.connection.commit()

    def list_evaluations(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM evaluation_cases ORDER BY case_id")]

    def add_evaluation(self, question: str, expected_source_id: str | None, expected_behavior: str, tags: str) -> dict[str, Any]:
        self.connection.execute(
            "INSERT INTO evaluation_cases(question, expected_source_id, expected_behavior, tags) VALUES (?, ?, ?, ?)",
            (question, expected_source_id, expected_behavior, tags),
        )
        self.connection.commit()
        case_id = self._scalar("SELECT MAX(case_id) FROM evaluation_cases")
        return dict(self.connection.execute("SELECT * FROM evaluation_cases WHERE case_id = ?", (case_id,)).fetchone())

    def run_evaluations(self) -> dict[str, Any]:
        details = []
        retrieval_total = retrieval_hits = behavior_hits = 0
        for case in self.connection.execute("SELECT * FROM evaluation_cases ORDER BY case_id"):
            outcome = self.guarded_query(case["question"], record_incident=False)
            retrieved = [item["source_id"] for item in outcome["results"][:3]]
            retrieval_ok = None
            if case["expected_source_id"]:
                retrieval_total += 1
                retrieval_ok = case["expected_source_id"] in retrieved
                retrieval_hits += int(retrieval_ok)
            behavior_ok = outcome["status"] == ("blocked" if case["expected_behavior"] == "block" else "answered")
            behavior_hits += int(behavior_ok)
            details.append(
                {
                    "case_id": case["case_id"], "question": case["question"], "status": outcome["status"],
                    "retrieval_at_3": retrieval_ok, "behavior_pass": behavior_ok,
                }
            )
        return {
            "run_id": f"eval-{uuid.uuid4().hex[:8]}",
            "cases": len(details),
            "retrieval_recall_at_3": round(retrieval_hits / retrieval_total, 3) if retrieval_total else None,
            "guardrail_behavior_accuracy": round(behavior_hits / len(details), 3) if details else None,
            "details": details,
        }

    def list_incidents(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM incidents ORDER BY incident_id DESC LIMIT 100")]
