"""Seed Groundwire with a synthetic demand-planning corpus for load tests.

This script creates public-demo-safe documents through the normal source ingestion
API. After seeding, call /api/studio/vector/sync so the chunks are reconciled
with the configured vector store, including Pinecone in production.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import requests


TOPICS = [
    "forecast bias",
    "WAPE",
    "inventory shortage risk",
    "promotion uplift",
    "supplier lead time",
    "regional demand spikes",
    "safety stock",
    "launch planning",
    "constrained allocation",
    "SQL governance",
]


@dataclass
class SeedResult:
    source_id: str
    chunk_count: int


def synthetic_paragraph(document_number: int, section_number: int, topic: str) -> str:
    policy_id = f"GW-DP-{document_number:03d}-{section_number:03d}"
    return (
        f"{policy_id}: This synthetic demand-planning guidance explains {topic} for a "
        "high-volume manufacturing planning team. Planners should compare the current "
        "weekly signal against the rolling baseline, inspect regional variance, and "
        "record the reason code before changing a forecast. When the evidence is weak, "
        "the assistant should cite the source, avoid unsupported claims, and escalate "
        "the decision to a human owner. This paragraph is generated test data created "
        "for Groundwire retrieval benchmarking."
    )


def build_document(document_number: int, sections_per_document: int) -> str:
    sections = []
    for section_number in range(1, sections_per_document + 1):
        topic = TOPICS[(document_number + section_number) % len(TOPICS)]
        sections.append(
            f"## Section {section_number}: {topic.title()}\n\n"
            f"{synthetic_paragraph(document_number, section_number, topic)}"
        )
    return "\n\n".join(sections)


def post_source(base_url: str, name: str, content: str, api_key: str | None) -> SeedResult:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{base_url.rstrip('/')}/api/studio/sources",
        json={"name": name, "content": content, "source_type": "markdown"},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return SeedResult(payload["source_id"], payload["chunk_count"])


def sync_vectors(base_url: str, api_key: str | None) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{base_url.rstrip('/')}/api/studio/vector/sync",
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Groundwire URL, for example https://production-rag-1.onrender.com")
    parser.add_argument("--api-key", default=None, help="Optional bearer token if auth is enabled")
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--sections-per-document", type=int, default=170)
    parser.add_argument("--skip-vector-sync", action="store_true")
    args = parser.parse_args()

    results: list[SeedResult] = []
    average_lengths: list[int] = []
    started = time.perf_counter()

    for document_number in range(1, args.documents + 1):
        content = build_document(document_number, args.sections_per_document)
        average_lengths.append(round(len(content) / args.sections_per_document))
        result = post_source(
            args.base_url,
            f"Synthetic Demand Planning Corpus {document_number:03d}",
            content,
            args.api_key,
        )
        results.append(result)
        print(f"seeded {result.source_id}: {result.chunk_count} chunks")

    vector_sync = None
    if not args.skip_vector_sync:
        vector_sync = sync_vectors(args.base_url, args.api_key)

    elapsed = time.perf_counter() - started
    print("\nCorpus summary")
    print(f"documents: {len(results)}")
    print(f"chunks: {sum(result.chunk_count for result in results)}")
    print(f"average synthetic section length: {round(statistics.mean(average_lengths))} characters")
    print(f"elapsed_seconds: {round(elapsed, 2)}")
    if vector_sync:
        print(f"vector_provider: {vector_sync.get('provider')}")
        print(f"indexed_chunks: {vector_sync.get('indexed_chunks')}")


if __name__ == "__main__":
    main()
