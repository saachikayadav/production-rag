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


def post_source_with_retries(
    base_url: str,
    name: str,
    content: str,
    api_key: str | None,
    retries: int,
    continue_on_conflict: bool,
) -> SeedResult | None:
    for attempt in range(1, retries + 2):
        try:
            return post_source(base_url, name, content, api_key)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            if status == 409 and continue_on_conflict:
                print(f"skipped duplicate: {name}")
                return None
            if status and status >= 500 and attempt <= retries + 1:
                wait_seconds = min(30, 2**attempt)
                print(
                    f"server error while seeding {name} "
                    f"(attempt {attempt}/{retries + 1}, status {status}); retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"failed to seed {name}: HTTP {status}: {detail}") from exc


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


def sync_vectors_with_retries(base_url: str, api_key: str | None, retries: int) -> dict | None:
    for attempt in range(1, retries + 2):
        try:
            return sync_vectors(base_url, api_key)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            if status and status >= 500 and attempt <= retries + 1:
                wait_seconds = min(60, 5 * attempt)
                print(
                    f"vector sync failed temporarily "
                    f"(attempt {attempt}/{retries + 1}, status {status}); retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue
            print(f"vector sync did not complete: HTTP {status}: {detail}")
            return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Groundwire URL, for example https://production-rag-1.onrender.com")
    parser.add_argument("--api-key", default=None, help="Optional bearer token if auth is enabled")
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--start-document", type=int, default=1, help="First synthetic document number to create")
    parser.add_argument("--sections-per-document", type=int, default=170)
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Pause between ingestions to avoid overwhelming the deployed service")
    parser.add_argument("--retries", type=int, default=3, help="Retries for temporary 5xx responses")
    parser.add_argument("--sync-retries", type=int, default=5, help="Retries for temporary vector sync failures")
    parser.add_argument("--stop-on-conflict", action="store_true", help="Fail instead of skipping duplicate synthetic documents")
    parser.add_argument("--skip-vector-sync", action="store_true")
    args = parser.parse_args()

    results: list[SeedResult] = []
    average_lengths: list[int] = []
    started = time.perf_counter()

    end_document = args.start_document + args.documents
    for document_number in range(args.start_document, end_document):
        content = build_document(document_number, args.sections_per_document)
        average_lengths.append(round(len(content) / args.sections_per_document))
        result = post_source_with_retries(
            args.base_url,
            f"Synthetic Demand Planning Corpus {document_number:03d}",
            content,
            args.api_key,
            args.retries,
            not args.stop_on_conflict,
        )
        if result is None:
            continue
        results.append(result)
        print(f"seeded {result.source_id}: {result.chunk_count} chunks")
        if args.delay_seconds:
            time.sleep(args.delay_seconds)

    vector_sync = None
    if not args.skip_vector_sync:
        vector_sync = sync_vectors_with_retries(args.base_url, args.api_key, args.sync_retries)

    elapsed = time.perf_counter() - started
    print("\nCorpus summary")
    print(f"documents_seeded_this_run: {len(results)}")
    print(f"chunks: {sum(result.chunk_count for result in results)}")
    if average_lengths:
        print(f"average synthetic section length: {round(statistics.mean(average_lengths))} characters")
    print(f"elapsed_seconds: {round(elapsed, 2)}")
    if vector_sync:
        print(f"vector_provider: {vector_sync.get('provider')}")
        print(f"indexed_chunks: {vector_sync.get('indexed_chunks')}")


if __name__ == "__main__":
    main()
