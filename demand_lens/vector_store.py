"""Dense retrieval providers with a deterministic local fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .retrieval import cosine, hashed_embedding


@dataclass(frozen=True)
class DenseMatch:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    provider: str

    def upsert(self, namespace: str, documents: list[dict[str, str]]) -> None: ...
    def search(self, namespace: str, query: str, documents: list[dict[str, str]], limit: int) -> list[DenseMatch]: ...


class LocalVectorStore:
    provider = "local-feature-hashing"

    def upsert(self, namespace: str, documents: list[dict[str, str]]) -> None:
        return None

    def search(self, namespace: str, query: str, documents: list[dict[str, str]], limit: int) -> list[DenseMatch]:
        vector = hashed_embedding(query)
        ranked = sorted(
            (DenseMatch(doc["id"], cosine(vector, hashed_embedding(doc["content"]))) for doc in documents),
            key=lambda match: match.score,
            reverse=True,
        )
        return ranked[:limit]


class PineconeVectorStore:
    provider = "pinecone-llama-text-embed-v2"

    def __init__(self, api_key: str, host: str, text_field: str = "chunk_text"):
        from pinecone import Pinecone

        self.index = Pinecone(api_key=api_key).Index(host=host)
        self.text_field = text_field

    def upsert(self, namespace: str, documents: list[dict[str, str]]) -> None:
        records = [
            {
                "_id": document["id"],
                self.text_field: document["content"],
                "source_id": document["source_id"],
                "title": document["title"],
                "section": document["section"],
            }
            for document in documents
        ]
        for start in range(0, len(records), 96):
            self.index.upsert_records(namespace, records[start : start + 96])

    def search(self, namespace: str, query: str, documents: list[dict[str, str]], limit: int) -> list[DenseMatch]:
        response = self.index.search(
            namespace=namespace,
            query={"inputs": {"text": query}, "top_k": limit},
            fields=["source_id", "title", "section"],
        )
        result = response.to_dict() if hasattr(response, "to_dict") else response
        hits = result.get("result", {}).get("hits", result.get("hits", []))
        return [DenseMatch(hit.get("_id") or hit.get("id"), float(hit.get("_score", hit.get("score", 0)))) for hit in hits]


def create_vector_store(settings) -> VectorStore:
    if settings.pinecone_api_key and settings.pinecone_index_host:
        return PineconeVectorStore(settings.pinecone_api_key, settings.pinecone_index_host, settings.pinecone_text_field)
    return LocalVectorStore()
