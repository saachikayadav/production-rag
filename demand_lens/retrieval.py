"""Dependency-light BM25 + semantic retrieval with explicit weighted RRF."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hashed_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Deterministic local feature hashing for an offline semantic-search baseline."""
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        vector[raw % dimensions] += 1.0 if raw & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class RetrievalResult:
    document: dict[str, str]
    fused_score: float
    bm25_rank: int | None
    semantic_rank: int | None
    bm25_score: float
    semantic_score: float


class HybridRetriever:
    def __init__(self, documents: list[dict[str, str]], rrf_k: int = 60):
        self.documents = documents
        self.rrf_k = rrf_k
        self.tokens = [tokenize(doc["content"] + " " + doc["title"]) for doc in documents]
        self.embeddings: list[list[float]] | None = None
        self.doc_freq = Counter(token for tokens in self.tokens for token in set(tokens))
        self.avg_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = []
        total = len(self.documents)
        for index, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                df = self.doc_freq[token]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                tf = frequencies[token]
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / self.avg_length)
                score += idf * (tf * 2.5 / denominator) if denominator else 0
            scores.append((index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _semantic(self, query: str) -> list[tuple[int, float]]:
        if self.embeddings is None:
            self.embeddings = [hashed_embedding(doc["content"] + " " + doc["title"]) for doc in self.documents]
        query_vector = hashed_embedding(query)
        return sorted(
            [(index, cosine(query_vector, vector)) for index, vector in enumerate(self.embeddings)],
            key=lambda item: item[1],
            reverse=True,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        bm25_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> list[RetrievalResult]:
        bm25 = self._bm25(query)
        semantic = self._semantic(query)
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25, 1)}
        semantic_ranks = {index: rank for rank, (index, _) in enumerate(semantic, 1)}
        bm25_scores = dict(bm25)
        semantic_scores = dict(semantic)
        candidates = set(index for index, _ in bm25[:limit * 3]) | set(
            index for index, _ in semantic[:limit * 3]
        )
        results = []
        for index in candidates:
            score = (
                bm25_weight / (self.rrf_k + bm25_ranks[index])
                + semantic_weight / (self.rrf_k + semantic_ranks[index])
            )
            results.append(
                RetrievalResult(
                    document=self.documents[index],
                    fused_score=score,
                    bm25_rank=bm25_ranks.get(index),
                    semantic_rank=semantic_ranks.get(index),
                    bm25_score=bm25_scores.get(index, 0.0),
                    semantic_score=semantic_scores.get(index, 0.0),
                )
            )
        return sorted(results, key=lambda item: item.fused_score, reverse=True)[:limit]
