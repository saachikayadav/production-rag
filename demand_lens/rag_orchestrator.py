"""Bounded, observable agentic retrieval orchestration."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

from .retrieval import tokenize


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    modalities: tuple[str, ...]
    query_variants: tuple[str, ...]
    requires_structured_data: bool


class AgenticRAGOrchestrator:
    """Plans, retrieves, fuses, grades, and either returns evidence or abstains."""

    def __init__(self, studio):
        self.studio = studio

    def plan(self, question: str) -> RetrievalPlan:
        lower = question.lower()
        intent = "comparison" if any(term in lower for term in ("compare", "difference", "versus")) else "definition" if any(term in lower for term in ("what is", "define", "meaning")) else "fact"
        modalities = ["text"]
        if any(term in lower for term in ("image", "diagram", "screenshot", "figure")):
            modalities.append("image-caption")
        if any(term in lower for term in ("table", "row", "column")):
            modalities.append("table")
        tokens = [token for token in tokenize(question) if len(token) > 2]
        focused = " ".join(tokens)
        variants = [question.strip()]
        if focused and focused.lower() != question.strip().lower():
            variants.append(focused)
        if intent == "definition":
            variants.append(f"policy definition {focused}")
        elif intent == "comparison":
            variants.append(f"differences requirements {focused}")
        return RetrievalPlan(
            intent=intent,
            modalities=tuple(modalities),
            query_variants=tuple(dict.fromkeys(variants[:3])),
            requires_structured_data=any(term in lower for term in ("how many", "total", "average", "trend", "forecast")),
        )

    def run(self, question: str, limit: int = 5) -> dict[str, Any]:
        run_id = f"rag-{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        plan = self.plan(question)
        trace = [{"stage": "plan", "status": "ok", "detail": asdict(plan)}]
        ranked_lists = []
        for variant in plan.query_variants:
            comparison = self.studio.compare_retrieval(variant, limit=max(limit, 8))
            ranked_lists.append(comparison["hybrid"]["results"])
        trace.append({"stage": "multi_query_retrieval", "status": "ok", "detail": {"queries": len(ranked_lists)}})

        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}
        for results in ranked_lists:
            for rank, result in enumerate(results, 1):
                records[result["chunk_id"]] = result
                scores[result["chunk_id"]] = scores.get(result["chunk_id"], 0.0) + 1 / (60 + rank)
        fused_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        fused = [{**records[chunk_id], "multi_query_rrf_score": round(scores[chunk_id], 6)} for chunk_id in fused_ids]
        trace.append({"stage": "cross_query_fusion", "status": "ok", "detail": {"candidates": len(records), "selected": len(fused)}})

        source_diversity = len({result["source_id"] for result in fused})
        evidence_strength = min(1.0, (scores[fused_ids[0]] * 20) if fused_ids else 0.0)
        status = "answered" if fused and evidence_strength >= 0.2 else "abstained"
        trace.append({"stage": "evidence_grade", "status": status, "detail": {"strength": round(evidence_strength, 3), "source_diversity": source_diversity}})
        context = self.studio._pack_context(fused) if status == "answered" else []
        trace.append({"stage": "context_expand_and_pack", "status": "ok", "detail": {"chunks": len(context), "characters": sum(item["character_count"] for item in context)}})
        return {
            "run_id": run_id,
            "status": status,
            "plan": asdict(plan),
            "results": fused,
            "context": context,
            "trace": trace,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
