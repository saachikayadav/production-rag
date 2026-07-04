from demand_lens.database import connect, initialize
from demand_lens.rag_orchestrator import AgenticRAGOrchestrator
from demand_lens.studio import KnowledgeOpsStudio


def make_orchestrator():
    connection = connect()
    initialize(connection)
    return AgenticRAGOrchestrator(KnowledgeOpsStudio(connection))


def test_planner_detects_modalities_and_structured_questions():
    plan = make_orchestrator().plan("Compare the forecast table with the policy diagram and show the total")
    assert plan.intent == "comparison"
    assert "table" in plan.modalities
    assert "image-caption" in plan.modalities
    assert plan.requires_structured_data is True


def test_agentic_retrieval_emits_observable_bounded_trace():
    result = make_orchestrator().run("What is forecast bias?")
    assert result["status"] == "answered"
    assert result["results"][0]["source_id"] == "forecast-metrics-1"
    assert [event["stage"] for event in result["trace"]] == [
        "plan", "multi_query_retrieval", "cross_query_fusion", "evidence_grade", "context_expand_and_pack"
    ]
    assert result["context"]
