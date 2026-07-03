from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    thread_id: str = Field(default="default")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


class DemandQuestion(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class DemandAnswer(BaseModel):
    question: str
    answer: str
    route: str
    sql: dict | None = None
    citations: list[dict] = Field(default_factory=list)
    warning: str | None = None


class SourceCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    content: str = Field(..., min_length=20, max_length=200_000)
    source_type: str = Field(default="text", pattern="^(text|markdown|policy|url)$")


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    limit: int = Field(default=5, ge=1, le=10)


class GuardrailUpdate(BaseModel):
    enabled: bool
    config: dict = Field(default_factory=dict)


class EvaluationCreate(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    expected_source_id: str | None = None
    expected_behavior: str = Field(default="answer", pattern="^(answer|block)$")
    tags: str = ""


class StudioQuery(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
