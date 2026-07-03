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
