import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from config import get_settings
from models import ChatRequest, ChatResponse, DemandAnswer, DemandQuestion, HealthResponse, MetricsResponse
from security_patterns import SecurePipeline
from cache import ResponseCache
from monitoring import get_logger, MetricsCollector, RequestTimer
from agent import ProductionAgent
from demand_lens import DemandLensService
from demand_lens.database import connect

load_dotenv()

security = None
cache = None
metrics = None
agent = None
demand_lens = None
logger = get_logger()

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global security, cache, metrics, agent, demand_lens

    settings = get_settings()

    logger.info("Starting production API...")

    security = SecurePipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()
    demand_lens = DemandLensService(connect())

    logger.info("All components initialized. Ready to serve requests.")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Production LangGraph API",
    description="Production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""

    return cache.stats

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc)},
    )

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("frontend.html")


@app.get("/api/examples")
def demand_examples():
    return {
        "examples": [
            "Which region has the largest forecast miss for Model Beta, and how should WAPE be interpreted?",
            "Which products are at inventory shortage risk?",
            "Show forecast bias by product and region.",
            "What controls are required for AI-generated SQL?",
        ]
    }


@app.post("/api/ask", response_model=DemandAnswer)
@limiter.limit(get_settings().rate_limit)
async def demand_question(request: Request, body: DemandQuestion):
    try:
        return DemandAnswer(**demand_lens.ask(body.question))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""

    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
        "demand_lens": demand_lens is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards."""

    summary = metrics.get_summary()
    return MetricsResponse(**summary)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    with RequestTimer() as timer:
        security_notes = []

        cleaned_message, input_warnings = security.check_input(body.message)
        security_notes.extend(input_warnings)

        if cleaned_message is None:
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        cached_response = cache.get(cleaned_message)

        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]

        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        cache.set(cleaned_message, validated_response)

        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)

        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        if security_notes:
            logger.info(
                "Security notes",
                extra={
                    "extra_data": {
                        "notes": security_notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

        logger.info(
            "Request completed",
            extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                    "model_used": model_used,
                    "latency_ms": round(timer.elapsed_ms, 2),
                }
            },
        )

        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
        )
