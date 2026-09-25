"""
api/app.py
==========
FastAPI Application — Production Entry Point

Startup sequence (runs once per container boot on Render):
  1. Download missing model artefacts from HuggingFace Hub
  2. Initialise Neon.tech Postgres tables (idempotent CREATE IF NOT EXISTS)
  3. Verify Chroma vector-store connectivity
  4. Register routes, middleware, exception handlers

Environment variables required for full cloud mode:
  GROQ_API_KEY             — primary LLM
  DATABASE_URL             — Neon.tech postgres connection string
  HUGGINGFACE_REPO_ID      — e.g. "yourname/loan-decision-assets"
  HUGGINGFACE_API_TOKEN    — HF access token (private repos)
  CHROMA_CLOUD_API_KEY     — Chroma Cloud API key
  CHROMA_CLOUD_TENANT      — Chroma Cloud tenant ID
  CORS_ORIGINS             — comma-separated allowed origins
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging — configure before anything else so startup messages are visible
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — runs at container start and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ordered startup:
      1. Download models from HuggingFace Hub (skips cached files)
      2. CREATE tables in Neon Postgres (idempotent)
      3. Verify Chroma Cloud / local vector-store
    Shutdown: graceful, nothing to flush.
    """

    # ── Step 1: Model artefacts ──────────────────────────────────────────
    logger.info("━━━ [1/3] Model artefacts ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        from utils.model_loader import download_models, check_model_health
        result = download_models()        # no-op if files already cached
        health = check_model_health()
        if result["success"]:
            logger.info("✅ Models ready  (repo=%s  downloaded=%d  cached=%d)",
                        result["repo"], len(result["downloaded"]), len(result["skipped"]))
        else:
            logger.error("❌ Some required models missing: %s", result["failed"])
        app.state.model_health = health
    except Exception as exc:
        logger.error("Model loader failed: %s", exc)
        app.state.model_health = {"status": "error", "error": str(exc)}

    # ── Step 2: Neon Postgres ────────────────────────────────────────────
    logger.info("━━━ [2/3] Database (Neon) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        from utils.db_utils import init_db
        init_db()                         # CREATE TABLE IF NOT EXISTS
        db_status = "healthy"
        logger.info("✅ Neon tables initialised")
    except Exception as exc:
        db_status = f"error: {exc}"
        logger.error("❌ DB init failed: %s", exc)
    app.state.db_status = db_status

    # ── Step 3: Chroma vector-store ──────────────────────────────────────
    logger.info("━━━ [3/3] Vector store (Chroma) ━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        from agents.rag_agent import check_rag_health
        rag_health = check_rag_health()
        logger.info("✅ Chroma %s  mode=%s  docs=%s",
                    rag_health["status"],
                    rag_health.get("mode", "?"),
                    rag_health.get("document_count", "?"))
    except Exception as exc:
        rag_health = {"status": "error", "error": str(exc)}
        logger.error("❌ RAG health check failed: %s", exc)
    app.state.rag_health = rag_health

    logger.info("🚀 API ready  env=%s", os.getenv("ENVIRONMENT", "development"))
    yield

    # Shutdown
    logger.info("🛑 Shutting down Agentic Loan Decision API")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _parse_cors_origins() -> list:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]

def _parse_allowed_hosts() -> list:
    raw = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1")
    return [h.strip() for h in raw.split(",") if h.strip()]

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

app = FastAPI(
    title       = "Agentic Loan Decision API",
    description = (
        "Production-grade loan approval system with AI agents, "
        "RBI regulatory compliance, and explainable decisions.\n\n"
        "Backend: Render · DB: Neon Postgres · VectorDB: Chroma Cloud · "
        "LLM: Groq (Llama 3.1) · Models: HuggingFace Hub"
    ),
    version     = "2.0.0",
    docs_url    = None if IS_PRODUCTION else "/docs",
    redoc_url   = None if IS_PRODUCTION else "/redoc",
    lifespan    = lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = _parse_cors_origins(),
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

# ── Trusted hosts (production only) ─────────────────────────────────────────
if IS_PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=_parse_allowed_hosts(),
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
from api.routes.evaluate import router as evaluate_router   # noqa: E402
app.include_router(evaluate_router, prefix="/api/v1", tags=["Loan Evaluation"])

# ---------------------------------------------------------------------------
# Health & status endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
async def root():
    return {
        "service":       "Agentic Loan Decision API",
        "version":       "2.0.0",
        "environment":   os.getenv("ENVIRONMENT", "development"),
        "docs":          "/docs",
        "health":        "/health",
    }


@app.get("/health", tags=["Meta"])
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Comprehensive health check used by Render's health-check probe,
    Streamlit dashboard, and CI smoke tests.

    Returns HTTP 200 if all required components are healthy.
    Returns HTTP 503 if any required component is degraded.
    """
    # Models
    model_health = getattr(request.app.state, "model_health", {"status": "unknown"})

    # Database
    db_status = getattr(request.app.state, "db_status", "unknown")
    try:
        from utils.db_utils import SessionLocal, get_decision_stats
        db = SessionLocal()
        stats = get_decision_stats(db)
        db.close()
        db_health = {"status": "healthy", "stats": stats}
    except Exception as exc:
        db_health = {"status": "error", "error": str(exc)}

    # Vector store
    rag_health = getattr(request.app.state, "rag_health", {"status": "unknown"})

    # LLM
    try:
        from utils.llm_utility_cloud import check_llm_health
        llm_health = check_llm_health()
    except Exception as exc:
        llm_health = {"status": "error", "error": str(exc)}

    components = {
        "models":      model_health,
        "database":    db_health,
        "vector_store": rag_health,
        "llm":         llm_health,
    }

    # Overall: healthy only if every required component is healthy
    required_ok = all(
        components[c].get("status") == "healthy"
        for c in ("models", "database", "llm")
    )
    overall = "healthy" if required_ok else "degraded"

    payload = {
        "status":      overall,
        "version":     "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "components":  components,
    }
    return JSONResponse(content=payload, status_code=200 if required_ok else 503)


@app.get("/status", tags=["Meta"])
async def detailed_status() -> Dict[str, Any]:
    """Deployment topology snapshot — useful for debugging."""
    return {
        "environment":       os.getenv("ENVIRONMENT", "development"),
        "llm_provider":      "groq" if os.getenv("GROQ_API_KEY") else "mock",
        "groq_model":        os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "database_backend":  "neon" if (os.getenv("DATABASE_URL", "").startswith("postgresql")) else "sqlite",
        "vector_db_backend": "cloud" if os.getenv("CHROMA_CLOUD_API_KEY") else "local",
        "model_storage":     "huggingface" if os.getenv("HUGGINGFACE_REPO_ID") else "local",
        "cors_origins":      _parse_cors_origins(),
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    if not IS_PRODUCTION:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__, "path": str(request.url)},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host      = "0.0.0.0",
        port      = int(os.getenv("PORT", 8000)),
        reload    = not IS_PRODUCTION,
        log_level = os.getenv("LOG_LEVEL", "info").lower(),
    )
