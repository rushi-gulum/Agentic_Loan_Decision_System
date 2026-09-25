"""
api/routes/evaluate.py
======================
POST /api/v1/evaluate  — core loan evaluation endpoint

Changes from v1:
  • Depends(get_db)  injects a Neon Postgres session into every request
  • Every evaluation is persisted to `loan_decisions` via log_decision()
  • Normalised violations written to `compliance_violations`
  • application_id generated once; echoed in response metadata
  • Structured error responses with request-id tracing
"""

import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from api.schemas import (
    LoanApplicationRequest,
    FinalDecisionResponse,
    ErrorResponse,
    create_error_response,
)
from utils.db_utils import get_db, log_decision

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Orchestrator — singleton, initialised lazily on first request
# ---------------------------------------------------------------------------

_orchestrator = None

def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import LoanDecisionOrchestrator
        _orchestrator = LoanDecisionOrchestrator()
        logger.info("✅ LoanDecisionOrchestrator initialised")
    return _orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def _extract_violations(result: dict) -> list:
    """Pull normalised violation dicts from the orchestrator result."""
    violations = []
    compliance = result.get("compliance_result", {})

    # Hard violations (from HardConstraintResult)
    for v in compliance.get("violations", []):
        if isinstance(v, dict):
            violations.append(v)
        elif hasattr(v, "__dict__"):
            violations.append(v.__dict__)

    # Soft violations
    for v in compliance.get("soft_violations", []):
        if isinstance(v, dict):
            violations.append(v)
        elif hasattr(v, "__dict__"):
            violations.append(v.__dict__)

    return violations


def _build_db_payload(
    application_id: str,
    request_data: dict,
    result: dict,
    llm_provider: str,
) -> dict:
    """Extract scalar fields from the orchestrator result for the DB row."""
    decision_str  = str(result.get("decision", "UNKNOWN")).upper()
    confidence    = _safe_float(result.get("confidence_score", result.get("confidence", 0)))
    model_used    = str(result.get("selected_model", result.get("model_used", "unknown")))

    risk_block    = result.get("risk_assessment", {})
    risk_score    = _safe_float(risk_block.get("risk_score_10", risk_block.get("risk_score", 0)))
    risk_grade    = str(risk_block.get("grade", risk_block.get("risk_grade", "UNKNOWN")))

    comp_block        = result.get("compliance_result", {})
    is_compliant      = bool(comp_block.get("is_compliant", False))
    compliance_score  = _safe_float(comp_block.get("compliance_score", 0))
    hard_violations   = len([
        v for v in comp_block.get("violations", [])
        if isinstance(v, dict) and v.get("severity") == "violation"
    ])

    return dict(
        application_id   = application_id,
        decision         = decision_str,
        confidence       = confidence,
        model_used       = model_used,
        risk_score       = risk_score,
        risk_grade       = risk_grade,
        compliance_status= "PASS" if is_compliant else "FAIL",
        compliance_score = compliance_score,
        hard_violations  = hard_violations,
        loan_type        = str(request_data.get("loan_type", "unknown")),
        requested_amount = _safe_float(request_data.get("requested_amount_inr", 0)),
        bureau_score     = _safe_int(request_data.get("bureau_score", 0)),
        foir_pct         = _safe_float(request_data.get("foir_total_obligations_pct", 0)),
        raw_request      = request_data,
        raw_response     = result,
        llm_provider     = llm_provider,
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/evaluate",
    response_model     = FinalDecisionResponse,
    status_code        = status.HTTP_200_OK,
    summary            = "Evaluate a loan application",
    description        = (
        "Runs the full agentic pipeline:\n"
        "1. Unified preprocessing\n"
        "2. Hard-constraint guardrail (short-circuit on violations)\n"
        "3. RAG-powered soft compliance check\n"
        "4. Risk scoring\n"
        "5. Final decision + SHAP/LIME explanations\n\n"
        "Every evaluation is persisted to Neon Postgres for audit."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
)
async def evaluate_loan_application(
    request_body : LoanApplicationRequest,
    request      : Request,
    db           : Session = Depends(get_db),
) -> FinalDecisionResponse:
    """
    Evaluate a loan application and persist the result to Neon Postgres.

    The DB write is fire-and-forget inside a try/except so a DB hiccup
    never blocks the applicant from getting their decision.
    """

    # ── 1. Generate a stable application ID ──────────────────────────────
    application_id = f"LOAN-{uuid.uuid4().hex[:12].upper()}"
    logger.info("▶ Evaluating  application_id=%s  loan_type=%s  bureau=%s",
                application_id,
                request_body.loan_type,
                request_body.bureau_score)

    # ── 2. Convert Pydantic model → plain dict for agents ─────────────────
    application_data = request_body.model_dump()
    application_data["application_id"] = application_id   # propagate to orchestrator

    # ── 3. Determine active LLM provider for telemetry ────────────────────
    import os
    llm_provider = (
        "groq"      if os.getenv("GROQ_API_KEY")      else
        "openai"    if os.getenv("OPENAI_API_KEY")     else
        "anthropic" if os.getenv("ANTHROPIC_API_KEY")  else
        "mock"
    )

    # ── 4. Run orchestrator (CPU-bound — offload to thread pool) ──────────
    try:
        orchestrator = _get_orchestrator()
        loop   = asyncio.get_event_loop()
        result: dict = await loop.run_in_executor(
            None,
            orchestrator.evaluate_application,
            application_data,
        )
    except ValueError as exc:
        logger.warning("Validation error for %s: %s", application_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid application data: {exc}",
        )
    except Exception as exc:
        logger.error("Pipeline error for %s: %s", application_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation pipeline failed: {exc}",
        )

    # ── 5. Persist to Neon Postgres (non-blocking best-effort) ────────────
    try:
        payload    = _build_db_payload(application_id, application_data, result, llm_provider)
        violations = _extract_violations(result)
        log_entry  = log_decision(
            db         = db,
            violations = violations,
            processing_ms = _safe_int(result.get("processing_time_ms", 0)),
            **payload,
        )
        logger.info("✅ Persisted  application_id=%s  db_row_id=%s  decision=%s",
                    application_id, log_entry.id, payload["decision"])
    except Exception as db_exc:
        # DB failure must NOT break the response — log and continue
        logger.error("⚠️  DB persist failed for %s: %s (decision still returned)",
                     application_id, db_exc)

    # ── 6. Inject application_id into metadata before serialising ─────────
    if isinstance(result.get("metadata"), dict):
        result["metadata"]["application_id"] = application_id
    elif hasattr(result.get("metadata"), "__dict__"):
        result["metadata"].application_id = application_id

    logger.info("◀ Completed  application_id=%s  decision=%s  confidence=%.2f",
                application_id,
                result.get("decision", "?"),
                _safe_float(result.get("confidence_score", result.get("confidence", 0))))

    # ── 7. Serialise and return ────────────────────────────────────────────
    try:
        return FinalDecisionResponse(**result)
    except Exception as exc:
        logger.error("Response serialisation failed for %s: %s", application_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Decision completed but response formatting failed.",
        )


# ---------------------------------------------------------------------------
# Decision history endpoint (Streamlit dashboard / audit panel)
# ---------------------------------------------------------------------------

@router.get(
    "/history",
    summary     = "Recent loan decisions",
    description = "Returns the 50 most recent decisions from Neon Postgres.",
    tags        = ["Audit"],
)
async def get_decision_history(
    limit : int     = 50,
    db    : Session = Depends(get_db),
) -> Dict[str, Any]:
    """Paginated audit log for the Streamlit dashboard."""
    from utils.db_utils import LoanDecisionLog
    from sqlalchemy import desc

    try:
        rows = (
            db.query(LoanDecisionLog)
            .order_by(desc(LoanDecisionLog.timestamp))
            .limit(min(limit, 200))
            .all()
        )
        records = [
            {
                "id":                row.id,
                "application_id":   row.application_id,
                "timestamp":        row.timestamp.isoformat() if row.timestamp else None,
                "decision":         row.decision,
                "confidence":       row.confidence,
                "risk_score":       row.risk_score,
                "risk_grade":       row.risk_grade,
                "compliance_status": row.compliance_status,
                "loan_type":        row.loan_type,
                "requested_amount": row.requested_amount,
                "bureau_score":     row.bureau_score,
                "foir_pct":         row.foir_pct,
                "llm_provider":     row.llm_provider,
                "processing_ms":    row.processing_ms,
            }
            for row in rows
        ]
        return {"count": len(records), "decisions": records}

    except Exception as exc:
        logger.error("History query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Decision history unavailable.",
        )


# ---------------------------------------------------------------------------
# Aggregate stats endpoint (Streamlit dashboard)
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    summary     = "Aggregate decision statistics",
    description = "Approval rates, average risk scores, and counts from Neon.",
    tags        = ["Audit"],
)
async def get_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Live aggregate stats from the Neon Postgres audit table."""
    from utils.db_utils import get_decision_stats
    try:
        return get_decision_stats(db)
    except Exception as exc:
        logger.error("Stats query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Statistics unavailable.",
        )
