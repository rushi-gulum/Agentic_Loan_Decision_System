"""
api/routes/evaluate.py
======================
POST /api/v1/evaluate  — core loan evaluation endpoint

Key responsibility: bridge the orchestrator's flat response dict into the
strict FinalDecisionResponse Pydantic schema via _normalize_to_schema().
Every evaluation is also persisted to Neon Postgres for audit.
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime
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

# ── Orchestrator singleton ────────────────────────────────────────────────

_orchestrator = None

def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import LoanDecisionOrchestrator
        _orchestrator = LoanDecisionOrchestrator()
        logger.info("✅ LoanDecisionOrchestrator initialised")
    return _orchestrator


# ── Schema normalisation ──────────────────────────────────────────────────
# Maps the orchestrator's internal dict into FinalDecisionResponse's exact shape.
# This isolates schema-breaking changes in the orchestrator from the API contract.

_VALID_DECISIONS = {"APPROVED", "REJECTED", "ESCALATED", "REVIEW_REQUIRED"}
_VALID_GRADES    = {"A+", "A", "B", "C", "D", "E"}


def _normalize_to_schema(raw: Dict[str, Any], application_id: str) -> Dict[str, Any]:
    """Bridge orchestrator dict → FinalDecisionResponse-compatible dict."""

    # decision (enum guard)
    decision = str(raw.get("decision", "REJECTED")).upper()
    if decision not in _VALID_DECISIONS:
        decision = "REJECTED"

    # risk_assessment (grade enum guard)
    ra    = raw.get("risk_assessment", {})
    grade = str(ra.get("grade", "E")).upper()
    if grade not in _VALID_GRADES:
        grade = "E"

    risk_assessment = {
        "risk_score_10": float(ra.get("risk_score_10", 10.0)),
        "grade":         grade,
        "components":    ra.get("components", {}),
        "drivers":       ra.get("drivers",    []),
        "context":       ra.get("context",    {}),
        "reasons":       ra.get("reasons",    []),
    }

    # compliance_result (fill required fields the orchestrator may omit)
    cr        = raw.get("compliance_result", {})
    loan_type = str(raw.get("loan_type", "") or cr.get("loan_type", "") or "unknown")

    compliance_result = {
        "is_compliant":        bool(cr.get("is_compliant", False)),
        "compliance_score":    float(cr.get("compliance_score", 0.0)),
        "hard_violations":     cr.get("hard_violations",     []),
        "hard_warnings":       cr.get("hard_warnings",       []),  # required by schema
        "soft_violations":     cr.get("soft_violations",     []),
        "explanation":         str(cr.get("explanation",     "")),
        "loan_type":           loan_type,                          # required by schema
        "rules_applied":       cr.get("rules_applied",       []),
        "rag_guidelines_used": cr.get("rag_guidelines_used", []),  # required by schema
    }

    # xai_report (Optional — map orchestrator "explanations" dict into XAIReport shape)
    exp = raw.get("explanations", {})
    xai_report = {
        "model_used":              raw.get("selected_model", "none"),
        "prediction_probability":  {
            "approved": float(raw.get("approval_probability", 0.0)),
            "rejected": round(1.0 - float(raw.get("approval_probability", 0.0)), 4),
        },
        "shap_summary":            exp.get("raw_data", {}),
        "lime_explanation":        [],
        "customer_summary":        str(exp.get("customer_explanation",  "")),
        "regulator_summary":       str(exp.get("technical_explanation", "")),
        "top_features":            [],
    } if exp else None

    # metadata (DecisionMetadata — fill all required fields)
    processing_ms = float(raw.get("processing_time_ms", 0))
    metadata = {
        "processing_time_ms": processing_ms,
        "model_version":      "2.0.0",
        "decision_id":        application_id,  # required, no schema default
        "short_circuited":    bool(raw.get("metadata", {}).get("short_circuit", False)),
    }

    return {
        "decision":         decision,
        "decision_reason":  str(raw.get("decision_reason", "")),
        "confidence_score": float(raw.get("confidence_score", 0.0)),
        "risk_assessment":  risk_assessment,
        "compliance_result":compliance_result,
        "xai_report":       xai_report,
        "metadata":         metadata,
        "loan_eligible":    bool(raw.get("loan_eligible", False)),
        "selected_model":   str(raw.get("selected_model", "none")),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0):
    try:    return float(v) if v is not None else default
    except: return default

def _safe_int(v, default=0):
    try:    return int(v) if v is not None else default
    except: return default

def _extract_violations(result: dict) -> list:
    out = []
    for v in result.get("compliance_result", {}).get("hard_violations", []):
        out.append(v if isinstance(v, dict) else vars(v))
    for v in result.get("compliance_result", {}).get("soft_violations", []):
        out.append(v if isinstance(v, dict) else vars(v))
    return out

def _build_db_payload(app_id, req_data, result, llm_provider):
    ra = result.get("risk_assessment", {})
    cr = result.get("compliance_result", {})
    return dict(
        application_id   = app_id,
        decision         = str(result.get("decision", "UNKNOWN")).upper(),
        confidence       = _safe_float(result.get("confidence_score")),
        model_used       = str(result.get("selected_model", "unknown")),
        risk_score       = _safe_float(ra.get("risk_score_10")),
        risk_grade       = str(ra.get("grade", "UNKNOWN")),
        compliance_status= "PASS" if cr.get("is_compliant") else "FAIL",
        compliance_score = _safe_float(cr.get("compliance_score")),
        hard_violations  = len(cr.get("hard_violations", [])),
        loan_type        = str(req_data.get("loan_type", "unknown")),
        requested_amount = _safe_float(req_data.get("requested_amount_inr")),
        bureau_score     = _safe_int(req_data.get("bureau_score")),
        foir_pct         = _safe_float(req_data.get("foir_total_obligations_pct")),
        raw_request      = req_data,
        raw_response     = result,
        llm_provider     = llm_provider,
    )


# ── Main endpoint ─────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model  = FinalDecisionResponse,
    status_code     = status.HTTP_200_OK,
    summary         = "Evaluate a loan application",
    description     = (
        "Runs the full agentic pipeline:\n"
        "1. Preprocessing\n"
        "2. Hard-constraint guardrail (short-circuit on violations)\n"
        "3. RAG soft compliance (Groq + Chroma Cloud)\n"
        "4. Risk scoring\n"
        "5. Decision + SHAP/LIME explanations\n\n"
        "Every evaluation is persisted to Neon Postgres for audit."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
)
async def evaluate_loan_application(
    request_body: LoanApplicationRequest,
    request:      Request,
    db:           Session = Depends(get_db),
) -> FinalDecisionResponse:

    application_id = f"LOAN-{uuid.uuid4().hex[:12].upper()}"
    logger.info("▶ [%s] loan_type=%s bureau=%s", application_id,
                request_body.loan_type, request_body.bureau_score)

    application_data = request_body.model_dump()
    application_data["application_id"] = application_id

    llm_provider = (
        "groq"   if os.getenv("GROQ_API_KEY")  else
        "openai" if os.getenv("OPENAI_API_KEY") else
        "mock"
    )

    # ── 1. Run orchestrator ───────────────────────────────────────────────
    try:
        orch = _get_orchestrator()
        loop = asyncio.get_event_loop()
        raw: dict = await loop.run_in_executor(
            None, orch.evaluate_application, application_data
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid application data: {exc}")
    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", application_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline failed: {exc}")

    # ── 2. Normalise → FinalDecisionResponse ─────────────────────────────
    try:
        normalised = _normalize_to_schema(raw, application_id)
        response   = FinalDecisionResponse(**normalised)
    except Exception as exc:
        logger.error("Schema normalisation failed [%s]: %s  raw_keys=%s",
                     application_id, exc, list(raw.keys()))
        raise HTTPException(
            status_code=500,
            detail=f"Response formatting failed: {exc}",
        )

    # ── 3. Persist to Neon (best-effort, never blocks response) ──────────
    try:
        payload   = _build_db_payload(application_id, application_data, raw, llm_provider)
        log_entry = log_decision(
            db=db,
            violations=_extract_violations(raw),
            processing_ms=_safe_int(raw.get("processing_time_ms", 0)),
            **payload,
        )
        logger.info("✅ Persisted [%s] row=%s decision=%s",
                    application_id, log_entry.id, payload["decision"])
    except Exception as db_exc:
        logger.error("⚠️  DB persist failed [%s]: %s", application_id, db_exc)

    logger.info("◀ [%s] decision=%s confidence=%.2f",
                application_id, response.decision, response.confidence_score)
    return response


# ── Audit endpoints ───────────────────────────────────────────────────────

@router.get("/history", summary="Recent loan decisions", tags=["Audit"])
async def get_decision_history(
    limit: int = 50, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    from utils.db_utils import LoanDecisionLog
    from sqlalchemy import desc
    try:
        rows = (
            db.query(LoanDecisionLog)
            .order_by(desc(LoanDecisionLog.timestamp))
            .limit(min(limit, 200))
            .all()
        )
        return {"count": len(rows), "decisions": [
            {
                "id": r.id, "application_id": r.application_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "decision": r.decision, "confidence": r.confidence,
                "risk_score": r.risk_score, "risk_grade": r.risk_grade,
                "compliance_status": r.compliance_status,
                "loan_type": r.loan_type, "bureau_score": r.bureau_score,
                "llm_provider": r.llm_provider, "processing_ms": r.processing_ms,
            }
            for r in rows
        ]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Decision history unavailable.")


@router.get("/stats", summary="Aggregate decision statistics", tags=["Audit"])
async def get_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    from utils.db_utils import get_decision_stats
    try:
        return get_decision_stats(db)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Statistics unavailable.")
