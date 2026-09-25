"""
utils/db_utils.py
=================
Neon.tech Serverless Postgres — Audit Trail & Loan Decision Logs

Why Neon?
- Free forever (no 90-day expiry like Render Postgres)
- Serverless: scales to zero, no idle charges
- Full Postgres 15 compatibility with SSL

Tables:
  loan_decisions  — every evaluation request + response
  compliance_logs — per-application rule violations
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, JSON, Text, Boolean,
    Index
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine — Neon requires sslmode=require for all external connections
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")

def _build_engine():
    """Create the SQLAlchemy engine with the right settings per environment."""
    if DATABASE_URL and "neon.tech" in DATABASE_URL:
        logger.info("☁️  Connecting to Neon.tech Postgres")
        return create_engine(
            DATABASE_URL,
            connect_args={"sslmode": "require"},
            pool_pre_ping=True,          # handle Neon cold-start reconnects
            pool_recycle=300,            # recycle connections every 5 min
            echo=False,
        )

    if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
        logger.info("🐘 Connecting to external Postgres")
        return create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

    # Local fallback — SQLite (no psycopg2 needed for dev)
    sqlite_path = os.getenv("SQLITE_PATH", "./data/local_loan_decisions.db")
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    logger.warning("💾 DATABASE_URL not set — falling back to SQLite at %s", sqlite_path)
    return create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class LoanDecisionLog(Base):
    """
    Immutable audit record for every loan evaluation.
    Raw request + response stored as JSONB for full replay capability.
    """
    __tablename__ = "loan_decisions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    application_id  = Column(String(64), unique=True, index=True, nullable=False,
                             default=lambda: f"LOAN-{uuid.uuid4().hex[:12].upper()}")
    timestamp       = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Core decision fields — indexed for dashboard queries
    decision        = Column(String(20), nullable=False, index=True)   # APPROVED | REJECTED | REVIEW
    confidence      = Column(Float, nullable=True)
    model_used      = Column(String(64), nullable=True)

    # Risk snapshot
    risk_score      = Column(Float, nullable=True)
    risk_grade      = Column(String(10), nullable=True)                # LOW | MEDIUM | HIGH

    # Compliance snapshot
    compliance_status   = Column(String(10), nullable=True)            # PASS | FAIL
    compliance_score    = Column(Float, nullable=True)
    hard_violations     = Column(Integer, default=0)

    # Applicant summary (non-PII analytics fields)
    loan_type           = Column(String(32), nullable=True, index=True)
    requested_amount    = Column(Float, nullable=True)
    bureau_score        = Column(Integer, nullable=True)
    foir_pct            = Column(Float, nullable=True)

    # Full payloads for replay / audit
    raw_request     = Column(JSON, nullable=True)
    raw_response    = Column(JSON, nullable=True)

    # LLM provider used (for cost tracking)
    llm_provider    = Column(String(32), nullable=True)
    processing_ms   = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_loan_decisions_timestamp", "timestamp"),
        Index("ix_loan_decisions_decision_type", "decision", "loan_type"),
    )

    def __repr__(self):
        return f"<LoanDecisionLog id={self.id} app={self.application_id} decision={self.decision}>"


class ComplianceViolationLog(Base):
    """
    Normalised per-rule violation records for compliance reporting.
    Linked to LoanDecisionLog via application_id.
    """
    __tablename__ = "compliance_violations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    application_id  = Column(String(64), index=True, nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    rule_type       = Column(String(64), nullable=False)
    feature         = Column(String(64), nullable=True)
    severity        = Column(String(20), nullable=True)    # violation | warning
    message         = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_compliance_app_id", "application_id"),
        Index("ix_compliance_rule_type", "rule_type"),
    )

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call on every API startup (idempotent).
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialised (Neon / SQLite)")
    except Exception as exc:
        logger.error("❌ Failed to initialise database: %s", exc)
        raise


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a scoped DB session, closes after request.

    Usage:
        @router.post("/evaluate")
        async def evaluate(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# High-level write helpers used by the API route
# ---------------------------------------------------------------------------

def log_decision(
    db: Session,
    application_id: str,
    decision: str,
    confidence: float,
    model_used: str,
    risk_score: float,
    risk_grade: str,
    compliance_status: str,
    compliance_score: float,
    hard_violations: int,
    loan_type: str,
    requested_amount: float,
    bureau_score: int,
    foir_pct: float,
    raw_request: dict,
    raw_response: dict,
    llm_provider: str = "groq",
    processing_ms: int = 0,
    violations: list = None,
) -> LoanDecisionLog:
    """
    Persist one loan evaluation to Neon (or SQLite in dev).
    Also writes normalised violation rows if any.
    Returns the saved ORM object.
    """
    entry = LoanDecisionLog(
        application_id  = application_id,
        decision        = decision,
        confidence      = confidence,
        model_used      = model_used,
        risk_score      = risk_score,
        risk_grade      = risk_grade,
        compliance_status   = compliance_status,
        compliance_score    = compliance_score,
        hard_violations = hard_violations,
        loan_type       = loan_type,
        requested_amount= requested_amount,
        bureau_score    = bureau_score,
        foir_pct        = foir_pct,
        raw_request     = raw_request,
        raw_response    = raw_response,
        llm_provider    = llm_provider,
        processing_ms   = processing_ms,
    )
    db.add(entry)

    # Persist normalised violation rows
    if violations:
        for v in violations:
            vlog = ComplianceViolationLog(
                application_id = application_id,
                rule_type      = v.get("rule_type", "UNKNOWN"),
                feature        = v.get("feature", ""),
                severity       = v.get("severity", "violation"),
                message        = v.get("message", ""),
            )
            db.add(vlog)

    db.commit()
    db.refresh(entry)
    return entry


def get_decision_stats(db: Session) -> dict:
    """
    Aggregate stats for the Streamlit dashboard.
    Returns counts and averages from the last 1000 decisions.
    """
    from sqlalchemy import func

    total     = db.query(func.count(LoanDecisionLog.id)).scalar() or 0
    approved  = db.query(func.count(LoanDecisionLog.id)).filter(LoanDecisionLog.decision == "APPROVED").scalar() or 0
    rejected  = db.query(func.count(LoanDecisionLog.id)).filter(LoanDecisionLog.decision == "REJECTED").scalar() or 0
    avg_risk  = db.query(func.avg(LoanDecisionLog.risk_score)).scalar() or 0.0
    avg_conf  = db.query(func.avg(LoanDecisionLog.confidence)).scalar() or 0.0

    return {
        "total_evaluations": total,
        "approved": approved,
        "rejected": rejected,
        "review": total - approved - rejected,
        "approval_rate_pct": round((approved / total * 100), 1) if total else 0.0,
        "avg_risk_score": round(float(avg_risk), 3),
        "avg_confidence": round(float(avg_conf), 1),
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    print("🔧 Testing database connection...")
    init_db()
    print("✅ Tables created / verified")

    # Quick smoke test
    db = SessionLocal()
    stats = get_decision_stats(db)
    db.close()
    print("📊 Stats:", json.dumps(stats, indent=2))
