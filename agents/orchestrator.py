"""
agents/orchestrator.py
======================
LoanDecisionOrchestrator — single authoritative execution path.

5-stage pipeline (short-circuit pattern):
  1. Preprocess         utils/preprocessing.py
  2. Hard compliance    rules/rule_engine.py          (deterministic, no LLM)
  3. Soft compliance    agents/compliance_agent.py    (RAG + Groq)
  4. Risk assessment    agents/risk_agent.py          (deterministic math)
  5. Decision + XAI     (Groq via get_llm)
                        agents/xai_agent.py           (SHAP/LIME + Groq)

LLM provider: ALL LLM calls go through utils/llm_utility.get_llm()
              which routes Groq → OpenAI → MockLLM in priority order.
              No CrewAI dependency anywhere in this file.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(override=True)

# ── Core pipeline imports ─────────────────────────────────────────────────
from utils.preprocessing import preprocess_single_application
from rules.rule_engine import evaluate_hard_constraints
from agents.compliance_agent import check_rbi_compliance
from agents.risk_agent import compute_risk_score
from agents.rag_agent import retrieve_feature_guidelines

# ── Unified LLM (Groq-primary, pure LangChain — no CrewAI) ───────────────
from utils.llm_utility import get_llm, TaskType

logger = logging.getLogger(__name__)


class LoanDecisionOrchestrator:
    """
    Authoritative orchestrator for the loan decision pipeline.

    Every public method is synchronous; the API layer wraps calls in
    asyncio.run_in_executor so FastAPI stays non-blocking.
    """

    def __init__(self):
        """Lazy init — LLM created on first request, not at import time."""
        self._llm: Optional[Any] = None
        logger.info("LoanDecisionOrchestrator created (lazy init)")

    # ── Lazy LLM accessor ────────────────────────────────────────────────

    def _get_llm(self):
        """Return a LangChain LLM routed through unified utility (Groq-primary)."""
        if self._llm is None:
            self._llm = get_llm(TaskType.SMART)
            logger.info("LLM initialised: %s", type(self._llm).__name__)
        return self._llm

    # ── Public entry point ────────────────────────────────────────────────

    def evaluate_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full 5-stage evaluation pipeline.

        Returns a dict that is schema-compatible with FinalDecisionResponse.
        Guaranteed to return *something* — never raises to the caller.
        """
        start_time = datetime.utcnow()
        application_id = application_data.get("application_id", "UNKNOWN")

        try:
            # ── Stage 1: Preprocess ──────────────────────────────────────
            logger.info("[%s] Stage 1: Preprocessing", application_id)
            preprocessed = preprocess_single_application(application_data)

            # ── Stage 2: Hard compliance (deterministic, short-circuit) ──
            logger.info("[%s] Stage 2: Hard compliance", application_id)
            loan_type    = str(application_data.get("loan_type", "personal")).lower()
            # Rule engine needs the ORIGINAL dict (feature values), not the preprocessed array
            hard_result  = evaluate_hard_constraints(application_data, loan_type)

            if not hard_result.is_compliant:
                logger.info("[%s] SHORT-CIRCUIT: %d hard violations",
                            application_id, len(hard_result.violations))
                return self._rejection_response(
                    application_id   = application_id,
                    violations       = [v.model_dump() for v in hard_result.violations],
                    rules_applied    = hard_result.rules_applied,
                    processing_ms    = self._elapsed_ms(start_time),
                    application_data = application_data,
                    preprocessed     = preprocessed,
                )

            # ── Stage 3: Soft compliance (RAG + Groq) ────────────────────
            logger.info("[%s] Stage 3: Soft compliance", application_id)
            try:
                guidelines      = retrieve_feature_guidelines(application_data)
                compliance_dict = check_rbi_compliance(application_data, guidelines)
            except Exception as exc:
                logger.warning("[%s] Soft compliance error (non-fatal): %s", application_id, exc)
                compliance_dict = self._compliance_pass_fallback(hard_result)

            # ── Stage 4: Risk assessment (pure math, no LLM) ─────────────
            logger.info("[%s] Stage 4: Risk assessment", application_id)
            risk_dict = compute_risk_score(application_data)

            # ── Stage 5: Decision + XAI ───────────────────────────────────
            logger.info("[%s] Stage 5: Decision + XAI", application_id)
            decision_dict = self._make_decision(compliance_dict, risk_dict, application_id)
            xai_dict      = self._generate_xai(preprocessed, application_data,
                                               compliance_dict, risk_dict, decision_dict,
                                               application_id)

            return self._build_response(
                application_id  = application_id,
                compliance_dict = compliance_dict,
                risk_dict       = risk_dict,
                decision_dict   = decision_dict,
                xai_dict        = xai_dict,
                processing_ms   = self._elapsed_ms(start_time),
            )

        except Exception as exc:
            logger.error("[%s] Orchestration error: %s", application_id, exc, exc_info=True)
            return self._error_response(application_id, str(exc), self._elapsed_ms(start_time))

    # ── Stage 5a: Decision ────────────────────────────────────────────────

    def _make_decision(
        self,
        compliance: Dict[str, Any],
        risk:       Dict[str, Any],
        app_id:     str,
    ) -> Dict[str, Any]:
        """
        Use Groq (via get_llm) to determine final eligibility and model selection.
        Falls back to rule-based decision if LLM is unavailable.
        """
        risk_score = float(risk.get("risk_score_10", 5.0))
        is_compliant = compliance.get("is_compliant", False)

        # Rule-based fast-path when LLM unavailable or for very clear cases
        if not is_compliant:
            return {
                "loan_eligible": False,
                "decision_reason": "Soft compliance issues detected by RAG/LLM review.",
                "selected_model": "none",
                "approval_probability": 0.05,
            }

        try:
            llm = get_llm(TaskType.SMART)
            prompt = f"""You are an RBI-regulated credit decision officer.

PolicyAgent Output:
{json.dumps(compliance, indent=2, default=str)}

RiskAgent Output:
{json.dumps(risk, indent=2, default=str)}

Rules:
- risk_score_10 > 7 → reject (high risk)
- 4 ≤ risk_score_10 ≤ 7 → cautious approval, model="blackbox"
- risk_score_10 < 4 → confident approval, model="interpretable"
- Any compliance violation → reject

Return STRICTLY valid JSON (no markdown):
{{
  "loan_eligible": true | false,
  "decision_reason": "<2-3 sentences>",
  "selected_model": "blackbox" | "interpretable" | "none",
  "approval_probability": 0.0
}}"""

            raw = llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)

            # Extract JSON block
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                result = json.loads(m.group())
                result.setdefault("loan_eligible",        risk_score < 7.0)
                result.setdefault("decision_reason",      "Decision by LLM.")
                result.setdefault("selected_model",       "blackbox")
                result.setdefault("approval_probability", 1 - risk_score / 10)
                return result

        except Exception as exc:
            logger.warning("[%s] Decision LLM failed (%s), using rule fallback", app_id, exc)

        # Pure rule-based fallback
        eligible = risk_score < 7.0
        return {
            "loan_eligible":        eligible,
            "decision_reason":      (
                f"Rule-based decision: risk_score={risk_score:.1f} "
                f"({'below' if eligible else 'above'} threshold of 7.0)."
            ),
            "selected_model":       "interpretable" if risk_score < 4 else "blackbox",
            "approval_probability": round(max(0.05, 1 - risk_score / 10), 2),
        }

    # ── Stage 5b: XAI ─────────────────────────────────────────────────────

    def _generate_xai(
        self,
        preprocessed:    Any,
        raw_application: Dict[str, Any],
        compliance:      Dict[str, Any],
        risk:            Dict[str, Any],
        decision:        Dict[str, Any],
        app_id:          str,
    ) -> Dict[str, Any]:
        """
        Generate SHAP/LIME + LLM explanations.
        Degrades gracefully: if the Keras model / explainers are unavailable,
        falls back to a Groq-generated natural-language summary.
        """
        try:
            from agents.xai_agent import explain_prediction
            return explain_prediction(preprocessed, raw_application, compliance, risk)
        except Exception as exc:
            logger.warning("[%s] XAI full pipeline failed (%s), using LLM fallback", app_id, exc)

        # Groq-generated explanation fallback (always works if API key is set)
        return self._llm_xai_fallback(raw_application, compliance, risk, decision, app_id)

    def _llm_xai_fallback(
        self,
        raw_application: Dict[str, Any],
        compliance:      Dict[str, Any],
        risk:            Dict[str, Any],
        decision:        Dict[str, Any],
        app_id:          str,
    ) -> Dict[str, Any]:
        """Generate explanation using Groq when SHAP/LIME/Keras unavailable."""
        try:
            llm        = get_llm(TaskType.SMART)
            eligible   = decision.get("loan_eligible", False)
            risk_score = risk.get("risk_score_10", "N/A")
            grade      = risk.get("grade", "N/A")
            bureau     = raw_application.get("bureau_score", "N/A")
            foir       = raw_application.get("foir_total_obligations_pct", "N/A")
            loan_type  = raw_application.get("loan_type", "loan")

            user_prompt = f"""You are a loan officer explaining a decision to an applicant.

Decision: {"APPROVED" if eligible else "REJECTED"}
Risk Grade: {grade} (score {risk_score}/10)
Key factors: bureau_score={bureau}, FOIR={foir}%, loan_type={loan_type}
Compliance: {"PASS" if compliance.get("is_compliant") else "FAIL"}

Write 3-4 plain-English sentences explaining why the {loan_type} loan was {"approved" if eligible else "rejected"}.
Mention specific numbers. Be empathetic but factual."""

            user_raw  = llm.invoke(user_prompt)
            user_text = user_raw.content if hasattr(user_raw, "content") else str(user_raw)

            reg_prompt = f"""You are a financial auditor preparing a regulatory compliance report.

Application: {loan_type} loan
Decision: {"APPROVED" if eligible else "REJECTED"}
Risk: grade={grade}, score={risk_score}/10
Compliance score: {compliance.get("compliance_score", 0):.2f}
Hard violations: {len(compliance.get("hard_violations", []))}
Soft violations: {len(compliance.get("soft_violations", []))}
Risk drivers: {risk.get("drivers", [])}

Write a 2-paragraph structured report suitable for regulatory audit.
Paragraph 1: Executive summary of the decision.
Paragraph 2: Technical risk and compliance analysis."""

            reg_raw  = llm.invoke(reg_prompt)
            reg_text = reg_raw.content if hasattr(reg_raw, "content") else str(reg_raw)

            return {
                "user_explanation":      user_text.strip(),
                "regulator_explanation": reg_text.strip(),
                "raw_data":              {
                    "explanation_source": "groq_fallback",
                    "shap_available":     False,
                    "lime_available":     False,
                    "decision":           "approved" if eligible else "rejected",
                    "risk_drivers":       risk.get("drivers", []),
                },
            }

        except Exception as exc:
            logger.error("[%s] LLM XAI fallback failed: %s", app_id, exc)
            eligible = decision.get("loan_eligible", False)
            return {
                "user_explanation":      (
                    "Your application has been reviewed. "
                    f"Decision: {'Approved' if eligible else 'Rejected'}. "
                    "Please contact your branch for detailed reasoning."
                ),
                "regulator_explanation": (
                    f"Automated decision: {'APPROVED' if eligible else 'REJECTED'}. "
                    f"Risk grade: {risk.get('grade','N/A')}. "
                    "Full XAI pipeline unavailable; decision based on deterministic rules and risk score."
                ),
                "raw_data": {"explanation_source": "static_fallback", "error": str(exc)},
            }

    # ── Response builders ─────────────────────────────────────────────────

    def _build_response(
        self,
        application_id:  str,
        compliance_dict: Dict[str, Any],
        risk_dict:       Dict[str, Any],
        decision_dict:   Dict[str, Any],
        xai_dict:        Dict[str, Any],
        processing_ms:   int,
    ) -> Dict[str, Any]:
        """Assemble the final structured response matching FinalDecisionResponse."""

        eligible   = decision_dict.get("loan_eligible", False)
        decision   = "APPROVED" if eligible else "REJECTED"
        risk_score = float(risk_dict.get("risk_score_10", 5.0))

        # Confidence: high at extremes, lower in borderline zone
        if risk_score < 3.0 and compliance_dict.get("is_compliant"):
            confidence = 0.92
        elif risk_score > 7.5 or not compliance_dict.get("is_compliant"):
            confidence = 0.93
        else:
            confidence = 0.68

        return {
            "application_id":   application_id,
            "decision":         decision,
            "confidence_score": confidence,
            "processing_time_ms": processing_ms,
            "loan_eligible":    eligible,
            "selected_model":   decision_dict.get("selected_model", "none"),

            # ── Risk Assessment ───────────────────────────────────────
            "risk_assessment": {
                "risk_score_10":  risk_score,
                "grade":          risk_dict.get("grade", "UNKNOWN"),
                "components":     risk_dict.get("components", {}),
                "drivers":        risk_dict.get("drivers", []),
                "reasons":        risk_dict.get("reasons", []),
                "context":        risk_dict.get("context", {}),
            },

            # ── Compliance ────────────────────────────────────────────
            "compliance_result": {
                "is_compliant":    compliance_dict.get("is_compliant", False),
                "compliance_score":compliance_dict.get("compliance_score", 0.0),
                "violations":      compliance_dict.get("violations", []),
                "hard_violations": compliance_dict.get("hard_violations", []),
                "soft_violations": compliance_dict.get("soft_violations", []),
                "explanation":     compliance_dict.get("explanation", ""),
                "rules_applied":   compliance_dict.get("rules_applied", []),
            },

            # ── Decision Summary ──────────────────────────────────────
            "decision_reason":  decision_dict.get("decision_reason", ""),
            "approval_probability": float(decision_dict.get("approval_probability", 0.0)),

            # ── Explanations ──────────────────────────────────────────
            "explanations": {
                "customer_explanation":  xai_dict.get("user_explanation", ""),
                "technical_explanation": xai_dict.get("regulator_explanation", ""),
                "user_explanation":      xai_dict.get("user_explanation", ""),
                "regulator_explanation": xai_dict.get("regulator_explanation", ""),
                "raw_data":              xai_dict.get("raw_data", {}),
            },

            # ── Metadata ─────────────────────────────────────────────
            "metadata": {
                "application_id":  application_id,
                "pipeline_version":"v2.0",
                "llm_provider":    os.getenv("GROQ_API_KEY") and "groq" or "mock",
                "timestamp":       datetime.utcnow().isoformat(),
            },
        }

    def _rejection_response(
        self,
        application_id:   str,
        violations:       List[Dict],
        rules_applied:    List[str],
        processing_ms:    int,
        application_data: Optional[Dict[str, Any]] = None,
        preprocessed:     Optional[Any] = None,
    ) -> Dict[str, Any]:
        messages = [v.get("message", "Violation") for v in violations]

        customer_exp = None
        technical_exp = None
        raw_xai = {"violations": violations, "rules_applied": rules_applied}

        if application_data is not None:
            try:
                from agents.xai_agent import explain_short_circuit_rejection
                sc_res = explain_short_circuit_rejection(
                    applicant_data = application_data,
                    violations     = violations,
                    rules_applied  = rules_applied,
                    preprocessed   = preprocessed,
                )
                customer_exp = sc_res.get("user_explanation")
                technical_exp = sc_res.get("regulator_explanation")
                raw_xai = sc_res.get("raw_data", raw_xai)
            except Exception as exc:
                logger.warning("[%s] Short-circuit XAI generation failed: %s", application_id, exc)

        if not customer_exp:
            customer_exp = (
                "Your application could not be approved at this time because it does not "
                "meet one or more mandatory regulatory requirements. "
                f"Issues: {'; '.join(messages)}. "
                "Please contact your branch for guidance on resolving these issues."
            )
        if not technical_exp:
            technical_exp = (
                f"Hard constraint evaluation failed for {len(violations)} rule(s). "
                f"Rules applied: {', '.join(rules_applied)}. "
                "Application short-circuited before ML inference."
            )

        return {
            "application_id":   application_id,
            "decision":         "REJECTED",
            "confidence_score": 0.98,
            "processing_time_ms": processing_ms,
            "loan_eligible":    False,
            "selected_model":   "rule_engine",

            "risk_assessment": {
                "risk_score_10": 10.0,
                "grade":         "E",
                "components":    {},
                "drivers":       ["hard_compliance_violation"],
                "reasons":       messages,
                "context":       {},
            },

            "compliance_result": {
                "is_compliant":    False,
                "compliance_score":0.0,
                "violations":      violations,
                "hard_violations": violations,
                "soft_violations": [],
                "explanation":     f"Application rejected: {'; '.join(messages)}",
                "rules_applied":   rules_applied,
            },

            "decision_reason": f"Hard regulatory violations: {'; '.join(messages)}",
            "approval_probability": 0.0,

            "explanations": {
                "customer_explanation":  customer_exp,
                "technical_explanation": technical_exp,
                "user_explanation":      customer_exp,
                "regulator_explanation": technical_exp,
                "raw_data":              raw_xai,
            },

            "metadata": {
                "application_id":   application_id,
                "pipeline_version": "v2.0",
                "short_circuit":    True,
                "timestamp":        datetime.utcnow().isoformat(),
            },
        }

    def _error_response(self, application_id: str, error: str, processing_ms: int) -> Dict[str, Any]:
        return {
            "application_id":   application_id,
            "decision":         "ERROR",
            "confidence_score": 0.0,
            "processing_time_ms": processing_ms,
            "loan_eligible":    False,
            "selected_model":   "none",

            "risk_assessment":  {"risk_score_10": 0.0, "grade": "UNKNOWN", "components": {},
                                 "drivers": [], "reasons": [error], "context": {}},
            "compliance_result":{"is_compliant": False, "compliance_score": 0.0,
                                 "violations": [], "hard_violations": [], "soft_violations": [],
                                 "explanation": error, "rules_applied": []},

            "decision_reason":  f"System error: {error}",
            "approval_probability": 0.0,

            "explanations": {
                "customer_explanation":  "We were unable to process your application. Please try again.",
                "technical_explanation": f"Pipeline error: {error}",
                "raw_data":              {"error": error},
            },
            "metadata": {
                "application_id":   application_id,
                "pipeline_version": "v2.0",
                "error":            error,
                "timestamp":        datetime.utcnow().isoformat(),
            },
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _elapsed_ms(start: datetime) -> int:
        return int((datetime.utcnow() - start).total_seconds() * 1000)

    @staticmethod
    def _compliance_pass_fallback(hard_result) -> Dict[str, Any]:
        """Return a minimal PASS compliance dict when soft evaluation errors out."""
        return {
            "is_compliant":    True,
            "compliance_score":0.85,
            "violations":      [],
            "hard_violations": [],
            "soft_violations": [],
            "explanation":     "Hard constraints passed. Soft evaluation unavailable.",
            "rules_applied":   hard_result.rules_applied,
        }
