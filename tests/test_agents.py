"""
tests/test_agents.py
====================
Unit tests for the autonomous agents layer:
- CreditRiskAgent
- ComplianceAgent
- XAIAgent (SHAP attributions & Dual-Audience Justification Dossier)
- LoanDecisionOrchestrator
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.risk_agent import compute_risk_score
from agents.compliance_agent import evaluate_hard_compliance
from agents.xai_agent import (
    FEATURE_METADATA,
    _fmt_currency,
    _fmt_pct,
    _generate_user_explanation,
    _generate_regulator_explanation,
    explain_short_circuit_rejection,
    preprocess,
    explain_prediction,
)
from agents.orchestrator import LoanDecisionOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Credit Risk Agent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskAgent:
    def test_prime_borrower_low_risk(self):
        prime_data = {
            "loan_type": "housing",
            "bureau_score": 790,
            "foir_total_obligations_pct": 23.18,
            "monthly_income_inr": 150000,
            "existing_monthly_obligations_inr": 15000,
            "requested_amount_inr": 3500000,
            "proposed_emi_inr": 31000,
            "ltv_ratio": 0.70,
            "pep_flag": False,
        }
        res = compute_risk_score(prime_data)
        assert "risk_score_10" in res
        assert "grade" in res
        assert res["risk_score_10"] < 4.0
        assert res["grade"] in ("A+", "A")

    def test_overleveraged_borrower_high_risk(self):
        risky_data = {
            "loan_type": "housing",
            "bureau_score": 645,
            "foir_total_obligations_pct": 66.12,
            "monthly_income_inr": 350000,
            "existing_monthly_obligations_inr": 190000,
            "requested_amount_inr": 4000000,
            "proposed_emi_inr": 41800,
            "ltv_ratio": 0.89,
            "pep_flag": False,
        }
        res = compute_risk_score(risky_data)
        assert res["risk_score_10"] >= 5.0
        assert res["grade"] in ("C", "D", "E")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compliance Agent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComplianceAgent:
    def test_compliant_application_passes(self):
        clean_app = {
            "loan_type": "housing",
            "age_years": 34,
            "bureau_score": 780,
            "foir_total_obligations_pct": 30.0,
            "ltv_ratio": 0.75,
            "property_value_inr": 5000000,
            "pep_flag": False,
        }
        res = evaluate_hard_compliance(clean_app)
        assert res.is_compliant is True
        assert len(res.violations) == 0

    def test_pep_flag_causes_short_circuit_violation(self):
        pep_app = {
            "loan_type": "personal",
            "age_years": 45,
            "bureau_score": 760,
            "foir_total_obligations_pct": 25.0,
            "pep_flag": True,
        }
        res = evaluate_hard_compliance(pep_app)
        assert res.is_compliant is False
        assert any("pep" in v.rule_type.lower() or "aml" in v.rule_type.lower() for v in res.violations)


# ─────────────────────────────────────────────────────────────────────────────
# 3. XAI Agent & Explainability Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestXAIAgent:
    def test_feature_metadata_coverage(self):
        assert len(FEATURE_METADATA) >= 20
        for f_key, meta in FEATURE_METADATA.items():
            assert "name" in meta, f"{f_key} missing name"
            assert "category" in meta, f"{f_key} missing category"
            assert "fmt" in meta, f"{f_key} missing fmt formatter"
            assert "pos" in meta, f"{f_key} missing pos impact description"
            assert "neg" in meta, f"{f_key} missing neg impact description"

    def test_formatters(self):
        assert _fmt_currency(150000) == "₹1.50 L"
        assert _fmt_currency(12000000) == "₹1.20 Cr"
        assert _fmt_currency(25000) == "₹25,000"
        assert _fmt_pct(23.18) == "23.2%"
        assert _fmt_pct(0.70) == "70.0%"

    def test_user_explanation_structure_approved(self):
        applicant = {
            "loan_type": "housing",
            "bureau_score": 790,
            "foir_total_obligations_pct": 23.18,
            "monthly_income_inr": 150000,
            "existing_monthly_obligations_inr": 15000,
            "proposed_emi_inr": 31000,
            "requested_amount_inr": 3500000,
            "ltv_ratio": 0.70,
            "apr_pct": 9.2,
        }
        text = _generate_user_explanation(
            decision="approved",
            applicant=applicant,
            top_shap=[],
            p_approve=0.998,
            threshold=0.555
        )
        assert "APPLICATION STATUS: APPROVED" in text
        assert "DIRECT DECISION FACTORS" in text
        assert "ACTIONABLE NEXT STEPS" in text
        assert "Credit Bureau Score (790" in text
        assert "Debt-to-Income Ratio (23.18%" in text

    def test_user_explanation_structure_rejected(self):
        applicant = {
            "loan_type": "personal",
            "bureau_score": 645,
            "foir_total_obligations_pct": 66.12,
            "monthly_income_inr": 350000,
            "existing_monthly_obligations_inr": 190000,
            "proposed_emi_inr": 41800,
            "requested_amount_inr": 4000000,
            "ltv_ratio": 0.89,
        }
        text = _generate_user_explanation(
            decision="rejected",
            applicant=applicant,
            top_shap=[],
            p_approve=0.12,
            threshold=0.555
        )
        assert "APPLICATION STATUS: REJECTED" in text
        assert "DIRECT DECISION FACTORS" in text
        assert "ACTIONABLE REMEDIATION ROADMAP" in text

    def test_regulator_explanation_structure(self):
        applicant = {"application_id": "TEST-APP-001", "bureau_score": 750, "foir_total_obligations_pct": 32.0}
        raw_data = {"p_approve": 0.89, "model_used": "XGBoost Loan Classifier", "shap_top_features": []}
        compliance = {"compliance_score": 1.0, "hard_violations": [], "soft_violations": []}
        risk = {"risk_score_10": 2.8, "grade": "A"}

        memo = _generate_regulator_explanation("approved", applicant, raw_data, compliance, risk)
        assert "REGULATORY COMPLIANCE & UNDERWRITING AUDIT MEMORANDUM" in memo
        assert "STATUTORY GUARDRAIL VERIFICATION MATRIX" in memo
        assert "QUANTITATIVE MODEL INFERENCE & SHAP ATTRIBUTION" in memo
        assert "EXPLAINABILITY & FAIR LENDING AUDIT ATTESTATION" in memo

    def test_short_circuit_explainability(self):
        applicant = {
            "application_id": "TEST-PEP-001",
            "loan_type": "personal",
            "bureau_score": 750,
            "monthly_income_inr": 100000,
            "existing_monthly_obligations_inr": 15000,
            "requested_amount_inr": 500000,
            "sanctioned_amount_inr": 500000,
            "tenure_months": 36,
            "interest_rate_annual_pct": 12.0,
            "proposed_emi_inr": 16000,
            "foir_total_obligations_pct": 31.0,
            "pep_flag": True,
            "kfs_provided": True,
            "ovd_provided": True,
            "age_years": 40,
            "pin_code": 110001
        }
        violations = [{"rule": "AML_PEP_POLICY", "message": "PEP status detected"}]
        res = explain_short_circuit_rejection(applicant, violations, ["AML_PEP_POLICY"])
        assert "user_explanation" in res
        assert "regulator_explanation" in res
        assert "raw_data" in res
        assert len(res["raw_data"]["shap_top_features"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Orchestrator Initialization Test
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_initialization():
    orc = LoanDecisionOrchestrator()
    assert hasattr(orc, "evaluate_application")
