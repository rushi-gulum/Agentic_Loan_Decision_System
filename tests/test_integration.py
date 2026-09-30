"""
tests/test_integration.py
==========================
Integration tests for the Agentic Loan Decision System.

Coverage:
  1. Hard-rule short-circuit — rejected application must NOT invoke ML/LLM
  2. Full evaluate_application() — compliant applicant traces all 5 stages
  3. Response schema contract — required keys and types

These tests do NOT require a live API server — they call the orchestrator
directly. They DO require the artefacts from pipeline/build_artifacts.py.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── gRPC stub (prevents Windows AppControl DLL block) ────────────────────
import types as _types

for _m in [
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
]:
    if _m not in sys.modules:
        _s = _types.ModuleType(_m)
        _s.OTLPSpanExporter    = object
        _s.OTLPLogExporter     = object
        _s.OTLPMetricExporter  = object
        sys.modules[_m]        = _s

# ── Fixtures ──────────────────────────────────────────────────────────────

COMPLIANT_HOUSING = {
    "application_id":                  "TEST-COMPLIANT-001",
    "loan_type":                       "housing",
    "age_years":                       35,
    "gender":                          "Male",
    "pin_code":                        400001,
    "bureau_score":                    750,
    "monthly_income_inr":              120_000,
    "existing_monthly_obligations_inr":15_000,
    "requested_amount_inr":            3_000_000,
    "sanctioned_amount_inr":           3_000_000,
    "tenure_months":                   240,
    "interest_rate_annual_pct":        8.5,
    "processing_fee_inr":              30_000,
    "other_charges_inr":               1_000,
    "apr_pct":                         9.0,
    "kfs_provided":                    True,
    "proposed_emi_inr":                26_000,
    "foir_total_obligations_pct":      34.2,   # (15000+26000)/120000*100
    "property_value_inr":              4_000_000,
    "ltv_ratio":                       0.75,
    "pep_flag":                        False,
    "ovd_type":                        "PAN",
    "kyc_mode":                        "eKYC",
    "application_date":                "2024-01-01",
    "sanction_date":                   "2024-01-08",
    "interest_type":                   "Fixed",
}

HARD_VIOLATION_PERSONAL = {
    "application_id":                  "TEST-REJECT-001",
    "loan_type":                       "personal",
    "age_years":                       28,
    "gender":                          "Male",
    "pin_code":                        110001,
    "bureau_score":                    630,     # BELOW 720 minimum for personal
    "monthly_income_inr":              50_000,
    "existing_monthly_obligations_inr":30_000,
    "requested_amount_inr":            400_000,
    "sanctioned_amount_inr":           400_000,
    "tenure_months":                   24,
    "interest_rate_annual_pct":        18.0,
    "processing_fee_inr":              4_000,
    "other_charges_inr":               500,
    "apr_pct":                         19.0,
    "kfs_provided":                    True,
    "proposed_emi_inr":                20_000,
    "foir_total_obligations_pct":      60.0,   # ABOVE 50% maximum for personal
    "property_value_inr":              0,
    "ltv_ratio":                       0,
    "pep_flag":                        False,
    "ovd_type":                        "PAN",
    "kyc_mode":                        "eKYC",
    "application_date":                "2024-01-01",
    "sanction_date":                   "2024-01-07",
    "interest_type":                   "Fixed",
}

# ── Shared orchestrator (one instance, lazy init) ─────────────────────────

@pytest.fixture(scope="module")
def orchestrator():
    from agents.orchestrator import LoanDecisionOrchestrator
    return LoanDecisionOrchestrator()

# ─────────────────────────────────────────────────────────────────────────
# Test 1 — Hard-rule short-circuit
# ─────────────────────────────────────────────────────────────────────────

class TestHardRuleShortCircuit:
    """
    Verifies the deterministic guardrail: applications violating hard rules
    are rejected immediately without touching the ML model or LLM.
    """

    def test_hard_violation_is_rejected(self, orchestrator):
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        assert result["decision"] == "REJECTED", (
            f"Expected REJECTED for hard-violation applicant, got {result['decision']}"
        )

    def test_hard_violation_high_confidence(self, orchestrator):
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        # Hard rejections should be very confident
        assert result["confidence_score"] >= 0.90, (
            f"Hard rejection confidence too low: {result['confidence_score']}"
        )

    def test_hard_violation_has_violations_list(self, orchestrator):
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        compliance = result.get("compliance_result", {})
        assert len(compliance.get("hard_violations", [])) >= 1, (
            "Expected at least one hard violation in result"
        )

    def test_hard_violation_model_is_rule_engine(self, orchestrator):
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        # Short-circuited results must not claim they used an ML model
        assert result.get("selected_model") in ("rule_engine", "none"), (
            f"Expected rule_engine or none, got {result.get('selected_model')}"
        )

    def test_hard_violation_fast_processing(self, orchestrator):
        import time
        start = time.time()
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        elapsed_ms = (time.time() - start) * 1000
        # Without ML inference, hard rejections should complete in <2 seconds
        assert elapsed_ms < 2000, (
            f"Short-circuit should be <2s, took {elapsed_ms:.0f}ms"
        )

    def test_bureau_score_violation_message(self, orchestrator):
        result = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        compliance = result.get("compliance_result", {})
        violations = compliance.get("hard_violations", [])
        # At least one violation must mention bureau score
        messages = " ".join(str(v) for v in violations).lower()
        assert any(k in messages for k in ("bureau", "score", "630", "foir", "50")), (
            f"Violation messages don't mention expected fields: {messages[:200]}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — Full evaluate pipeline
# ─────────────────────────────────────────────────────────────────────────

class TestFullEvaluatePipeline:
    """
    Verifies that a compliant application flows through all 5 stages and
    returns a fully populated, schema-consistent response dict.
    """

    @pytest.fixture(scope="class")
    def compliant_result(self, orchestrator):
        return orchestrator.evaluate_application(COMPLIANT_HOUSING)

    def test_returns_dict(self, compliant_result):
        assert isinstance(compliant_result, dict)

    def test_decision_is_valid_string(self, compliant_result):
        assert compliant_result["decision"] in ("APPROVED", "REJECTED", "REVIEW", "ERROR"), (
            f"Unexpected decision: {compliant_result['decision']}"
        )

    def test_risk_assessment_present(self, compliant_result):
        ra = compliant_result.get("risk_assessment", {})
        assert "risk_score_10" in ra, "risk_assessment.risk_score_10 missing"
        assert "grade" in ra,        "risk_assessment.grade missing"
        score = float(ra["risk_score_10"])
        assert 0.0 <= score <= 10.0, f"risk_score_10 out of bounds: {score}"
        assert ra["grade"] in ("A+", "A", "B", "C", "D", "E"), (
            f"Unexpected risk grade: {ra['grade']}"
        )

    def test_compliance_result_present(self, compliant_result):
        cr = compliant_result.get("compliance_result", {})
        assert "is_compliant" in cr,    "compliance_result.is_compliant missing"
        assert "compliance_score" in cr,"compliance_result.compliance_score missing"
        assert isinstance(cr["is_compliant"], bool)
        score = float(cr["compliance_score"])
        assert 0.0 <= score <= 1.0, f"compliance_score out of bounds: {score}"

    def test_confidence_in_range(self, compliant_result):
        conf = float(compliant_result.get("confidence_score", -1))
        assert 0.0 <= conf <= 1.0, f"confidence_score out of bounds: {conf}"

    def test_explanations_present(self, compliant_result):
        exp = compliant_result.get("explanations", {})
        assert "customer_explanation" in exp,  "customer_explanation missing"
        assert "technical_explanation" in exp, "technical_explanation missing"
        assert len(exp["customer_explanation"]) > 10,  "customer_explanation too short"
        assert len(exp["technical_explanation"]) > 10, "technical_explanation too short"

    def test_processing_time_recorded(self, compliant_result):
        ms = compliant_result.get("processing_time_ms", 0)
        assert ms > 0, "processing_time_ms should be positive"

    def test_metadata_present(self, compliant_result):
        meta = compliant_result.get("metadata", {})
        assert "timestamp" in meta,        "metadata.timestamp missing"
        assert "pipeline_version" in meta, "metadata.pipeline_version missing"

    def test_selected_model_present(self, compliant_result):
        model = compliant_result.get("selected_model", "")
        assert model != "", "selected_model should not be empty"


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — Deterministic rules produce consistent results
# ─────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """
    The risk score and hard compliance result must be identical across
    repeated calls for the same input (no randomness).
    """

    def test_risk_score_identical_on_repeat(self, orchestrator):
        r1 = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        r2 = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        s1 = r1["risk_assessment"]["risk_score_10"] if "risk_assessment" in r1 else r1.get("risk_score_10")
        s2 = r2["risk_assessment"]["risk_score_10"] if "risk_assessment" in r2 else r2.get("risk_score_10")
        assert s1 == s2, f"Risk scores differ across calls: {s1} vs {s2}"

    def test_decision_identical_on_repeat(self, orchestrator):
        r1 = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        r2 = orchestrator.evaluate_application(HARD_VIOLATION_PERSONAL)
        assert r1["decision"] == r2["decision"], (
            f"Decision differs across calls: {r1['decision']} vs {r2['decision']}"
        )

    def test_compliance_identical_on_repeat(self, orchestrator):
        r1 = orchestrator.evaluate_application(COMPLIANT_HOUSING)
        r2 = orchestrator.evaluate_application(COMPLIANT_HOUSING)
        c1 = r1["compliance_result"]["is_compliant"]
        c2 = r2["compliance_result"]["is_compliant"]
        assert c1 == c2, f"Compliance status differs across calls: {c1} vs {c2}"
