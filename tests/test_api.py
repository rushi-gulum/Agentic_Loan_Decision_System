"""
Integration tests for FastAPI backend (Phase 3 & Production)

Tests the complete API integration:
1. FastAPI app initialization and root/health checks
2. Request/response validation with Pydantic schemas (LoanApplicationRequest, FinalDecisionResponse)
3. Orchestrator integration with short-circuit and normal execution patterns
4. Error handling and edge cases
5. CORS headers on preflight requests
6. Audit endpoints (/api/v1/stats)
7. Performance and latency limits

Run with: pytest tests/test_api.py -v
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app
from api.schemas import LoanApplicationRequest, FinalDecisionResponse
from agents.orchestrator import LoanDecisionOrchestrator

# Test client
client = TestClient(app)

# Canonical valid application payload conforming strictly to LoanApplicationRequest
VALID_APPLICATION = {
    "applicant_name": "John Doe",
    "application_id": "TEST-001",
    "loan_type": "housing",
    "age_years": 35,
    "bureau_score": 750,
    "monthly_income_inr": 75000.0,
    "requested_amount_inr": 500000.0,
    "tenure_months": 60,
    "interest_rate_annual_pct": 8.5,
    "foir_total_obligations_pct": 30.0,
    "existing_monthly_obligations_inr": 22500.0,
    "gender": "Male",
    "state": "Karnataka",
    "pin_code": 560001,
    "kyc_mode": "Video KYC",
    "ovd_type": "Aadhaar",
    "interest_type": "Fixed",
    "pep_flag": False,
    "kfs_provided": True,
    "processing_fee_inr": 5000.0,
    "other_charges_inr": 1000.0,
    "apr_pct": 8.5,
    "property_value_inr": 2000000.0,
    "ltv_ratio": 0.25,
}

INVALID_APPLICATION = {
    "applicant_name": "Jane Doe",
    "application_id": "TEST-002",
    "loan_type": "personal",
    "age_years": 17,  # Invalid age (< 18)
    "bureau_score": 250,  # Below minimum 300
    "monthly_income_inr": 0,  # Must be > 0
    "requested_amount_inr": 1000000,
    # Missing required core fields like tenure_months, interest_rate_annual_pct, foir_total_obligations_pct
}


class TestAPIHealthChecks:
    """Test API health and status endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Agentic Loan Decision API"
        assert data["version"] == "2.0.0"
        assert data["health"] == "/health"

    def test_health_check(self):
        """Test detailed health check structure"""
        response = client.get("/health")
        # 200 when healthy, 503 when cloud DB / external LLM keys are absent in local test
        assert response.status_code in (200, 503)
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "2.0.0"
        assert "components" in data
        assert "models" in data["components"]
        assert "database" in data["components"]
        assert "llm" in data["components"]


class TestEvaluationEndpoint:
    """Test loan evaluation endpoint"""

    def test_valid_application_schema_validation(self):
        """Test that valid application passes Pydantic validation"""
        request = LoanApplicationRequest(**VALID_APPLICATION)
        assert request.age_years == 35
        assert request.bureau_score == 750
        assert request.interest_rate_annual_pct == 8.5

    def test_invalid_application_schema_validation(self):
        """Test that invalid application fails Pydantic validation"""
        with pytest.raises(ValueError):
            LoanApplicationRequest(**INVALID_APPLICATION)

    def test_successful_evaluation_flow(self):
        """Test successful loan evaluation with mocked orchestrator"""
        mock_raw = {
            "application_id": "TEST-001",
            "decision": "APPROVED",
            "decision_reason": "Low credit risk and fully compliant with statutory norms.",
            "confidence_score": 0.88,
            "processing_time_ms": 125.0,
            "loan_eligible": True,
            "selected_model": "xgboost_calibrated",
            "approval_probability": 0.89,
            "risk_assessment": {
                "risk_score_10": 2.5,
                "grade": "A",
                "components": {"bureau": 0.2, "dti": 0.3},
                "drivers": ["Good credit score", "Low debt burden"],
                "context": {},
                "reasons": ["Good credit score", "Stable income"],
            },
            "compliance_result": {
                "is_compliant": True,
                "compliance_score": 1.0,
                "hard_violations": [],
                "hard_warnings": [],
                "soft_violations": [],
                "explanation": "All criteria met.",
                "loan_type": "housing",
                "rules_applied": ["age", "bureau", "foir"],
                "rag_guidelines_used": [],
            },
            "explanations": {
                "customer_explanation": "Loan approved based on strong credit profile",
                "technical_explanation": "Risk score: 2.5/10, Grade: A",
                "raw_data": {"shap_values": [0.1, -0.2]},
            },
            "metadata": {
                "short_circuit": False,
            },
        }

        with patch.object(LoanDecisionOrchestrator, "evaluate_application", return_value=mock_raw):
            response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
            assert response.status_code == 200

            data = response.json()
            assert data["decision"] == "APPROVED"
            assert data["confidence_score"] == 0.88
            assert data["risk_assessment"]["risk_score_10"] == 2.5
            assert data["risk_assessment"]["grade"] == "A"
            assert data["compliance_result"]["is_compliant"] is True
            assert data["selected_model"] == "xgboost_calibrated"
            assert data["loan_eligible"] is True

    def test_rejection_with_hard_violations(self):
        """Test rejection due to hard compliance violations"""
        mock_raw = {
            "application_id": "TEST-002",
            "decision": "REJECTED",
            "decision_reason": "Statutory constraint breach detected.",
            "confidence_score": 0.98,
            "processing_time_ms": 25.0,
            "loan_eligible": False,
            "selected_model": "rule_engine",
            "approval_probability": 0.0,
            "risk_assessment": {
                "risk_score_10": 10.0,
                "grade": "E",
                "components": {},
                "drivers": ["Hard policy violation"],
                "context": {},
                "reasons": ["Hard policy violation"],
            },
            "compliance_result": {
                "is_compliant": False,
                "compliance_score": 0.0,
                "hard_violations": [
                    {
                        "feature": "age_years",
                        "rule_type": "min_value",
                        "rule_value": 21,
                        "actual_value": 17,
                        "severity": "violation",
                        "message": "Applicant age 17 is below minimum requirement of 21",
                    }
                ],
                "hard_warnings": [],
                "soft_violations": [],
                "explanation": "Hard rule violations detected",
                "loan_type": "housing",
                "rules_applied": ["age"],
                "rag_guidelines_used": [],
            },
            "explanations": {},
            "metadata": {
                "short_circuit": True,
            },
        }

        with patch.object(LoanDecisionOrchestrator, "evaluate_application", return_value=mock_raw):
            response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
            assert response.status_code == 200
            data = response.json()
            assert data["decision"] == "REJECTED"
            assert data["metadata"]["short_circuited"] is True
            assert len(data["compliance_result"]["hard_violations"]) == 1

    def test_malformed_request_validation(self):
        """Test validation of malformed requests returns 422 Unprocessable Entity"""
        malformed_requests = [
            {},  # Empty request
            {"applicant_name": "Test"},  # Missing required fields
            {"age_years": "invalid"},  # Wrong data type
            {"bureau_score": 1000},  # Out of range value (> 900)
        ]

        for malformed_data in malformed_requests:
            response = client.post("/api/v1/evaluate", json=malformed_data)
            assert response.status_code == 422

    def test_orchestrator_exception_handling(self):
        """Test API error handling when orchestrator fails"""
        with patch.object(LoanDecisionOrchestrator, "evaluate_application", side_effect=Exception("Database connection down")):
            response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
            assert response.status_code == 500
            data = response.json()
            assert "Evaluation pipeline failed" in data["detail"]


class TestAuditEndpoints:
    """Test audit log and stats endpoints"""

    def test_stats_endpoint(self):
        """Test /api/v1/stats endpoint responds or handles missing db gracefully"""
        response = client.get("/api/v1/stats")
        assert response.status_code in (200, 503)

    def test_history_endpoint(self):
        """Test /api/v1/history endpoint responds or handles missing db gracefully"""
        response = client.get("/api/v1/history?limit=10")
        assert response.status_code in (200, 503)


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""

    def test_cors_headers(self):
        """Test CORS headers are present on valid OPTIONS preflight request"""
        response = client.options(
            "/api/v1/evaluate",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestPerformanceAndLimits:
    """Test performance characteristics and limits"""

    def test_response_time_reasonable(self):
        """Test that API responds quickly when orchestrator executes"""
        mock_raw = {
            "application_id": "TEST-FAST",
            "decision": "APPROVED",
            "decision_reason": "Low risk",
            "confidence_score": 0.9,
            "processing_time_ms": 45.0,
            "loan_eligible": True,
            "selected_model": "xgboost_calibrated",
            "approval_probability": 0.92,
            "risk_assessment": {
                "risk_score_10": 2.0,
                "grade": "A",
                "components": {},
                "drivers": [],
                "context": {},
                "reasons": [],
            },
            "compliance_result": {
                "is_compliant": True,
                "compliance_score": 1.0,
                "hard_violations": [],
                "hard_warnings": [],
                "soft_violations": [],
                "explanation": "OK",
                "loan_type": "housing",
                "rules_applied": [],
                "rag_guidelines_used": [],
            },
            "explanations": {},
            "metadata": {"short_circuit": False},
        }

        with patch.object(LoanDecisionOrchestrator, "evaluate_application", return_value=mock_raw):
            start_time = time.time()
            response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
            duration = time.time() - start_time

            assert response.status_code == 200
            assert duration < 5.0


def test_response_schema_compliance():
    """Test that API responses strictly comply with Pydantic schemas"""
    sample_response = {
        "decision": "APPROVED",
        "decision_reason": "Low credit risk and fully compliant with statutory norms.",
        "confidence_score": 0.85,
        "loan_eligible": True,
        "selected_model": "xgboost_calibrated",
        "risk_assessment": {
            "risk_score_10": 3.5,
            "grade": "B",
            "components": {"bureau_score": 0.3},
            "drivers": ["bureau_score"],
            "context": {},
            "reasons": ["Good credit"],
        },
        "compliance_result": {
            "is_compliant": True,
            "compliance_score": 0.95,
            "hard_violations": [],
            "hard_warnings": [],
            "soft_violations": [],
            "explanation": "Compliant with RBI digital lending guidelines",
            "loan_type": "housing",
            "rules_applied": ["age_range", "foir_limit"],
            "rag_guidelines_used": [],
        },
        "xai_report": None,
        "metadata": {
            "processing_time_ms": 120.0,
            "model_version": "2.0.0",
            "decision_id": "TEST-001",
            "short_circuited": False,
        },
    }

    response_obj = FinalDecisionResponse(**sample_response)
    assert response_obj.decision == "APPROVED"
    assert response_obj.risk_assessment.risk_score_10 == 3.5
    assert response_obj.risk_assessment.grade == "B"
    assert response_obj.compliance_result.is_compliant is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])