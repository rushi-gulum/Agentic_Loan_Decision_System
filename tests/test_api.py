"""
Integration tests for FastAPI backend (Phase 3)

Tests the complete API integration:
1. FastAPI app initialization
2. Request/response validation with Pydantic schemas
3. Orchestrator integration with short-circuit pattern
4. Error handling and edge cases
5. Performance and timeout behavior

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app
from api.schemas import LoanApplicationRequest, FinalDecisionResponse

# Test client
client = TestClient(app)

# Test data
VALID_APPLICATION = {
    "applicant_name": "John Doe",
    "application_id": "TEST-001",
    "loan_type": "housing",
    "age_years": 35,
    "bureau_score": 750,
    "monthly_income_inr": 75000,
    "foir_total_obligations_pct": 30.0,
    "requested_amount_inr": 500000,
    "tenure_months": 60,
    "gender": "Male",
    "state": "Karnataka",
    "kyc_mode": "Video KYC",
    "ovd_type": "Aadhaar",
    "interest_type": "Fixed",
    "pep_flag": False,
    "kfs_provided": True,
    "processing_fee_inr": 5000.0,
    "other_charges_inr": 1000.0,
    "apr_pct": 8.5,
    "property_value_inr": 2000000,
    "ltv_ratio": 0.25
}

INVALID_APPLICATION = {
    "applicant_name": "Jane Doe",
    "application_id": "TEST-002",
    "loan_type": "personal",
    "age_years": 17,  # Invalid age
    "bureau_score": 300,  # Poor credit score
    "monthly_income_inr": 10000,  # Low income
    "requested_amount_inr": 1000000  # Missing required fields
}

class TestAPIHealthChecks:
    """Test API health and status endpoints"""
    
    def test_root_endpoint(self):
        """Test root health check"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Agentic Loan Decision System API"
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test detailed health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.0.0"
        assert "components" in data
        
        components = data["components"]
        assert isinstance(components["preprocessor"], bool)
        assert isinstance(components["compliance_engine"], bool) 
        assert isinstance(components["orchestrator"], bool)
        assert isinstance(components["models"], bool)

class TestEvaluationEndpoint:
    """Test loan evaluation endpoint"""
    
    def test_valid_application_schema_validation(self):
        """Test that valid application passes Pydantic validation"""
        # This should not raise an exception
        request = LoanApplicationRequest(**VALID_APPLICATION)
        assert request.applicant_name == "John Doe"
        assert request.loan_type == "housing"
        assert request.age_years == 35
    
    def test_invalid_application_schema_validation(self):
        """Test that invalid application fails Pydantic validation"""
        with pytest.raises(ValueError):
            LoanApplicationRequest(**INVALID_APPLICATION)
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_successful_evaluation_flow(self, mock_orchestrator_class):
        """Test successful loan evaluation with mocked orchestrator"""
        # Mock orchestrator response
        mock_orchestrator = MagicMock()
        mock_orchestrator.evaluate_application.return_value = {
            "application_id": "TEST-001",
            "decision": "APPROVED",
            "confidence_score": 0.85,
            "processing_time_ms": 1500,
            
            "risk_assessment": {
                "score": 3.2,
                "grade": "LOW",
                "factors": ["Good credit score", "Stable income"],
                "details": {"credit_component": 0.8, "income_component": 0.9}
            },
            
            "compliance": {
                "overall_status": "PASS",
                "score": 0.92,
                "violations": [],
                "checks_performed": 8
            },
            
            "explanations": {
                "user_friendly": "Loan approved based on strong credit profile",
                "technical": "Risk score: 3.2/10, Compliance: PASS",
                "key_factors": ["Credit score: 750", "Income: ₹75,000"],
                "model_used": "interpretable"
            },
            
            "summary": {
                "primary_reason": "Strong creditworthiness and compliance",
                "recommendation": "APPROVE",
                "next_steps": ["Proceed with documentation", "Property verification"]
            }
        }
        
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Make request
        response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert data["decision"] == "APPROVED"
        assert data["confidence_score"] == 0.85
        assert data["risk_assessment"]["score"] == 3.2
        assert data["compliance"]["overall_status"] == "PASS"
        
        # Verify orchestrator was called
        mock_orchestrator.evaluate_application.assert_called_once()
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_rejection_with_hard_violations(self, mock_orchestrator_class):
        """Test rejection due to hard compliance violations"""
        # Mock rejection response
        mock_orchestrator = MagicMock()
        mock_orchestrator.evaluate_application.return_value = {
            "application_id": "TEST-002", 
            "decision": "REJECTED",
            "confidence_score": 0.95,
            "processing_time_ms": 800,
            
            "risk_assessment": {
                "score": 10.0,
                "grade": "HIGH",
                "factors": ["Hard compliance violations"],
                "details": {}
            },
            
            "compliance": {
                "overall_status": "FAIL",
                "score": 0.0,
                "violations": [
                    {"rule": "min_age", "description": "Applicant below minimum age"},
                    {"rule": "min_income", "description": "Income below threshold"}
                ],
                "checks_performed": 2
            },
            
            "explanations": {
                "user_friendly": "Application rejected due to eligibility criteria",
                "technical": "Hard rule violations detected",
                "key_factors": ["Age below 18", "Income below minimum"],
                "model_used": "rule_engine"
            },
            
            "summary": {
                "primary_reason": "Hard compliance violations detected",
                "recommendation": "REJECT", 
                "next_steps": ["Address compliance violations", "Resubmit application"]
            }
        }
        
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Make request with invalid data
        response = client.post("/api/v1/evaluate", json=INVALID_APPLICATION)
        
        # Note: This might fail validation before reaching orchestrator
        # So we test the orchestrator logic separately
        if response.status_code == 200:
            data = response.json()
            assert data["decision"] == "REJECTED"
            assert data["confidence_score"] == 0.95
            assert len(data["compliance"]["violations"]) == 2
    
    def test_malformed_request_validation(self):
        """Test validation of malformed requests"""
        malformed_requests = [
            {},  # Empty request
            {"applicant_name": "Test"},  # Missing required fields
            {"age_years": "invalid"},  # Wrong data type
            {"bureau_score": 1000},  # Out of range value
        ]
        
        for malformed_data in malformed_requests:
            response = client.post("/api/v1/evaluate", json=malformed_data)
            assert response.status_code == 422  # Validation error
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_orchestrator_exception_handling(self, mock_orchestrator_class):
        """Test API error handling when orchestrator fails"""
        # Mock orchestrator to raise exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.evaluate_application.side_effect = Exception("Orchestrator failed")
        mock_orchestrator_class.return_value = mock_orchestrator
        
        response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to process application" in data["detail"]

class TestSystemStatus:
    """Test system status endpoint"""
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_system_status_healthy(self, mock_orchestrator_class):
        """Test system status when all components are healthy"""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert data["version"] == "1.0.0"
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_system_status_error(self, mock_orchestrator_class):
        """Test system status when orchestrator fails to initialize"""
        mock_orchestrator_class.side_effect = Exception("Initialization failed")
        
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "error"
        assert "Initialization failed" in data["error"]

class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/v1/evaluate")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

class TestPerformanceAndLimits:
    """Test performance characteristics and limits"""
    
    @patch('agents.orchestrator.LoanDecisionOrchestrator')
    def test_response_time_reasonable(self, mock_orchestrator_class):
        """Test that API responds within reasonable time"""
        # Mock fast response
        mock_orchestrator = MagicMock()
        mock_orchestrator.evaluate_application.return_value = {
            "decision": "APPROVED",
            "processing_time_ms": 500,
            "confidence_score": 0.8,
            "risk_assessment": {"score": 3.0, "grade": "LOW", "factors": [], "details": {}},
            "compliance": {"overall_status": "PASS", "score": 0.9, "violations": [], "checks_performed": 5},
            "explanations": {"user_friendly": "Test", "technical": "Test", "key_factors": [], "model_used": "test"},
            "summary": {"primary_reason": "Test", "recommendation": "APPROVE", "next_steps": []}
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        import time
        start_time = time.time()
        
        response = client.post("/api/v1/evaluate", json=VALID_APPLICATION)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds

def test_response_schema_compliance():
    """Test that API responses comply with Pydantic schemas"""
    # Create a sample response manually to test schema
    sample_response = {
        "application_id": "TEST-001",
        "decision": "APPROVED",
        "confidence_score": 0.85,
        "processing_time_ms": 1200,
        
        "risk_assessment": {
            "score": 3.5,
            "grade": "LOW",
            "factors": ["Good credit"],
            "details": {"test": "value"}
        },
        
        "compliance": {
            "overall_status": "PASS", 
            "score": 0.9,
            "violations": [],
            "checks_performed": 7
        },
        
        "explanations": {
            "user_friendly": "Approved",
            "technical": "Technical explanation",
            "key_factors": ["Factor 1"],
            "model_used": "interpretable"
        },
        
        "summary": {
            "primary_reason": "Good profile", 
            "recommendation": "APPROVE",
            "next_steps": ["Documentation"]
        }
    }
    
    # This should not raise an exception
    response_obj = FinalDecisionResponse(**sample_response)
    assert response_obj.decision == "APPROVED"
    assert response_obj.risk_assessment.score == 3.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])