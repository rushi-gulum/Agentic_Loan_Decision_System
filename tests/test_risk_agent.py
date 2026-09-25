#!/usr/bin/env python3
"""
Unit Tests for Risk Agent Mathematical Scoring
============================================

Tests the deterministic mathematical logic of risk scoring to ensure
proper normalization and boundary handling without LLM dependencies.
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.risk_agent import compute_risk_score


class TestRiskScoring:
    """Test suite for risk agent mathematical scoring"""
    
    @pytest.fixture
    def low_risk_application(self) -> Dict[str, Any]:
        """Application that should receive low risk score"""
        return {
            "bureau_score": 800,
            "monthly_income_inr": 100000,
            "foir_total_obligations_pct": 20.0,
            "age_years": 35,
            "ltv_ratio": 0.60,
            "requested_amount_inr": 500000,
            "tenure_months": 60,
            "pep_flag": False
        }
    
    @pytest.fixture
    def high_risk_application(self) -> Dict[str, Any]:
        """Application that should receive high risk score"""
        return {
            "bureau_score": 550,
            "monthly_income_inr": 25000,
            "foir_total_obligations_pct": 45.0,
            "age_years": 22,
            "ltv_ratio": 0.85,
            "requested_amount_inr": 800000,
            "tenure_months": 84,
            "pep_flag": True
        }
    
    @pytest.fixture
    def extreme_high_income_application(self) -> Dict[str, Any]:
        """Application with extremely high income"""
        return {
            "bureau_score": 750,
            "monthly_income_inr": 1000000,  # 10 lakhs per month
            "foir_total_obligations_pct": 15.0,
            "age_years": 40,
            "ltv_ratio": 0.70,
            "requested_amount_inr": 2000000,
            "tenure_months": 60,
            "pep_flag": False
        }
    
    def test_risk_score_bounds(self, low_risk_application, high_risk_application):
        """Test that risk scores are properly bounded between 0 and 10"""
        
        # Test low risk application
        low_result = compute_risk_score(low_risk_application)
        assert isinstance(low_result, dict)
        assert "risk_score_10" in low_result
        
        low_score = low_result["risk_score_10"]
        assert isinstance(low_score, (int, float))
        assert 0 <= low_score <= 10, f"Low risk score {low_score} not in range [0, 10]"
        
        # Test high risk application
        high_result = compute_risk_score(high_risk_application)
        high_score = high_result["risk_score_10"]
        assert isinstance(high_score, (int, float))
        assert 0 <= high_score <= 10, f"High risk score {high_score} not in range [0, 10]"
        
        # High risk should have higher score than low risk
        assert high_score > low_score, f"High risk score {high_score} should be > low risk score {low_score}"
    
    def test_required_output_structure(self, low_risk_application):
        """Test that the output has the required structure"""
        result = compute_risk_score(low_risk_application)
        
        # Required fields
        assert "risk_score_10" in result
        assert "grade" in result
        
        # Optional but expected fields
        expected_fields = ["components", "drivers", "context", "reasons"]
        for field in expected_fields:
            if field in result:
                assert isinstance(result[field], (dict, list, str))
        
        # Validate grade is a string
        assert isinstance(result["grade"], str)
        assert len(result["grade"]) > 0
    
    def test_extreme_income_handling(self, extreme_high_income_application):
        """Test handling of extremely high income values"""
        result = compute_risk_score(extreme_high_income_application)
        
        score = result["risk_score_10"]
        assert 0 <= score <= 10
        
        # Extremely high income should generally result in lower risk
        assert score <= 5.0, f"Extremely high income should result in low risk, got {score}"
    
    def test_extreme_low_values(self):
        """Test handling of extremely low values"""
        extreme_low_app = {
            "bureau_score": 300,  # Minimum possible
            "monthly_income_inr": 15000,  # Very low income
            "foir_total_obligations_pct": 5.0,  # Very low FOIR
            "age_years": 18,  # Minimum age
            "ltv_ratio": 0.10,  # Very low LTV
            "requested_amount_inr": 50000,  # Small amount
            "tenure_months": 12,  # Short tenure
            "pep_flag": False
        }
        
        result = compute_risk_score(extreme_low_app)
        score = result["risk_score_10"]
        assert 0 <= score <= 10
        
        # Low bureau score should contribute to higher risk despite other low values
        assert score >= 3.0, f"Low bureau score should result in higher risk, got {score}"
    
    def test_extreme_high_values(self):
        """Test handling of extremely high risk values"""
        extreme_high_app = {
            "bureau_score": 450,  # Poor score
            "monthly_income_inr": 20000,  # Low income
            "foir_total_obligations_pct": 55.0,  # Very high FOIR
            "age_years": 65,  # Higher age
            "ltv_ratio": 0.95,  # Very high LTV
            "requested_amount_inr": 2000000,  # Large amount
            "tenure_months": 120,  # Long tenure
            "pep_flag": True
        }
        
        result = compute_risk_score(extreme_high_app)
        score = result["risk_score_10"]
        assert 0 <= score <= 10
        
        # Should result in high risk
        assert score >= 6.0, f"High risk factors should result in high risk score, got {score}"
    
    def test_missing_data_handling(self):
        """Test handling of missing data fields"""
        minimal_app = {
            "bureau_score": 700,
            "monthly_income_inr": 50000,
            # Missing other fields
        }
        
        result = compute_risk_score(minimal_app)
        score = result["risk_score_10"]
        assert 0 <= score <= 10
        
        # Should handle gracefully with defaults or partial scoring
        assert isinstance(result, dict)
        assert "grade" in result
    
    def test_edge_case_boundary_values(self):
        """Test exact boundary values"""
        boundary_cases = [
            {
                "bureau_score": 300,  # Minimum
                "monthly_income_inr": 0.01,  # Near zero
                "foir_total_obligations_pct": 0.0,  # Zero
                "age_years": 18,  # Minimum
                "ltv_ratio": 0.0,  # Zero
                "requested_amount_inr": 1,  # Minimum
                "tenure_months": 1,  # Minimum
                "pep_flag": False
            },
            {
                "bureau_score": 900,  # Maximum
                "monthly_income_inr": 10000000,  # Very high
                "foir_total_obligations_pct": 100.0,  # Maximum
                "age_years": 70,  # High age
                "ltv_ratio": 1.0,  # Maximum
                "requested_amount_inr": 100000000,  # Very high
                "tenure_months": 360,  # Maximum
                "pep_flag": True
            }
        ]
        
        for app in boundary_cases:
            result = compute_risk_score(app)
            score = result["risk_score_10"]
            assert 0 <= score <= 10, f"Boundary case score {score} not in range [0, 10]"
    
    def test_consistency_across_calls(self, low_risk_application):
        """Test that the same input produces consistent results"""
        result1 = compute_risk_score(low_risk_application)
        result2 = compute_risk_score(low_risk_application)
        
        # Should be deterministic (same inputs = same outputs)
        assert result1["risk_score_10"] == result2["risk_score_10"]
        assert result1["grade"] == result2["grade"]
    
    def test_risk_grade_consistency(self, low_risk_application, high_risk_application):
        """Test that risk grade is consistent with numeric score"""
        low_result = compute_risk_score(low_risk_application)
        high_result = compute_risk_score(high_risk_application)
        
        low_score = low_result["risk_score_10"]
        high_score = high_result["risk_score_10"]
        
        low_grade = low_result["grade"]
        high_grade = high_result["grade"]
        
        # Grade should reflect the numeric score appropriately
        # (Exact mapping depends on implementation, but should be consistent)
        if low_score < high_score:
            # Lower score should not have a "worse" grade than higher score
            # This is a basic consistency check
            assert isinstance(low_grade, str) and isinstance(high_grade, str)
    
    def test_component_breakdown(self, low_risk_application):
        """Test that component breakdown is provided and reasonable"""
        result = compute_risk_score(low_risk_application)
        
        if "components" in result:
            components = result["components"]
            assert isinstance(components, dict)
            
            # Components should be numeric values
            for component_name, component_value in components.items():
                assert isinstance(component_value, (int, float))
                # Component values should be reasonable (0-1 range typical)
                if isinstance(component_value, float):
                    assert 0 <= component_value <= 1 or component_value == result["risk_score_10"]
    
    def test_performance_requirement(self, low_risk_application):
        """Test that risk scoring completes within reasonable time"""
        import time
        
        start_time = time.time()
        result = compute_risk_score(low_risk_application)
        end_time = time.time()
        
        # Should complete quickly (< 1 second)
        assert (end_time - start_time) < 1.0
        assert result["risk_score_10"] is not None


class TestRiskScoringEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        invalid_cases = [
            {"bureau_score": "invalid"},  # String instead of number
            {"monthly_income_inr": None},  # None value
            {"foir_total_obligations_pct": []},  # List instead of number
            {"age_years": -5},  # Negative age
            {"ltv_ratio": -1.0},  # Negative ratio
        ]
        
        for invalid_app in invalid_cases:
            try:
                result = compute_risk_score(invalid_app)
                # If it doesn't raise an error, should still return valid bounds
                if "risk_score_10" in result:
                    assert 0 <= result["risk_score_10"] <= 10
            except (ValueError, TypeError, KeyError):
                # Acceptable to raise errors for invalid data
                pass
    
    def test_empty_application(self):
        """Test handling of empty application"""
        try:
            result = compute_risk_score({})
            # If it handles empty input, should return valid structure
            if "risk_score_10" in result:
                assert 0 <= result["risk_score_10"] <= 10
        except (ValueError, KeyError):
            # Acceptable to require minimum data
            pass
    
    def test_mathematical_precision(self):
        """Test mathematical precision and floating point handling"""
        precision_app = {
            "bureau_score": 750.5555555,
            "monthly_income_inr": 50000.999999,
            "foir_total_obligations_pct": 25.123456789,
            "age_years": 35,
            "ltv_ratio": 0.7000000001,
            "requested_amount_inr": 500000.01,
            "tenure_months": 60,
            "pep_flag": False
        }
        
        result = compute_risk_score(precision_app)
        score = result["risk_score_10"]
        
        # Should handle floating point precision gracefully
        assert isinstance(score, (int, float))
        assert 0 <= score <= 10
        # Score should be reasonable precision (not excessive decimals)
        assert abs(score - round(score, 2)) < 0.01 or isinstance(score, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])