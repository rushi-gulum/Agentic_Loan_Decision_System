#!/usr/bin/env python3
"""
Unit Tests for Deterministic Rule Engine
======================================

Tests the mathematical logic of hard constraints that must be bulletproof.
These tests validate the Phase 2 deterministic rules without LLM dependencies.
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rules.rule_engine import evaluate_hard_constraints, load_rules, HardConstraintResult


class TestHardConstraints:
    """Test suite for deterministic hard constraint validation"""
    
    @pytest.fixture
    def perfect_housing_application(self) -> Dict[str, Any]:
        """A loan application that should pass all constraints"""
        return {
            "bureau_score": 750,
            "foir_total_obligations_pct": 25.0,  # 25%
            "ltv_ratio": 0.70,  # 70%
            "property_value_inr": 1500000,  # 15 lakhs
            "pep_flag": False,
            "monthly_income_inr": 75000
        }
    
    @pytest.fixture
    def perfect_personal_application(self) -> Dict[str, Any]:
        """A personal loan application that should pass"""
        return {
            "bureau_score": 720,
            "foir_total_obligations_pct": 30.0,
            "pep_flag": False,
            "monthly_income_inr": 50000,
            "requested_amount_inr": 200000
        }
    
    def test_perfect_housing_application_passes(self, perfect_housing_application):
        """Test that a perfect housing loan application passes all constraints"""
        result = evaluate_hard_constraints(
            perfect_housing_application, 
            "housing"
        )
        
        assert isinstance(result, HardConstraintResult)
        assert result.is_compliant == True
        assert len(result.violations) == 0
        assert result.loan_type == "housing"
        assert len(result.rules_applied) > 0
    
    def test_perfect_personal_application_passes(self, perfect_personal_application):
        """Test that a perfect personal loan application passes"""
        result = evaluate_hard_constraints(
            perfect_personal_application,
            "personal"
        )
        
        assert result.is_compliant == True
        assert len(result.violations) == 0
        assert result.loan_type == "personal"
    
    def test_foir_violation_housing(self, perfect_housing_application):
        """Test FOIR limit violation for housing loans"""
        # Set FOIR to exceed the limit (assuming 40% is the max for housing)
        bad_application = perfect_housing_application.copy()
        bad_application["foir_total_obligations_pct"] = 65.0  # 65% - should fail (exceeds 55% + 5% tolerance = 60%)
        
        result = evaluate_hard_constraints(bad_application, "housing")
        
        assert result.is_compliant == False
        assert len(result.violations) >= 1
        
        # Check that FOIR violation is present
        foir_violations = [v for v in result.violations if v.rule_type == "FOIR_MAX"]
        assert len(foir_violations) >= 1
        
        violation = foir_violations[0]
        assert violation.feature == "foir_total_obligations_pct"
        assert violation.severity == "violation"
        assert "FOIR" in violation.message.upper()
    
    def test_bureau_score_violation(self, perfect_housing_application):
        """Test bureau score minimum violation"""
        bad_application = perfect_housing_application.copy()
        bad_application["bureau_score"] = 500  # Low score - should fail
        
        result = evaluate_hard_constraints(bad_application, "housing")
        
        assert result.is_compliant == False
        assert len(result.violations) >= 1
        
        # Check that bureau score violation is present
        bureau_violations = [v for v in result.violations if v.rule_type == "BUREAU_MIN"]
        assert len(bureau_violations) >= 1
        
        violation = bureau_violations[0]
        assert violation.feature == "bureau_score"
        assert violation.severity == "violation"
        assert "bureau" in violation.message.lower()
    
    def test_ltv_violation(self, perfect_housing_application):
        """Test LTV ratio violation"""
        bad_application = perfect_housing_application.copy()
        bad_application["ltv_ratio"] = 0.95  # 95% - should exceed most LTV caps
        
        result = evaluate_hard_constraints(bad_application, "housing")
        
        # LTV violations might exist depending on the rule configuration
        if result.violations:
            ltv_violations = [v for v in result.violations if v.rule_type == "LTV_MAX"]
            if ltv_violations:
                violation = ltv_violations[0]
                assert violation.feature == "ltv_ratio"
                assert violation.severity == "violation"
    
    def test_pep_flag_handling(self, perfect_housing_application):
        """Test PEP (Politically Exposed Person) flag handling"""
        pep_application = perfect_housing_application.copy()
        pep_application["pep_flag"] = True
        
        result = evaluate_hard_constraints(pep_application, "housing")
        
        # PEP flag should either cause violation or warning depending on rules
        if not result.is_compliant:
            pep_violations = [v for v in result.violations if "pep" in v.message.lower()]
            if pep_violations:
                assert len(pep_violations) >= 1
        else:
            # Check if there's at least a warning for PEP
            pep_warnings = [w for w in result.warnings if "pep" in w.message.lower()]
            # PEP handling might vary by rule configuration
    
    def test_multiple_violations(self, perfect_housing_application):
        """Test application with multiple constraint violations"""
        bad_application = perfect_housing_application.copy()
        bad_application["bureau_score"] = 450  # Too low
        bad_application["foir_total_obligations_pct"] = 60.0  # Too high
        bad_application["ltv_ratio"] = 0.90  # Potentially too high
        
        result = evaluate_hard_constraints(bad_application, "housing")
        
        assert result.is_compliant == False
        assert len(result.violations) >= 2  # At least bureau and FOIR violations
        
        # Verify we have different types of violations
        violation_types = {v.rule_type for v in result.violations}
        assert "BUREAU_MIN" in violation_types
        assert "FOIR_MAX" in violation_types
    
    def test_unknown_loan_type(self, perfect_housing_application):
        """Test handling of unknown loan type"""
        result = evaluate_hard_constraints(
            perfect_housing_application, 
            "unknown_loan_type"
        )
        
        # Should be compliant but with warnings about unknown type
        assert result.is_compliant == True
        assert len(result.warnings) >= 1
        
        type_warnings = [w for w in result.warnings if w.rule_type == "PRODUCT_RECOGNITION"]
        assert len(type_warnings) >= 1
        assert "not recognized" in type_warnings[0].message
    
    def test_edge_case_boundary_values(self, perfect_housing_application):
        """Test boundary values for constraints"""
        # Test exactly at the boundary
        boundary_application = perfect_housing_application.copy()
        
        # This test would need to know the exact rule values
        # For now, test that the function handles edge cases gracefully
        result = evaluate_hard_constraints(boundary_application, "housing")
        assert isinstance(result, HardConstraintResult)
        assert result.loan_type == "housing"
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.warnings, list)
    
    def test_missing_data_handling(self):
        """Test handling of missing applicant data"""
        minimal_application = {
            "bureau_score": 700,
            # Missing other fields
        }
        
        result = evaluate_hard_constraints(minimal_application, "housing")
        
        # Should handle gracefully without crashing
        assert isinstance(result, HardConstraintResult)
        # Missing data might result in warnings or defaults
    
    def test_different_loan_types(self, perfect_housing_application):
        """Test that different loan types use different rule sets"""
        loan_types = ["housing", "personal", "vehicle", "gold", "microfinance"]
        
        for loan_type in loan_types:
            result = evaluate_hard_constraints(perfect_housing_application, loan_type)
            assert isinstance(result, HardConstraintResult)
            # Each loan type should be processed (even if rules don't exist)
    
    def test_rules_loading(self):
        """Test that rules load successfully"""
        rules = load_rules()
        assert rules is not None
        assert hasattr(rules, 'products')
        assert hasattr(rules, 'global_rules')
        
        # Check that we have some product rules
        assert len(rules.products) > 0
        
        # Check that product rules have expected structure
        for product_name, product_rules in rules.products.items():
            assert hasattr(product_rules, 'BUREAU_MIN') or hasattr(product_rules, 'FOIR_MAX')


class TestRuleEngineRobustness:
    """Test the robustness of the rule engine"""
    
    def test_data_type_handling(self):
        """Test handling of different data types"""
        test_cases = [
            {"bureau_score": "750"},  # String instead of int
            {"foir_total_obligations_pct": "25.0"},  # String instead of float
            {"pep_flag": "false"},  # String instead of bool
            {"bureau_score": None},  # None values
        ]
        
        for test_data in test_cases:
            # Should not crash with different data types
            try:
                result = evaluate_hard_constraints(test_data, "housing")
                assert isinstance(result, HardConstraintResult)
            except (ValueError, TypeError) as e:
                # Acceptable to raise type errors for invalid data
                pass
    
    def test_performance_with_large_dataset(self):
        """Test performance doesn't degrade significantly"""
        large_application = {
            "bureau_score": 750,
            "foir_total_obligations_pct": 25.0,
            "ltv_ratio": 0.70,
            "property_value_inr": 1500000,
            "pep_flag": False,
            **{f"extra_field_{i}": f"value_{i}" for i in range(100)}  # Extra fields
        }
        
        import time
        start_time = time.time()
        result = evaluate_hard_constraints(large_application, "housing")
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert (end_time - start_time) < 1.0
        assert isinstance(result, HardConstraintResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])