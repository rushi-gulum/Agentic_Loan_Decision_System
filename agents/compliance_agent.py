"""
Hybrid Compliance Agent — Deterministic + RAG-based Policy Enforcement
====================================================================
Implements a guardrail pattern:
1. Hard Constraints: Deterministic mathematical checks (cannot be hallucinated)
2. Soft Constraints: RAG + LLM for nuanced policy interpretation

Input:
  {
    "applicant": {...},
    "guidelines": {
        "feature_guidelines": {
            "age_years": {"summary": "..."},
            "ltv_ratio": {"summary": "..."},
            ...
        }
    }
  }

Output:
  {
    "is_compliant": bool,
    "hard_violations": [...],  # Deterministic rule failures
    "soft_violations": [...],  # LLM-identified nuanced violations  
    "warnings": [...],
    "compliance_score": float (0–1),
    "explanation": str  # Human-readable summary
  }
"""

import sys
import os
import re
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_utility import get_llm
from rules.rule_engine import (
    evaluate_hard_constraints, 
    HardConstraintResult,
    PolicyViolation,
    load_rules
)


# ============================================================================
# PYDANTIC MODELS FOR COMPLIANCE OUTPUT
# ============================================================================

class SoftViolation(BaseModel):
    """Represents a soft policy violation identified by LLM."""
    feature: str = Field(description="Feature with potential issue")
    guideline_reference: str = Field(description="RBI guideline reference")
    concern: str = Field(description="Policy concern identified")
    severity: str = Field(description="Severity: 'minor', 'moderate', 'significant'")
    recommendation: str = Field(description="Recommended action")


class ComplianceResult(BaseModel):
    """Complete compliance evaluation result."""
    is_compliant: bool = Field(description="Overall compliance status")
    
    # Hard constraint results
    hard_violations: List[PolicyViolation] = Field(default_factory=list)
    hard_warnings: List[PolicyViolation] = Field(default_factory=list)
    
    # Soft constraint results  
    soft_violations: List[SoftViolation] = Field(default_factory=list)
    
    # Metadata
    loan_type: str = Field(description="Loan type evaluated")
    compliance_score: float = Field(ge=0, le=1, description="Overall compliance score")
    explanation: str = Field(description="Human-readable explanation")
    
    # RAG context
    rag_guidelines_used: List[str] = Field(default_factory=list)
    rules_applied: List[str] = Field(default_factory=list)


# ============================================================================
# HYBRID COMPLIANCE FUNCTIONS
# ============================================================================

def evaluate_hard_compliance(applicant_data: Dict[str, Any]) -> HardConstraintResult:
    """
    Step 1: Evaluate deterministic hard constraints.
    This cannot be hallucinated or misinterpreted.
    """
    loan_type = applicant_data.get("loan_type", "personal")
    
    try:
        return evaluate_hard_constraints(applicant_data, loan_type)
    except Exception as e:
        # If rule engine fails, create a safe fallback
        return HardConstraintResult(
            is_compliant=False,
            violations=[PolicyViolation(
                feature="rule_engine",
                rule_type="SYSTEM_ERROR",
                rule_value="functional_rule_engine",
                actual_value="error",
                severity="violation",
                message=f"Rule engine error: {str(e)}"
            )],
            loan_type=loan_type,
            rules_applied=["SYSTEM_ERROR"]
        )


def evaluate_soft_compliance(
    applicant_data: Dict[str, Any], 
    guidelines: Dict[str, Any],
    hard_result: HardConstraintResult
) -> List[SoftViolation]:
    """
    Step 2: Use RAG + LLM for nuanced policy evaluation.
    Only called if hard constraints pass or for explanation purposes.
    """
    
    # Skip soft evaluation if no guidelines available
    if not guidelines or not guidelines.get("feature_guidelines"):
        return []
    
    try:
        llm = get_llm()
        
        # Prepare hard constraint summary
        hard_status = "PASSED" if hard_result.is_compliant else f"FAILED ({len(hard_result.violations)} violations)"
        
        prompt_text = f"""
You are an RBI compliance specialist conducting a nuanced policy review.

CONTEXT:
- Hard mathematical constraints have already been checked separately.
- Your job is to evaluate SOFT CONSTRAINTS and policy nuances based on RBI guidelines.
- Focus on the spirit of regulations, not just mathematical limits.

GUIDELINES FOR EVALUATION:
1. Look for income structure concerns (e.g., irregular income, cash-heavy businesses)
2. Check loan purpose alignment with regulatory priorities  
3. Assess documentation quality and completeness
4. Identify potential regulatory flags (sector restrictions, etc.)
5. Consider applicant profile vs. product suitability

INSTRUCTIONS:
- If you find NO soft policy concerns, return: {{"soft_violations": []}}
- If you identify concerns, structure them as shown below
- Be conservative but reasonable - don't create violations where none exist
- Focus on actionable regulatory guidance

Return STRICTLY valid JSON in this format:

{{
  "soft_violations": [
    {{
      "feature": "feature_name",
      "guideline_reference": "RBI guideline reference",  
      "concern": "specific policy concern",
      "severity": "minor/moderate/significant",
      "recommendation": "suggested action"
    }}
  ]
}}

APPLICANT DATA:
{json.dumps(applicant_data, indent=2)}

RBI GUIDELINES:
{json.dumps(guidelines, indent=2)}

HARD CONSTRAINT STATUS:
{hard_status}
"""
        
        result = llm.invoke(prompt_text)
        if hasattr(result, 'content'):
            result = result.content
        else:
            result = str(result)
        
        # Parse LLM response
        json_block = re.search(r"\{[\s\S]*\}", result)
        if json_block:
            parsed = json.loads(json_block.group())
            soft_violations_data = parsed.get("soft_violations", [])
            
            # Convert to Pydantic models
            soft_violations = []
            for violation_data in soft_violations_data:
                try:
                    soft_violations.append(SoftViolation(**violation_data))
                except Exception as e:
                    print(f"⚠️ Warning: Invalid soft violation format: {e}")
            
            return soft_violations
        
    except Exception as e:
        print(f"⚠️ Warning: Soft compliance evaluation failed: {e}")
    
    return []  # Safe fallback


def format_compliance_explanation(
    hard_result: HardConstraintResult,
    soft_violations: List[SoftViolation],
    overall_compliant: bool
) -> str:
    """
    Generate a professional, human-readable compliance explanation.
    """
    try:
        llm = get_llm()
        
        # Create prompt manually for MockLLM compatibility
        hard_violations_count = len(hard_result.violations)
        soft_concerns_count = len(soft_violations)
        rules_applied = ", ".join(hard_result.rules_applied)
        
        hard_violation_messages = [v.message for v in hard_result.violations]
        soft_concerns = [v.concern for v in soft_violations]
        
        prompt_text = f"""
You are a senior compliance officer writing a loan decision explanation.

Create a clear, professional explanation that covers:
1. Overall compliance status
2. Key regulatory requirements checked
3. Any violations or concerns found
4. Recommendations for next steps

Keep it concise (2-3 paragraphs) and suitable for both internal teams and regulatory review.

COMPLIANCE DATA:
- Overall Status: {"COMPLIANT" if overall_compliant else "NON-COMPLIANT"}
- Hard Violations: {hard_violations_count}
- Soft Concerns: {soft_concerns_count}
- Rules Applied: {rules_applied}

HARD VIOLATIONS:
{hard_violation_messages}

SOFT CONCERNS:
{soft_concerns}

Write a professional compliance explanation:
"""
        
        explanation = llm.invoke(prompt_text)
        if hasattr(explanation, 'content'):
            return explanation.content.strip()
        else:
            return str(explanation).strip()
            
    except Exception as e:
        print(f"⚠️ Warning: LLM explanation generation failed: {e}")
        # Safe fallback
        if overall_compliant:
            return f"Application meets all regulatory requirements for {hard_result.loan_type} loans. All hard constraints satisfied with {len(hard_result.rules_applied)} rules verified."
        else:
            violations_summary = "; ".join([v.message for v in hard_result.violations])
            return f"Application does not meet regulatory requirements: {violations_summary}"


def check_rbi_compliance(applicant: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main hybrid compliance check function.
    
    Implements the guardrail pattern:
    1. Hard constraints first (deterministic)
    2. Soft constraints only if needed (LLM-based)
    3. Professional explanation generation
    
    Args:
        applicant: Applicant data dictionary
        guidelines: RAG-retrieved guidelines dictionary
        
    Returns:
        Dict containing complete compliance results
    """
    
    # STEP 1: Evaluate hard constraints (deterministic)
    print("🔒 Checking hard constraints...")
    hard_result = evaluate_hard_compliance(applicant)
    
    # STEP 2: Evaluate soft constraints (LLM-based, only if guidelines available)
    soft_violations = []
    rag_guidelines_used = []
    
    if guidelines and guidelines.get("feature_guidelines"):
        print("📋 Checking soft constraints with RAG guidelines...")
        soft_violations = evaluate_soft_compliance(applicant, guidelines, hard_result)
        
        # Extract guideline keys that were used
        rag_guidelines_used = list(guidelines.get("feature_guidelines", {}).keys())
    else:
        print("⚠️ No RAG guidelines provided, skipping soft constraint evaluation")
    
    # STEP 3: Determine overall compliance
    # Fail if hard constraints fail OR if significant soft violations found
    significant_soft_violations = [v for v in soft_violations if v.severity == "significant"]
    overall_compliant = hard_result.is_compliant and len(significant_soft_violations) == 0
    
    # STEP 4: Calculate compliance score
    # Base score from hard constraints, reduced by soft violations
    base_score = 1.0 if hard_result.is_compliant else 0.0
    
    # Reduce score for violations
    violation_penalty = len(hard_result.violations) * 0.25  # Major penalty for hard violations
    warning_penalty = len(hard_result.warnings) * 0.05    # Minor penalty for warnings
    soft_penalty = sum({
        "significant": 0.15,
        "moderate": 0.08, 
        "minor": 0.03
    }.get(v.severity, 0.05) for v in soft_violations)
    
    compliance_score = max(0.0, base_score - violation_penalty - warning_penalty - soft_penalty)
    
    # STEP 5: Generate explanation
    print("📝 Generating compliance explanation...")
    explanation = format_compliance_explanation(hard_result, soft_violations, overall_compliant)
    
    # STEP 6: Create comprehensive result
    result = ComplianceResult(
        is_compliant=overall_compliant,
        hard_violations=hard_result.violations,
        hard_warnings=hard_result.warnings,
        soft_violations=soft_violations,
        loan_type=hard_result.loan_type,
        compliance_score=compliance_score,
        explanation=explanation,
        rag_guidelines_used=rag_guidelines_used,
        rules_applied=hard_result.rules_applied
    )
    
    print(f"✅ Compliance check complete: {'PASS' if overall_compliant else 'FAIL'}")
    
    # Return as dictionary for compatibility with existing orchestrator
    return {
        "is_compliant": result.is_compliant,
        "compliance_score": result.compliance_score,
        "hard_violations": [v.model_dump() for v in result.hard_violations],
        "hard_warnings": [v.model_dump() for v in result.hard_warnings], 
        "soft_violations": [v.model_dump() for v in result.soft_violations],
        "explanation": result.explanation,
        "loan_type": result.loan_type,
        "rules_applied": result.rules_applied,
        "rag_guidelines_used": result.rag_guidelines_used,
        
        # Legacy compatibility fields
        "violations": [
            {
                "feature": v.feature,
                "rule": v.message,
                "status": "violation"
            } for v in result.hard_violations
        ] + [
            {
                "feature": v.feature, 
                "rule": v.concern,
                "status": "soft_violation"
            } for v in result.soft_violations
        ],
        "compliance_summary": result.explanation
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Hybrid Compliance Agent")
    print("=" * 50)
    
    # Test 1: Compliant applicant
    print("\n📊 Test 1: COMPLIANT Applicant")
    compliant_applicant = {
        "loan_type": "housing",
        "age_years": 35,
        "bureau_score": 750,
        "foir_total_obligations_pct": 45.0,
        "ltv_ratio": 0.75,
        "property_value_inr": 5_000_000,
        "pep_flag": False,
        "monthly_income_inr": 100_000
    }
    
    # Mock guidelines
    mock_guidelines = {
        "feature_guidelines": {
            "age_years": {"summary": "Minimum age 21 years for housing loans"},
            "ltv_ratio": {"summary": "LTV should not exceed 80% for housing loans"},
            "monthly_income_inr": {"summary": "Stable income source required"}
        }
    }
    
    try:
        result1 = check_rbi_compliance(compliant_applicant, mock_guidelines)
        print(f"✅ Compliance Status: {result1['is_compliant']}")
        print(f"✅ Score: {result1['compliance_score']:.2f}")
        print(f"✅ Hard Violations: {len(result1['hard_violations'])}")
        print(f"✅ Soft Violations: {len(result1['soft_violations'])}")
        print(f"📝 Explanation: {result1['explanation'][:100]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 2: Non-compliant applicant
    print("\n📊 Test 2: NON-COMPLIANT Applicant")
    non_compliant_applicant = {
        "loan_type": "personal",
        "age_years": 25,
        "bureau_score": 650,  # Below 720 minimum
        "foir_total_obligations_pct": 65.0,  # Above 50% limit
        "pep_flag": False,
        "monthly_income_inr": 50_000
    }
    
    try:
        result2 = check_rbi_compliance(non_compliant_applicant, mock_guidelines)
        print(f"❌ Compliance Status: {result2['is_compliant']}")
        print(f"📊 Score: {result2['compliance_score']:.2f}")
        print(f"🚫 Hard Violations: {len(result2['hard_violations'])}")
        print(f"📝 Explanation: {result2['explanation'][:100]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n🎯 Hybrid Compliance Agent Test Complete!")
    print(f"✅ Hard constraints checked deterministically")
    print(f"✅ Soft constraints handled by LLM when guidelines available")
    print(f"✅ Professional explanations generated")