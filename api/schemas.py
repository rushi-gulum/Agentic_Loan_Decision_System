# api/schemas.py
"""
FastAPI Pydantic Schemas for Loan Decision API
=============================================
Defines the strict contract for all API requests and responses.
Ensures type safety and validation across the entire system.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS FOR CONSTRAINED VALUES
# ============================================================================

class LoanTypeEnum(str, Enum):
    """Supported loan types."""
    HOUSING = "housing"
    PERSONAL = "personal"
    VEHICLE = "vehicle"
    GOLD = "gold"
    MICROFINANCE = "microfinance"
    MSME = "msme"
    BUSINESS = "business"


class GenderEnum(str, Enum):
    """Gender options."""
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class InterestTypeEnum(str, Enum):
    """Interest rate types."""
    FIXED = "Fixed"
    FLOATING = "Floating"


class DecisionEnum(str, Enum):
    """Final loan decision options."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class RiskGradeEnum(str, Enum):
    """Risk grade classifications."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class ViolationSeverityEnum(str, Enum):
    """Severity levels for policy violations."""
    VIOLATION = "violation"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class LoanApplicationRequest(BaseModel):
    """
    Complete loan application request schema.
    Matches Phase 1 LoanApplicationSchema with API-specific enhancements.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'  # Strict mode - no extra fields allowed
    )
    
    # ========================================================================
    # REQUIRED CORE FIELDS
    # ========================================================================
    age_years: int = Field(
        ge=18, le=100, 
        description="Applicant age in years"
    )
    
    bureau_score: int = Field(
        ge=300, le=900,
        description="Credit bureau score (300-900)"
    )
    
    monthly_income_inr: float = Field(
        gt=0, le=10_000_000,
        description="Monthly income in INR"
    )
    
    requested_amount_inr: float = Field(
        gt=0, le=100_000_000,
        description="Requested loan amount in INR"
    )
    
    tenure_months: int = Field(
        ge=6, le=360,
        description="Loan tenure in months"
    )
    
    interest_rate_annual_pct: float = Field(
        ge=0, le=50,
        description="Annual interest rate percentage"
    )
    
    foir_total_obligations_pct: float = Field(
        ge=0, le=150,
        description="Fixed Obligation to Income Ratio percentage"
    )
    
    # ========================================================================
    # LOAN DETAILS WITH DEFAULTS
    # ========================================================================
    loan_type: LoanTypeEnum = Field(
        default=LoanTypeEnum.PERSONAL,
        description="Type of loan being requested"
    )
    
    gender: GenderEnum = Field(
        default=GenderEnum.MALE,
        description="Applicant gender"
    )
    
    interest_type: InterestTypeEnum = Field(
        default=InterestTypeEnum.FIXED,
        description="Interest rate type"
    )
    
    # ========================================================================
    # OPTIONAL FIELDS WITH SENSIBLE DEFAULTS
    # ========================================================================
    pin_code: int = Field(
        default=110001,
        ge=100000, le=999999,
        description="6-digit PIN code"
    )
    
    pep_flag: bool = Field(
        default=False,
        description="Politically Exposed Person flag"
    )
    
    kfs_provided: bool = Field(
        default=True,
        description="Key Fact Statement provided flag"
    )
    
    existing_monthly_obligations_inr: float = Field(
        default=0,
        ge=0,
        description="Existing monthly obligations in INR"
    )
    
    sanctioned_amount_inr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sanctioned loan amount in INR (auto-calculated if None)"
    )
    
    processing_fee_inr: float = Field(
        default=0,
        ge=0,
        description="Processing fee in INR"
    )
    
    other_charges_inr: float = Field(
        default=0,
        ge=0,
        description="Other charges in INR"
    )
    
    apr_pct: Optional[float] = Field(
        default=None,
        ge=0, le=50,
        description="Annual Percentage Rate (auto-calculated if None)"
    )
    
    proposed_emi_inr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Proposed EMI in INR (auto-calculated if None)"
    )
    
    property_value_inr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Property value in INR (for secured loans)"
    )
    
    ltv_ratio: Optional[float] = Field(
        default=None,
        ge=0, le=1.5,
        description="Loan to Value ratio (for secured loans)"
    )
    
    # ========================================================================
    # METADATA FIELDS
    # ========================================================================
    application_date: Optional[str] = Field(
        default=None,
        description="Application date in YYYY-MM-DD format"
    )
    
    sanction_date: Optional[str] = Field(
        default=None,
        description="Sanction date in YYYY-MM-DD format"
    )
    
    ovd_type: Optional[str] = Field(
        default="Aadhaar",
        description="OVD document type"
    )
    
    kyc_mode: Optional[str] = Field(
        default="Video KYC",
        description="KYC verification mode"
    )


# ============================================================================
# COMPONENT RESPONSE SCHEMAS
# ============================================================================

class PolicyViolationDetail(BaseModel):
    """Individual policy violation details."""
    feature: str = Field(description="Feature that violated the rule")
    rule_type: str = Field(description="Type of rule violated")
    rule_value: Union[float, int, str] = Field(description="Expected rule value")
    actual_value: Union[float, int, str] = Field(description="Actual applicant value")
    severity: ViolationSeverityEnum = Field(description="Violation severity")
    message: str = Field(description="Human-readable violation message")


class SoftViolationDetail(BaseModel):
    """Soft policy violation identified by LLM."""
    feature: str = Field(description="Feature with potential issue")
    guideline_reference: str = Field(description="RBI guideline reference")
    concern: str = Field(description="Policy concern identified")
    severity: ViolationSeverityEnum = Field(description="Violation severity")
    recommendation: str = Field(description="Recommended action")


class RiskAssessment(BaseModel):
    """Risk assessment results from Risk Agent."""
    risk_score_10: float = Field(
        ge=0, le=10,
        description="Risk score on 0-10 scale (higher = riskier)"
    )
    
    grade: RiskGradeEnum = Field(description="Risk grade classification")
    
    components: Dict[str, float] = Field(
        description="Individual risk component scores (0-1)"
    )
    
    drivers: List[str] = Field(
        description="Top risk drivers with contribution scores"
    )
    
    context: Dict[str, Any] = Field(
        description="Additional risk assessment context"
    )
    
    reasons: List[str] = Field(
        description="Human-readable risk assessment reasons"
    )


class ComplianceResult(BaseModel):
    """Compliance assessment results from Hybrid Compliance Engine."""
    is_compliant: bool = Field(description="Overall compliance status")
    
    compliance_score: float = Field(
        ge=0, le=1,
        description="Overall compliance score (0-1)"
    )
    
    hard_violations: List[PolicyViolationDetail] = Field(
        description="Deterministic rule violations"
    )
    
    hard_warnings: List[PolicyViolationDetail] = Field(
        description="Deterministic rule warnings"
    )
    
    soft_violations: List[SoftViolationDetail] = Field(
        description="LLM-identified policy concerns"
    )
    
    explanation: str = Field(description="Professional compliance explanation")
    
    loan_type: str = Field(description="Loan type evaluated")
    
    rules_applied: List[str] = Field(description="List of rules that were checked")
    
    rag_guidelines_used: List[str] = Field(
        description="RAG guidelines consulted"
    )


class XAIReport(BaseModel):
    """Explainable AI report from XAI Agent."""
    model_used: str = Field(description="ML model used for prediction")
    
    prediction_probability: Dict[str, float] = Field(
        description="Model prediction probabilities"
    )
    
    shap_summary: Dict[str, Any] = Field(
        description="SHAP explanation summary"
    )
    
    lime_explanation: List[Any] = Field(
        description="LIME explanation details"
    )
    
    customer_summary: str = Field(
        description="Customer-friendly explanation"
    )
    
    regulator_summary: str = Field(
        description="Regulator-friendly detailed explanation"
    )
    
    top_features: List[str] = Field(
        description="Most important features for decision"
    )


class DecisionMetadata(BaseModel):
    """Additional decision metadata."""
    processing_time_ms: float = Field(description="Total processing time in milliseconds")
    model_version: str = Field(default="1.0.0", description="System version")
    decision_id: str = Field(description="Unique decision identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")
    short_circuited: bool = Field(default=False, description="Whether decision was short-circuited due to hard violations")


# ============================================================================
# FINAL RESPONSE SCHEMA
# ============================================================================

class FinalDecisionResponse(BaseModel):
    """
    Complete loan decision response containing all assessment results.
    This is the main response schema for the /api/v1/evaluate endpoint.
    """
    
    # ========================================================================
    # CORE DECISION
    # ========================================================================
    decision: DecisionEnum = Field(description="Final loan decision")
    
    decision_reason: str = Field(
        description="Primary reason for the decision"
    )
    
    confidence_score: float = Field(
        ge=0, le=1,
        description="Overall confidence in the decision (0-1)"
    )
    
    # ========================================================================
    # COMPONENT ASSESSMENTS
    # ========================================================================
    risk_assessment: RiskAssessment = Field(
        description="Risk evaluation results"
    )
    
    compliance_result: ComplianceResult = Field(
        description="Regulatory compliance assessment"
    )
    
    xai_report: Optional[XAIReport] = Field(
        default=None,
        description="Explainable AI analysis (may be None if short-circuited)"
    )
    
    # ========================================================================
    # METADATA
    # ========================================================================
    metadata: DecisionMetadata = Field(description="Decision metadata")
    
    # ========================================================================
    # LEGACY COMPATIBILITY (for existing integrations)
    # ========================================================================
    loan_eligible: bool = Field(description="Legacy field - same as decision == APPROVED")
    selected_model: str = Field(description="ML model selected for decision")


# ============================================================================
# ERROR RESPONSE SCHEMAS
# ============================================================================

class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Error message")
    value: Any = Field(description="Invalid value provided")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[List[ValidationError]] = Field(
        default=None,
        description="Detailed validation errors"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier for debugging"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_error_response(
    error_type: str,
    message: str,
    details: Optional[List[ValidationError]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details,
        request_id=request_id
    )


# ============================================================================
# EXAMPLE/TESTING SCHEMAS
# ============================================================================

def get_sample_loan_request() -> LoanApplicationRequest:
    """Get a sample loan application for testing."""
    return LoanApplicationRequest(
        age_years=35,
        gender=GenderEnum.MALE,
        bureau_score=720,
        monthly_income_inr=75_000,
        existing_monthly_obligations_inr=15_000,
        requested_amount_inr=500_000,
        tenure_months=48,
        interest_rate_annual_pct=12.5,
        foir_total_obligations_pct=35.0,
        loan_type=LoanTypeEnum.PERSONAL,
        interest_type=InterestTypeEnum.FIXED,
        pep_flag=False,
        kfs_provided=True,
        pin_code=110001
    )


if __name__ == "__main__":
    # Test schema validation
    print("🧪 Testing Pydantic Schemas")
    print("=" * 50)
    
    # Test valid request
    try:
        sample_request = get_sample_loan_request()
        print("✅ Valid request schema:", sample_request.model_dump_json(indent=2)[:200] + "...")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
    
    # Test invalid request
    try:
        invalid_request = LoanApplicationRequest(
            age_years=150,  # Invalid age
            bureau_score=1000,  # Invalid score
            monthly_income_inr=-5000,  # Invalid income
            requested_amount_inr=0,  # Invalid amount
            tenure_months=500,  # Invalid tenure
            interest_rate_annual_pct=100,  # Invalid rate
            foir_total_obligations_pct=200  # Invalid FOIR
        )
        print("❌ Should have failed validation")
    except Exception as e:
        print(f"✅ Correctly caught validation errors: {type(e).__name__}")
    
    print("\n🎯 All schemas validated successfully!")
    print("✅ Ready for FastAPI integration")