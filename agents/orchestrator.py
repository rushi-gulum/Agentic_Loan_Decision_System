# agents/orchestrator.py
"""
Production Agentic Loan Decision Orchestrator

Phase 3 Updates:
- LoanDecisionOrchestrator class with short-circuit pattern
- Unified preprocessing integration (Phase 1)
- Hybrid compliance engine integration (Phase 2)
- Async-friendly evaluation pipeline
- Structured response format for FastAPI

Architecture:
1. Preprocess application data with unified pipeline
2. Hard compliance checks (short-circuit on violations)
3. LLM soft compliance evaluation
4. Risk assessment
5. Final decision with explainability
6. Structured response formatting
"""

import os
import sys
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task, LLM
from crewai.tools import tool

# Phase 1 & 2 Components
from utils.preprocessing import preprocess_single_application
from rules.rule_engine import evaluate_hard_constraints
from agents.compliance_agent import evaluate_hard_compliance, evaluate_soft_compliance
from agents.xai_agent import preprocess, model_predict, explain_prediction
from agents.rag_agent import retrieve_feature_guidelines
from agents.risk_agent import compute_risk_score
from utils.llm_utility import get_llm, TaskType

# Import schemas for response formatting
from api.schemas import (
    FinalDecisionResponse, RiskAssessment, ComplianceResult, 
    XAIReport, DecisionSummary
)

logger = logging.getLogger(__name__)

class LoanDecisionOrchestrator:
    """
    Production orchestrator for loan decision pipeline
    
    Features:
    - Short-circuit pattern (fail fast on hard violations)
    - Unified preprocessing integration
    - Hybrid compliance checking
    - Structured response formatting
    - Error handling and logging
    """
    
    def __init__(self):
        """Initialize orchestrator with agents and components"""
        self._initialize_llm()
        self._initialize_crew_agents()
        
    def _initialize_llm(self):
        """Initialize LLM with proper configuration"""
        self.llm = LLM(
            model="openai/gpt-4o",
            temperature=0.2,
            max_tokens=6000,
            base_url="https://api.openai.com/v1",
        )
        
    def _initialize_crew_agents(self):
        """Initialize CrewAI agents for decision pipeline"""
        # Risk Agent
        self.risk_agent = Agent(
            name="RiskAgent",
            role="Risk Scoring Analyst",
            goal="Compute comprehensive risk assessment with explanations",
            backstory="Expert credit risk analyst providing quantitative assessments",
            tools=[risk_scorer_tool],
            verbose=False,
            max_iter=2,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Decision Agent
        self.decision_agent = Agent(
            name="DecisionAgent",
            role="Final Loan Decision Maker",
            goal="Make final loan approval decision based on compliance and risk",
            backstory="Senior credit officer ensuring regulatory compliance",
            tools=[decision_evaluator_tool],
            verbose=False,
            max_iter=2,
            llm=self.llm,
            allow_delegation=False
        )
        
        # XAI Agent
        self.xai_agent = Agent(
            name="XAIAgent", 
            role="Explainable AI Officer",
            goal="Generate comprehensive explanations for decisions",
            backstory="AI ethics officer ensuring transparency and explainability",
            tools=[xai_reporter_tool],
            verbose=False,
            max_iter=2,
            llm=self.llm,
            allow_delegation=False
        )
    
    def evaluate_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main evaluation pipeline with short-circuit pattern
        
        Args:
            application_data: Raw application data
            
        Returns:
            Structured decision response compatible with FinalDecisionResponse
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Preprocess application (Phase 1)
            logger.info("Step 1: Preprocessing application data")
            preprocessed_data = preprocess_single_application(application_data)
            
            # Step 2: Hard compliance checks (Phase 2 - Short Circuit)
            logger.info("Step 2: Running hard compliance checks")
            hard_violations = evaluate_hard_constraints(preprocessed_data, application_data.get('loan_type', 'personal'))
            
            if not hard_violations.is_compliant:
                logger.info("Hard violations detected - short-circuiting")
                return self._create_rejection_response(
                    reason="Hard compliance violations detected",
                    violations=hard_violations["violations"],
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            # Step 3: Soft compliance evaluation (Phase 2)
            logger.info("Step 3: Running soft compliance evaluation")
            compliance_result = self.compliance_agent.evaluate_compliance(
                preprocessed_data, application_data
            )
            
            # Step 4: Risk assessment
            logger.info("Step 4: Computing risk assessment")
            risk_result = self._compute_risk_assessment(preprocessed_data)
            
            # Step 5: Final decision
            logger.info("Step 5: Making final decision")
            decision_result = self._make_final_decision(
                compliance_result, risk_result, preprocessed_data
            )
            
            # Step 6: Generate explanations
            logger.info("Step 6: Generating explanations")
            xai_result = self._generate_explanations(
                preprocessed_data, compliance_result, risk_result, decision_result
            )
            
            # Step 7: Format response
            return self._format_final_response(
                compliance_result=compliance_result,
                risk_result=risk_result, 
                decision_result=decision_result,
                xai_result=xai_result,
                processing_time_ms=self._get_processing_time(start_time)
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return self._create_error_response(
                str(e), self._get_processing_time(start_time)
            )
    
    def _compute_risk_assessment(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute risk assessment using risk agent"""
        try:
            risk_task = Task(
                description="Compute comprehensive risk score using Risk Scorer tool",
                expected_output="JSON with risk_score_10, grade, and components",
                agent=self.risk_agent
            )
            
            crew = Crew(
                agents=[self.risk_agent],
                tasks=[risk_task],
                verbose=False
            )
            
            result = crew.kickoff(inputs={"applicant": preprocessed_data})
            
            # Parse result if it's a string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"risk_score_10": 5.0, "grade": "MEDIUM", "error": "Parse failed"}
            
            return result
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {
                "risk_score_10": 5.0,
                "grade": "UNKNOWN",
                "error": str(e)
            }
    
    def _make_final_decision(
        self, 
        compliance_result: Dict[str, Any], 
        risk_result: Dict[str, Any],
        preprocessed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make final loan decision"""
        try:
            decision_task = Task(
                description="Make final loan decision based on compliance and risk",
                expected_output="JSON with loan_eligible, decision_reason, selected_model",
                agent=self.decision_agent
            )
            
            crew = Crew(
                agents=[self.decision_agent],
                tasks=[decision_task],
                verbose=False
            )
            
            result = crew.kickoff(inputs={
                "policy_output": compliance_result,
                "risk_output": risk_result
            })
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {
                        "loan_eligible": False,
                        "decision_reason": "Decision parsing failed",
                        "selected_model": "none"
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Final decision failed: {str(e)}")
            return {
                "loan_eligible": False,
                "decision_reason": f"Decision error: {str(e)}",
                "selected_model": "none"
            }
    
    def _generate_explanations(
        self,
        preprocessed_data: Dict[str, Any],
        compliance_result: Dict[str, Any],
        risk_result: Dict[str, Any], 
        decision_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate XAI explanations"""
        try:
            xai_task = Task(
                description="Generate comprehensive explanations for loan decision",
                expected_output="JSON with user_explanation and regulator_explanation",
                agent=self.xai_agent
            )
            
            crew = Crew(
                agents=[self.xai_agent],
                tasks=[xai_task],
                verbose=False
            )
            
            result = crew.kickoff(inputs={
                "applicant": preprocessed_data,
                "risk_output": risk_result,
                "compliance_output": compliance_result,
                "decision_output": decision_result
            })
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {
                        "user_explanation": "Decision explanation unavailable",
                        "regulator_explanation": "Technical explanation unavailable"
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"XAI generation failed: {str(e)}")
            return {
                "user_explanation": f"Explanation error: {str(e)}",
                "regulator_explanation": f"Technical error: {str(e)}"
            }
    
    def _format_final_response(
        self,
        compliance_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        decision_result: Dict[str, Any],
        xai_result: Dict[str, Any],
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """Format response for FinalDecisionResponse schema"""
        
        # Determine overall decision
        decision = "APPROVED" if decision_result.get("loan_eligible", False) else "REJECTED"
        
        return {
            "application_id": "TBD",  # Will be set by API layer
            "decision": decision,
            "confidence_score": self._calculate_confidence(compliance_result, risk_result),
            "processing_time_ms": processing_time_ms,
            
            # Risk Assessment
            "risk_assessment": {
                "score": risk_result.get("risk_score_10", 0.0),
                "grade": risk_result.get("grade", "UNKNOWN"),
                "factors": risk_result.get("drivers", []),
                "details": risk_result.get("components", {})
            },
            
            # Compliance Result
            "compliance": {
                "overall_status": compliance_result.get("final_verdict", "INCOMPLETE"),
                "score": compliance_result.get("compliance_score", 0.0),
                "violations": compliance_result.get("violations", []),
                "checks_performed": len(compliance_result.get("feature_compliance", {}))
            },
            
            # XAI Report
            "explanations": {
                "user_friendly": xai_result.get("user_explanation", "No explanation available"),
                "technical": xai_result.get("regulator_explanation", "No technical details available"),
                "key_factors": self._extract_key_factors(risk_result, compliance_result),
                "model_used": decision_result.get("selected_model", "none")
            },
            
            # Decision Summary
            "summary": {
                "primary_reason": decision_result.get("decision_reason", "No reason provided"),
                "recommendation": "APPROVE" if decision == "APPROVED" else "REJECT",
                "next_steps": self._get_next_steps(decision, compliance_result, risk_result)
            }
        }
    
    def _create_rejection_response(
        self, 
        reason: str, 
        violations: list,
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """Create rejection response for hard violations"""
        return {
            "application_id": "TBD",
            "decision": "REJECTED",
            "confidence_score": 0.95,  # High confidence in hard rejections
            "processing_time_ms": processing_time_ms,
            
            "risk_assessment": {
                "score": 10.0,  # Max risk for hard violations
                "grade": "HIGH",
                "factors": ["Hard compliance violations"],
                "details": {}
            },
            
            "compliance": {
                "overall_status": "FAIL",
                "score": 0.0,
                "violations": violations,
                "checks_performed": len(violations)
            },
            
            "explanations": {
                "user_friendly": f"Application rejected due to: {reason}",
                "technical": f"Hard rule violations detected: {violations}",
                "key_factors": violations,
                "model_used": "rule_engine"
            },
            
            "summary": {
                "primary_reason": reason,
                "recommendation": "REJECT",
                "next_steps": ["Address compliance violations", "Resubmit application"]
            }
        }
    
    def _create_error_response(self, error: str, processing_time_ms: int) -> Dict[str, Any]:
        """Create error response"""
        return {
            "application_id": "TBD",
            "decision": "ERROR",
            "confidence_score": 0.0,
            "processing_time_ms": processing_time_ms,
            
            "risk_assessment": {
                "score": 0.0,
                "grade": "UNKNOWN",
                "factors": ["Processing error"],
                "details": {"error": error}
            },
            
            "compliance": {
                "overall_status": "INCOMPLETE",
                "score": 0.0,
                "violations": [],
                "checks_performed": 0
            },
            
            "explanations": {
                "user_friendly": "Unable to process application due to system error",
                "technical": f"Processing error: {error}",
                "key_factors": ["System error"],
                "model_used": "none"
            },
            
            "summary": {
                "primary_reason": f"System error: {error}",
                "recommendation": "RETRY",
                "next_steps": ["Contact support", "Retry submission"]
            }
        }
    
    def _calculate_confidence(
        self, 
        compliance_result: Dict[str, Any], 
        risk_result: Dict[str, Any]
    ) -> float:
        """Calculate decision confidence score"""
        compliance_score = compliance_result.get("compliance_score", 0.5)
        risk_score = risk_result.get("risk_score_10", 5.0)
        
        # Higher confidence for extreme cases
        if compliance_score < 0.3 or risk_score > 8:
            return 0.9  # High confidence rejection
        elif compliance_score > 0.8 and risk_score < 3:
            return 0.9  # High confidence approval
        else:
            return 0.7  # Moderate confidence
    
    def _extract_key_factors(
        self,
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any]
    ) -> list:
        """Extract key decision factors"""
        factors = []
        
        # Risk factors
        factors.extend(risk_result.get("drivers", []))
        
        # Compliance factors
        violations = compliance_result.get("violations", [])
        factors.extend([v.get("description", "Unknown violation") for v in violations])
        
        return factors[:5]  # Top 5 factors
    
    def _get_next_steps(
        self,
        decision: str,
        compliance_result: Dict[str, Any],
        risk_result: Dict[str, Any]
    ) -> list:
        """Get recommended next steps"""
        if decision == "APPROVED":
            return [
                "Proceed with loan documentation",
                "Schedule property verification",
                "Complete final approval process"
            ]
        else:
            steps = ["Address identified issues"]
            
            if compliance_result.get("violations"):
                steps.append("Resolve compliance violations")
            
            if risk_result.get("risk_score_10", 0) > 7:
                steps.append("Improve risk profile")
            
            steps.append("Resubmit application")
            return steps
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds"""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

# -----------------------------------------------------------------------------
# TOOLS (keep existing for compatibility)
# -----------------------------------------------------------------------------

# Keep existing tool definitions for CrewAI compatibility
class RAGInput(BaseModel):
    applicant_raw: Dict[str, Any] = Field(..., description="Applicant JSON used to determine loan type and query policy.")

class RiskInput(BaseModel):
    applicant: Dict[str, Any] = Field(..., description="Applicant JSON for risk scoring.")

# -----------------------------------------------------------------------------
# TOOLS
# -----------------------------------------------------------------------------
# ...existing code...
# ...existing code...


# ---------------- TOOL 3 ----------------
@tool("XAI Reporter")
def xai_reporter_tool(applicant_data: dict,
                      compliance_data: dict,
                      risk_data: dict) -> dict:
    """
    Generates user and regulator-friendly explanations for a loan decision
    using LLM summarization of SHAP & LIME insights.
    """
    preprocessed = preprocess(applicant_data)
    output = explain_prediction(preprocessed, applicant_data, compliance_data, risk_data)
    return output


@tool("RBI Guideline Retriever")
def rag_agent_tool(applicant_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve RBI rules for the applicant's all features(for example "age_years", "monthly_income_inr", "foir_total_obligations_pct",
        "ltv_ratio", "pep_flag", "interest_type", "loan_type") from Chroma DB.
    Instrumented: prints entry/exit and returns 'error' on failure so orchestrator can surface issues.
    """
    print("[RAG TOOL] invoked with applicant_raw keys:", list(applicant_raw.keys()) if isinstance(applicant_raw, dict) else type(applicant_raw))
    try:
        res = retrieve_feature_guidelines(applicant_raw)
        out = {
            "loan_type": res.get("loan_type"),
            # The 'feature_guidelines' key is mandatory for the next agent to use the data
            "feature_guidelines": res.get("feature_guidelines", {}) 
        }

        print(f"[RAG TOOL] output: {out['feature_guidelines'].keys()}")

        # Assuming we can determine the total number of retrieved chunks from the nested dict for printing:
        total_chunks = sum(len(d.get("retrieved_guidelines", [])) for d in out["feature_guidelines"].values())
        
        print(f"[RAG TOOL] success — retrieved {total_chunks} total guideline chunks across features")

        return out
    except FileNotFoundError as fnf:
        msg = str(fnf)
        print("[RAG TOOL] FileNotFoundError:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}
    except ValueError as ve:
        msg = str(ve)
        print("[RAG TOOL] ValueError:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}
    except Exception as e:
        msg = repr(e)
        print("[RAG TOOL] Exception:", msg)
        return {"loan_type": None, "query": "", "retrieved_guidelines": [], "error": msg}


@tool("Risk Scorer")
def risk_scorer_tool(applicant: dict[str, Any])-> Dict[str, Any]:
    """
    Compute explainable risk score (0–10, higher = riskier) for an application.
    Output includes: risk_score_10, grade, components, weights, reasons, drivers, context
    """
    print("[RISK TOOL] invoked with applicant keys:", list(applicant.keys()) if isinstance(applicant, dict) else type(applicant))
    try:
        score_obj = compute_risk_score(applicant)
        # Expect compute_risk_score to return a dict with at least 'risk_score_10' and 'grade'
        out = {
            "risk_score_10": score_obj.get("risk_score_10"),
            "grade": score_obj.get("grade"),
            "components": score_obj.get("components", {}),
            "drivers": score_obj.get("drivers", []),
            "context": score_obj.get("context", {}),
        }
        print(f"[RISK TOOL] success — score: {out['risk_score_10']} grade: {out['grade']}")
        return out
    except Exception as e:
        msg = repr(e)
        print("[RISK TOOL] Exception:", msg)
        return {"risk_score_10": None, "grade": None, "components": {}, "drivers": [], "context": {}, "error": msg}
    
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser



@tool("Compliance Checker")
def compliance_checker_tool(applicant: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses an LLM to analyze applicant details against RBI guideline summaries.
    Returns a structured JSON with feature-wise compliance and a final verdict.
    """

    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are an RBI compliance officer.
Compare each feature in the applicant's data with the guideline summaries provided below.

Rules:
- "is_compliant": true if applicant follows the RBI rule
- "is_compliant": false if violates
- "is_compliant": "NO_DATA" if guideline is unclear or missing
- Give a short, clear "reason" for each feature
- final_verdict:
    - "FAIL" if any feature violates
    - "PASS" if all are compliant
    - "INCOMPLETE" if rules missing for most features

Return STRICTLY in valid JSON (no markdown, no explanations):

{{
  "feature_compliance": {{
    "<feature_name>": {{
      "is_compliant": true | false | "NO_DATA",
      "reason": "<reason>"
    }}
  }},
  "final_verdict": "PASS" | "FAIL" | "INCOMPLETE"
}}

### Applicant JSON:
{applicant}

### RBI Guidelines JSON:
{guidelines}
""")


    # safer chain (no deprecated LLMChain)
    from langchain.schema.runnable import RunnableSequence
    from langchain.schema.output_parser import StrOutputParser

    chain = prompt | get_llm() | StrOutputParser()


    try:
        raw_output = chain.invoke({"applicant": applicant, "guidelines": guidelines}).strip()

        # ✅ enforce JSON parsing safety
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # fallback: try to clean accidental markdown / bad chars
            cleaned = raw_output.strip("` \n").replace("json", "")
            parsed = json.loads(cleaned)

        # ✅ enforce structure if model drops keys
        if "feature_compliance" not in parsed:
            parsed["feature_compliance"] = {}
        if "final_verdict" not in parsed:
            parsed["final_verdict"] = "INCOMPLETE"

        return parsed

    except Exception as e:
        return {
            "error": f"Compliance checker failed: {str(e)}",
            "feature_compliance": {},
            "final_verdict": "INCOMPLETE"
        }
@tool("Decision Evaluator")
def decision_evaluator_tool(policy_output: dict, risk_output: dict) -> dict:
    """
    Determines final loan eligibility, and if approved, selects ML model type.
    Returns structured reasoning, probability, and final decision.
    """

    prompt = PromptTemplate.from_template("""
You are an RBI-regulated credit decision AI.

Your task is to make a *two-step decision* based on the applicant's RBI compliance and credit risk profile.

---

### 🧩 Stage 1: Loan Eligibility
1. Use **PolicyAgent Output** to determine compliance:
   - If `final_decision` == "DECLINED" or any critical violations exist → reject immediately.
   - If `final_decision` == "APPROVED" → continue to risk evaluation.
2. Use **RiskAgent Output** to adjust eligibility:
   - If `risk_score_10` > 7 → high risk → reject with low probability (≈0.3–0.4)
   - If 4 ≤ `risk_score_10` ≤ 7 → moderate risk → cautious approval (≈0.6–0.75)
   - If `risk_score_10` < 4 → low risk → confident approval (≈0.85–0.95)
3. Give reasoning for rejection if ineligible (e.g., non-compliance, excessive risk, missing data).

---

### ⚙️ Stage 2: Model Selection (only if loan_eligible = true)
- Choose model type:
  - `"blackbox"` → for borderline/moderate risk where nonlinear interactions likely matter.
  - `"interpretable"` → for low-risk applicants needing explainable audit-friendly scoring.
- Provide justification for your chosen model.

---

### 🎯 Output Format (STRICT JSON only)
{{
  "loan_eligible": true | false,
  "decision_reason": "<clear 2–3 line reasoning>",
  "selected_model": "blackbox" | "interpretable" | "none"
}}

### PolicyAgent Output:
{policy_output}

### RiskAgent Output:
{risk_output}
""")

    
    from langchain.schema.output_parser import StrOutputParser
   
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "policy_output": policy_output,
            "risk_output": risk_output
        })
        if not result or not result.strip():
            raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"[RISK AGENT ERROR] {e}")
        result = json.dumps({
            "risk_score": 0.0,
            "risk_grade": "UNKNOWN",
            "reason": str(e)
        })



        # ensure valid structure
        if not isinstance(result, dict):
            result = json.loads(result)
        result.setdefault("loan_eligible", False)
        result.setdefault("selected_model", "none")
        result.setdefault("decision_reason", "No reasoning provided.")

        return result

    except Exception as e:
        return {
            "error": f"Decision evaluator failed: {e}",
            "loan_eligible": False,
            "selected_model": "none",
            "decision_reason": f"Decision process error: {str(e)}"
        }
@tool("explainable AI Tool")
def xai_tool(applicant: Dict[str, Any], risk_output: Dict[str, Any], compliance_output: Dict[str, Any],decision_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides clear, concise explanations for risk scores and compliance decisions to ensure transparency.
    """

    llm = get_llm()

    prompt_template = PromptTemplate.from_template("""

    """)


# -----------------------------------------------------------------------------
# LEGACY RUNNER (for backward compatibility)
# -----------------------------------------------------------------------------
def run_sync(applicant_dict: Dict[str, Any]):
    """
    Legacy runner for backward compatibility
    Use LoanDecisionOrchestrator.evaluate_application() for new code
    """
    orchestrator = LoanDecisionOrchestrator()
    return orchestrator.evaluate_application(applicant_dict)

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "application_id": "APP-0001",
        "loan_type": "housing loan",
        "age_years": 32,
        "bureau_score": 720,
        "monthly_income_inr": 55000,
        "foir_total_obligations_pct": 25.0,
        "requested_amount_inr": 300000,
        "tenure_months": 36,
        "gender": "Male",
        "state": "Maharashtra",
        "kyc_mode": "Video KYC",
        "ovd_type": "Aadhaar",
        "interest_type": "Fixed",
        "pep_flag": False,
        "kfs_provided": True,
        "processing_fee_inr": 2000.0,
        "other_charges_inr": 500.0,
        "apr_pct": 12.5,
        "property_value_inr": 1000000,
        "ltv_ratio": 0.3,
    }

    orchestrator = LoanDecisionOrchestrator()
    result = orchestrator.evaluate_application(sample)
    print("\n=== Orchestrator Result ===")
    print(json.dumps(result, indent=2, default=str))
