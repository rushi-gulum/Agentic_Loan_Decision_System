"""
Cloud-Ready LLM Utility for Agentic Loan Decision System
========================================================

Production-ready LLM interface optimized for cloud deployment.
Primary: Groq API (free Llama 3.1), Fallbacks: OpenAI, Anthropic, Mock
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time

# Environment configuration
from utils.config import config

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai" 
    ANTHROPIC = "anthropic"
    MOCK = "mock"

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None

class CloudLLM:
    """
    Cloud-optimized LLM utility with Groq as primary provider.
    Handles provider fallbacks, rate limiting, and cost tracking.
    """
    
    def __init__(self):
        """Initialize with configured providers"""
        self.config = config.get_primary_llm_config()
        self.provider = LLMProvider(self.config["provider"])
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate LLM client"""
        try:
            if self.provider == LLMProvider.GROQ:
                self._init_groq()
            elif self.provider == LLMProvider.OPENAI:
                self._init_openai()
            elif self.provider == LLMProvider.ANTHROPIC:
                self._init_anthropic()
            else:
                logger.info("Using MockLLM for development")
                self.client = MockLLMClient()
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider.value}: {e}")
            logger.info("Falling back to MockLLM")
            self.client = MockLLMClient()
            self.provider = LLMProvider.MOCK
    
    def _init_groq(self) -> None:
        """Initialize Groq client"""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.config["api_key"])
            logger.info("✅ Groq client ready")
        except ImportError:
            raise ImportError("Install groq: pip install groq")
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config["api_key"])
            logger.info("✅ OpenAI client ready")
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def _init_anthropic(self) -> None:
        """Initialize Anthropic client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config["api_key"])
            logger.info("✅ Anthropic client ready")
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with the configured provider
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system context
            max_tokens: Max tokens (default from config)
            temperature: Sampling temperature (default from config)
            
        Returns:
            LLMResponse: Standardized response
        """
        start_time = time.time()
        max_tokens = max_tokens or self.config.get("max_tokens", 1000)
        temperature = temperature or self.config.get("temperature", 0.1)
        
        try:
            if self.provider == LLMProvider.GROQ:
                response = self._groq_generate(prompt, system_prompt, max_tokens, temperature, **kwargs)
            elif self.provider == LLMProvider.OPENAI:
                response = self._openai_generate(prompt, system_prompt, max_tokens, temperature, **kwargs)
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self._anthropic_generate(prompt, system_prompt, max_tokens, temperature, **kwargs)
            else:
                response = self._mock_generate(prompt, system_prompt, max_tokens, temperature, **kwargs)
                
            response.latency_ms = int((time.time() - start_time) * 1000)
            return response
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResponse(
                content=f"Error: LLM unavailable. Fallback response generated. ({str(e)[:50]}...)",
                provider="error_fallback",
                model="fallback",
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=int((time.time() - start_time) * 1000)
            )
    
    def _groq_generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, **kwargs) -> LLMResponse:
        """Generate using Groq API (Primary)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider="groq",
            model=self.config["model"],
            tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None,
            cost_usd=0.0  # Groq is free tier
        )
    
    def _openai_generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, **kwargs) -> LLMResponse:
        """Generate using OpenAI API (Fallback)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        cost = self._estimate_openai_cost(response.usage.total_tokens, self.config["model"])
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=self.config["model"],
            tokens_used=response.usage.total_tokens,
            cost_usd=cost
        )
    
    def _anthropic_generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, **kwargs) -> LLMResponse:
        """Generate using Anthropic API (Fallback)"""
        response = self.client.messages.create(
            model=self.config["model"],
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        cost = self._estimate_anthropic_cost(response.usage.input_tokens, response.usage.output_tokens)
        
        return LLMResponse(
            content=response.content[0].text,
            provider="anthropic", 
            model=self.config["model"],
            tokens_used=total_tokens,
            cost_usd=cost
        )
    
    def _mock_generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, **kwargs) -> LLMResponse:
        """Generate mock response for development"""
        return LLMResponse(
            content="MockLLM: This is a development placeholder response for testing without API costs.",
            provider="mock",
            model="mock-model",
            tokens_used=25,
            cost_usd=0.0
        )
    
    def _estimate_openai_cost(self, tokens: int, model: str) -> float:
        """Estimate OpenAI cost (approximate rates)"""
        costs_per_1k = {
            "gpt-3.5-turbo": 0.001,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-4o": 0.005,
        }
        rate = costs_per_1k.get(model, 0.001)
        return (tokens / 1000) * rate
    
    def _estimate_anthropic_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic cost (approximate rates)"""
        input_rate = 0.008   # per 1K input tokens
        output_rate = 0.024  # per 1K output tokens
        return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate

class MockLLMClient:
    """Mock client for development without API calls"""
    
    def __init__(self):
        self.mock_responses = {
            "compliance": """
            {
                "loan_eligible": true,
                "compliance_score": 0.85,
                "decision_reason": "Application meets regulatory requirements",
                "regulatory_concerns": [],
                "recommended_actions": ["Proceed with standard verification"]
            }
            """,
            "explanation": "This application shows strong creditworthiness with a bureau score of 750 and stable income. The requested loan amount is within acceptable debt-to-income ratios.",
            "insights": [
                "Strong credit profile with bureau score above 720",
                "Stable income verification completed", 
                "Low existing debt obligations",
                "Property valuation within market norms"
            ]
        }

# Global LLM instance
llm = CloudLLM()

# ============================================================================
# Task-Specific Functions for Loan Decision System
# ============================================================================

def analyze_compliance(applicant_data: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze regulatory compliance using LLM with RBI guidelines
    
    Args:
        applicant_data: Loan application details
        guidelines: Retrieved RBI guidelines context
        
    Returns:
        Dict with compliance assessment
    """
    system_prompt = """You are a regulatory compliance expert for Indian banking, specialized in RBI guidelines.
    
Analyze loan applications for regulatory compliance. Always respond in valid JSON with these fields:
- loan_eligible (boolean): Whether loan meets basic regulatory requirements  
- compliance_score (float 0.0-1.0): Overall compliance rating
- decision_reason (string): Brief explanation for the decision
- regulatory_concerns (array): List of specific compliance issues
- recommended_actions (array): Suggested next steps

Consider: KYC/AML, income verification, LTV ratios, FOIR limits, borrower eligibility."""

    prompt = f"""Analyze regulatory compliance for this loan application:

APPLICANT DATA:
{json.dumps(applicant_data, indent=2)}

RELEVANT RBI GUIDELINES:
{json.dumps(guidelines, indent=2)}

Provide compliance assessment in JSON format."""

    response = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=800,
        temperature=0.1
    )
    
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from LLM, using fallback")
        return {
            "loan_eligible": False,
            "compliance_score": 0.5,
            "decision_reason": "Automated compliance analysis unavailable",
            "regulatory_concerns": ["System analysis failed"],
            "recommended_actions": ["Manual regulatory review required"]
        }

def generate_explanation(
    applicant_data: Dict[str, Any],
    decision_result: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    audience: str = "customer"
) -> str:
    """
    Generate audience-specific explanations for loan decisions
    
    Args:
        applicant_data: Application details
        decision_result: Final loan decision
        risk_assessment: Risk scoring results  
        audience: Target audience (customer, regulator, internal)
        
    Returns:
        Human-readable explanation text
    """
    audience_contexts = {
        "customer": "Explain to the customer in simple, empathetic language without technical jargon.",
        "regulator": "Provide detailed regulatory explanation with compliance rationale for audit purposes.", 
        "internal": "Brief internal risk team with technical insights and business implications."
    }
    
    system_prompt = f"""You are explaining loan decisions. {audience_contexts.get(audience, audience_contexts['customer'])}
    
Keep explanations clear, factual, and under 250 words. Focus on key decision factors."""

    prompt = f"""Explain this loan decision for a {audience}:

APPLICANT:
- Age: {applicant_data.get('age_years', 'N/A')}
- Monthly Income: ₹{applicant_data.get('monthly_income_inr', 'N/A'):,}
- Bureau Score: {applicant_data.get('bureau_score', 'N/A')}
- Loan Amount: ₹{applicant_data.get('requested_amount_inr', 'N/A'):,}

DECISION: {decision_result.get('decision', 'N/A')}
CONFIDENCE: {decision_result.get('confidence', 'N/A')}%
RISK SCORE: {risk_assessment.get('risk_score', 'N/A')}
RISK GRADE: {risk_assessment.get('risk_grade', 'N/A')}

Provide clear explanation:"""

    response = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=400,
        temperature=0.2
    )
    
    return response.content.strip()

def extract_key_insights(data: Dict[str, Any], context: str) -> List[str]:
    """
    Extract bullet-point insights from complex data
    
    Args:
        data: Data to analyze
        context: Context description
        
    Returns:
        List of key insights (3-5 bullet points)
    """
    system_prompt = """Extract 3-5 key insights as bullet points. Each point should be under 40 words and actionable."""

    prompt = f"""Extract key insights from this {context}:

{json.dumps(data, indent=2)}

Provide insights as bullet points:"""

    response = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=300,
        temperature=0.1
    )
    
    # Parse bullet points
    insights = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
            clean_line = line.lstrip('•-*123456789. ').strip()
            if clean_line:
                insights.append(clean_line)
    
    return insights[:5]

# ============================================================================
# Health Check & Monitoring
# ============================================================================

def check_llm_health() -> Dict[str, Any]:
    """Check LLM service health for monitoring"""
    try:
        test_response = llm.generate(
            prompt="Respond with 'HEALTHY' if operational.",
            max_tokens=10,
            temperature=0.0
        )
        
        return {
            "status": "healthy",
            "provider": llm.provider.value,
            "model": llm.config["model"],
            "test_response": test_response.content.strip(),
            "latency_ms": test_response.latency_ms,
            "cost_estimate_usd": test_response.cost_usd or 0.0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "provider": llm.provider.value, 
            "error": str(e),
            "latency_ms": 0
        }

if __name__ == "__main__":
    # Test configuration
    print("🚀 Testing Cloud LLM Configuration...")
    
    health = check_llm_health()
    print(f"Health: {health}")
    
    # Test basic functionality
    test_insights = extract_key_insights(
        {"bureau_score": 750, "income": 80000}, 
        "loan application"
    )
    print(f"Test insights: {test_insights}")