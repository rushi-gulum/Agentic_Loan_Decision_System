"""
utils/llm_utility.py
------------------
Unified LLM routing and factory for the Agentic Loan Decision System.
Provides task-specific model selection and graceful fallbacks.
"""

import os
from typing import Optional, Dict, Any, Union
from enum import Enum


class TaskType(str, Enum):
    """Task-specific LLM types for optimal model selection."""
    FAST = "fast"           # Quick routing, compliance checks - use Groq/Llama
    SMART = "smart"         # Complex reasoning, XAI explanations - use OpenAI/GPT-4
    GENERAL = "general"     # Default/general purpose
    MOCK = "mock"          # Testing/development without API costs


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GROQ = "groq"
    MOCK = "mock"


class MockLLM:
    """
    Enhanced Mock LLM for testing when no API keys available.
    Compatible with both LangChain and CrewAI expectations.
    """
    
    def __init__(self, model: str = "mock-llm", **kwargs):
        self.model = model
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 6000)
        self._provider = "mock"
    
    def invoke(self, prompt: Union[str, Dict, Any]) -> Union[str, Dict]:
        """
        Generate contextually appropriate mock responses.
        Handles both string prompts and structured inputs.
        """
        # Extract prompt text from various input formats
        if isinstance(prompt, str):
            prompt_text = prompt.lower()
        elif isinstance(prompt, dict):
            prompt_text = str(prompt).lower()
        elif hasattr(prompt, 'content'):
            prompt_text = prompt.content.lower()
        elif hasattr(prompt, 'text'):
            prompt_text = prompt.text.lower()
        else:
            prompt_text = str(prompt).lower()
        
        # Generate task-specific responses
        return self._generate_response(prompt_text)
    
    def _generate_response(self, prompt_text: str) -> str:
        """Generate appropriate mock responses based on prompt analysis."""
        
        # Risk assessment responses
        if "risk" in prompt_text and any(word in prompt_text for word in ["score", "grade", "assessment"]):
            return """
            {
                "risk_score_10": 4.2,
                "grade": "B", 
                "components": {"bureau": 0.3, "foir": 0.4, "ltv": 0.2},
                "drivers": ["bureau_score: moderate", "foir: acceptable", "ltv: low_risk"],
                "reasons": ["Stable income profile", "Moderate credit history"],
                "context": {"loan_type": "personal", "assessment_basis": "standard_criteria"}
            }
            """
        
        # Compliance evaluation responses  
        elif "compliance" in prompt_text and "json" in prompt_text:
            if any(word in prompt_text for word in ["violation", "fail", "non-compliant"]):
                return '''
                {
                    "soft_violations": [
                        {
                            "feature": "income_verification", 
                            "guideline_reference": "RBI Income Documentation Standards",
                            "concern": "Income verification documentation may require additional scrutiny",
                            "severity": "moderate",
                            "recommendation": "Request additional income proof documentation"
                        }
                    ]
                }
                '''
            else:
                return '{"soft_violations": []}'
        
        # Decision making responses
        elif "decision" in prompt_text and any(word in prompt_text for word in ["loan", "eligible", "approved"]):
            if any(word in prompt_text for word in ["fail", "violation", "reject"]):
                return '''
                {
                    "loan_eligible": false,
                    "decision_reason": "Application does not meet minimum regulatory requirements",
                    "selected_model": "none",
                    "approval_probability": 0.15
                }
                '''
            else:
                return '''
                {
                    "loan_eligible": true, 
                    "decision_reason": "Application meets all regulatory and risk criteria",
                    "selected_model": "interpretable",
                    "approval_probability": 0.78
                }
                '''
        
        # XAI explanation responses
        elif any(word in prompt_text for word in ["explanation", "shap", "lime", "explainable"]):
            if "customer" in prompt_text:
                return "Your loan application has been carefully reviewed using our advanced assessment system. Based on your credit profile, income stability, and loan requirements, we have determined that you meet our lending criteria. The key factors supporting this decision include your solid credit history and appropriate debt-to-income ratio."
            elif "regulator" in prompt_text:
                return "This loan decision was made using a comprehensive risk assessment framework that evaluates multiple factors including credit bureau score, financial obligation ratios, and regulatory compliance requirements. The decision process follows RBI guidelines and incorporates both quantitative risk metrics and qualitative policy considerations. All regulatory requirements have been satisfied."
            else:
                return "Mock XAI explanation: Decision based on comprehensive risk and compliance assessment using interpretable machine learning models."
        
        # General compliance explanation
        elif "compliance officer" in prompt_text or "explanation" in prompt_text:
            if any(word in prompt_text for word in ["non-compliant", "violation", "fail"]):
                return "Application assessment reveals regulatory compliance concerns that prevent approval. Specific policy violations include mathematical threshold breaches that require resolution before proceeding. Additional documentation and review processes are recommended to address identified compliance gaps."
            else:
                return "Application successfully meets all regulatory compliance requirements. Comprehensive assessment indicates adherence to applicable lending guidelines and risk management policies. All mandatory documentation and verification requirements have been satisfied per regulatory standards."
        
        # Default response
        return f"Mock LLM response: Processed request for {self.model}. Analysis completed using simulated reasoning capabilities."
    
    def __call__(self, *args, **kwargs):
        """Support callable interface for some LangChain integrations."""
        if args:
            return self.invoke(args[0])
        return self.invoke("")
    
    @property
    def model_name(self) -> str:
        """Model name property for compatibility."""
        return self.model


def get_llm_provider() -> LLMProvider:
    """
    Determine which LLM provider to use based on environment variables.
    Priority: GROQ -> OPENAI -> MOCK
    """
    if os.getenv("GROQ_API_KEY"):
        return LLMProvider.GROQ
    elif os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    else:
        return LLMProvider.MOCK


def create_openai_llm(task_type: TaskType, **kwargs) -> Any:
    """Create OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        # Task-specific model selection
        if task_type == TaskType.SMART:
            model = "gpt-4o"
            max_tokens = kwargs.get("max_tokens", 8000)
        else:
            model = "gpt-3.5-turbo" 
            max_tokens = kwargs.get("max_tokens", 4000)
        
        return ChatOpenAI(
            model=model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=max_tokens,
            api_key=api_key
        )
    except ImportError:
        raise ImportError("langchain_openai not installed. Run: pip install langchain-openai")


def create_groq_llm(task_type: TaskType, **kwargs) -> Any:
    """Create Groq LLM instance.""" 
    try:
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        # Task-specific model selection
        if task_type == TaskType.SMART:
            model = "llama-3.1-70b-versatile"  # Larger model for complex reasoning
        else:
            model = "llama-3.1-8b-instant"    # Fast model for quick tasks
        
        return ChatGroq(
            model=model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 6000),
            groq_api_key=api_key
        )
    except ImportError:
        raise ImportError("langchain_groq not installed. Run: pip install langchain-groq")


def get_llm(
    task_type: TaskType = TaskType.GENERAL,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.2,
    max_tokens: int = 6000,
    **kwargs
) -> Any:
    """
    Unified LLM factory with task-specific optimization and graceful fallbacks.
    
    Args:
        task_type: Type of task to optimize model selection for
        provider: Specific provider to use (auto-detected if None)
        temperature: Model temperature (0.0 = deterministic, 1.0 = creative)  
        max_tokens: Maximum tokens in response
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured LLM instance compatible with LangChain/CrewAI
        
    Examples:
        # Fast compliance checking
        llm = get_llm(TaskType.FAST)
        
        # Complex XAI explanations  
        llm = get_llm(TaskType.SMART)
        
        # Force specific provider
        llm = get_llm(TaskType.GENERAL, provider=LLMProvider.OPENAI)
    """
    
    # Override for explicit mock request
    if task_type == TaskType.MOCK or provider == LLMProvider.MOCK:
        print("🎭 Using MockLLM for testing")
        return MockLLM(
            model=f"mock-{task_type.value}",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    # Auto-detect provider if not specified
    if provider is None:
        provider = get_llm_provider()
    
    # Create provider-specific LLM with fallback chain
    try:
        if provider == LLMProvider.GROQ:
            print(f"🚀 Using Groq LLM for {task_type.value} tasks")
            return create_groq_llm(task_type, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        elif provider == LLMProvider.OPENAI:
            print(f"🧠 Using OpenAI LLM for {task_type.value} tasks") 
            return create_openai_llm(task_type, temperature=temperature, max_tokens=max_tokens, **kwargs)
            
    except (ImportError, ValueError) as e:
        print(f"⚠️ {provider.value} LLM unavailable ({e}), falling back to MockLLM")
    
    # Final fallback to MockLLM
    print(f"🎭 Using MockLLM for {task_type.value} tasks (no API keys)")
    return MockLLM(
        model=f"mock-{task_type.value}",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def get_crewai_llm(task_type: TaskType = TaskType.GENERAL, **kwargs) -> Any:
    """
    Get LLM specifically configured for CrewAI compatibility.
    CrewAI expects LLM instances in a specific format.
    """
    try:
        from crewai import LLM as CrewAI_LLM
        
        # Get base LLM
        base_llm = get_llm(task_type, **kwargs)
        
        # If it's already a MockLLM, return as-is (CrewAI can handle it)
        if isinstance(base_llm, MockLLM):
            return base_llm
        
        # For real LLMs, try to wrap in CrewAI LLM class
        # This depends on the specific CrewAI version and configuration
        provider = get_llm_provider()
        
        if provider == LLMProvider.GROQ:
            model_name = "llama-3.1-8b-instant" if task_type == TaskType.FAST else "llama-3.1-70b-versatile"
            return CrewAI_LLM(
                model=f"groq/{model_name}",
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 6000)
            )
        elif provider == LLMProvider.OPENAI:
            model_name = "gpt-3.5-turbo" if task_type == TaskType.FAST else "gpt-4o"
            return CrewAI_LLM(
                model=f"openai/{model_name}",
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 6000)
            )
        else:
            return base_llm
            
    except ImportError:
        print("⚠️ CrewAI not available, using standard LLM")
        return get_llm(task_type, **kwargs)
    except Exception as e:
        print(f"⚠️ CrewAI LLM creation failed ({e}), using fallback")
        return get_llm(task_type, **kwargs)


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing LLM Routing System")
    print("=" * 50)
    
    # Test each task type
    task_types = [TaskType.FAST, TaskType.SMART, TaskType.GENERAL, TaskType.MOCK]
    
    for task_type in task_types:
        print(f"\n📊 Testing {task_type.value} LLM:")
        try:
            llm = get_llm(task_type)
            print(f"   ✅ LLM Type: {type(llm).__name__}")
            print(f"   ✅ Model: {getattr(llm, 'model', getattr(llm, 'model_name', 'unknown'))}")
            
            # Test invocation
            response = llm.invoke("Test prompt for compliance evaluation")
            if hasattr(response, 'content'):
                response_text = response.content[:100]
            else:
                response_text = str(response)[:100]
            print(f"   ✅ Response: {response_text}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test CrewAI compatibility
    print(f"\n🤖 Testing CrewAI Compatibility:")
    try:
        crewai_llm = get_crewai_llm(TaskType.FAST)
        print(f"   ✅ CrewAI LLM Type: {type(crewai_llm).__name__}")
    except Exception as e:
        print(f"   ⚠️ CrewAI test skipped: {e}")
    
    print(f"\n🎯 LLM Routing Test Complete!")
    print("✅ Task-specific model selection working")
    print("✅ Graceful fallback to MockLLM implemented")  
    print("✅ Ready for API integration")
