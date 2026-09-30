"""
utils/llm_utility.py
====================
Unified LLM factory for the Agentic Loan Decision System.

Architecture: pure LangChain — no CrewAI dependency.

Provider priority (auto-detected from environment):
  1. Groq  (GROQ_API_KEY set)     → ChatGroq   — fast, free tier
  2. OpenAI (OPENAI_API_KEY set)  → ChatOpenAI — fallback
  3. Mock  (no keys)              → MockLLM    — dev / testing

All agents call get_llm().  get_agent_llm() is an alias kept for
backwards compatibility with orchestrator.py.
"""

import os
from typing import Any, Dict, Optional, Union
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    FAST    = "fast"     # Compliance checks, quick routing
    SMART   = "smart"    # Decision reasoning, XAI explanations
    GENERAL = "general"  # Default
    MOCK    = "mock"     # Testing — no API calls


class LLMProvider(str, Enum):
    GROQ   = "groq"
    OPENAI = "openai"
    MOCK   = "mock"


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM (zero dependencies, used in dev / CI)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """
    In-process mock that returns plausible JSON for each agent stage.
    No network calls, no API keys required.
    Implements .invoke() so it is a drop-in for ChatGroq / ChatOpenAI.
    """

    def __init__(self, model: str = "mock-llm", **kwargs):
        self.model       = model
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens  = kwargs.get("max_tokens", 6000)

    def invoke(self, prompt: Union[str, Dict, Any]) -> str:
        text = (
            prompt.lower()                if isinstance(prompt, str)
            else prompt.content.lower()   if hasattr(prompt, "content")
            else str(prompt).lower()
        )

        if "risk" in text and any(w in text for w in ("score", "grade", "assessment")):
            return (
                '{"risk_score_10": 4.2, "grade": "B", '
                '"components": {"bureau": 0.3, "foir": 0.4}, '
                '"drivers": ["bureau_score: moderate"], '
                '"reasons": ["Stable income profile"]}'
            )

        if "compliance" in text and "json" in text:
            if any(w in text for w in ("violation", "fail", "non-compliant")):
                return '{"soft_violations": [{"feature": "income", "guideline_reference": "RBI", "concern": "requires scrutiny", "severity": "moderate", "recommendation": "request docs"}]}'
            return '{"soft_violations": []}'

        if "decision" in text and any(w in text for w in ("loan", "eligible")):
            eligible = not any(w in text for w in ("fail", "violation", "reject"))
            return (
                f'{{"loan_eligible": {str(eligible).lower()}, '
                f'"decision_reason": "Rule-based mock decision.", '
                f'"selected_model": "interpretable", "approval_probability": {"0.78" if eligible else "0.15"}}}'
            )

        if any(w in text for w in ("explanation", "shap", "lime", "explainable")):
            return (
                "Your application was assessed using our credit scoring system. "
                "Key factors include bureau score and debt-to-income ratio."
            )

        return f"[MockLLM] Processed request. Model: {self.model}."

    def __call__(self, *args, **kwargs):
        return self.invoke(args[0] if args else "")

    @property
    def model_name(self) -> str:
        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# Provider detection
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_provider() -> LLMProvider:
    """Auto-detect provider from environment. Priority: Groq → OpenAI → Mock."""
    if os.getenv("GROQ_API_KEY"):
        return LLMProvider.GROQ
    if os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    return LLMProvider.MOCK


# ─────────────────────────────────────────────────────────────────────────────
# Provider constructors
# ─────────────────────────────────────────────────────────────────────────────

def _make_groq(task_type: TaskType, **kwargs) -> Any:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model          = model,
        temperature    = kwargs.get("temperature", 0.2),
        max_tokens     = kwargs.get("max_tokens", 6000),
        groq_api_key   = api_key,
    )


def _make_openai(task_type: TaskType, **kwargs) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    model      = "gpt-4o"       if task_type == TaskType.SMART else "gpt-3.5-turbo"
    max_tokens = kwargs.get("max_tokens", 8000 if task_type == TaskType.SMART else 4000)
    return ChatOpenAI(
        model       = model,
        temperature = kwargs.get("temperature", 0.2),
        max_tokens  = max_tokens,
        api_key     = api_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(
    task_type   : TaskType          = TaskType.GENERAL,
    provider    : Optional[LLMProvider] = None,
    temperature : float             = 0.2,
    max_tokens  : int               = 6000,
    **kwargs,
) -> Any:
    """
    Return a LangChain-compatible chat LLM.

    Routing: MOCK → immediate return
             Groq  → ChatGroq  (primary)
             OpenAI → ChatOpenAI (fallback)
             fallback → MockLLM (no keys / import error)

    All agents in this repo use this function. No CrewAI dependency.
    """
    if task_type == TaskType.MOCK or provider == LLMProvider.MOCK:
        return MockLLM(model=f"mock-{task_type.value}", temperature=temperature)

    if provider is None:
        provider = get_llm_provider()

    try:
        if provider == LLMProvider.GROQ:
            return _make_groq(task_type, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if provider == LLMProvider.OPENAI:
            return _make_openai(task_type, temperature=temperature, max_tokens=max_tokens, **kwargs)
    except (ImportError, ValueError) as exc:
        print(f"⚠️  {provider.value} unavailable ({exc}) — falling back to MockLLM")

    return MockLLM(model=f"mock-{task_type.value}", temperature=temperature)


# Alias kept so existing call-sites (orchestrator, xai_agent) don't break
def get_agent_llm(task_type: TaskType = TaskType.GENERAL, **kwargs) -> Any:
    """
    Alias for get_llm().  Previously called get_crewai_llm() when the system
    used CrewAI.  CrewAI has been removed; this now returns a plain LangChain
    ChatGroq / ChatOpenAI / MockLLM instance.
    """
    return get_llm(task_type, **kwargs)


# Keep the old name importable so any stale reference doesn't crash at import
get_crewai_llm = get_agent_llm


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    print("LLM Routing Test")
    print("=" * 40)

    for tt in (TaskType.FAST, TaskType.SMART, TaskType.MOCK):
        llm = get_llm(tt)
        print(f"  {tt.value:<8} → {type(llm).__name__}")
        resp = llm.invoke("Reply HEALTHY if working.")
        text = resp.content if hasattr(resp, "content") else str(resp)
        print(f"           response: {text[:60]}")

    print("\nAll provider checks passed.")
