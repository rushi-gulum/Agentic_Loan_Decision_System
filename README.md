# 🏦 Agentic Explainable AI Framework for Loan Approval

**Autonomous Multi-Agent Credit Underwriting, Regulatory Guardrails & Explainable AI (SHAP) for Banking & Fintech.**

[![Tests](https://img.shields.io/badge/pytest-76%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

Traditional retail lending platforms suffer from two major deficiencies: opaque "black-box" machine learning models that violate fair lending regulations, and slow manual underwriting processes that scale poorly. 

The **Agentic Explainable AI Framework for Loan Approval** solves this by uniting **deterministic regulatory guardrails**, **mathematical credit risk scoring**, **calibrated machine learning (XGBoost / Logistic Regression)**, and **local SHAP (SHapley Additive exPlanations)** within a high-throughput multi-agent architecture.

### Key Highlights
- **40,000+ Loan Applications Analyzed**: Built on a real-world banking dataset with 25+ engineered demographic, financial, collateral, and bureau attributes.
- **Calibrated XGBoost Model ($T^* = 0.555$)**: Tuned decision threshold minimizing costly false approvals (Type II errors) by **14.9%**, achieving **85.4% accuracy** and **0.85+ AUC-ROC**.
- **Point-by-point Dual-Audience Justification Dossier**: Automatically synthesizes both a direct, non-technical 5-point **Borrower Decision Notice** and an exhaustive 10-point **RBI Regulatory & Statutory Compliance Audit Memo**.
- **Visual SHAP Attribution Engine**: Decomposes model predictions into intuitive directional pressure forces relative to base expected value $E[f(x)] = 0.50$, accompanied by an easy-to-digest Factor Impact Table.
- **Sub-100ms Short-Circuit Guardrails**: Non-compliant applications (PEP violations, FOIR breaches, underage applicants) are declined deterministically in $<30\text{ ms}$ without consuming model compute.

---

## 🏗️ System Architecture & Workflow

```
                                 ┌─────────────────────────────────┐
                                 │   Applicant Data / API Payload  │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │      Preprocessor Agent         │
                                 │  • 25-feature matrix alignment  │
                                 │  • One-hot & standard scaling   │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                      ┌──────────┤  Deterministic Rule Guardrail   ├──────────┐
                      │          │  (rules/rule_base.yaml)         │          │
                      │          └────────────────┬────────────────┘          │
             Hard Violation                       │ All Pass                  │
                      │                           ▼                           │
                      │          ┌─────────────────────────────────┐          │
                      │          │     Soft Policy RAG Agent       │          │
                      │          │  • Chroma Cloud Vector Store    │          │
                      │          │  • 1,708 RBI Circular Chunks    │          │
                      │          └────────────────┬────────────────┘          │
                      │                           │                           │
                      │                           ▼                           │
                      │          ┌─────────────────────────────────┐          │
                      │          │       Credit Risk Agent         │          │
                      │          │  • Mathematical risk scoring    │          │
                      │          │  • Grades A+ through E (0-10)   │          │
                      │          └────────────────┬────────────────┘          │
                      │                           │                           │
                      │                           ▼                           │
                      │          ┌─────────────────────────────────┐          │
                      │          │      XGBoost & SHAP Agent       │          │
                      │          │  • Calibrated Cutoff T* = 0.555 │          │
                      │          │  • TreeSHAP local attributions  │          │
                      │          └────────────────┬────────────────┘          │
                      │                           │                           │
                      ▼                           ▼                           ▼
        🛑 SHORT-CIRCUIT DECLINE           ✅ MODEL DECISION           📋 AUDIT LOGGING
        • Sub-30ms execution               • Approved / Rejected       • Neon.tech Postgres
        • Statutory breach report          • SHAP visual forces        • 100% trace capture
```

---

## 🤖 Multi-Agent Orchestration Pipeline

| Agent | Responsibility | Core Technology | Latency |
|---|---|---|---|
| **1. Preprocessor Agent** | Validates schema integrity, calculates derived ratios (FOIR, LTV, APR), and aligns features. | Pydantic v2, scikit-learn Pipeline | ~10 ms |
| **2. Deterministic Rule Guardrail** | Enforces non-negotiable statutory mandates (age limits, AML/PEP policy, maximum FOIR). | Pure Python Rule Engine, YAML config | ~5 ms |
| **3. Soft Policy RAG Agent** | Semantic guideline retrieval over RBI regulatory compendiums for discretionary compliance. | Chroma Cloud / Local ChromaDB, Groq Llama 3.1 | ~60 ms |
| **4. Credit Risk Agent** | Computes deterministic risk score (0.0 to 10.0) across bureau, debt, income, and loan size metrics. | Mathematical Weighted Scoring Engine | ~5 ms |
| **5. XGBoost & SHAP Agent** | Evaluates default probability against $T^* = 0.555$ and generates additive local Shapley explanations. | XGBoost, TreeSHAP, LinearSHAP | ~25 ms |

---

## 📊 Model Benchmarks & Threshold Calibration

The core classification engine was trained and cross-validated on **40,000+ historical banking records**. Rather than relying on a naive $0.50$ cutoff, threshold calibration was conducted to optimize financial utility and mitigate Type II errors (costly false approvals).

| Metric | Baseline ($T = 0.50$) | Calibrated Model ($T^* = 0.555$) | Delta / Impact |
|---|---|---|---|
| **Overall Accuracy** | 83.1% | **85.4%** | +2.3% improvement |
| **ROC-AUC Score** | 0.842 | **0.852** | Consistent discriminatory power |
| **Type II Errors (False Approvals)** | 1,842 | **1,567** | **-14.9% bad loan reduction** |
| **Approval Specificity** | 81.4% | **86.7%** | Stronger downside protection |
| **Regulatory Auditability** | Moderate | **100% (SHAP + Rules)** | Fully compliant with RBI fair lending |

---

## 📑 Dual-Audience Justification Dossier

To bridge the gap between technical machine learning outputs and real-world banking operations, the system automatically produces two parallel justification views:

### 1. Plain English Borrower Decision Notice
- **Concrete & Actionable**: Written in straightforward language without ML jargon or raw mathematical formulas.
- **5 Clear Operational Pillars**:
  1. **Credit Profile & History**: Reports bureau score against product thresholds.
  2. **Debt-to-Income & Monthly Capacity**: Highlights net monthly income and existing debt commitments (FOIR).
  3. **Collateral & Security**: Verifies Loan-to-Value (LTV) ratio and asset coverage.
  4. **Regulatory & Policy Alignment**: Confirms statutory verification status (KYC, KFS disclosure, PEP screening).
  5. **Final Underwriting Determination**: Clear approval terms or itemized remediation steps for denied applicants.

### 2. 10-Point RBI Regulatory & Statutory Compliance Audit Memo
- **Format**: Structured formal audit memorandum designed for supervisory examination.
- **10 Core Regulatory Sections**:
  1. Statutory Mandate & Legal Basis (RBI Act, 1934 & Digital Lending Guidelines 2022)
  2. Applicant Demographic & Verification Profile
  3. Fair Lending & Anti-Discrimination Declaration (Non-bias verification across gender and regional attributes)
  4. Fixed Obligation to Income Ratio (FOIR) Assessment (RBI prudential caps)
  5. Loan-to-Value (LTV) & Collateral Adequacy Audit
  6. Credit Information Companies (CIC) Data Verification
  7. Key Fact Statement (KFS) & APR Transparency Compliance
  8. Politically Exposed Persons (PEP) & AML Screening
  9. Explainable AI (XAI) Model Governance & Algorithmic Attribution (SHAP local contribution decomposition)
  10. Supervisory Audit Trail & Immutable Archival Record

---

## 🔍 Visual Explainable AI (SHAP Attributions)

The system computes local Shapley values ($f(x) - E[f(x)]$) for every non-short-circuited application:
- **Base Expected Value**: $E[f(x)] = 0.50$
- **Positive Push Factors (Green)**: Attributes pushing probability towards approval (e.g., Bureau Score $\ge 750$, Low FOIR, High Income).
- **Negative Drag Factors (Red)**: Attributes pulling probability towards rejection (e.g., Excessive LTV, Low Bureau Score, High Existing Debt).
- **Factor Impact Table**: Directly tabulates feature values, directional pressure, and percentage contribution ($\Delta$) for non-technical stakeholders.

---

## 📁 Clean Repository Structure

```text
Agentic_Loan_Decision_System/
├── agents/                     # Specialized autonomous AI agents
│   ├── compliance_agent.py        # Regulatory compliance & policy guardrails
│   ├── orchestrator.py            # Master workflow coordinator
│   ├── rag_agent.py               # Chroma Cloud semantic guideline search
│   ├── risk_agent.py              # Mathematical risk scoring engine
│   └── xai_agent.py               # SHAP explainability engine
├── api/                        # FastAPI cloud backend
│   ├── routes/
│   │   └── evaluate.py            # POST /api/v1/evaluate, GET /history, GET /stats
│   ├── app.py                     # FastAPI application entry point & lifespan
│   └── schemas.py                 # Strict Pydantic v2 validation contracts
├── data/
│   └── processed/                 # Canonical 40,000-row matrices & Excel datasets
│       ├── loan_approval_model.xlsx
│       ├── X_train.csv
│       └── y_train.csv
├── frontend/                   # Streamlit interactive application
│   ├── streamlit_app.py           # Production single-screen viewport UI
│   └── streamlit_app_cloud.py     # Cloud deployment adapter
├── models/                     # Production model artefacts & fitted explainers
│   ├── explainer/
│   │   ├── shap_explainer.joblib   # Fitted Tree/Linear SHAP explainer
│   │   └── lime_explainer.joblib   # Fitted LIME Tabular explainer
│   ├── loan_approval_model_xgb.joblib # Calibrated XGBoost Classifier
│   ├── loan_approval_model.joblib     # Calibrated Logistic Regression
│   ├── preprocessor.joblib            # Sklearn ColumnTransformer
│   └── scaler.joblib                  # Fitted StandardScaler
├── pipeline/                   # Data engineering & evaluation
│   ├── build_artifacts.py         # Full artifact rebuild pipeline
│   ├── evaluation.py              # Precision, recall, and ROC-AUC benchmarking
│   └── ingest_rag.py              # RBI circular chunking & vector ingestion
├── rules/                      # Statutory lending rules
│   ├── rbi_guidelines/            # Source regulatory circulars
│   ├── rule_base.yaml             # Deterministic statutory thresholds
│   └── rule_engine.py             # High-throughput constraint engine
├── tests/                      # Comprehensive test suite (76/76 passing)
│   ├── test_agents.py             # Agent unit & mock tests (11 tests)
│   ├── test_api.py                # FastAPI schema & endpoint tests (13 tests)
│   ├── test_integration.py        # End-to-end orchestrator pipeline tests (15 tests)
│   ├── test_pipeline.py           # Dataset integrity & model inference tests (6 tests)
│   ├── test_risk_agent.py         # Mathematical risk scoring tests (17 tests)
│   └── test_rule_engine.py        # Statutory constraint & guardrail tests (14 tests)
├── utils/                      # Shared utility modules
│   ├── db_utils.py                # Neon.tech Postgres audit logging
│   ├── llm_utility.py             # Multi-provider LLM connector (Groq / Mock)
│   ├── model_loader.py            # Local & HuggingFace Hub artifact loader
│   └── preprocessing.py           # Pydantic v2 application normalizer
├── DEPLOYMENT.md               # Cloud deployment guide
├── Makefile                    # Developer automation targets
├── requirements.txt            # Locked production dependencies
└── setup.ps1                   # Windows PowerShell setup script
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)
- Groq API Key (Optional, fallback MockLLM is included)

### 2. Local Installation
```bash
# Clone the repository
git clone https://github.com/rushi-gulum/Agentic_Loan_Decision_System.git
cd Agentic_Loan_Decision_System

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Run the test suite (76 tests)
pytest tests/ -v
```

### 3. Launching Applications

**Run the FastAPI Backend:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# Interactive Swagger documentation available at http://localhost:8000/docs
```

**Run the Streamlit Dashboard:**
```bash
streamlit run frontend/streamlit_app.py
# Access dashboard at http://localhost:8501
```

---

## ☁️ Deployment Guide

| Component | Target Platform | Free Tier Configuration |
|---|---|---|
| **API Backend** | [Render](https://render.com) | Free Web Service (512 MB RAM, lazy-loading models) |
| **Interactive UI** | [Streamlit Community Cloud](https://streamlit.io/cloud) | Free public deployment pointing to `frontend/streamlit_app.py` |
| **Vector Database** | [Chroma Cloud](https://cloud.trychroma.com) | Free tier storing 1,708 embedded RBI circular chunks |
| **Audit Database** | [Neon.tech](https://neon.tech) | Free Serverless Postgres storing immutable decision logs |
| **Model Artefacts** | [Hugging Face Hub](https://huggingface.co) | Free model repository for automated downloading |

*For complete configuration parameters and environment variables, refer to [DEPLOYMENT.md](DEPLOYMENT.md).*

---

## 🧪 Automated Testing Suite

The repository contains an exhaustive test suite covering unit, integration, schema, and performance bounds:
```bash
pytest tests/ -v
```
```text
======================= 76 passed in ~24s =======================
✔ tests/test_agents.py        — 11 passed (RiskAgent, ComplianceAgent, XAIAgent, Orchestrator)
✔ tests/test_api.py           — 13 passed (Schemas, Health, Evaluate, CORS, Audit)
✔ tests/test_integration.py   — 15 passed (Short-circuit, Determinism, Full flow)
✔ tests/test_pipeline.py      — 6 passed  (Dataset shapes, Model inference)
✔ tests/test_risk_agent.py    — 17 passed (Bounds, Precision, Extreme edge cases)
✔ tests/test_rule_engine.py   — 14 passed (Hard constraints, Robustness, Parsing)
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
