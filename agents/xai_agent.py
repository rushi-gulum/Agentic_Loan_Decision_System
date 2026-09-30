"""
agents/xai_agent.py
===================
Explainable AI agent for the Agentic Loan Decision System.

What it does
------------
1. Loads the sklearn loan approval model  (loan_approval_model.joblib)
2. Runs SHAP LinearExplainer to get feature attributions
3. Runs LIME TabularExplainer for instance-level explanation
4. Uses Groq (via get_llm) to generate human-readable summaries
   — customer-facing paragraph
   — regulator-facing structured report

All LLM calls go through utils/llm_utility.get_llm() — Groq-primary,
never direct OpenAI() instantiation.

The .h5 Keras model is NOT required. The pipeline uses sklearn models
produced by pipeline/build_artifacts.py.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm_utility import get_llm, TaskType

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
_MODEL_PATH       = "models/loan_approval_model.joblib"
_SHAP_PATH        = "models/explainer/shap_explainer.joblib"
_LIME_PATH        = "models/explainer/lime_explainer.joblib"
_PREPROCESSOR_PATH= "models/preprocessor.joblib"
_X_TRAIN_PATH     = "data/processed/X_train.csv"

# ── Module-level cache so we only load once per process ───────────────────
_model            = None
_shap_explainer   = None
_lime_explainer   = None
_feature_names    = None


# ─────────────────────────────────────────────────────────────────────────
# Loader helpers
# ─────────────────────────────────────────────────────────────────────────

def _load_artefacts():
    """Lazy-load all ML artefacts. Raises FileNotFoundError with clear message."""
    global _model, _shap_explainer, _lime_explainer, _feature_names

    if _model is not None:
        return  # already loaded

    import joblib
    from utils.model_loader import load_approval_model, load_shap_explainer, load_lime_explainer

    _model, model_name = load_approval_model()
    logger.info("Loaded model: %s", model_name)

    _shap_explainer = load_shap_explainer()
    _lime_explainer = load_lime_explainer()

    # Feature names — prefer X_train.csv header, fall back to SHAP's stored names
    if os.path.exists(_X_TRAIN_PATH):
        _feature_names = list(pd.read_csv(_X_TRAIN_PATH, nrows=0).columns)
    elif hasattr(_shap_explainer, "feature_names") and _shap_explainer.feature_names is not None:
        _feature_names = list(_shap_explainer.feature_names)
    else:
        _feature_names = [f"feature_{i}" for i in range(25)]

    logger.info("Artefacts ready — %d features", len(_feature_names))


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing helper (used by orchestrator and standalone tests)
# ─────────────────────────────────────────────────────────────────────────

def preprocess(raw_data: dict) -> np.ndarray:
    """
    Preprocess a raw application dict into a (1, n_features) numpy array.
    Uses the unified LoanPreprocessor first; falls back to a simple
    column-alignment approach if the preprocessor is unavailable.
    """
    try:
        from utils.preprocessing import preprocess_single_application
        return preprocess_single_application(raw_data, _PREPROCESSOR_PATH)
    except Exception as exc:
        logger.warning("Unified preprocessor failed (%s), using fallback", exc)
        return _fallback_preprocess(raw_data)


def _fallback_preprocess(raw: dict) -> np.ndarray:
    """Minimal feature alignment when preprocessor is unavailable."""
    COLS = [
        "age_years", "pin_code", "pep_flag", "bureau_score",
        "monthly_income_inr", "existing_monthly_obligations_inr",
        "requested_amount_inr", "sanctioned_amount_inr", "tenure_months",
        "interest_rate_annual_pct", "processing_fee_inr", "other_charges_inr",
        "apr_pct", "kfs_provided", "proposed_emi_inr",
        "foir_total_obligations_pct", "property_value_inr", "ltv_ratio",
        "time_to_sanction_days", "application_month",
        "interest_type_encoded", "gender_Female", "gender_Male", "gender_Other",
        "ovd_provided",
    ]
    row = {c: 0 for c in COLS}
    row.update({k: v for k, v in raw.items() if k in COLS})
    row["pep_flag"]  = int(bool(row.get("pep_flag", False)))
    row["kfs_provided"] = int(bool(row.get("kfs_provided", True)))
    return np.array([[row[c] for c in COLS]], dtype=np.float32)


def model_predict(features: np.ndarray) -> float:
    """Return P(approved) for a preprocessed feature array."""
    _load_artefacts()
    proba = _model.predict_proba(features)
    return float(proba[0, 1])


# ─────────────────────────────────────────────────────────────────────────
# Core explanation function
# ─────────────────────────────────────────────────────────────────────────

def explain_prediction(
    features_array: np.ndarray,
    applicant_data: dict,
    compliance_data: dict,
    risk_data: dict,
    model_name: str = "LoanApprovalClassifier",
) -> dict:
    """
    Generate SHAP + LIME explanations and Groq-powered summaries.

    Parameters
    ----------
    features_array : preprocessed (1, n) array
    applicant_data : raw applicant dict
    compliance_data: output of compliance agent
    risk_data      : output of risk agent
    model_name     : label for reports

    Returns
    -------
    {
      "user_explanation":      str,
      "regulator_explanation": str,
      "raw_data":              dict,
    }
    """
    _load_artefacts()

    # ── Prediction ────────────────────────────────────────────────────────
    proba    = _model.predict_proba(features_array)
    p_approve= float(proba[0, 1])
    decision = "approved" if p_approve >= 0.5 else "rejected"

    # ── SHAP ──────────────────────────────────────────────────────────────
    shap_values = _shap_explainer.shap_values(features_array)

    # LinearExplainer returns 1-D; CoeffExplainer returns list of lists
    if isinstance(shap_values, list):
        if len(shap_values) > 0 and isinstance(shap_values[0], list):
            sv_arr = np.array(shap_values[0])       # CoeffExplainer: [[vals]]
        else:
            sv_arr = np.array(shap_values)          # Linear single-output
    else:
        sv_arr = np.array(shap_values)
    sv_arr = sv_arr.flatten()
    feat_names = _feature_names[:len(sv_arr)]

    # Top 10 features by absolute impact
    top_idx   = np.argsort(np.abs(sv_arr))[::-1][:10]
    top_shap  = [
        {
            "feature": feat_names[i],
            "shap_value": round(float(sv_arr[i]), 4),
            "direction": "positive" if sv_arr[i] > 0 else "negative",
        }
        for i in top_idx
    ]

    # ── LIME ──────────────────────────────────────────────────────────────
    def _predict_fn(arr):
        return _model.predict_proba(arr)

    try:
        lime_exp   = _lime_explainer.explain_instance(
            features_array[0].astype(float),
            _predict_fn,
            num_features=10,
        )
        lime_list  = lime_exp.as_list()
    except Exception as exc:
        logger.warning("LIME explanation failed: %s", exc)
        lime_list  = []

    # ── Build raw data block ───────────────────────────────────────────────
    raw_data = {
        "decision":             decision,
        "p_approve":            round(p_approve, 4),
        "p_reject":             round(1 - p_approve, 4),
        "model_used":           model_name,
        "shap_top_features":    top_shap,
        "lime_explanation":     lime_list,
        "risk_drivers":         risk_data.get("drivers", []),
        "compliance_summary":   compliance_data.get("explanation", ""),
    }

    # ── LLM summaries ─────────────────────────────────────────────────────
    user_text   = _generate_user_explanation(decision, applicant_data, top_shap, p_approve)
    reg_text    = _generate_regulator_explanation(decision, applicant_data, raw_data,
                                                   compliance_data, risk_data)

    return {
        "user_explanation":      user_text,
        "regulator_explanation": reg_text,
        "raw_data":              raw_data,
    }


# ─────────────────────────────────────────────────────────────────────────
# LLM explanation generators (Groq via get_llm)
# ─────────────────────────────────────────────────────────────────────────

def _generate_user_explanation(
    decision:      str,
    applicant:     dict,
    top_shap:      list,
    p_approve:     float,
) -> str:
    """Customer-facing explanation in plain English."""
    try:
        llm = get_llm(TaskType.FAST)   # Groq fast model

        # Summarise top positive and negative factors
        pos = [f["feature"] for f in top_shap if f["direction"] == "positive"][:3]
        neg = [f["feature"] for f in top_shap if f["direction"] == "negative"][:3]

        prompt = f"""You are a friendly loan officer explaining a decision.

Loan type : {applicant.get('loan_type', 'loan')}
Bureau score: {applicant.get('bureau_score', 'N/A')}
FOIR       : {applicant.get('foir_total_obligations_pct', 'N/A')}%
Decision   : {decision.upper()}  (approval probability {p_approve:.0%})
Top supporting factors : {', '.join(pos) or 'none'}
Top risk factors       : {', '.join(neg) or 'none'}

Write 3-5 plain-English sentences explaining the {decision} decision.
Mention actual numbers. Be empathetic and constructive. No markdown."""

        raw = llm.invoke(prompt)
        return (raw.content if hasattr(raw, "content") else str(raw)).strip()

    except Exception as exc:
        logger.warning("User explanation LLM failed: %s", exc)
        return (
            f"Your {applicant.get('loan_type','loan')} application has been {decision}. "
            f"Key factors include your bureau score of {applicant.get('bureau_score','N/A')} "
            f"and FOIR of {applicant.get('foir_total_obligations_pct','N/A')}%."
        )


def _generate_regulator_explanation(
    decision:    str,
    applicant:   dict,
    raw_data:    dict,
    compliance:  dict,
    risk:        dict,
) -> str:
    """Structured regulatory audit report."""
    try:
        llm = get_llm(TaskType.SMART)

        prompt = f"""You are a financial auditor preparing a regulatory compliance report.

Decision        : {decision.upper()}
Approval P      : {raw_data['p_approve']:.4f}
Model used      : {raw_data['model_used']}
Compliance score: {compliance.get('compliance_score', 0):.2f}
Hard violations : {len(compliance.get('hard_violations', []))}
Soft violations : {len(compliance.get('soft_violations', []))}
Risk grade      : {risk.get('grade', 'N/A')}  (score {risk.get('risk_score_10', 'N/A')}/10)
Risk drivers    : {risk.get('drivers', [])}
Top SHAP        : {json.dumps(raw_data['shap_top_features'][:5], indent=2)}

Write a 2-section regulatory report:
Section 1 — Executive Summary (2 short paragraphs)
Section 2 — Technical Appendix (feature impacts, risk assessment, compliance checks)
Tone: professional, precise, suitable for RBI regulatory review. No markdown headers."""

        raw = llm.invoke(prompt)
        return (raw.content if hasattr(raw, "content") else str(raw)).strip()

    except Exception as exc:
        logger.warning("Regulator explanation LLM failed: %s", exc)
        return (
            f"AUTOMATED DECISION REPORT — {decision.upper()}\n"
            f"Model: {raw_data.get('model_used', 'N/A')} | "
            f"P(approve)={raw_data.get('p_approve', 0):.4f} | "
            f"Risk grade: {risk.get('grade', 'N/A')} | "
            f"Compliance score: {compliance.get('compliance_score', 0):.2f}\n"
            f"Risk drivers: {risk.get('drivers', [])}"
        )


# ─────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample = {
        "application_id": "TEST-001",
        "loan_type": "housing",
        "age_years": 35,
        "gender": "Male",
        "bureau_score": 720,
        "monthly_income_inr": 80_000,
        "existing_monthly_obligations_inr": 15_000,
        "requested_amount_inr": 2_500_000,
        "sanctioned_amount_inr": 2_500_000,
        "tenure_months": 240,
        "interest_rate_annual_pct": 9.0,
        "processing_fee_inr": 25_000,
        "other_charges_inr": 1_000,
        "apr_pct": 9.5,
        "kfs_provided": True,
        "proposed_emi_inr": 22_000,
        "foir_total_obligations_pct": 46.25,
        "property_value_inr": 3_500_000,
        "ltv_ratio": 0.71,
        "pep_flag": False,
    }

    arr = preprocess(sample)
    print("Preprocessed shape:", arr.shape)

    result = explain_prediction(
        arr, sample,
        compliance_data={"compliance_score": 0.9, "hard_violations": [],
                         "soft_violations": [], "explanation": "All checks passed."},
        risk_data={"risk_score_10": 3.5, "grade": "A", "drivers": ["bureau_score", "foir"]},
    )
    print("Decision:", result["raw_data"]["decision"])
    print("P(approve):", result["raw_data"]["p_approve"])
    print("\nUser explanation:\n", result["user_explanation"][:300])
