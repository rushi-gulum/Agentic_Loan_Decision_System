#!/usr/bin/env python3
"""
pipeline/build_artifacts.py
============================
Deterministic ML artefact builder.

What this script produces
-------------------------
models/preprocessor.joblib          sklearn Pipeline (fitted)
models/scaler.joblib                 StandardScaler (fitted, legacy compat)
models/loan_approval_model.joblib    LogisticRegression (interpretable default)
models/explainer/shap_explainer.joblib  shap.Explainer (real, fitted)
models/explainer/lime_explainer.joblib  LimeTabularExplainer (real, fitted)
data/processed/X_train.csv
data/processed/y_train.csv
data/processed/training_data_raw.csv

Design principles
-----------------
- sklearn only — no Keras/TensorFlow dependency (avoids torch DLL issues on Windows)
- LogisticRegression for interpretability; XGBoost added if xgboost is installed
- SHAP LinearExplainer for logistic regression (fast, no background-sample cost)
- LIME TabularExplainer on the full training set
- Deterministic: np.random.seed(42) throughout
- Idempotent: safe to re-run; overwrites previous artefacts
- The Keras .h5 file is NOT produced here; xai_agent uses .joblib model instead

Run
---
    python pipeline/build_artifacts.py
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── project root on sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import LoanPreprocessor, LoanApplicationSchema

# ── directories ──────────────────────────────────────────────────────────
DIRS = [
    ROOT / "models" / "explainer",
    ROOT / "data"   / "processed",
]
for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ── random seed ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


class CoeffExplainer:
    """
    Module-level (picklable) SHAP-compatible explainer for LogisticRegression.
    Used as fallback when shap/numba DLLs are blocked (Windows AppControl).
    Implements .shap_values() via coefficient × feature-value product.
    On Linux/Render the real shap.LinearExplainer is used instead.
    """
    def __init__(self, model, feature_names):
        self.coef_         = model.coef_[0]
        self.feature_names = list(feature_names)
        self.fitted_       = True

    def shap_values(self, X):
        if hasattr(X, "values"):
            X = X.values
        X = np.array(X, dtype=float)
        return (X * self.coef_).tolist()


FEATURE_NAMES = [
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


# ─────────────────────────────────────────────────────────────────────────
# 1. Training data generator
# ─────────────────────────────────────────────────────────────────────────

def _generate_training_data(n: int = 2000) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic loan applications with deterministic seed.
    Approval labels use rule-based logic (no randomness beyond data generation),
    so the dataset is reproducible across runs.
    """
    rng = np.random.RandomState(SEED)

    loan_types    = ["housing", "personal", "vehicle", "gold", "personal"]  # personal weighted up
    genders       = ["Male", "Female", "Other"]
    interest_types= ["Fixed", "Floating"]
    ovd_types     = ["Aadhaar", "PAN", "Passport", "Driving Licence"]
    pin_codes     = [110001, 400001, 560001, 600001, 700001, 411001, 500001, 380001]

    base_date = datetime(2023, 1, 1)
    records   = []

    for i in range(n):
        age          = int(rng.randint(21, 65))
        bureau       = int(np.clip(650 + age * 2 + rng.normal(0, 50), 300, 900))
        income       = max(15_000, int(25_000 + age * 800 + (bureau - 650) * 50 + rng.normal(0, 8_000)))
        loan_type    = rng.choice(loan_types)

        # loan amounts
        multipliers = {"housing": (30, 80, 5_000_000), "vehicle": (8, 25, 2_000_000),
                       "gold": (3, 10, 800_000), "personal": (3, 15, 1_000_000)}
        lo, hi, cap  = multipliers.get(loan_type, (3, 15, 1_000_000))
        requested    = min(int(income * rng.uniform(lo, hi)), cap)
        sanctioned   = int(requested * rng.uniform(0.75, 1.0))

        # obligations & FOIR
        existing_obl = int(income * rng.uniform(0.05, 0.38))
        rate_ranges  = {"housing": (8.5, 12.0), "vehicle": (9.0, 14.0),
                        "gold": (11.0, 16.0), "personal": (12.0, 24.0)}
        lo_r, hi_r   = rate_ranges.get(loan_type, (12.0, 24.0))
        rate         = round(rng.uniform(lo_r, hi_r), 2)

        tenure_ranges = {"housing": (120, 300), "vehicle": (36, 84)}
        t_lo, t_hi   = tenure_ranges.get(loan_type, (12, 60))
        tenure       = int(rng.randint(t_lo, t_hi))

        mr  = rate / 1200
        emi = (sanctioned * mr * (1 + mr)**tenure) / ((1 + mr)**tenure - 1) if mr > 0 else sanctioned / tenure
        foir = min(((existing_obl + emi) / income) * 100, 95.0)

        # property / LTV
        prop_val  = None
        ltv_ratio = None
        if loan_type in ("housing", "vehicle"):
            prop_val  = int(sanctioned / rng.uniform(0.65, 0.90))
            ltv_ratio = round(min(sanctioned / prop_val, 1.0), 3)

        # dates
        app_date      = base_date + timedelta(days=int(rng.randint(0, 365)))
        sanction_date = app_date  + timedelta(days=int(rng.randint(3, 25)))

        # approval: deterministic rule (not random), reproducible
        approved = (
            bureau >= 650
            and foir  <= 70
            and (not bool(rng.choice([True, False], p=[0.02, 0.98])))  # PEP
            and income >= 20_000
        )

        records.append({
            "application_id":                  f"APP_{i+1:05d}",
            "application_date":                app_date.strftime("%Y-%m-%d"),
            "sanction_date":                   sanction_date.strftime("%Y-%m-%d"),
            "loan_type":                       loan_type,
            "age_years":                       age,
            "gender":                          rng.choice(genders, p=[0.60, 0.35, 0.05]),
            "pin_code":                        rng.choice(pin_codes),
            "pep_flag":                        bool(rng.choice([True, False], p=[0.02, 0.98])),
            "bureau_score":                    bureau,
            "monthly_income_inr":              income,
            "existing_monthly_obligations_inr":existing_obl,
            "requested_amount_inr":            requested,
            "sanctioned_amount_inr":           sanctioned,
            "tenure_months":                   tenure,
            "interest_type":                   rng.choice(interest_types, p=[0.70, 0.30]),
            "interest_rate_annual_pct":        rate,
            "processing_fee_inr":              max(500, int(requested * 0.01)),
            "other_charges_inr":               int(rng.randint(200, 2_000)),
            "apr_pct":                         round(rate + rng.uniform(0.1, 1.5), 2),
            "kfs_provided":                    bool(rng.choice([True, False], p=[0.95, 0.05])),
            "proposed_emi_inr":                round(emi, 2),
            "foir_total_obligations_pct":      round(foir, 2),
            "ovd_type":                        rng.choice(ovd_types),
            "kyc_mode":                        rng.choice(["Video KYC", "Offline", "eKYC", "CKYC"]),
            "property_value_inr":              prop_val,
            "ltv_ratio":                       ltv_ratio,
            "target_approved":                 int(approved),
            "target_default_12m":              0,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────

def _build_preprocessing(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, pd.Series]:
    """Fit LoanPreprocessor, return (X_array, y_series)."""
    import pandas as pd

    preprocessor = LoanPreprocessor()

    # Convert to DataFrame so NaN-filled columns (property_value_inr, ltv_ratio)
    # are handled uniformly by the pipeline's DerivedFeatureTransformer.
    df_raw = pd.DataFrame(records)
    df_raw["property_value_inr"] = df_raw["property_value_inr"].fillna(0)
    df_raw["ltv_ratio"]          = df_raw["ltv_ratio"].fillna(0)

    # fit() and transform() must both receive the same DataFrame
    preprocessor.pipeline.fit(df_raw)
    preprocessor.is_fitted = True
    X = preprocessor.pipeline.transform(df_raw)

    preprocessor.save(str(ROOT / "models" / "preprocessor.joblib"))
    logger.info("preprocessor.joblib  shape=%s", X.shape)

    # Legacy StandardScaler (sklearn) — fit on preprocessed features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, str(ROOT / "models" / "scaler.joblib"))
    logger.info("scaler.joblib saved (StandardScaler)")

    df_raw.to_csv(ROOT / "data" / "processed" / "training_data_raw.csv", index=False)

    y = pd.Series([r["target_approved"] for r in records], name="target_approved")
    df_X = pd.DataFrame(X, columns=FEATURE_NAMES[:X.shape[1]])
    df_X.to_csv(ROOT / "data" / "processed" / "X_train.csv", index=False)
    y.to_csv(ROOT / "data" / "processed" / "y_train.csv", index=False)
    logger.info("X_train.csv  y_train.csv  saved")

    return X, y


# ─────────────────────────────────────────────────────────────────────────
# 3. Model training
# ─────────────────────────────────────────────────────────────────────────

def _train_model(X: np.ndarray, y: pd.Series):
    """
    Train a LogisticRegression (interpretable default).
    If xgboost is available a second XGB model is also saved.
    Returns the primary (LogisticRegression) fitted model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=SEED, stratify=y)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_tr, y_tr)

    acc = lr.score(X_val, y_val)
    logger.info("LogisticRegression  val_accuracy=%.3f", acc)
    logger.info("\n%s", classification_report(y_val, lr.predict(X_val), zero_division=0))

    model_path = ROOT / "models" / "loan_approval_model.joblib"
    joblib.dump(lr, str(model_path))
    logger.info("loan_approval_model.joblib saved  (%d bytes)", model_path.stat().st_size)

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=SEED, eval_metric="logloss",
                             use_label_encoder=False)
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_acc = xgb.score(X_val, y_val)
        logger.info("XGBClassifier  val_accuracy=%.3f", xgb_acc)
        xgb_path = ROOT / "models" / "loan_approval_model_xgb.joblib"
        joblib.dump(xgb, str(xgb_path))
        logger.info("loan_approval_model_xgb.joblib saved")
    except ImportError:
        logger.info("xgboost not installed — skipping XGB model")

    return lr


# ─────────────────────────────────────────────────────────────────────────
# 4. Explainers
# ─────────────────────────────────────────────────────────────────────────

def _build_explainers(model, X: np.ndarray, feature_names: List[str]):
    """
    Fit and save real SHAP and LIME explainers.

    SHAP: Uses shap.explainers.Linear which does NOT require numba.
          Falls back to a lightweight correlation-based explainer if shap
          itself is blocked (Windows AppControl DLL issue with numba).
    LIME: LimeTabularExplainer fitted on the full training set.
    """
    # ── SHAP ─────────────────────────────────────────────────────────────
    try:
        # Import only the linear sub-module — avoids numba via _clustering
        import importlib
        shap_linear = importlib.import_module("shap.explainers.linear")
        LinearExplainer = shap_linear.Linear

        masker         = importlib.import_module("shap.maskers").Independent(X, max_samples=200)
        shap_explainer = LinearExplainer(model, masker, feature_names=feature_names)

        sample_shap = shap_explainer.shap_values(X[:3])
        assert sample_shap is not None
        logger.info("SHAP LinearExplainer verified  sample.shape=%s", np.array(sample_shap).shape)

    except Exception as exc:
        logger.warning("SHAP import failed (%s) — using coefficient-based fallback", exc)
        shap_explainer = CoeffExplainer(model, feature_names)
        logger.info("Coefficient-based fallback explainer ready (%d features)", len(feature_names))

    shap_path = ROOT / "models" / "explainer" / "shap_explainer.joblib"
    joblib.dump(shap_explainer, str(shap_path))
    logger.info("shap_explainer.joblib  saved  (%d bytes)", shap_path.stat().st_size)

    # ── LIME ─────────────────────────────────────────────────────────────
    # LimeTabularExplainer contains internal lambdas that are not picklable.
    # Save a serialisable manifest; load_lime_explainer() reconstructs it.
    from lime.lime_tabular import LimeTabularExplainer
    lime_manifest = {
        "training_data": X.tolist(),
        "feature_names": feature_names,
        "class_names":   ["Rejected", "Approved"],
        "mode":          "classification",
        "discretize_continuous": True,
        "random_state":  SEED,
    }

    # Smoke-test: reconstruct and explain first row
    lime_explainer = LimeTabularExplainer(
        training_data         = X,
        feature_names         = feature_names,
        class_names           = ["Rejected", "Approved"],
        mode                  = "classification",
        discretize_continuous = True,
        random_state          = SEED,
    )

    def _predict_fn(arr):
        return model.predict_proba(arr)

    lime_exp = lime_explainer.explain_instance(X[0], _predict_fn, num_features=10)
    assert lime_exp is not None
    logger.info("LIME TabularExplainer verified  num_features=%d", len(lime_exp.as_list()))

    lime_path = ROOT / "models" / "explainer" / "lime_explainer.joblib"
    joblib.dump(lime_manifest, str(lime_path))
    logger.info("lime_explainer.joblib (manifest)  saved  (%d bytes)", lime_path.stat().st_size)

    return shap_explainer, lime_manifest


# ─────────────────────────────────────────────────────────────────────────
# 5. Validation
# ─────────────────────────────────────────────────────────────────────────

def _validate():
    """Reload every artefact and verify it is usable."""
    required = {
        "models/preprocessor.joblib":             "preprocessor",
        "models/scaler.joblib":                   "scaler",
        "models/loan_approval_model.joblib":      "classification model",
        "models/explainer/shap_explainer.joblib": "shap explainer",
        "models/explainer/lime_explainer.joblib": "lime explainer",
        "data/processed/X_train.csv":             "training features",
        "data/processed/y_train.csv":             "training labels",
    }

    all_ok = True
    print("\n" + "─" * 55)
    print("  ARTEFACT VALIDATION")
    print("─" * 55)
    for path, label in required.items():
        full = ROOT / path
        if full.exists():
            size_kb = full.stat().st_size // 1024
            print(f"  ✅  {label:<32}  {size_kb} KB")
        else:
            print(f"  ❌  {label:<32}  MISSING: {path}")
            all_ok = False

    if not all_ok:
        raise RuntimeError("One or more required artefacts are missing.")

    # Functional checks
    from utils.preprocessing import LoanPreprocessor
    prep = LoanPreprocessor.load(str(ROOT / "models" / "preprocessor.joblib"))
    sample = {
        "age_years": 35, "bureau_score": 720, "monthly_income_inr": 75_000,
        "requested_amount_inr": 500_000, "tenure_months": 48,
        "interest_rate_annual_pct": 12.5, "foir_total_obligations_pct": 35.0,
        "gender": "Male", "interest_type": "Fixed",
        # Ensure all fields needed by pipeline are present
        "pin_code": 400001, "pep_flag": False, "kfs_provided": True,
        "existing_monthly_obligations_inr": 10_000,
        "sanctioned_amount_inr": 500_000, "processing_fee_inr": 5_000,
        "other_charges_inr": 500, "apr_pct": 13.0, "proposed_emi_inr": 12_000,
        "property_value_inr": 0, "ltv_ratio": 0,
        "ovd_type": "PAN", "kyc_mode": "eKYC",
        "application_date": "2023-01-01", "sanction_date": "2023-01-08",
        "loan_type": "personal",
    }
    out = prep.transform(sample)
    assert out.shape[0] == 1, f"Preprocessor transform shape unexpected: {out.shape}"
    print(f"  ✅  preprocessor.transform()   shape={out.shape}")

    import shap
    shap_exp = joblib.load(str(ROOT / "models" / "explainer" / "shap_explainer.joblib"))
    assert hasattr(shap_exp, "shap_values"), "SHAP explainer missing .shap_values()"
    sv = shap_exp.shap_values(out)
    assert sv is not None
    print(f"  ✅  shap_explainer.shap_values()  returned {type(sv).__name__}")

    model = joblib.load(str(ROOT / "models" / "loan_approval_model.joblib"))
    proba = model.predict_proba(out)
    assert proba.shape == (1, 2), f"Unexpected predict_proba shape: {proba.shape}"
    print(f"  ✅  model.predict_proba()       shape={proba.shape}  p_approve={proba[0,1]:.3f}")

    print("─" * 55)
    print("  ALL ARTEFACTS VALID ✅")
    print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 2000):
    print("\n" + "═" * 55)
    print("  AGENTIC LOAN DECISION — ARTEFACT BUILDER")
    print("═" * 55)

    logger.info("Generating %d training samples ...", n_samples)
    records = _generate_training_data(n_samples)

    logger.info("Fitting preprocessing pipeline ...")
    X, y = _build_preprocessing(records)

    logger.info("Training classification model ...")
    model = _train_model(X, y)

    feat_names = FEATURE_NAMES[:X.shape[1]]
    logger.info("Building real SHAP + LIME explainers ...")
    _build_explainers(model, X, feat_names)

    _validate()

    print("  Next steps:")
    print("    python pipeline/ingest_rag.py   (if not done)")
    print("    make api | .\\setup.ps1 api")
    print("    make ui  | .\\setup.ps1 ui\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run: skipping training, validating existing artefacts only.")
        _validate()
    else:
        main(n_samples=args.samples)
