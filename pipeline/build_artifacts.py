#!/usr/bin/env python3
"""
pipeline/build_artifacts.py
============================
Deterministic ML artefact & dataset builder.

What this script produces
-------------------------
loan_approval_model.xlsx              Full 40,000 rows x 26 columns Excel dataset
data/processed/training_data_raw.csv  Full 40,000 rows raw application records
data/processed/X_train.csv            40,000 rows preprocessed feature matrix
data/processed/y_train.csv            40,000 rows labels
models/preprocessor.joblib            sklearn Pipeline (fitted)
models/scaler.joblib                  StandardScaler (fitted, legacy compat)
models/loan_approval_model.joblib      LogisticRegression (interpretable default)
models/explainer/shap_explainer.joblib   shap.Explainer (real, fitted)
models/explainer/lime_explainer.joblib   LimeTabularExplainer manifest (real, fitted)

Run
---
    python pipeline/build_artifacts.py
    python pipeline/build_artifacts.py --samples 40000
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
from utils.explain_utils import CoeffExplainer

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
# 1. Training data generator (40,000 records)
# ─────────────────────────────────────────────────────────────────────────

def _generate_training_data(n: int = 40000) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic loan applications with deterministic seed.
    Produces comprehensive credit profiles with realistic credit scoring,
    risk parameters, and regulatory boundary adherence.
    """
    rng = np.random.RandomState(SEED)

    loan_types     = ["housing", "personal", "vehicle", "gold", "personal"]  # personal weighted up
    genders        = ["Male", "Female", "Other"]
    interest_types = ["Fixed", "Floating"]
    ovd_types      = ["Aadhaar", "PAN", "Passport", "Driving Licence", "Voter ID"]
    kyc_modes      = ["Video KYC", "eKYC", "CKYC", "Offline"]
    pin_codes      = [110001, 400001, 560001, 600001, 700001, 411001, 500001, 380001, 302001, 226001]

    base_date = datetime(2023, 1, 1)
    records   = []

    for i in range(n):
        age          = int(rng.randint(21, 65))
        # Bureau score: 300 to 900, centered around 680
        bureau       = int(np.clip(670 + age * 1.2 + rng.normal(0, 65), 300, 900))
        # Monthly income: INR 15,000 to 500,000+ (log-normal distribution)
        income       = max(15_000, int(np.exp(rng.normal(10.8, 0.45))))
        loan_type    = rng.choice(loan_types)

        # Multipliers based on loan product
        multipliers = {
            "housing": (30, 80, 7_500_000),
            "vehicle": (8, 25, 2_500_000),
            "gold":    (3, 10, 1_000_000),
            "personal":(2, 12, 1_500_000)
        }
        lo, hi, cap = multipliers.get(loan_type, (2, 12, 1_500_000))
        requested   = min(int(income * rng.uniform(lo, hi)), cap)
        sanctioned  = int(requested * rng.uniform(0.80, 1.0))

        # Obligations & FOIR
        existing_obl = int(income * rng.uniform(0.05, 0.35))
        rate_ranges = {
            "housing": (8.25, 11.5),
            "vehicle": (9.0, 14.0),
            "gold":    (10.5, 15.5),
            "personal":(12.0, 22.0)
        }
        lo_r, hi_r = rate_ranges.get(loan_type, (12.0, 22.0))
        rate       = round(rng.uniform(lo_r, hi_r), 2)

        tenure_ranges = {"housing": (120, 300), "vehicle": (36, 84), "gold": (12, 36)}
        t_lo, t_hi    = tenure_ranges.get(loan_type, (12, 60))
        tenure        = int(rng.randint(t_lo, t_hi))

        mr  = rate / 1200
        emi = (sanctioned * mr * (1 + mr)**tenure) / ((1 + mr)**tenure - 1) if mr > 0 else sanctioned / tenure
        foir = min(((existing_obl + emi) / income) * 100, 95.0)

        # property / LTV
        prop_val  = None
        ltv_ratio = None
        if loan_type in ("housing", "vehicle", "gold"):
            haircut   = rng.uniform(0.65, 0.88) if loan_type == "housing" else rng.uniform(0.70, 0.90)
            prop_val  = int(sanctioned / haircut)
            ltv_ratio = round(min(sanctioned / prop_val, 1.0), 3)

        # dates
        app_date      = base_date + timedelta(days=int(rng.randint(0, 365)))
        sanction_date = app_date  + timedelta(days=int(rng.randint(2, 20)))

        # PEP & KFS flags
        pep_flag     = bool(rng.choice([True, False], p=[0.02, 0.98]))
        kfs_provided = bool(rng.choice([True, False], p=[0.96, 0.04]))

        # Realistic Credit Scoring Decision Logic
        hard_reject = (
            bureau < 580
            or foir > 75.0
            or income < 18_000
            or (ltv_ratio is not None and ltv_ratio > 0.92)
            or (pep_flag and rng.uniform(0, 1) < 0.80)
            or (not kfs_provided and rng.uniform(0, 1) < 0.70)
        )

        if hard_reject:
            approved = 0
        else:
            # Credit propensity logit
            z = (
                (bureau - 650.0) / 60.0
                - (foir - 40.0) / 15.0
                + (np.log(income) - np.log(30000)) * 1.4
                - (0.8 if foir > 55.0 else 0.0)
                - (1.2 if ltv_ratio and ltv_ratio > 0.80 else 0.0)
            )
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -5.0, 5.0)))
            approved = int(rng.uniform(0, 1) < prob)

        # Default probability for approved loans
        default_12m = 0
        if approved:
            p_def = 1.0 / (1.0 + np.exp(-(-2.8 - (bureau - 650) / 80.0 + (foir - 40) / 20.0)))
            default_12m = int(rng.uniform(0, 1) < p_def)

        records.append({
            "application_id":                  f"APP_{i+1:06d}",
            "application_date":                app_date.strftime("%Y-%m-%d"),
            "sanction_date":                   sanction_date.strftime("%Y-%m-%d"),
            "loan_type":                       loan_type,
            "age_years":                       age,
            "gender":                          rng.choice(genders, p=[0.58, 0.38, 0.04]),
            "pin_code":                        rng.choice(pin_codes),
            "pep_flag":                        pep_flag,
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
            "kfs_provided":                    kfs_provided,
            "proposed_emi_inr":                round(emi, 2),
            "foir_total_obligations_pct":      round(foir, 2),
            "ovd_type":                        rng.choice(ovd_types),
            "kyc_mode":                        rng.choice(kyc_modes),
            "property_value_inr":              prop_val,
            "ltv_ratio":                       ltv_ratio,
            "target_approved":                 int(approved),
            "target_default_12m":              int(default_12m),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────
# 2. Preprocessing & Dataset Export
# ─────────────────────────────────────────────────────────────────────────

def _build_preprocessing(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, pd.Series]:
    """Fit LoanPreprocessor, return (X_array, y_series) and export datasets."""
    preprocessor = LoanPreprocessor()

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

    # 1. Raw applications CSV
    raw_path = ROOT / "data" / "processed" / "training_data_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info("training_data_raw.csv saved (%d rows)", len(df_raw))

    # 2. Processed feature matrices
    y = pd.Series([r["target_approved"] for r in records], name="target_approved")
    df_X = pd.DataFrame(X, columns=FEATURE_NAMES[:X.shape[1]])
    df_X.to_csv(ROOT / "data" / "processed" / "X_train.csv", index=False)
    y.to_csv(ROOT / "data" / "processed" / "y_train.csv", index=False)
    logger.info("X_train.csv and y_train.csv saved (%d rows)", len(df_X))

    # 3. 26-column loan_approval_model dataset (25 features + target)
    df_excel = df_X.copy()
    df_excel["target"] = y.values
    logger.info("Exporting loan_approval_model datasets (%d rows x %d columns) ...",
                len(df_excel), df_excel.shape[1])

    # Save Excel at primary locations
    excel_paths = [
        ROOT / "loan_approval_model.xlsx",
        ROOT / "data" / "loan_approval_model.xlsx",
        ROOT / "data" / "processed" / "loan_approval_model.xlsx",
    ]
    for p in excel_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        df_excel.to_excel(p, index=False)
        logger.info("Saved %s (%.1f MB)", str(p.relative_to(ROOT)), p.stat().st_size / (1024 * 1024))

    # Save companion CSV for rapid reading
    csv_paths = [
        ROOT / "data" / "loan_approval_model.csv",
        ROOT / "data" / "processed" / "loan_approval_model.csv",
    ]
    for cp in csv_paths:
        df_excel.to_csv(cp, index=False)
        logger.info("Saved %s (%.1f MB)", str(cp.relative_to(ROOT)), cp.stat().st_size / (1024 * 1024))

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
    SHAP: Uses shap.LinearExplainer. Falls back to CoeffExplainer.
    LIME: LimeTabularExplainer fitted on a representative 5,000 sample background.
    """
    # ── SHAP ─────────────────────────────────────────────────────────────
    try:
        import shap
        shap_explainer = shap.LinearExplainer(model, X[:200], feature_names=feature_names)
        sample_shap = shap_explainer.shap_values(X[:3])
        assert sample_shap is not None
        logger.info("SHAP LinearExplainer verified  sample.shape=%s", np.array(sample_shap).shape)

    except Exception as exc:
        logger.warning("SHAP LinearExplainer failed (%s) — using CoeffExplainer fallback", exc)
        shap_explainer = CoeffExplainer(model, feature_names)
        logger.info("Coefficient-based fallback explainer ready (%d features)", len(feature_names))

    shap_path = ROOT / "models" / "explainer" / "shap_explainer.joblib"
    joblib.dump(shap_explainer, str(shap_path))
    logger.info("shap_explainer.joblib  saved  (%d bytes)", shap_path.stat().st_size)

    # ── LIME ─────────────────────────────────────────────────────────────
    from lime.lime_tabular import LimeTabularExplainer
    # Use 5,000 background sample for fast inference and compact serialization
    lime_bg = X[:5000] if len(X) > 5000 else X
    lime_manifest = {
        "training_data": lime_bg.tolist(),
        "feature_names": feature_names,
        "class_names":   ["Rejected", "Approved"],
        "mode":          "classification",
        "discretize_continuous": True,
        "random_state":  SEED,
    }

    # Smoke-test: reconstruct and explain first row
    lime_explainer = LimeTabularExplainer(
        training_data         = lime_bg,
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
        "data/processed/training_data_raw.csv":   "raw applications CSV",
        "loan_approval_model.xlsx":               "40,000 row Excel dataset",
    }

    all_ok = True
    print("\n" + "─" * 55)
    print("  ARTEFACT & DATASET VALIDATION")
    print("─" * 55)
    for path, label in required.items():
        full = ROOT / path
        if full.exists():
            size_kb = full.stat().st_size // 1024
            print(f"  ✅  {label:<32}  {size_kb:>8} KB")
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

    from utils.model_loader import load_shap_explainer
    shap_exp = load_shap_explainer()
    assert hasattr(shap_exp, "shap_values"), "SHAP explainer missing .shap_values()"
    sv = shap_exp.shap_values(out)
    assert sv is not None
    print(f"  ✅  shap_explainer.shap_values()  returned {type(sv).__name__}")

    model = joblib.load(str(ROOT / "models" / "loan_approval_model.joblib"))
    proba = model.predict_proba(out)
    assert proba.shape == (1, 2), f"Unexpected predict_proba shape: {proba.shape}"
    print(f"  ✅  model.predict_proba()       shape={proba.shape}  p_approve={proba[0,1]:.3f}")

    print("─" * 55)
    print("  ALL ARTEFACTS & DATASETS VALID ✅")
    print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 40000):
    print("\n" + "═" * 55)
    print(f"  AGENTIC LOAN DECISION — ARTEFACT & DATASET BUILDER ({n_samples:,} rows)")
    print("═" * 55)

    logger.info("Generating %d training samples ...", n_samples)
    records = _generate_training_data(n_samples)

    logger.info("Fitting preprocessing pipeline & exporting datasets ...")
    X, y = _build_preprocessing(records)

    logger.info("Training classification model ...")
    model = _train_model(X, y)

    feat_names = FEATURE_NAMES[:X.shape[1]]
    logger.info("Building real SHAP + LIME explainers ...")
    _build_explainers(model, X, feat_names)

    _validate()

    print("  Dataset and Model Artefacts Ready:")
    print("    • loan_approval_model.xlsx (40,000 rows x 26 columns)")
    print("    • data/processed/training_data_raw.csv (40,000 rows)")
    print("    • data/processed/X_train.csv & y_train.csv (40,000 rows)")
    print("    • models/loan_approval_model.joblib")
    print("    • models/preprocessor.joblib\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run: skipping training, validating existing artefacts only.")
        _validate()
    else:
        main(n_samples=args.samples)
