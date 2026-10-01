"""
tests/test_pipeline.py
=======================
Tests for the 40,000 dataset, preprocessing, and inference pipeline.
"""

import pytest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

class TestDatasetIntegrity:
    """Test suite for checking the 40,000 applicant dataset."""

    def test_training_raw_shape(self):
        csv_path = ROOT / "data" / "processed" / "training_data_raw.csv"
        assert csv_path.exists(), "training_data_raw.csv does not exist"
        df = pd.read_csv(csv_path)
        assert len(df) == 40000, f"Expected 40,000 rows, got {len(df)}"
        assert "target_approved" in df.columns

    def test_feature_matrix_shape(self):
        x_path = ROOT / "data" / "processed" / "X_train.csv"
        y_path = ROOT / "data" / "processed" / "y_train.csv"
        assert x_path.exists() and y_path.exists()
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        assert len(X) == 40000
        assert len(y) == 40000
        assert X.shape[1] == 25

    def test_excel_dataset_shape(self):
        excel_path = ROOT / "data" / "processed" / "loan_approval_model.xlsx"
        if not excel_path.exists():
            excel_path = ROOT / "loan_approval_model.xlsx"
        assert excel_path.exists(), "Excel dataset does not exist"
        df = pd.read_excel(excel_path, nrows=10)
        assert df.shape[1] in (25, 26)

class TestModelInference:
    """Test suite for verifying the trained model on 40k data."""

    def test_model_loaded(self):
        model_path = ROOT / "models" / "loan_approval_model.joblib"
        assert model_path.exists()
        model = joblib.load(str(model_path))
        assert hasattr(model, "predict_proba")

    def test_preprocessor_transform(self):
        from utils.preprocessing import LoanPreprocessor
        preprocessor = LoanPreprocessor.load(str(ROOT / "models" / "preprocessor.joblib"))
        sample = {
            "age_years": 32,
            "bureau_score": 750,
            "monthly_income_inr": 80000,
            "requested_amount_inr": 400000,
            "tenure_months": 36,
            "interest_rate_annual_pct": 11.5,
            "foir_total_obligations_pct": 32.0,
            "loan_type": "personal",
            "gender": "Male",
            "interest_type": "Fixed",
            "pin_code": 400001,
            "pep_flag": False,
            "kfs_provided": True
        }
        out = preprocessor.transform(sample)
        assert out.shape == (1, 25)

    def test_prediction_probabilities(self):
        from utils.preprocessing import LoanPreprocessor
        preprocessor = LoanPreprocessor.load(str(ROOT / "models" / "preprocessor.joblib"))
        model = joblib.load(str(ROOT / "models" / "loan_approval_model.joblib"))

        sample = {
            "age_years": 32,
            "bureau_score": 750,
            "monthly_income_inr": 80000,
            "existing_monthly_obligations_inr": 15000,
            "requested_amount_inr": 400000,
            "sanctioned_amount_inr": 400000,
            "tenure_months": 36,
            "interest_rate_annual_pct": 11.5,
            "processing_fee_inr": 4000,
            "other_charges_inr": 500,
            "apr_pct": 12.0,
            "proposed_emi_inr": 13000,
            "foir_total_obligations_pct": 35.0,
            "loan_type": "personal",
            "gender": "Male",
            "interest_type": "Fixed",
            "pin_code": 400001,
            "pep_flag": False,
            "kfs_provided": True,
            "property_value_inr": 0,
            "ltv_ratio": 0,
            "ovd_type": "Aadhaar",
            "kyc_mode": "eKYC",
            "application_date": "2024-01-01",
            "sanction_date": "2024-01-08"
        }
        features = preprocessor.transform(sample)
        proba = model.predict_proba(features)
        assert 0.0 <= proba[0, 1] <= 1.0
