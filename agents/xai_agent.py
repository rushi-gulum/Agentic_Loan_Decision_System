

from openai import OpenAI
import shap
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import json
from lime.lime_tabular import LimeTabularExplainer
import os
from crewai import LLM


def preprocess(raw_data_row: dict) -> np.ndarray:
    """
    Preprocesses a single raw loan application data row for model prediction.
    Uses the unified preprocessing pipeline for consistency.

    Args:
        raw_data_row (dict): A dictionary representing a single new loan application,
                             containing all original features.

    Returns:
        np.ndarray: A scaled numpy array of the preprocessed single data row,
                    ready for input into a trained machine learning model.
    """
    try:
        # Import the unified preprocessor
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from utils.preprocessing import preprocess_single_application
        
        # Use the unified preprocessing pipeline
        processed_data = preprocess_single_application(raw_data_row, "models/preprocessor.joblib")
        return processed_data
        
    except Exception as e:
        print(f"⚠️ Warning: Unified preprocessor failed ({e}), falling back to legacy method")
        
        # Fallback to legacy preprocessing for backward compatibility
        df_single = pd.DataFrame([raw_data_row])

        # Drop irrelevant columns
        initial_irrelevant_columns = [
            'application_id', 'applicant_name', 'email', 'mobile', 'pan',
            'aadhaar_masked', 'borrower_data_consent_timestamp',
            'target_approved', 'target_default_12m', 'target'
        ]
        df_single = df_single.drop(columns=[col for col in initial_irrelevant_columns if col in df_single.columns], errors='ignore')

        # Handle dates
        if 'application_date' in df_single.columns and 'sanction_date' in df_single.columns:
            df_single['application_date'] = pd.to_datetime(df_single['application_date'], errors='coerce')
            df_single['sanction_date'] = pd.to_datetime(df_single['sanction_date'], errors='coerce')
            df_single['time_to_sanction_days'] = (df_single['sanction_date'] - df_single['application_date']).dt.days
            df_single['application_month'] = df_single['application_date'].dt.month
        else:
            df_single['time_to_sanction_days'] = 7  # Default
            df_single['application_month'] = pd.Timestamp.now().month

        df_single = df_single.drop(columns=['application_date', 'sanction_date', 'state'], errors='ignore')

        # Convert boolean columns
        df_single['pep_flag'] = df_single['pep_flag'].astype(int)
        df_single['kfs_provided'] = df_single['kfs_provided'].astype(int)

        # Encode interest_type
        df_single['interest_type_encoded'] = df_single['interest_type'].map({'Fixed': 0, 'Floating': 1}).fillna(0)
        df_single = df_single.drop(columns=['interest_type'], errors='ignore')

        # One-hot encode gender
        for gender_type in ['Female', 'Male', 'Other']:
            df_single[f'gender_{gender_type}'] = 0
        if 'gender' in df_single.columns:
            gender_value = df_single['gender'].iloc[0]
            if f'gender_{gender_value}' in df_single.columns:
                df_single[f'gender_{gender_value}'] = 1
            df_single = df_single.drop(columns=['gender'])

        # Create ovd_provided feature
        df_single['ovd_provided'] = df_single['ovd_type'].notna().astype(int) if 'ovd_type' in df_single.columns else 1

        # Drop final columns
        final_cols_to_drop = ['kyc_mode', 'ovd_type', 'loan_type']
        df_single = df_single.drop(columns=[col for col in final_cols_to_drop if col in df_single.columns], errors='ignore')

        # Ensure feature order
        training_features_order = ['age_years', 'pin_code', 'pep_flag', 'bureau_score',
           'monthly_income_inr', 'existing_monthly_obligations_inr',
           'requested_amount_inr', 'sanctioned_amount_inr', 'tenure_months',
           'interest_rate_annual_pct', 'processing_fee_inr', 'other_charges_inr',
           'apr_pct', 'kfs_provided', 'proposed_emi_inr',
           'foir_total_obligations_pct', 'property_value_inr', 'ltv_ratio',
            'time_to_sanction_days', 'application_month',
           'interest_type_encoded', 'gender_Female', 'gender_Male', 'gender_Other',
           'ovd_provided']
        
        for col in training_features_order:
            if col not in df_single.columns:
                df_single[col] = 0
        df_single = df_single[training_features_order]

        # Apply scaling
        try:
            scaler_data = joblib.load('models/scaler.joblib')
            if isinstance(scaler_data, dict) and scaler_data.get('type') == 'simple_scaler':
                # New format: dictionary with mean and std
                mean = np.array(scaler_data['mean'])
                std = np.array(scaler_data['std'])
                std[std == 0] = 1.0  # Avoid division by zero
                scaled_array = (df_single.values - mean) / std
            else:
                # Old format: sklearn scaler object
                scaled_array = scaler_data.transform(df_single)
        except Exception as e:
            print(f"⚠️ Warning: Could not apply scaling ({e})")
            scaled_array = df_single.values
            
        return scaled_array


def model_predict(features_array: np.array) -> np.ndarray:
    """
    Dummy function to represent model training.
    In practice, this would involve fitting a machine learning model.

    Args:
        features_array (np.ndarray): Preprocessed scaled feature array for training.

    Returns:
        np.ndarray: Dummy return value.
    """

    # Load the saved Keras model
    loaded_model = load_model('models/loan_approval_model.h5')
    print("Keras model 'loan_approval_model.h5' loaded successfully.")
    
    prediction_probability = loaded_model.predict(features_array)[0][0]

    return prediction_probability


MODEL_PATH = "models/loan_approval_model.h5"
SHAP_EXPLAINER_PATH = "models/explainer/shap_explainer.joblib"
LIME_EXPLAINER_PATH = "models/explainer/lime_explainer.joblib"

# ---------------------------------------------------------------------
# 🔍 LOADERS
# ---------------------------------------------------------------------
def _load_model_and_explainers():
    """Load model, SHAP, and LIME explainers from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # 1️⃣ Load Keras model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # 2️⃣ Load X_train for background + feature reference
    try:
        x_train = pd.read_csv('data/processed/X_train.csv')
        feature_names = x_train.columns.tolist()
    except Exception as e:
        raise FileNotFoundError("❌ Failed to load training data for explainers.") from e

    # 3️⃣ Sample background for SHAP
    background_data = x_train.sample(n=min(100, len(x_train)), random_state=42)
    background_array = background_data.values.astype(np.float32)

    # 4️⃣ Initialize SHAP safely
    shap_explainer = None
    try:
        shap_explainer = shap.Explainer(model, background_array, feature_names=feature_names)
        print("✅ SHAP universal Explainer initialized successfully.")
    except Exception as e:
        print(f"⚠️ shap.Explainer failed ({e}), falling back to KernelExplainer.")
        def predict_fn(x):
            preds = model.predict(x)
            if preds.ndim == 1 or preds.shape[1] == 1:
                preds = np.hstack([1 - preds, preds])
            return preds
        shap_explainer = shap.KernelExplainer(predict_fn, background_array[:50])
        shap_explainer.feature_names = feature_names
        print("✅ SHAP KernelExplainer fallback initialized successfully.")

    # 5️⃣ Initialize LIME
    lime_explainer = LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=feature_names,
        class_names=["Rejected", "Approved"],
        mode="classification",
        discretize_continuous=True
    )
    print("✅ LIME TabularExplainer initialized successfully.")

    return model, shap_explainer, lime_explainer

def explain_prediction(
    features_array: np.ndarray,
    applicant_data: dict,
    compliance_data: dict,
    risk_data: dict,
    model_name: str = "Credit Risk Classifier",
) -> dict:
    """
    Generate two-tiered explanations for a loan decision using SHAP, LIME, and LLM.

    Args:
        features_array (np.ndarray): Scaled applicant features.
        applicant_data (dict): Raw applicant input.
        compliance_data (dict): ComplianceAgent outputs.
        risk_data (dict): RiskAgent outputs.
        model_name (str): Model used for prediction.

    Returns:
        dict: {
          "user_explanation": str,
          "regulator_explanation": str,
          "raw_data": {...}
        }
    """

    # Load model + explainers
    model, shap_explainer, lime_explainer = _load_model_and_explainers()

    # Run prediction
    prediction = model.predict(features_array)[0]
    pred_proba = np.hstack([1 - prediction, prediction])
    decision = "approved" if prediction == 1 else "rejected"

    # -----------------------------------------------------------------
    # 🟦 SHAP Explanation
    # -----------------------------------------------------------------
    shap_exp = shap_explainer.shap_values(features_array)
    shap_values = shap_exp[1] if isinstance(shap_exp, list) else shap_exp
    feature_names = shap_explainer.feature_names
    top_shap_features = feature_names[:10]
    # -----------------------------------------------------------------
    # 🟩 LIME Explanation
    # -----------------------------------------------------------------
    def predict_fn(x):
        preds = model.predict(x)
        if preds.ndim == 1 or preds.shape[1] == 1:
            preds = np.hstack([1 - preds, preds])
        return preds

    lime_exp = lime_explainer.explain_instance(
        features_array[0],
        predict_fn,
        num_features=25
    )

    lime_explanation = lime_exp.as_list()

    # -----------------------------------------------------------------
    # 🧾 Combine Raw Explanation Data
    # -----------------------------------------------------------------
    raw_explanation_data = {
        "decision": decision,
        "prediction_probability": {
            "approved": float(pred_proba[1]),
            "rejected": float(pred_proba[0]),
        },
        "model_used": model_name,
        "applicant_data": applicant_data,
        "compliance_data": compliance_data,
        "risk_data": risk_data,
        "explainability": {
            "top_shap_features": top_shap_features,
            "lime_explanation": lime_explanation,
        },
    }

    # -----------------------------------------------------------------
    # 🤖 LLM Summarization (Readable Reports)
    # -----------------------------------------------------------------
    llm = LLM(
    model="openai/gpt-4o",
    temperature=0.2,
    max_tokens=6000,
    base_url="https://api.openai.com/v1",
)

    # User Explanation Prompt
    user_prompt = f"""
    You are a loan officer. Based on this decision data:
    Decision: {decision}
    Model: {model_name}
    Top SHAP features: {top_shap_features}
    LIME explanation: {lime_explanation}

    Write a short, easy-to-understand paragraph (3–5 sentences)
    explaining why the loan was {decision}. Use plain language suitable
    for the applicant (non-technical). Mention positive and negative factors.
    """
    client = OpenAI()
    user_explanation = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        
        temperature=0.2,
    ).choices[0].message.content

    # Regulator Explanation Prompt
    regulator_prompt = f"""
    You are a financial auditor preparing a compliance-friendly report.
    Using the following data, produce a detailed, structured summary.

    Decision: {decision}
    Model Used: {model_name}
    Prediction Probabilities: {pred_proba.tolist()}
    Applicant Data: {json.dumps(applicant_data, indent=2)}
    Compliance Data: {json.dumps(compliance_data, indent=2)}
    Risk Data: {json.dumps(risk_data, indent=2)}
    Top SHAP Features: {top_shap_features}
    LIME Explanation: {lime_explanation}

    Generate a 2-part report:
    1. Executive Summary (2–3 short paragraphs)
    2. Technical Appendix (include feature impacts, risk assessment, and compliance checks)

    The tone should be professional and clear, suitable for regulatory review.
    """

    regulator_explanation = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": regulator_prompt}
        ],
        temperature=0.2,
    ).choices[0].message.content
    # ------------------------------------------------------
    # ✅ Final Output
    # -----------------------------------------------------------------
    return {
        "user_explanation": user_explanation,
        "regulator_explanation": regulator_explanation,
        "raw_data": raw_explanation_data,
    }

if __name__ == "__main__":
    sample_new_application_data = {
        'application_id': 'app_987654321',
        'application_date': '2023-05-10',
        'sanction_date': '2023-05-18',
        'loan_type': 'Personal Loan',
        'applicant_name': 'Jane Doe',
        'age_years': 42,
        'gender': 'Female',
        'state': 'Delhi',
        'pin_code': 110001,
        'email': 'jane.doe@example.com',
        'mobile': 9988776655,
        'pan': 'FGHIJ5678K',
        'aadhaar_masked': 'XXXX XXXX 5678',
        'kyc_mode': 'Offline',
        'ovd_type': 'Driving Licence',
        'pep_flag': False,
        'bureau_score': 710,
        'monthly_income_inr': 90000,
        'existing_monthly_obligations_inr': 25000,
        'requested_amount_inr': 2500000,
        'sanctioned_amount_inr': 2300000,
        'tenure_months': 180,
        'interest_type': 'Floating',
        'interest_rate_annual_pct': 9.8,
        'processing_fee_inr': 25000.0,
        'other_charges_inr': 1000.0,
        'apr_pct': 10.1,
        'kfs_provided': True,
        'borrower_data_consent_timestamp': '2023-05-10 09:30:00',
        'proposed_emi_inr': 28000.0,
        'foir_total_obligations_pct': 0.45,
        'property_value_inr': 3000000,
        'ltv_ratio': 0.75,
        'target_approved': 1, # These targets would not be available for actual new data
        'target_default_12m': 0,
    }

    # Preprocess the sample data
    preprocessed_sample = preprocess(sample_new_application_data)
    applicant_data = {
        "loan_type": "housing loan",
        "bureau_score": 720,
        "monthly_income_inr": 55000,
    }
    compliance_data = {"rbi_rules_followed": True, "violations": []}
    risk_data = {"risk_score": 0.23, "risk_category": "Low"}

    output = explain_prediction(preprocessed_sample, applicant_data, compliance_data, risk_data)
    print(output)