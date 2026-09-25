#!/usr/bin/env python3
# pipeline/build_artifacts.py
"""
Script to build all preprocessing artifacts and dummy training data.
This ensures consistent preprocessing between training and inference.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import joblib
from datetime import datetime, timedelta
import random

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import LoanPreprocessor, LoanApplicationSchema


def create_dummy_training_data(n_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate realistic dummy training data for loan applications.
    
    Args:
        n_samples: Number of training samples to generate
    
    Returns:
        List of dictionaries containing loan application data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic value ranges
    loan_types = ["housing", "personal", "vehicle", "gold", "business"]
    genders = ["Male", "Female", "Other"]
    interest_types = ["Fixed", "Floating"]
    ovd_types = ["Aadhaar", "PAN", "Passport", "Driving Licence", "Voter ID"]
    kyc_modes = ["Video KYC", "Offline", "eKYC", "CKYC"]
    
    # Indian PIN codes (major cities)
    pin_codes = [
        110001, 110002, 110003,  # Delhi
        400001, 400002, 400003,  # Mumbai
        560001, 560002, 560003,  # Bangalore
        600001, 600002, 600003,  # Chennai
        700001, 700002, 700003,  # Kolkata
        411001, 411002, 411003,  # Pune
        500001, 500002, 500003,  # Hyderabad
        380001, 380002, 380003   # Ahmedabad
    ]
    
    training_data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        # Generate application and sanction dates
        app_date = base_date + timedelta(days=random.randint(0, 365))
        sanction_date = app_date + timedelta(days=random.randint(3, 30))
        
        # Generate correlated features
        age = np.random.randint(21, 65)
        
        # Bureau score with some correlation to age and income
        base_bureau = 650 + age * 2 + np.random.normal(0, 50)
        bureau_score = int(np.clip(base_bureau, 300, 900))
        
        # Monthly income with some correlation to age and bureau score
        base_income = 25000 + age * 800 + (bureau_score - 650) * 50 + np.random.normal(0, 10000)
        monthly_income = max(15000, int(base_income))
        
        # Loan amount based on income and type
        loan_type = np.random.choice(loan_types)
        if loan_type == "housing":
            multiplier = np.random.uniform(30, 80)  # Higher amounts
            max_amount = 5000000
        elif loan_type == "vehicle":
            multiplier = np.random.uniform(8, 25)
            max_amount = 2000000
        elif loan_type == "business":
            multiplier = np.random.uniform(15, 40)
            max_amount = 3000000
        else:  # personal, gold
            multiplier = np.random.uniform(3, 15)
            max_amount = 1000000
        
        requested_amount = min(int(monthly_income * multiplier), max_amount)
        sanctioned_amount = int(requested_amount * np.random.uniform(0.7, 1.0))
        
        # Existing obligations (impacts FOIR)
        existing_obligations = int(monthly_income * np.random.uniform(0.05, 0.4))
        
        # FOIR calculation and capping
        processing_fee = max(500, int(requested_amount * 0.01))
        other_charges = np.random.randint(200, 2000)
        
        # Interest rates by loan type
        if loan_type == "housing":
            interest_rate = np.random.uniform(8.5, 12.0)
        elif loan_type == "vehicle":
            interest_rate = np.random.uniform(9.0, 14.0)
        elif loan_type == "personal":
            interest_rate = np.random.uniform(12.0, 24.0)
        elif loan_type == "business":
            interest_rate = np.random.uniform(10.0, 18.0)
        else:  # gold
            interest_rate = np.random.uniform(11.0, 16.0)
        
        # Tenure based on loan type
        if loan_type == "housing":
            tenure = np.random.randint(120, 300)  # 10-25 years
        elif loan_type == "vehicle":
            tenure = np.random.randint(36, 84)    # 3-7 years
        else:
            tenure = np.random.randint(12, 60)    # 1-5 years
        
        # Calculate EMI
        monthly_rate = interest_rate / (12 * 100)
        if monthly_rate > 0:
            emi = (sanctioned_amount * monthly_rate * (1 + monthly_rate)**tenure) / ((1 + monthly_rate)**tenure - 1)
        else:
            emi = sanctioned_amount / tenure
        
        total_obligations = existing_obligations + emi
        foir = (total_obligations / monthly_income) * 100
        
        # Cap FOIR to make it realistic (max 100%)
        foir = min(foir, 95.0)
        
        # APR (slightly higher than interest rate)
        apr = interest_rate + np.random.uniform(0.1, 1.5)
        
        # Property value and LTV for secured loans
        property_value = None
        ltv_ratio = None
        
        if loan_type in ["housing", "vehicle"]:
            if loan_type == "housing":
                property_value = int(sanctioned_amount / np.random.uniform(0.6, 0.9))  # LTV 60-90%
            else:  # vehicle
                property_value = int(sanctioned_amount / np.random.uniform(0.7, 0.95))  # LTV 70-95%
            
            ltv_ratio = min(sanctioned_amount / property_value, 1.0)  # Cap at 100%
        
        # Generate the application data
        application = {
            "application_id": f"APP_{i+1:06d}",
            "application_date": app_date.strftime("%Y-%m-%d"),
            "sanction_date": sanction_date.strftime("%Y-%m-%d"),
            "loan_type": loan_type,
            "age_years": age,
            "gender": np.random.choice(genders, p=[0.6, 0.35, 0.05]),
            "pin_code": np.random.choice(pin_codes),
            "pep_flag": np.random.choice([True, False], p=[0.02, 0.98]),
            "bureau_score": bureau_score,
            "monthly_income_inr": monthly_income,
            "existing_monthly_obligations_inr": existing_obligations,
            "requested_amount_inr": requested_amount,
            "sanctioned_amount_inr": sanctioned_amount,
            "tenure_months": tenure,
            "interest_type": np.random.choice(interest_types, p=[0.7, 0.3]),
            "interest_rate_annual_pct": round(interest_rate, 2),
            "processing_fee_inr": processing_fee,
            "other_charges_inr": other_charges,
            "apr_pct": round(apr, 2),
            "kfs_provided": np.random.choice([True, False], p=[0.95, 0.05]),
            "proposed_emi_inr": round(emi, 2),
            "foir_total_obligations_pct": round(foir, 2),
            "ovd_type": np.random.choice(ovd_types),
            "kyc_mode": np.random.choice(kyc_modes),
        }
        
        # Add property details for secured loans
        if property_value is not None:
            application["property_value_inr"] = property_value
            application["ltv_ratio"] = round(ltv_ratio, 3)
        
        training_data.append(application)
    
    return training_data


def build_preprocessing_artifacts():
    """Build all preprocessing artifacts needed for the system."""
    
    print("🏗️ Building preprocessing artifacts...")
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/explainer", exist_ok=True)
    
    # Generate dummy training data
    print("📊 Generating dummy training data...")
    training_data = create_dummy_training_data(n_samples=2000)
    
    # Convert to DataFrame
    df_train = pd.DataFrame(training_data)
    
    # Save raw training data
    df_train.to_csv("data/processed/training_data_raw.csv", index=False)
    print(f"✅ Saved raw training data: {len(training_data)} samples")
    
    # Initialize and fit preprocessor
    print("🔧 Fitting preprocessing pipeline...")
    preprocessor = LoanPreprocessor()
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(training_data)
    
    # Save the fitted preprocessor
    preprocessor.save("models/preprocessor.joblib")
    
    # Create feature names for the processed data
    # Note: The pipeline outputs features in a specific order after all transformations
    feature_names = [
        'age_years', 'pin_code', 'pep_flag', 'bureau_score',
        'monthly_income_inr', 'existing_monthly_obligations_inr',
        'requested_amount_inr', 'sanctioned_amount_inr', 'tenure_months',
        'interest_rate_annual_pct', 'processing_fee_inr', 'other_charges_inr',
        'apr_pct', 'kfs_provided', 'proposed_emi_inr',
        'foir_total_obligations_pct', 'property_value_inr', 'ltv_ratio',
        'time_to_sanction_days', 'application_month',
        'interest_type_encoded', 'gender_Female', 'gender_Male', 'gender_Other',
        'ovd_provided'
    ]
    
    # Save processed training data with feature names
    df_X_train = pd.DataFrame(X_train_processed, columns=feature_names)
    df_X_train.to_csv("data/processed/X_train.csv", index=False)
    print(f"✅ Saved processed training features: shape {X_train_processed.shape}")
    
    # Generate dummy target variables for completeness
    # Simulate realistic approval rates based on FOIR and bureau score
    y_approved = []
    y_default = []
    
    for _, row in df_train.iterrows():
        # Approval logic based on risk factors
        bureau_score = row['bureau_score']
        foir = row['foir_total_obligations_pct']
        pep_flag = row['pep_flag']
        
        # Calculate approval probability
        approval_prob = 0.8  # Base probability
        
        # Adjust based on bureau score
        if bureau_score < 600:
            approval_prob -= 0.4
        elif bureau_score < 700:
            approval_prob -= 0.2
        elif bureau_score > 750:
            approval_prob += 0.1
        
        # Adjust based on FOIR
        if foir > 70:
            approval_prob -= 0.5
        elif foir > 50:
            approval_prob -= 0.3
        elif foir < 30:
            approval_prob += 0.1
        
        # PEP flag reduces approval probability
        if pep_flag:
            approval_prob -= 0.3
        
        # Final approval decision
        approved = np.random.random() < max(0.05, min(0.95, approval_prob))
        y_approved.append(int(approved))
        
        # Default probability (only for approved loans)
        if approved:
            default_prob = 0.05  # Base default rate
            
            # Higher default risk for lower bureau scores
            if bureau_score < 650:
                default_prob += 0.10
            elif bureau_score < 700:
                default_prob += 0.05
            
            # Higher default risk for high FOIR
            if foir > 60:
                default_prob += 0.08
            elif foir > 45:
                default_prob += 0.03
            
            defaulted = np.random.random() < min(0.25, default_prob)
            y_default.append(int(defaulted))
        else:
            y_default.append(0)  # Non-approved loans don't default
    
    # Save target variables
    df_targets = pd.DataFrame({
        'target_approved': y_approved,
        'target_default_12m': y_default
    })
    df_targets.to_csv("data/processed/y_train.csv", index=False)
    print(f"✅ Generated target variables - Approval rate: {np.mean(y_approved):.2%}, Default rate: {np.mean(y_default):.2%}")
    
    # Create a legacy scaler.joblib for backward compatibility
    # Just save the scaling parameters as a dictionary for now
    scaling_params = {
        'type': 'simple_scaler',
        'mean': np.mean(X_train_processed, axis=0).tolist(),
        'std': np.std(X_train_processed, axis=0).tolist(),
        'feature_names': feature_names
    }
    joblib.dump(scaling_params, "models/scaler.joblib")
    print("✅ Created legacy scaler.joblib for backward compatibility")
    
    # Create placeholder explainer files
    print("🔍 Creating placeholder explainer artifacts...")
    
    # Create dummy SHAP explainer
    shap_explainer_data = {
        'type': 'placeholder',
        'feature_names': feature_names,
        'background_data': X_train_processed[:100].tolist(),
        'created_at': datetime.now().isoformat()
    }
    joblib.dump(shap_explainer_data, "models/explainer/shap_explainer.joblib")
    
    # Create dummy LIME explainer
    lime_explainer_data = {
        'type': 'placeholder',
        'feature_names': feature_names,
        'training_data': X_train_processed[:500].tolist(),
        'created_at': datetime.now().isoformat()
    }
    joblib.dump(lime_explainer_data, "models/explainer/lime_explainer.joblib")
    
    print("✅ Created placeholder explainer artifacts")
    
    return X_train_processed, feature_names


def validate_artifacts():
    """Validate that all artifacts were created correctly."""
    
    print("\n🔍 Validating artifacts...")
    
    required_files = [
        "models/preprocessor.joblib",
        "models/scaler.joblib",
        "data/processed/X_train.csv",
        "data/processed/y_train.csv",
        "data/processed/training_data_raw.csv",
        "models/explainer/shap_explainer.joblib",
        "models/explainer/lime_explainer.joblib"
    ]
    
    for filepath in required_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filepath} ({size:,} bytes)")
        else:
            print(f"❌ Missing: {filepath}")
            return False
    
    # Test loading the preprocessor
    try:
        preprocessor = LoanPreprocessor.load("models/preprocessor.joblib")
        print("✅ Preprocessor loads successfully")
        
        # Test with a sample application
        sample_app = {
            "age_years": 35,
            "gender": "Male",
            "bureau_score": 720,
            "monthly_income_inr": 75000,
            "requested_amount_inr": 500000,
            "tenure_months": 48,
            "interest_rate_annual_pct": 12.5,
            "foir_total_obligations_pct": 35.0,
            "interest_type": "Fixed"
        }
        
        processed = preprocessor.transform(sample_app)
        expected_shape = (1, 25)  # Should match the feature count
        
        if processed.shape == expected_shape:
            print(f"✅ Sample transformation successful: {processed.shape}")
        else:
            print(f"❌ Shape mismatch: expected {expected_shape}, got {processed.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessor test failed: {e}")
        return False
    
    # Load and validate training data
    try:
        X_train = pd.read_csv("data/processed/X_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")
        
        print(f"✅ Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        if X_train.shape[0] != y_train.shape[0]:
            print("❌ Mismatch between X and y sample counts")
            return False
            
    except Exception as e:
        print(f"❌ Training data validation failed: {e}")
        return False
    
    print("🎉 All artifacts validated successfully!")
    return True


def main():
    """Main function to build and validate all artifacts."""
    
    print("=" * 60)
    print("🚀 AGENTIC LOAN DECISION SYSTEM - ARTIFACT BUILDER")
    print("=" * 60)
    
    try:
        # Build artifacts
        X_train, feature_names = build_preprocessing_artifacts()
        
        print(f"\n📈 Summary:")
        print(f"   • Training samples: {X_train.shape[0]:,}")
        print(f"   • Features: {X_train.shape[1]}")
        print(f"   • Feature names: {len(feature_names)}")
        
        # Validate everything
        if validate_artifacts():
            print("\n🎯 SUCCESS! All preprocessing artifacts built and validated.")
            print("\nNext steps:")
            print("1. Run the system: python agents/orchestrator.py")
            print("2. Test API: python api/app.py (after Phase 3)")
            print("3. Launch UI: streamlit run frontend/streamlit_app.py (after Phase 4)")
        else:
            print("\n❌ FAILED! Some artifacts are missing or invalid.")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())