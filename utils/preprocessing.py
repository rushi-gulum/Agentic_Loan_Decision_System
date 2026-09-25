# utils/preprocessing.py
"""
Unified preprocessing pipeline for the Agentic Loan Decision System.
Ensures consistent feature engineering between training and inference.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel, Field
from datetime import datetime
import warnings

# Try to import sklearn components, fall back to manual implementation
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: scikit-learn not available, using simplified preprocessing")
    SKLEARN_AVAILABLE = False
    
    # Create dummy base classes for compatibility
    class BaseEstimator:
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X
    
    class TransformerMixin:
        pass

warnings.filterwarnings("ignore", category=UserWarning)


class LoanApplicationSchema(BaseModel):
    """Pydantic schema for loan application input validation."""
    
    class Config:
        validate_assignment = True
        extra = 'allow'  # Allow extra fields
        
    # Required fields
    age_years: int = Field(ge=18, le=100, description="Applicant age in years")
    bureau_score: int = Field(ge=300, le=900, description="Credit bureau score")
    monthly_income_inr: float = Field(gt=0, description="Monthly income in INR")
    requested_amount_inr: float = Field(gt=0, description="Requested loan amount in INR")
    tenure_months: int = Field(ge=6, le=360, description="Loan tenure in months")
    interest_rate_annual_pct: float = Field(ge=0, le=50, description="Annual interest rate percentage")
    foir_total_obligations_pct: float = Field(ge=0, le=150, description="Fixed Obligation to Income Ratio percentage")  # Increased limit
    
    # Optional fields with defaults
    pin_code: Optional[int] = Field(default=110001, ge=100000, le=999999, description="6-digit PIN code")
    pep_flag: bool = Field(default=False, description="Politically Exposed Person flag")
    kfs_provided: bool = Field(default=True, description="Key Fact Statement provided flag")
    existing_monthly_obligations_inr: float = Field(default=0, ge=0, description="Existing monthly obligations in INR")
    sanctioned_amount_inr: Optional[float] = Field(default=None, ge=0, description="Sanctioned loan amount in INR")
    processing_fee_inr: float = Field(default=0, ge=0, description="Processing fee in INR")
    other_charges_inr: float = Field(default=0, ge=0, description="Other charges in INR")
    apr_pct: Optional[float] = Field(default=None, ge=0, le=50, description="Annual Percentage Rate")
    proposed_emi_inr: Optional[float] = Field(default=None, ge=0, description="Proposed EMI in INR")
    property_value_inr: Optional[float] = Field(default=None, ge=0, description="Property value in INR")
    ltv_ratio: Optional[float] = Field(default=None, ge=0, le=1.5, description="Loan to Value ratio")  # Slightly higher limit
    
    # Categorical fields
    gender: str = Field(default="Male", pattern="^(Male|Female|Other)$", description="Applicant gender")
    interest_type: str = Field(default="Fixed", pattern="^(Fixed|Floating)$", description="Interest rate type")
    loan_type: str = Field(default="personal", description="Type of loan")
    ovd_type: Optional[str] = Field(default="Aadhaar", description="OVD document type")
    kyc_mode: Optional[str] = Field(default="Video KYC", description="KYC verification mode")
    
    # Date fields (optional for inference)
    application_date: Optional[str] = Field(default=None, description="Application date in YYYY-MM-DD format")
    sanction_date: Optional[str] = Field(default=None, description="Sanction date in YYYY-MM-DD format")


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for date-based feature engineering."""
    
    def __init__(self):
        self.default_sanction_days = 7  # Default processing time
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Handle date fields if present
        if 'application_date' in X.columns and 'sanction_date' in X.columns:
            # Convert to datetime
            X['application_date'] = pd.to_datetime(X['application_date'], errors='coerce')
            X['sanction_date'] = pd.to_datetime(X['sanction_date'], errors='coerce')
            
            # Calculate time to sanction
            X['time_to_sanction_days'] = (X['sanction_date'] - X['application_date']).dt.days
            
            # Extract application month
            X['application_month'] = X['application_date'].dt.month
        else:
            # Use defaults for inference
            X['time_to_sanction_days'] = self.default_sanction_days
            X['application_month'] = datetime.now().month
        
        # Drop original date columns
        X = X.drop(columns=['application_date', 'sanction_date'], errors='ignore')
        
        return X


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for boolean fields."""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        boolean_cols = ['pep_flag', 'kfs_provided']
        
        for col in boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        return X


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for derived features."""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Create OVD provided flag
        if 'ovd_type' in X.columns:
            X['ovd_provided'] = X['ovd_type'].notna().astype(int)
        else:
            X['ovd_provided'] = 1  # Default to provided
        
        # Handle sanctioned amount (use requested if not provided)
        if 'sanctioned_amount_inr' in X.columns:
            X['sanctioned_amount_inr'] = X['sanctioned_amount_inr'].fillna(X['requested_amount_inr'])
        else:
            X['sanctioned_amount_inr'] = X['requested_amount_inr']
        
        # Calculate APR if not provided
        if 'apr_pct' not in X.columns or X['apr_pct'].isna().any():
            X['apr_pct'] = X['apr_pct'].fillna(X['interest_rate_annual_pct'])
        
        # Calculate proposed EMI if not provided (simple approximation)
        if 'proposed_emi_inr' not in X.columns or X['proposed_emi_inr'].isna().any():
            # Simple EMI calculation: P * r * (1+r)^n / ((1+r)^n - 1)
            P = X['sanctioned_amount_inr']
            r = X['interest_rate_annual_pct'] / (12 * 100)  # Monthly interest rate
            n = X['tenure_months']
            
            # Avoid division by zero
            r = r.replace(0, 0.001)
            emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
            X['proposed_emi_inr'] = X['proposed_emi_inr'].fillna(emi)
        
        # Handle property value and LTV for unsecured loans
        if 'property_value_inr' not in X.columns:
            X['property_value_inr'] = 0
        if 'ltv_ratio' not in X.columns:
            X['ltv_ratio'] = 0
        
        # Fill remaining NaN values
        X['property_value_inr'] = X['property_value_inr'].fillna(0)
        X['ltv_ratio'] = X['ltv_ratio'].fillna(0)
        
        return X


class FeatureOrderTransformer(BaseEstimator, TransformerMixin):
    """Ensures consistent feature ordering for model compatibility."""
    
    def __init__(self):
        # Fixed feature order matching the trained model
        self.feature_order = [
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
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Add missing columns with default values
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training
        X = X[self.feature_order]
        
        return X


def create_preprocessing_pipeline() -> Union['Pipeline', 'SimplifiedPipeline']:
    """
    Create a comprehensive preprocessing pipeline.
    Uses sklearn Pipeline if available, otherwise falls back to simplified version.
    
    Returns:
        Pipeline or SimplifiedPipeline: A fitted pipeline for preprocessing loan applications.
    """
    
    if SKLEARN_AVAILABLE:
        return create_sklearn_pipeline()
    else:
        return SimplifiedPipeline()


def create_sklearn_pipeline() -> 'Pipeline':
    """Create sklearn-based preprocessing pipeline (when sklearn is available)."""
    
    # Define column groups
    numeric_features = [
        'age_years', 'pin_code', 'bureau_score', 'monthly_income_inr',
        'existing_monthly_obligations_inr', 'requested_amount_inr', 'sanctioned_amount_inr',
        'tenure_months', 'interest_rate_annual_pct', 'processing_fee_inr',
        'other_charges_inr', 'apr_pct', 'proposed_emi_inr', 'foir_total_obligations_pct',
        'property_value_inr', 'ltv_ratio', 'time_to_sanction_days', 'application_month'
    ]
    
    binary_features = ['pep_flag', 'kfs_provided', 'ovd_provided']
    
    categorical_features = ['gender']  # Will be one-hot encoded
    
    ordinal_features = ['interest_type']  # Will be label encoded
    
    # Create transformers
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = OneHotEncoder(
        drop='first', 
        sparse_output=False, 
        handle_unknown='ignore',
        dtype=np.int32
    )
    
    # Custom function for interest type encoding
    def encode_interest_type(X):
        X = X.copy()
        X['interest_type_encoded'] = X['interest_type'].map({'Fixed': 0, 'Floating': 1}).fillna(0)
        return X[['interest_type_encoded']]
    
    interest_type_transformer = FunctionTransformer(
        func=encode_interest_type,
        validate=False
    )
    
    # Column transformer to handle different feature types
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', interest_type_transformer, ['interest_type']),
            ('bin', 'passthrough', binary_features)  # Binary features don't need scaling
        ],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1
    )
    
    # Complete preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('date_features', DateFeatureTransformer()),
        ('boolean_conversion', BooleanTransformer()),
        ('derived_features', DerivedFeatureTransformer()),
        ('column_transform', column_transformer),
        ('feature_order', FeatureOrderTransformer())
    ])
    
    return preprocessing_pipeline


class SimplifiedPipeline:
    """Simplified preprocessing pipeline that doesn't depend on sklearn."""
    
    def __init__(self):
        self.feature_order = [
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
        self.numeric_features = [
            'age_years', 'pin_code', 'bureau_score', 'monthly_income_inr',
            'existing_monthly_obligations_inr', 'requested_amount_inr', 'sanctioned_amount_inr',
            'tenure_months', 'interest_rate_annual_pct', 'processing_fee_inr',
            'other_charges_inr', 'apr_pct', 'proposed_emi_inr', 'foir_total_obligations_pct',
            'property_value_inr', 'ltv_ratio', 'time_to_sanction_days', 'application_month'
        ]
        self.scaler_params = {}  # Will store mean and std for each feature
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the simplified pipeline by computing scaling parameters."""
        # Apply all transformations first
        X_transformed = self._apply_transformations(X)
        
        # Compute scaling parameters for numeric features
        for col in self.numeric_features:
            if col in X_transformed.columns:
                self.scaler_params[col] = {
                    'mean': X_transformed[col].mean(),
                    'std': X_transformed[col].std()
                }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input data using the fitted pipeline."""
        # Apply all feature transformations
        X_transformed = self._apply_transformations(X)
        
        # Apply scaling to numeric features
        for col in self.numeric_features:
            if col in X_transformed.columns and col in self.scaler_params:
                mean = self.scaler_params[col]['mean']
                std = self.scaler_params[col]['std']
                if std > 0:  # Avoid division by zero
                    X_transformed[col] = (X_transformed[col] - mean) / std
        
        # Ensure feature order and return as numpy array
        X_ordered = X_transformed[self.feature_order]
        return X_ordered.values
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        X = X.copy()
        
        # Handle date features
        if 'application_date' in X.columns and 'sanction_date' in X.columns:
            X['application_date'] = pd.to_datetime(X['application_date'], errors='coerce')
            X['sanction_date'] = pd.to_datetime(X['sanction_date'], errors='coerce')
            X['time_to_sanction_days'] = (X['sanction_date'] - X['application_date']).dt.days
            X['application_month'] = X['application_date'].dt.month
        else:
            X['time_to_sanction_days'] = 7  # Default processing time
            X['application_month'] = datetime.now().month
        
        X = X.drop(columns=['application_date', 'sanction_date'], errors='ignore')
        
        # Convert boolean columns
        boolean_cols = ['pep_flag', 'kfs_provided']
        for col in boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        # Create derived features
        if 'ovd_type' in X.columns:
            X['ovd_provided'] = X['ovd_type'].notna().astype(int)
        else:
            X['ovd_provided'] = 1
        
        # Handle sanctioned amount
        if 'sanctioned_amount_inr' in X.columns:
            X['sanctioned_amount_inr'] = X['sanctioned_amount_inr'].fillna(X['requested_amount_inr'])
        else:
            X['sanctioned_amount_inr'] = X['requested_amount_inr']
        
        # Calculate APR if not provided
        if 'apr_pct' not in X.columns or X['apr_pct'].isna().any():
            X['apr_pct'] = X['apr_pct'].fillna(X['interest_rate_annual_pct'])
        
        # Calculate EMI if not provided
        if 'proposed_emi_inr' not in X.columns or X['proposed_emi_inr'].isna().any():
            P = X['sanctioned_amount_inr']
            r = X['interest_rate_annual_pct'] / (12 * 100)
            n = X['tenure_months']
            r = r.replace(0, 0.001)
            emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
            X['proposed_emi_inr'] = X['proposed_emi_inr'].fillna(emi)
        
        # Handle property value and LTV
        if 'property_value_inr' not in X.columns:
            X['property_value_inr'] = 0
        if 'ltv_ratio' not in X.columns:
            X['ltv_ratio'] = 0
        
        X['property_value_inr'] = X['property_value_inr'].fillna(0)
        X['ltv_ratio'] = X['ltv_ratio'].fillna(0)
        
        # Encode interest type
        X['interest_type_encoded'] = X['interest_type'].map({'Fixed': 0, 'Floating': 1}).fillna(0)
        
        # One-hot encode gender
        for gender_type in ['Female', 'Male', 'Other']:
            X[f'gender_{gender_type}'] = 0
        
        if 'gender' in X.columns:
            for idx, gender_value in enumerate(X['gender']):
                if f'gender_{gender_value}' in X.columns:
                    X.loc[idx, f'gender_{gender_value}'] = 1
        
        # Drop unnecessary columns
        drop_cols = ['gender', 'interest_type', 'kyc_mode', 'ovd_type', 'loan_type', 'state']
        X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')
        
        # Add missing columns with default values
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = 0
        
        return X


class LoanPreprocessor:
    """Main preprocessing class for loan applications."""
    
    def __init__(self, pipeline_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            pipeline_path: Path to a saved preprocessing pipeline. If None, creates new pipeline.
        """
        if pipeline_path and joblib.os.path.exists(pipeline_path):
            try:
                self.pipeline = joblib.load(pipeline_path)
                print(f"✅ Loaded preprocessing pipeline from {pipeline_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load pipeline ({e}), creating new one")
                self.pipeline = create_preprocessing_pipeline()
        else:
            self.pipeline = create_preprocessing_pipeline()
        
        self.is_fitted = False
    
    def fit(self, X: Union[pd.DataFrame, List[Dict]], y=None) -> 'LoanPreprocessor':
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Training data as DataFrame or list of dictionaries
            y: Target values (ignored)
        
        Returns:
            self: Fitted preprocessor instance
        """
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Validate schema for first row (if Pydantic is working)
        if len(X) > 0:
            sample_row = X.iloc[0].to_dict()
            try:
                LoanApplicationSchema(**sample_row)
            except Exception as e:
                print(f"⚠️ Warning: Schema validation failed: {e}")
        
        self.pipeline.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[Dict[str, Any], List[Dict], pd.DataFrame]) -> np.ndarray:
        """
        Transform input data using the fitted pipeline.
        
        Args:
            X: Input data as dictionary, list of dictionaries, or DataFrame
        
        Returns:
            np.ndarray: Preprocessed data ready for model prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Convert input to DataFrame
        if isinstance(X, dict):
            # Single application
            try:
                validated_data = LoanApplicationSchema(**X)
                df = pd.DataFrame([validated_data.model_dump()])
            except Exception as e:
                print(f"⚠️ Schema validation failed: {e}, proceeding without validation")
                df = pd.DataFrame([X])
        elif isinstance(X, list):
            # Multiple applications
            try:
                validated_data = [LoanApplicationSchema(**item) for item in X]
                df = pd.DataFrame([item.model_dump() for item in validated_data])
            except Exception as e:
                print(f"⚠️ Schema validation failed: {e}, proceeding without validation")
                df = pd.DataFrame(X)
        else:
            # Already a DataFrame
            df = X.copy()
        
        # Apply preprocessing pipeline
        processed_data = self.pipeline.transform(df)
        
        return processed_data
    
    def fit_transform(self, X: Union[pd.DataFrame, List[Dict]], y=None) -> np.ndarray:
        """
        Fit the pipeline and transform the data in one step.
        
        Args:
            X: Training data
            y: Target values (ignored)
        
        Returns:
            np.ndarray: Preprocessed training data
        """
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessing pipeline to disk.
        
        Args:
            filepath: Path where to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor. Call fit() first.")
        
        joblib.dump(self.pipeline, filepath)
        print(f"✅ Preprocessing pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LoanPreprocessor':
        """
        Load a preprocessing pipeline from disk.
        
        Args:
            filepath: Path to the saved pipeline
        
        Returns:
            LoanPreprocessor: Loaded preprocessor instance
        """
        instance = cls(filepath)
        instance.is_fitted = True
        return instance


def preprocess_single_application(applicant_data: Dict[str, Any], 
                                pipeline_path: str = "models/preprocessor.joblib") -> np.ndarray:
    """
    Convenience function to preprocess a single loan application.
    
    Args:
        applicant_data: Dictionary containing applicant information
        pipeline_path: Path to the saved preprocessing pipeline
    
    Returns:
        np.ndarray: Preprocessed data ready for model prediction
    """
    preprocessor = LoanPreprocessor.load(pipeline_path)
    return preprocessor.transform(applicant_data)


if __name__ == "__main__":
    # Example usage and testing
    sample_application = {
        "age_years": 32,
        "gender": "Male",
        "pin_code": 110001,
        "pep_flag": False,
        "bureau_score": 720,
        "monthly_income_inr": 55000,
        "existing_monthly_obligations_inr": 12000,
        "requested_amount_inr": 400000,
        "tenure_months": 36,
        "interest_rate_annual_pct": 10.5,
        "processing_fee_inr": 2000.0,
        "other_charges_inr": 500.0,
        "foir_total_obligations_pct": 45.0,
        "interest_type": "Fixed",
        "kfs_provided": True,
        "loan_type": "personal",
        "ovd_type": "Aadhaar",
        "application_date": "2024-01-15",
        "sanction_date": "2024-01-22"
    }
    
    # Test preprocessing
    preprocessor = LoanPreprocessor()
    
    # Generate some dummy training data for fitting
    training_data = [sample_application.copy() for _ in range(10)]
    for i, app in enumerate(training_data):
        app['age_years'] += i
        app['bureau_score'] += i * 10
    
    # Fit and transform
    processed_data = preprocessor.fit_transform(training_data)
    print(f"✅ Processed training data shape: {processed_data.shape}")
    
    # Test single application preprocessing
    single_processed = preprocessor.transform(sample_application)
    print(f"✅ Processed single application shape: {single_processed.shape}")
    print(f"✅ First few features: {single_processed[0][:5]}")