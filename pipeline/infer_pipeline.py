# pipeline/infer_pipeline.py
"""
Unified inference pipeline for loan decision making.
Uses the standardized preprocessing pipeline for consistency.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import LoanPreprocessor, preprocess_single_application


class LoanInferencePipeline:
    """
    Complete inference pipeline for loan applications.
    Handles preprocessing, model prediction, and post-processing.
    """
    
    def __init__(self, 
                 preprocessor_path: str = "models/preprocessor.joblib",
                 model_path: str = "models/loan_approval_model.h5"):
        """
        Initialize the inference pipeline.
        
        Args:
            preprocessor_path: Path to the fitted preprocessing pipeline
            model_path: Path to the trained ML model
        """
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        
        # Load preprocessor
        try:
            self.preprocessor = LoanPreprocessor.load(preprocessor_path)
            print(f"✅ Loaded preprocessor from {preprocessor_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load preprocessor ({e})")
            self.preprocessor = None
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model."""
        try:
            if os.path.exists(self.model_path):
                # Try loading as Keras model first
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(self.model_path)
                    print(f"✅ Loaded Keras model from {self.model_path}")
                except ImportError:
                    print("⚠️ TensorFlow not available, trying joblib...")
                    self.model = joblib.load(self.model_path)
                    print(f"✅ Loaded joblib model from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None
    
    def preprocess_application(self, application_data: Union[Dict[str, Any], List[Dict]]) -> np.ndarray:
        """
        Preprocess loan application data.
        
        Args:
            application_data: Single application dict or list of applications
            
        Returns:
            np.ndarray: Preprocessed feature matrix
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Cannot preprocess data.")
        
        return self.preprocessor.transform(application_data)
    
    def predict_approval(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict loan approval probability.
        
        Args:
            features: Preprocessed feature matrix
            
        Returns:
            Dict containing prediction results
        """
        if self.model is None:
            # Return dummy prediction if model not available
            return {
                "approval_probability": 0.5,
                "predicted_class": 0,
                "confidence": 0.0,
                "model_available": False
            }
        
        try:
            # Get prediction
            prediction = self.model.predict(features)
            
            # Handle different model output formats
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1 and prediction.shape[1] > 1:
                    # Multi-class output (probabilities for each class)
                    approval_prob = float(prediction[0][1])
                else:
                    # Single output (probability or raw score)
                    approval_prob = float(prediction[0])
            else:
                approval_prob = float(prediction)
            
            # Ensure probability is in [0, 1] range
            approval_prob = max(0.0, min(1.0, approval_prob))
            
            # Determine predicted class
            predicted_class = 1 if approval_prob >= 0.5 else 0
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(approval_prob - 0.5) * 2
            
            return {
                "approval_probability": approval_prob,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "model_available": True
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {
                "approval_probability": 0.5,
                "predicted_class": 0,
                "confidence": 0.0,
                "model_available": False,
                "error": str(e)
            }
    
    def predict_single_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete inference pipeline for a single loan application.
        
        Args:
            application_data: Dictionary containing applicant information
            
        Returns:
            Dict containing complete prediction results
        """
        try:
            # Preprocess the application
            features = self.preprocess_application(application_data)
            
            # Get prediction
            prediction_result = self.predict_approval(features)
            
            # Add input data reference
            prediction_result["input_data"] = application_data
            prediction_result["processed_features_shape"] = features.shape
            
            return prediction_result
            
        except Exception as e:
            return {
                "approval_probability": 0.0,
                "predicted_class": 0,
                "confidence": 0.0,
                "model_available": False,
                "error": str(e),
                "input_data": application_data
            }
    
    def predict_batch_applications(self, applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch inference for multiple loan applications.
        
        Args:
            applications: List of application dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        try:
            # Preprocess all applications at once
            features = self.preprocess_application(applications)
            
            # Get batch predictions
            if self.model is not None:
                predictions = self.model.predict(features)
                
                for i, (app, pred) in enumerate(zip(applications, predictions)):
                    # Handle prediction format
                    if isinstance(pred, np.ndarray) and len(pred) > 1:
                        approval_prob = float(pred[1])
                    else:
                        approval_prob = float(pred)
                    
                    approval_prob = max(0.0, min(1.0, approval_prob))
                    predicted_class = 1 if approval_prob >= 0.5 else 0
                    confidence = abs(approval_prob - 0.5) * 2
                    
                    results.append({
                        "approval_probability": approval_prob,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "model_available": True,
                        "input_data": app,
                        "batch_index": i
                    })
            else:
                # Dummy results if model not available
                for i, app in enumerate(applications):
                    results.append({
                        "approval_probability": 0.5,
                        "predicted_class": 0,
                        "confidence": 0.0,
                        "model_available": False,
                        "input_data": app,
                        "batch_index": i
                    })
                        
        except Exception as e:
            # Fallback to individual processing
            print(f"⚠️ Batch processing failed ({e}), falling back to individual processing")
            for app in applications:
                results.append(self.predict_single_application(app))
        
        return results


# Convenience functions for backward compatibility
def preprocess_loan_application(application_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess a single loan application using the unified pipeline.
    
    Args:
        application_data: Dictionary containing applicant information
        
    Returns:
        np.ndarray: Preprocessed feature array
    """
    return preprocess_single_application(application_data)


def predict_loan_approval(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete prediction pipeline for a single loan application.
    
    Args:
        application_data: Dictionary containing applicant information
        
    Returns:
        Dict containing prediction results
    """
    pipeline = LoanInferencePipeline()
    return pipeline.predict_single_application(application_data)


if __name__ == "__main__":
    # Test the inference pipeline
    print("🧪 Testing Loan Inference Pipeline")
    print("-" * 40)
    
    # Sample application
    sample_application = {
        "age_years": 32,
        "gender": "Male",
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
        "pep_flag": False,
        "loan_type": "personal"
    }
    
    # Test inference pipeline
    pipeline = LoanInferencePipeline()
    
    print("📊 Testing single application prediction...")
    result = pipeline.predict_single_application(sample_application)
    
    print(f"✅ Prediction Result:")
    print(f"   • Approval Probability: {result['approval_probability']:.3f}")
    print(f"   • Predicted Class: {result['predicted_class']}")
    print(f"   • Confidence: {result['confidence']:.3f}")
    print(f"   • Model Available: {result['model_available']}")
    
    if 'error' in result:
        print(f"   • Error: {result['error']}")
    
    # Test batch processing
    print("\n📊 Testing batch prediction...")
    batch_applications = [sample_application.copy() for _ in range(3)]
    
    # Modify applications slightly
    batch_applications[1]['bureau_score'] = 650
    batch_applications[1]['foir_total_obligations_pct'] = 65.0
    
    batch_applications[2]['bureau_score'] = 800
    batch_applications[2]['foir_total_obligations_pct'] = 25.0
    
    batch_results = pipeline.predict_batch_applications(batch_applications)
    
    for i, result in enumerate(batch_results):
        print(f"   Application {i+1}: {result['approval_probability']:.3f} (Class: {result['predicted_class']})")
    
    print("\n🎯 Inference pipeline test completed!")