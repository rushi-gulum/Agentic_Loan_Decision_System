"""
utils/explain_utils.py
======================
Explainability utilities and picklable explainer classes.
"""

import numpy as np

class CoeffExplainer:
    """
    Module-level (picklable) SHAP-compatible explainer for LogisticRegression.
    Used as fallback when shap/numba DLLs are blocked (Windows AppControl).
    Implements .shap_values() via coefficient * feature-value product.
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
