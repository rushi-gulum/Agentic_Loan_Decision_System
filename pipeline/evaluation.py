#!/usr/bin/env python3
"""
pipeline/evaluation.py
=======================
Evaluate model performance on the 40,000 applicant loan dataset.

Usage:
    python pipeline/evaluation.py
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

def evaluate():
    model_path = ROOT / "models" / "loan_approval_model.joblib"
    x_path = ROOT / "data" / "processed" / "X_train.csv"
    y_path = ROOT / "data" / "processed" / "y_train.csv"

    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        return

    logger.info("Loading model and dataset ...")
    model = joblib.load(str(model_path))
    X = pd.read_csv(x_path).values
    y = pd.read_csv(y_path).values.ravel()

    # Predict on entire dataset
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    auc = float(roc_auc_score(y, probas))
    cm = confusion_matrix(y, preds).tolist()

    metrics = {
        "dataset_rows": len(y),
        "model_type": type(model).__name__,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1]
        }
    }

    print("\n" + "=" * 55)
    print("  MODEL PERFORMANCE ON 40,000 APPLICANT DATASET")
    print("=" * 55)
    print(f"  Model Type       : {metrics['model_type']}")
    print(f"  Total Samples    : {metrics['dataset_rows']:,}")
    print(f"  Accuracy         : {metrics['accuracy']:.2%}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")
    print(f"  F1 Score         : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print("=" * 55)
    print("\nClassification Report:\n", classification_report(y, preds, digits=4))

    out_file = ROOT / "data" / "processed" / "evaluation_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", out_file.name)

    return metrics

if __name__ == "__main__":
    evaluate()
