#!/usr/bin/env python3
"""
pipeline/train_model.py
========================
Modular model training script for the 40,000 applicant loan dataset.

Usage:
    python pipeline/train_model.py
    python pipeline/train_model.py --model lr
    python pipeline/train_model.py --model xgb
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SEED = 42

def train(model_type: str = "lr"):
    x_path = ROOT / "data" / "processed" / "X_train.csv"
    y_path = ROOT / "data" / "processed" / "y_train.csv"

    if not x_path.exists() or not y_path.exists():
        logger.error("Dataset not found. Run: python pipeline/build_artifacts.py first.")
        sys.exit(1)

    logger.info("Loading training data from %s ...", x_path.name)
    X = pd.read_csv(x_path).values
    y = pd.read_csv(y_path).values.ravel()
    logger.info("Loaded dataset: %d samples with %d features", X.shape[0], X.shape[1])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            logger.info("Training XGBClassifier ...")
            clf = XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=SEED,
                eval_metric="logloss"
            )
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            model_file = "loan_approval_model_xgb.joblib"
        except ImportError:
            logger.warning("xgboost not installed, falling back to LogisticRegression")
            model_type = "lr"

    if model_type == "lr":
        logger.info("Training LogisticRegression (balanced) ...")
        clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        clf.fit(X_train, y_train)
        model_file = "loan_approval_model.joblib"

    # Evaluation
    preds = clf.predict(X_val)
    probas = clf.predict_proba(X_val)[:, 1]
    acc = clf.score(X_val, y_val)
    auc = roc_auc_score(y_val, probas)

    logger.info("Validation Accuracy : %.4f", acc)
    logger.info("Validation ROC-AUC  : %.4f", auc)
    print("\n" + classification_report(y_val, preds, digits=4))

    # Save artifact
    out_path = ROOT / "models" / model_file
    joblib.dump(clf, str(out_path))
    logger.info("Saved %s (%.1f KB)", model_file, out_path.stat().st_size / 1024)

    return clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit decisioning models")
    parser.add_argument("--model", choices=["lr", "xgb"], default="lr", help="Model type to train")
    args = parser.parse_args()
    train(model_type=args.model)
