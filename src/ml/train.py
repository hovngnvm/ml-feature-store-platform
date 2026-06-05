"""
Model Ensemble Training & Evaluation Pipeline.

Trains XGBoost and LightGBM models on training dataset, builds blended 
FraudModelEnsemble pipeline, and exports joblib artifact and evaluation report.
"""

import os
import json
from typing import Dict, Any
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger("train_model")

DATASET_PATH = os.path.join(settings.project_dir, "data", "ml_training_dataset.parquet")
MODEL_DIR = os.path.join(settings.project_dir, "models")
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DIR, "ensemble_fraud_model.joblib")
REPORT_JSON_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")


def evaluate_model_performance(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """Calculates PR-AUC, ROC-AUC, F1-Score, and Confusion Matrix.

    Args:
        name: Identifier name of the model being evaluated.
        y_true: Ground truth binary labels array.
        y_proba: Predicted positive class probability array.

    Returns:
        Dictionary of evaluation metrics.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = float(auc(recall, precision))
    roc_score = float(roc_auc_score(y_true, y_proba))

    y_pred = (y_proba >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    logger.info(
        f"[{name}] PR-AUC: {pr_auc_score:.4f} | ROC-AUC: {roc_score:.4f} | "
        f"F1-Score: {f1:.4f} | CM: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}"
    )

    return {
        "pr_auc": round(pr_auc_score, 4),
        "roc_auc": round(roc_score, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm
    }


def train_ensemble_pipeline(
    dataset_path: str = DATASET_PATH,
    model_output_path: str = MODEL_ARTIFACT_PATH,
    report_output_path: str = REPORT_JSON_PATH
) -> Dict[str, Any]:
    """Trains XGBoost and LightGBM model ensemble and exports serialized pipeline artifact.

    Args:
        dataset_path: Source Parquet training dataset file path.
        model_output_path: Target joblib model artifact file path.
        report_output_path: Target evaluation JSON report file path.

    Returns:
        Evaluation report metrics dictionary.
    """
    logger.info(f"Loading ML training dataset from '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    target_col = "is_fraud"
    feature_cols = [c for c in df.columns if c not in ["card_id", "TransactionID", target_col]]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Split dataset into Train ({len(X_train):,} samples) and Validation ({len(X_val):,} samples)")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Calculated scale_pos_weight for imbalanced fraud data: {scale_pos_weight:.2f}")

    logger.info("Training Model 1: XGBoost Classifier...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train)

    logger.info("Training Model 2: LightGBM Classifier...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model_lgb.fit(X_train, y_train)

    p_xgb = model_xgb.predict_proba(X_val)[:, 1]
    p_lgb = model_lgb.predict_proba(X_val)[:, 1]
    p_ensemble = 0.5 * p_xgb + 0.5 * p_lgb

    res_xgb = evaluate_model_performance("XGBoost", y_val.values, p_xgb)
    res_lgb = evaluate_model_performance("LightGBM", y_val.values, p_lgb)
    res_ensemble = evaluate_model_performance("Model Ensemble (XGB+LGB)", y_val.values, p_ensemble)

    ensemble_pipeline = FraudModelEnsemble(
        xgb_model=model_xgb,
        lgb_model=model_lgb,
        feature_names=feature_cols,
        xgb_weight=0.5,
        lgb_weight=0.5
    )

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(ensemble_pipeline, model_output_path)
    logger.info(f"Saved Ensemble Model Artifact to '{model_output_path}'")

    report_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "metrics": {
            "xgboost": res_xgb,
            "lightgbm": res_lgb,
            "ensemble": res_ensemble
        }
    }

    with open(report_output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Saved Model Evaluation Report JSON to '{report_output_path}'")
    return report_dict


if __name__ == "__main__":
    logger.info("Executing Phase 2: Model Ensemble Training & Evaluation...")
    train_ensemble_pipeline()
