"""
Model Ensemble Training & Evaluation Pipeline with Dynamic Decision Threshold Tuning.

Trains XGBoost and LightGBM models on offline features, optimizes decision thresholds
based on Financial Cost Matrix (minimizing transaction losses & false positive friction),
and exports production model artifacts.
"""

from typing import Any
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

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

load_dotenv()
logger = get_logger("train_model")


class FraudModelEnsemble:
    """Production Model Ensemble combining XGBoost & LightGBM with Weighted Blending & Dynamic Thresholding."""

    def __init__(
        self,
        xgb_model: Any,
        lgb_model: Any,
        feature_names: list[str],
        xgb_weight: float = 0.5,
        lgb_weight: float = 0.5,
        optimal_threshold: float = 0.5,
        threshold_metrics: dict | None = None
    ) -> None:
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.feature_names = feature_names
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.optimal_threshold = optimal_threshold
        self.threshold_metrics = threshold_metrics or {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 2D array of class probabilities [[p0, p1], ...]."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        p1_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p1_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p1_ensemble = (self.xgb_weight * p1_xgb) + (self.lgb_weight * p1_lgb)
        p0_ensemble = 1.0 - p1_ensemble
        return np.column_stack((p0_ensemble, p1_ensemble))

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Returns binary predictions based on decision threshold (defaults to optimal_threshold)."""
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


sys.modules['__main__'].FraudModelEnsemble = FraudModelEnsemble


def find_optimal_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    amounts: np.ndarray = None,
    cost_fp: float = settings.default_fp_cost
) -> dict:
    """
    Finds optimal decision threshold minimizing total financial loss using a Cost Matrix:
    - False Negative (FN): Lost transaction amount ($A_i$).
    - False Positive (FP): Verification/friction cost ($cost_fp).
    - True Positive (TP) & True Negative (TN): $0.
    
    Also computes F1-optimal and F2-optimal (Recall-prioritizing) thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if amounts is None:
        amounts = np.ones(len(y_true)) * 100.0
    else:
        amounts = np.asarray(amounts).astype(float)

    thresholds = np.linspace(0.01, 0.90, 90)
    best_cost_threshold = 0.5
    min_cost = float("inf")
    cost_at_05 = 0.0

    best_f1_threshold = 0.5
    max_f1 = -1.0

    best_f2_threshold = 0.5
    max_f2 = -1.0

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        
        # FN: Actual fraud missed
        fn_mask = (y_true == 1) & (preds == 0)
        cost_fn = float(amounts[fn_mask].sum())

        # FP: Legitimate transaction falsely flagged
        fp_mask = (y_true == 0) & (preds == 1)
        cost_fp_total = float(fp_mask.sum() * cost_fp)

        total_cost = cost_fn + cost_fp_total

        if abs(th - 0.5) < 0.005:
            cost_at_05 = total_cost

        if total_cost < min_cost:
            min_cost = total_cost
            best_cost_threshold = float(th)

        # F1 and F2 computation
        tp = float(((y_true == 1) & (preds == 1)).sum())
        fp = float(fp_mask.sum())
        fn = float(fn_mask.sum())

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)

        if f1 > max_f1:
            max_f1 = float(f1)
            best_f1_threshold = float(th)

        if f2 > max_f2:
            max_f2 = float(f2)
            best_f2_threshold = float(th)

    if cost_at_05 == 0.0:
        cost_at_05 = min_cost

    savings_amount = max(0.0, cost_at_05 - min_cost)
    savings_pct = (savings_amount / (cost_at_05 + 1e-9)) * 100.0

    logger.info("Optimal Decision Threshold Tuning Results:")
    logger.info(f"   Cost-Optimal Threshold : {best_cost_threshold:.4f} (Min Cost: ${min_cost:,.2f})")
    logger.info(f"   Default 0.5 Cost       : ${cost_at_05:,.2f}")
    logger.info(f"   Financial Loss Saved   : ${savings_amount:,.2f} ({savings_pct:.2f}% reduction)")
    logger.info(f"   F1-Optimal Threshold   : {best_f1_threshold:.4f} (Max F1: {max_f1:.4f})")
    logger.info(f"   F2-Optimal Threshold   : {best_f2_threshold:.4f} (Max F2: {max_f2:.4f})")

    return {
        "optimal_threshold": round(best_cost_threshold, 4),
        "min_cost": round(min_cost, 2),
        "cost_at_05": round(cost_at_05, 2),
        "savings_amount": round(savings_amount, 2),
        "savings_pct": round(savings_pct, 2),
        "f1_optimal_threshold": round(best_f1_threshold, 4),
        "f2_optimal_threshold": round(best_f2_threshold, 4),
        "best_f1_score": round(max_f1, 4),
        "best_f2_score": round(max_f2, 4)
    }


def evaluate_model_performance(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Calculates PR-AUC, ROC-AUC, F1-Score, and Confusion Matrix at given threshold."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = float(auc(recall, precision))
    roc_score = float(roc_auc_score(y_true, y_proba))
    
    y_pred = (y_proba >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    logger.info(f"[{name}] Evaluation Results (Threshold = {threshold:.4f}):")
    logger.info(f"   PR-AUC  : {pr_auc_score:.4f}")
    logger.info(f"   ROC-AUC : {roc_score:.4f}")
    logger.info(f"   F1-Score: {f1:.4f}")
    logger.info(f"   Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    return {
        "threshold": round(threshold, 4),
        "pr_auc": round(pr_auc_score, 4),
        "roc_auc": round(roc_score, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm
    }


def train_ensemble_pipeline(
    dataset_path: str = settings.ml_dataset_path,
    model_output_path: str = settings.model_artifact_path,
    report_output_path: str = settings.report_json_path
) -> dict:
    """Trains XGBoost + LightGBM Model Ensemble, tunes optimal threshold, and exports artifacts."""
    logger.info(f"Loading ML Training Dataset from '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    target_col = "is_fraud"
    feature_cols = [c for c in df.columns if c not in ["card_id", "TransactionID", target_col]]

    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Split Dataset into Train ({len(X_train):,} samples) and Validation ({len(X_val):,} samples)")

    # Compute scale_pos_weight for XGBoost to handle imbalanced fraud data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Calculated scale_pos_weight for Imbalanced Fraud Data: {scale_pos_weight:.2f}")

    # Train XGBoost Classifier
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

    # Train LightGBM Classifier
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

    # Predict Probabilities on Validation Set
    p_xgb = model_xgb.predict_proba(X_val)[:, 1]
    p_lgb = model_lgb.predict_proba(X_val)[:, 1]
    p_ensemble = 0.5 * p_xgb + 0.5 * p_lgb

    # Dynamic Decision Threshold Tuning via Cost Matrix
    val_amounts = X_val["TransactionAmt"].values if "TransactionAmt" in X_val.columns else np.ones(len(X_val)) * 100.0
    threshold_tuning = find_optimal_decision_threshold(
        y_true=y_val.values,
        y_proba=p_ensemble,
        amounts=val_amounts,
        cost_fp=settings.default_fp_cost
    )
    optimal_th = threshold_tuning["optimal_threshold"]

    # Evaluate Performance across baseline 0.5 and cost-optimal threshold
    res_xgb = evaluate_model_performance("XGBoost", y_val.values, p_xgb, threshold=0.5)
    res_lgb = evaluate_model_performance("LightGBM", y_val.values, p_lgb, threshold=0.5)
    res_ensemble_default = evaluate_model_performance("Model Ensemble (Default 0.5)", y_val.values, p_ensemble, threshold=0.5)
    res_ensemble_optimal = evaluate_model_performance("Model Ensemble (Cost-Optimal)", y_val.values, p_ensemble, threshold=optimal_th)

    # Construct & Export Model Ensemble Pipeline with Optimal Threshold
    ensemble_pipeline = FraudModelEnsemble(
        xgb_model=model_xgb,
        lgb_model=model_lgb,
        feature_names=feature_cols,
        xgb_weight=0.5,
        lgb_weight=0.5,
        optimal_threshold=optimal_th,
        threshold_metrics=threshold_tuning
    )

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(ensemble_pipeline, model_output_path)
    logger.info(f"Saved Ensemble Model Artifact with Optimal Threshold ({optimal_th:.4f}) to '{model_output_path}'")

    # Save Evaluation Report JSON
    report_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "optimal_threshold": optimal_th,
        "threshold_tuning": threshold_tuning,
        "metrics": {
            "xgboost": res_xgb,
            "lightgbm": res_lgb,
            "ensemble_default_05": res_ensemble_default,
            "ensemble_optimal": res_ensemble_optimal,
            "ensemble": res_ensemble_optimal
        }
    }

    with open(report_output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Saved Model Evaluation Report JSON to '{report_output_path}'")
    return report_dict


if __name__ == "__main__":
    logger.info("Training Fraud Model Ensemble & Tuning Decision Threshold...")
    train_ensemble_pipeline()
