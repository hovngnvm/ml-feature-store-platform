"""Model Ensemble Training & Evaluation Pipeline with Dynamic Decision Threshold Tuning.

Trains XGBoost and LightGBM models on offline features, optimizes decision thresholds
based on Financial Cost Matrix (minimizing transaction losses & false positive friction),
and exports production model artifacts.
"""

from pathlib import Path
import json
from typing import Any
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

from src.config import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger(__name__)

THRESHOLD_START: float = 0.01
THRESHOLD_END: float = 0.90
THRESHOLD_STEPS: int = 90
EPSILON: float = 1e-9
DEFAULT_TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42


def find_optimal_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    amounts: np.ndarray | None = None,
    cost_fp: float = settings.default_fp_cost,
) -> dict[str, Any]:
    """Finds optimal decision threshold minimizing total financial loss using a Cost Matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if amounts is None:
        amounts = np.ones(len(y_true)) * 100.0
    else:
        amounts = np.asarray(amounts).astype(float)

    thresholds = np.linspace(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEPS)
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

        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)

        f1 = (2 * precision * recall) / (precision + recall + EPSILON)
        f2 = (5 * precision * recall) / (4 * precision + recall + EPSILON)

        if f1 > max_f1:
            max_f1 = float(f1)
            best_f1_threshold = float(th)

        if f2 > max_f2:
            max_f2 = float(f2)
            best_f2_threshold = float(th)

    if cost_at_05 == 0.0:
        cost_at_05 = min_cost

    savings_amount = max(0.0, cost_at_05 - min_cost)
    savings_pct = (savings_amount / (cost_at_05 + EPSILON)) * 100.0

    logger.info("Optimal Decision Threshold Tuning Results:")
    logger.info(f"   Cost-Optimal Threshold : {best_cost_threshold:.4f} (Min Cost: ${min_cost:,.2f})")
    logger.info(f"   Default 0.5 Cost       : ${cost_at_05:,.2f}")
    logger.info(f"   Financial Loss Saved   : ${savings_amount:,.2f} ({savings_pct:.2f}% reduction)")
    logger.info(f"   F1-Optimal Threshold   : {best_f1_threshold:.4f} (Max F1: {max_f1:.4f})")
    logger.info(f"   F2-Optimal Threshold   : {best_f2_threshold:.4f} (Max F2: {max_f2:.4f})")

    return {
        "optimal_threshold": round(best_f1_threshold, 4),
        "cost_optimal_threshold": round(best_cost_threshold, 4),
        "f1_optimal_threshold": round(best_f1_threshold, 4),
        "f2_optimal_threshold": round(best_f2_threshold, 4),
        "min_cost": round(min_cost, 2),
        "cost_at_05": round(cost_at_05, 2),
        "savings_amount": round(savings_amount, 2),
        "savings_pct": round(savings_pct, 2),
        "best_f1_score": round(max_f1, 4),
        "best_f2_score": round(max_f2, 4),
    }


def evaluate_model_performance(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
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
        "confusion_matrix": cm,
    }


def train_ensemble_pipeline(
    dataset_path: str | Path = settings.ml_dataset_path,
    model_output_path: str | Path = settings.model_artifact_path,
    report_output_path: str | Path = settings.report_json_path,
) -> dict[str, Any]:
    """Trains XGBoost + LightGBM Model Ensemble, tunes optimal threshold, and exports artifacts."""
    path_dataset = Path(dataset_path)
    logger.info(f"Loading ML Training Dataset from '{path_dataset}'...")
    if not path_dataset.exists():
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    df = pd.read_parquet(path_dataset)
    target_col = "is_fraud"
    feature_cols = [c for c in df.columns if c not in ["card_id", "TransactionID", target_col]]

    # Group-based Card Split to prevent Data Leakage across transactions of the same card
    if "card_id" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_SEED)
        train_idx, val_idx = next(gss.split(df, groups=df["card_id"]))
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        X_val = df_val[feature_cols]
        y_val = df_val[target_col]
        logger.info(f"GroupSplit on card_id: Train ({len(X_train):,} samples, {df_train['card_id'].nunique():,} cards), Val ({len(X_val):,} samples, {df_val['card_id'].nunique():,} cards)")
    else:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        logger.info(f"Split Dataset into Train ({len(X_train):,} samples) and Validation ({len(X_val):,} samples)")

    # Save Reference Baseline Dataset for Evidently AI Data Drift Monitoring
    baseline_dir = Path(settings.lakehouse_base_dir) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_dir / "reference_baseline.parquet"
    X_train.head(5000).to_parquet(baseline_path, index=False)
    logger.info(f"Saved Evidently AI Reference Baseline Dataset to '{baseline_path}'")

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
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model_xgb.fit(X_train, y_train)

    # Train LightGBM Classifier
    logger.info("Training Model 2: LightGBM Classifier...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=-1,
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
        cost_fp=settings.default_fp_cost,
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
        threshold_metrics=threshold_tuning,
    )

    path_model = Path(model_output_path)
    path_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_pipeline, path_model)
    logger.info(f"Saved Ensemble Model Artifact with Optimal Threshold ({optimal_th:.4f}) to '{path_model}'")

    # Save Evaluation Report JSON
    report_dict: dict[str, Any] = {
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
            "ensemble": res_ensemble_optimal,
        }
    }

    path_report = Path(report_output_path)
    path_report.parent.mkdir(parents=True, exist_ok=True)
    with open(path_report, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Saved Model Evaluation Report JSON to '{path_report}'")

    try:
        from src.ml.explain import generate_global_shap_explanations
        generate_global_shap_explanations()
    except Exception as e:
        logger.warning(f"Could not generate SHAP explanations after training: {e}")

    return report_dict


if __name__ == "__main__":
    logger.info("Training Fraud Model Ensemble & Tuning Decision Threshold...")
    train_ensemble_pipeline()
