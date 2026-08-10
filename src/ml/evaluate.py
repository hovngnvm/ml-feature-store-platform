"""
Offline Model Evaluation & Benchmark Exporter.

Evaluates XGBoost, LightGBM, and Model Ensemble performance metrics,
generating ROC Curves, Precision-Recall Curves, and Confusion Matrix plots.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger("ml_evaluate")

MODEL_PATH = os.path.join(settings.project_dir, "models", "ensemble_fraud_model.joblib")
DATASET_PATH = os.path.join(settings.project_dir, "data", "ml_training_dataset.parquet")
OUTPUT_DIR = os.path.join(settings.project_dir, "models")


def evaluate_models() -> None:
    """Executes offline evaluation pipeline and exports benchmark plots."""
    logger.info("Starting Offline Model Evaluation & Plot Export Pipeline...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
        logger.error("Model artifact or dataset missing! Please run prepare_dataset.py and train.py first.")
        return

    model_pipeline: FraudModelEnsemble = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATASET_PATH)

    feature_cols = model_pipeline.feature_names
    X = df[feature_cols]
    y = df["is_fraud"].values

    val_size = int(len(df) * 0.2)
    X_val = X.iloc[-val_size:]
    y_val = y[-val_size:]

    logger.info(f"Evaluating Model Ensemble on Validation Set ({len(X_val):,} samples)...")

    y_prob_xgb = model_pipeline.xgb_model.predict_proba(X_val)[:, 1]
    y_prob_lgb = model_pipeline.lgb_model.predict_proba(X_val)[:, 1]
    y_prob_ens = model_pipeline.predict_proba(X_val)[:, 1]

    # Plot 1: ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_prob_xgb)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_val, y_prob_lgb)
    fpr_ens, tpr_ens, _ = roc_curve(y_val, y_prob_ens)

    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc(fpr_xgb, tpr_xgb):.4f})", linestyle="--")
    plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {auc(fpr_lgb, tpr_lgb):.4f})", linestyle=":")
    plt.plot(fpr_ens, tpr_ens, label=f"Model Ensemble (AUC = {auc(fpr_ens, tpr_ens):.4f})", linewidth=2.5, color="darkred")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC Curve plot to '{roc_path}'")

    # Plot 2: Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    p_xgb, r_xgb, _ = precision_recall_curve(y_val, y_prob_xgb)
    p_lgb, r_lgb, _ = precision_recall_curve(y_val, y_prob_lgb)
    p_ens, r_ens, _ = precision_recall_curve(y_val, y_prob_ens)

    plt.plot(r_xgb, p_xgb, label=f"XGBoost (PR-AUC = {auc(r_xgb, p_xgb):.4f})", linestyle="--")
    plt.plot(r_lgb, p_lgb, label=f"LightGBM (PR-AUC = {auc(r_lgb, p_lgb):.4f})", linestyle=":")
    plt.plot(r_ens, p_ens, label=f"Model Ensemble (PR-AUC = {auc(r_ens, p_ens):.4f})", linewidth=2.5, color="darkgreen")
    plt.title("Precision-Recall Curve (Imbalanced Fraud Evaluation)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    pr_path = os.path.join(OUTPUT_DIR, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Precision-Recall Curve plot to '{pr_path}'")

    # Plot 3: Confusion Matrix
    y_pred_ens = (y_prob_ens >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred_ens)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=["Legitimate (0)", "Fraud (1)"],
                yticklabels=["Legitimate (0)", "Fraud (1)"])
    plt.title("Model Ensemble Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Confusion Matrix plot to '{cm_path}'")

    logger.info("Model Ensemble Classification Report:")
    report_str = classification_report(y_val, y_pred_ens, target_names=["Legitimate", "Fraud"])
    for line in report_str.split("\n"):
        if line.strip():
            logger.info(line)


if __name__ == "__main__":
    evaluate_models()
