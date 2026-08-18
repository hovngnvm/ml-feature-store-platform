"""
Offline Model Evaluation & Benchmark Exporter with Dynamic Threshold & Cost Curve Analysis.

Evaluates XGBoost, LightGBM, and Model Ensemble performance metrics,
generating ROC Curves, Precision-Recall Curves, Confusion Matrix plots,
and Financial Cost Optimization Curves vs Decision Thresholds.
"""

import os
import sys
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
from src.ml.train import FraudModelEnsemble

sys.modules['__main__'].FraudModelEnsemble = FraudModelEnsemble
logger = get_logger("ml_evaluate")


def evaluate_models() -> None:
    """Executes offline evaluation pipeline and exports benchmark plots."""
    logger.info("Starting Offline Model Evaluation & Plot Export Pipeline...")
    if not os.path.exists(settings.model_artifact_path) or not os.path.exists(settings.ml_dataset_path):
        logger.error("Model artifact or dataset missing! Please run prepare_dataset.py and train.py first.")
        return

    model_pipeline = joblib.load(settings.model_artifact_path)
    df = pd.read_parquet(settings.ml_dataset_path)

    feature_cols = model_pipeline.feature_names
    X = df[feature_cols]
    y = df["is_fraud"].values

    # Use the exact same Stratified Validation Split as train.py
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    amounts_val = X_val["TransactionAmt"].values if "TransactionAmt" in X_val.columns else np.ones(len(X_val)) * 100.0

    optimal_th = getattr(model_pipeline, "optimal_threshold", 0.5)

    logger.info(f"Evaluating Model Ensemble on Stratified Validation Set ({len(X_val):,} samples, Trained Optimal Threshold = {optimal_th:.4f})...")


    y_prob_xgb = model_pipeline.xgb_model.predict_proba(X_val)[:, 1]
    y_prob_lgb = model_pipeline.lgb_model.predict_proba(X_val)[:, 1]
    y_prob_ens = model_pipeline.predict_proba(X_val)[:, 1]

    # ROC Curve
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
    roc_path = os.path.join(settings.model_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC Curve plot to '{roc_path}'")

    # Precision-Recall Curve
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
    pr_path = os.path.join(settings.model_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Precision-Recall Curve plot to '{pr_path}'")

    # Financial Cost Curve vs Decision Threshold
    thresholds = np.linspace(0.01, 0.90, 90)
    total_costs = []
    fn_costs = []
    fp_costs = []
    cost_fp = settings.default_fp_cost

    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []

    for th in thresholds:
        preds = (y_prob_ens >= th).astype(int)
        fn_mask = (y_val == 1) & (preds == 0)
        fp_mask = (y_val == 0) & (preds == 1)
        tp_mask = (y_val == 1) & (preds == 1)

        c_fn = float(amounts_val[fn_mask].sum())
        c_fp = float(fp_mask.sum() * cost_fp)
        total_costs.append(c_fn + c_fp)
        fn_costs.append(c_fn)
        fp_costs.append(c_fp)

        tp = float(tp_mask.sum())
        fp = float(fp_mask.sum())
        fn = float(fn_mask.sum())

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append((2 * p * r) / (p + r + 1e-9))
        f2_scores.append((5 * p * r) / (4 * p + r + 1e-9))

    optimal_th = float(getattr(model_pipeline, "optimal_threshold", thresholds[int(np.argmin(total_costs))]))
    opt_idx = int(np.argmin(np.abs(thresholds - optimal_th)))
    opt_cost_val = total_costs[opt_idx]

    idx_05 = int(np.argmin(np.abs(thresholds - 0.50)))
    cost_05_val = total_costs[idx_05]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, total_costs, label="Total Financial Loss ($)", color="crimson", linewidth=2.5)
    plt.plot(thresholds, fn_costs, label="False Negative Loss (Missed Fraud $)", color="darkorange", linestyle="--")
    plt.plot(thresholds, fp_costs, label=f"False Positive Loss (Verification @ ${cost_fp:.1f})", color="navy", linestyle=":")
    
    plt.scatter([optimal_th], [opt_cost_val], color="darkgreen", s=120, zorder=5,
                label=f"Optimal Threshold = {optimal_th:.4f} (Loss: ${opt_cost_val:,.2f})")
    plt.scatter([0.50], [cost_05_val], color="black", s=100, zorder=5,
                label=f"Default 0.50 Threshold (Loss: ${cost_05_val:,.2f})")


    plt.title("Financial Cost Curve vs Decision Threshold (Cost Matrix Optimization)", fontsize=13)
    plt.xlabel("Decision Threshold (θ)", fontsize=11)
    plt.ylabel("Total Financial Loss (USD)", fontsize=11)
    plt.legend(loc="upper center")
    plt.grid(True, alpha=0.3)

    cost_plot_path = os.path.join(settings.model_dir, "cost_vs_threshold.png")
    plt.savefig(cost_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Financial Cost Curve plot to '{cost_plot_path}'")

    # Precision, Recall, F1, F2 vs Decision Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", color="dodgerblue", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall (Fraud Coverage)", color="forestgreen", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1-Score", color="purple", linestyle="--", linewidth=2)
    plt.plot(thresholds, f2_scores, label="F2-Score (Recall Prioritized)", color="crimson", linestyle="-.", linewidth=2)
    plt.axvline(x=optimal_th, color="darkgreen", linestyle=":", label=f"Cost-Optimal θ = {optimal_th:.4f}")


    plt.title("Metric Trade-offs (Precision, Recall, F1, F2) vs Decision Threshold", fontsize=13)
    plt.xlabel("Decision Threshold (θ)", fontsize=11)
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=11)
    plt.legend(loc="center right")
    plt.grid(True, alpha=0.3)

    tradeoff_plot_path = os.path.join(settings.model_dir, "threshold_tradeoffs.png")
    plt.savefig(tradeoff_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Threshold Trade-offs plot to '{tradeoff_plot_path}'")

    # Confusion Matrix at Optimal Threshold
    y_pred_opt = (y_prob_ens >= optimal_th).astype(int)
    cm = confusion_matrix(y_val, y_pred_opt)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=["Legitimate (0)", "Fraud (1)"],
                yticklabels=["Legitimate (0)", "Fraud (1)"])
    plt.title(f"Model Ensemble Confusion Matrix (Optimal θ = {optimal_th:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(settings.model_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Confusion Matrix plot to '{cm_path}'")

    report_str = classification_report(y_val, y_pred_opt, target_names=["Legitimate", "Fraud"])
    logger.info(f"Model Ensemble Classification Report (Optimal Threshold):\n{report_str}")


if __name__ == "__main__":
    evaluate_models()
