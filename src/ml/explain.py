"""
Global Model Explainability Engine (SHAP).

Computes Global SHAP values across the dataset, generating
Global SHAP Beeswarm Summary plots and Feature Importance Bar charts.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger("ml_explain")

MODEL_PATH = os.path.join(settings.project_dir, "models", "ensemble_fraud_model.joblib")
DATASET_PATH = os.path.join(settings.project_dir, "data", "ml_training_dataset.parquet")
DASHBOARD_DIR = os.path.join(settings.project_dir, "dashboards")


def generate_global_shap_explanations(sample_size: int = 2000) -> None:
    """Computes Global SHAP values and exports global explainability plots.

    Args:
        sample_size: Number of dataset rows to sample for SHAP calculation.
    """
    logger.info("Starting Global SHAP Explainability Engine...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
        logger.error("Model artifact or dataset missing! Please run prepare_dataset.py and train.py first.")
        return

    import shap

    model_pipeline: FraudModelEnsemble = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATASET_PATH)

    feature_cols = model_pipeline.feature_names
    X = df[feature_cols]

    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    logger.info(f"Computing TreeExplainer SHAP values on {len(X_sample):,} samples for XGBoost & LightGBM...")

    explainer_xgb = shap.TreeExplainer(model_pipeline.xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_sample)

    explainer_lgb = shap.TreeExplainer(model_pipeline.lgb_model)
    shap_values_lgb = explainer_lgb.shap_values(X_sample)

    shap_values_ens = (shap_values_xgb + shap_values_lgb) / 2.0

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    # Plot 1: Global SHAP Summary Beeswarm Plot
    plt.clf()
    shap.summary_plot(shap_values_ens, X_sample, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 7.5)
    fig.suptitle("Global SHAP Summary (Beeswarm Impact Distribution)", fontsize=13, y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.25, right=0.95)
    beeswarm_path = os.path.join(DASHBOARD_DIR, "shap_summary_beeswarm.png")
    fig.savefig(beeswarm_path, dpi=300, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)
    logger.info(f"Saved Global SHAP Beeswarm Plot to '{beeswarm_path}'")

    # Plot 2: Global Feature Importance Bar Chart
    plt.clf()
    shap.summary_plot(shap_values_ens, X_sample, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 7.5)
    fig.suptitle("Global Feature Importance Ranking (Mean |SHAP Value|)", fontsize=13, y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.25, right=0.95)
    bar_path = os.path.join(DASHBOARD_DIR, "shap_feature_importance.png")
    fig.savefig(bar_path, dpi=300, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)
    logger.info(f"Saved Global Feature Importance Bar Chart to '{bar_path}'")

    logger.info("Global SHAP Explainability Generation Complete.")


if __name__ == "__main__":
    generate_global_shap_explanations()
