"""
Global Model Explainability Engine (SHAP).

Computes Global SHAP values across the dataset, generating
Global SHAP Beeswarm Summary plots and Feature Importance Bar charts.
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.train import FraudModelEnsemble

sys.modules['__main__'].FraudModelEnsemble = FraudModelEnsemble
logger = get_logger("ml_explain")


def generate_global_shap_explanations(sample_size: int = 2000) -> None:
    """Computes Global SHAP values and exports global explainability plots."""
    logger.info("Starting Global SHAP Explainability Engine...")
    if not os.path.exists(settings.model_artifact_path) or not os.path.exists(settings.ml_dataset_path):
        logger.error("Model artifact or dataset missing! Please run prepare_dataset.py and train.py first.")
        return

    import shap

    model_pipeline = joblib.load(settings.model_artifact_path)
    df = pd.read_parquet(settings.ml_dataset_path)

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

    os.makedirs(settings.dashboard_dir, exist_ok=True)

    # Global SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_ens, X_sample, show=False)
    plt.title("Global SHAP Summary (Beeswarm Impact Distribution)", fontsize=14, pad=15)
    beeswarm_path = os.path.join(settings.dashboard_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Global SHAP Beeswarm Plot to '{beeswarm_path}'")

    # Global Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_ens, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance Ranking (Mean |SHAP Value|)", fontsize=14, pad=15)
    bar_path = os.path.join(settings.dashboard_dir, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Global Feature Importance Bar Chart to '{bar_path}'")

    logger.info("Global SHAP Explainability Generation Complete.")


if __name__ == "__main__":
    generate_global_shap_explanations()
