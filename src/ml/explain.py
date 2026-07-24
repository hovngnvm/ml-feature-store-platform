"""Global Model Explainability Engine (SHAP).

Computes Global SHAP values across the dataset, generating
Global SHAP Beeswarm Summary plots and Feature Importance Bar charts.
"""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger(__name__)


def generate_global_shap_explanations(sample_size: int = 2000) -> None:
    """Computes Global SHAP values and exports global explainability plots."""
    logger.info("Starting Global SHAP Explainability Engine...")
    path_model = Path(settings.model_artifact_path)
    path_dataset = Path(settings.ml_dataset_path)
    if not path_model.exists() or not path_dataset.exists():
        raise FileNotFoundError("Model artifact or dataset missing! Run prepare_dataset.py and train.py first.")

    import shap

    model_pipeline: FraudModelEnsemble = joblib.load(path_model)
    df = pd.read_parquet(path_dataset)

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

    w_xgb = getattr(model_pipeline, "xgb_weight", 0.5)
    w_lgb = getattr(model_pipeline, "lgb_weight", 0.5)
    shap_values_ens = (w_xgb * shap_values_xgb) + (w_lgb * shap_values_lgb)

    dashboard_dir = Path(settings.dashboard_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # Global SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_ens, X_sample, show=False)
    plt.title("Global SHAP Summary (Beeswarm Impact Distribution)", fontsize=14, pad=15)
    beeswarm_path = dashboard_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Global SHAP Beeswarm Plot to '{beeswarm_path}'")

    # Global Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_ens, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance Ranking (Mean |SHAP Value|)", fontsize=14, pad=15)
    bar_path = dashboard_dir / "shap_feature_importance.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Global Feature Importance Bar Chart to '{bar_path}'")

    logger.info("Global SHAP Explainability Generation Complete.")


if __name__ == "__main__":
    generate_global_shap_explanations()
