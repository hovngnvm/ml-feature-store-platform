import os
import sys
import logging
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# Evidently 0.7+ legacy module provides high-level HTML report generator
try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("feature_monitoring")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DASHBOARD_DIR = os.path.join(PROJECT_DIR, "dashboards")
DEFAULT_REPORT_PATH = os.path.join(DASHBOARD_DIR, "feature_drift_report.html")

def generate_feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_html_path: str = DEFAULT_REPORT_PATH
) -> dict:
    """
    Generates Data Drift Monitoring Report comparing historical baseline (reference) 
    with recent batch features (current) using Evidently AI.
    
    Saves standalone HTML report to dashboards/feature_drift_report.html.
    """
    if reference_df is None or reference_df.empty or current_df is None or current_df.empty:
        logger.warning("[Evidently AI] Reference or Current DataFrame is empty. Skipping drift report generation.")
        return {"status": "SKIPPED", "reason": "Empty input DataFrames"}

    os.makedirs(os.path.dirname(report_html_path), exist_ok=True)
    logger.info(f"[Evidently AI] Running Feature Data Drift Analysis on {len(reference_df):,} reference rows vs {len(current_df):,} current rows...")

    # Filter numeric feature columns
    feature_cols = [
        col for col in current_df.columns
        if col not in ["card_id", "event_timestamp", "timestamp"] and pd.api.types.is_numeric_dtype(current_df[col])
    ]

    ref_features = reference_df[feature_cols].dropna()
    curr_features = current_df[feature_cols].dropna()

    if ref_features.empty or curr_features.empty:
        logger.warning("[Evidently AI] Numeric feature columns are empty after filtering.")
        return {"status": "SKIPPED", "reason": "No numeric feature columns available"}

    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=curr_features)
        
        # Save HTML Report for browser visualization
        report.save_html(report_html_path)
        logger.info(f"✅ [Evidently AI] Saved HTML Data Drift Report to '{report_html_path}'")

        # Extract summary dict
        report_dict = report.as_dict()
        drift_metrics = report_dict.get("metrics", [{}])[0].get("result", {})
        
        dataset_drift = drift_metrics.get("dataset_drift", False)
        drift_share = drift_metrics.get("drift_share", 0.0)
        number_of_drifted_features = drift_metrics.get("number_of_drifted_columns", 0)

        if dataset_drift:
            logger.warning(f"⚠️ [Data Drift Alert] Dataset drift DETECTED! ({number_of_drifted_features} features drifted, share: {drift_share:.2%})")
        else:
            logger.info(f"✅ [Data Drift Normal] No dataset drift detected. (Drift share: {drift_share:.2%})")

        return {
            "status": "SUCCESS",
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drifted_features_count": number_of_drifted_features,
            "report_html_path": report_html_path
        }
    except Exception as e:
        logger.error(f"Failed to generate Evidently AI drift report: {e}")
        return {"status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    logger.info("Running Evidently AI Feature Monitoring Self-Test...")
    import numpy as np

    # Synthetic baseline data (Reference)
    np.random.seed(42)
    ref_sample = pd.DataFrame({
        "trans_count_7d": np.random.poisson(lam=5, size=100),
        "trans_count_30d": np.random.poisson(lam=20, size=100),
        "avg_amount_30d": np.random.normal(loc=150.0, scale=30.0, size=100),
        "max_amount_30d": np.random.normal(loc=500.0, scale=100.0, size=100),
    })

    # Synthetic drifted data (Current)
    curr_sample = pd.DataFrame({
        "trans_count_7d": np.random.poisson(lam=5, size=100),
        "trans_count_30d": np.random.poisson(lam=20, size=100),
        "avg_amount_30d": np.random.normal(loc=450.0, scale=80.0, size=100), # Drifted!
        "max_amount_30d": np.random.normal(loc=1500.0, scale=300.0, size=100), # Drifted!
    })

    res = generate_feature_drift_report(ref_sample, curr_sample)
    logger.info(f"Self-test Result: {res}")
