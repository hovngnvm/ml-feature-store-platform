"""Evidently AI Feature Data Drift Monitoring Engine.

Compares statistical distribution of current inference/batch features against historical baseline,
detecting drift and exporting interactive HTML reports.
"""

from pathlib import Path
from typing import Any
import pandas as pd

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        Report = None
        DataDriftPreset = None

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REPORT_PATH = Path(settings.dashboard_dir) / "feature_drift_report.html"


def generate_feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_html_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Generates Data Drift Monitoring Report comparing historical baseline with recent batch features."""
    if Report is None:
        logger.warning("[Evidently AI] evidently package is not installed. Skipping drift report generation.")
        return {"status": "SKIPPED", "reason": "evidently package not installed"}

    if reference_df is None or reference_df.empty or current_df is None or current_df.empty:
        logger.warning("[Evidently AI] Reference or Current DataFrame is empty. Skipping drift report generation.")
        return {"status": "SKIPPED", "reason": "Empty input DataFrames"}

    path_report = Path(report_html_path)
    path_report.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Evidently AI] Running Feature Data Drift Analysis on {len(reference_df):,} reference rows vs {len(current_df):,} current rows...")

    feature_cols = [
        col for col in current_df.columns
        if col in reference_df.columns
        and col not in ["card_id", "event_timestamp", "timestamp"]
        and pd.api.types.is_numeric_dtype(current_df[col])
    ]

    ref_features = reference_df[feature_cols].dropna()
    curr_features = current_df[feature_cols].dropna()

    if ref_features.empty or curr_features.empty:
        logger.warning("[Evidently AI] Numeric feature columns are empty after filtering.")
        return {"status": "SKIPPED", "reason": "No numeric feature columns available"}

    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=curr_features)

        report.save_html(str(path_report))
        logger.info(f"[Evidently AI] Saved HTML Data Drift Report to '{path_report}'")

        report_dict = report.as_dict()
        metrics_list = report_dict.get("metrics", [{}])
        drift_metrics = metrics_list[0].get("result", {}) if metrics_list else {}

        dataset_drift = bool(drift_metrics.get("dataset_drift", False))
        drift_share = float(drift_metrics.get("drift_share", 0.0))
        number_of_drifted_features = int(drift_metrics.get("number_of_drifted_columns", 0))

        if dataset_drift:
            logger.warning(f"[Data Drift Alert] Dataset drift DETECTED! ({number_of_drifted_features} features drifted, share: {drift_share:.2%})")
        else:
            logger.info(f"[Data Drift Normal] No dataset drift detected. (Drift share: {drift_share:.2%})")

        return {
            "status": "SUCCESS",
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drifted_features_count": number_of_drifted_features,
            "report_html_path": str(path_report),
        }
    except Exception as e:
        logger.error(f"Failed to generate Evidently AI drift report: {e}")
        return {"status": "FAILED", "error": str(e)}


if __name__ == "__main__":
    logger.info("Running Evidently AI Feature Monitoring Self-Test...")
    import numpy as np

    np.random.seed(42)
    ref_sample = pd.DataFrame({
        "trans_count_7d": np.random.poisson(lam=5, size=100),
        "trans_count_30d": np.random.poisson(lam=20, size=100),
        "avg_amount_30d": np.random.normal(loc=150.0, scale=30.0, size=100),
        "max_amount_30d": np.random.normal(loc=500.0, scale=100.0, size=100),
    })

    curr_sample = pd.DataFrame({
        "trans_count_7d": np.random.poisson(lam=5, size=100),
        "trans_count_30d": np.random.poisson(lam=20, size=100),
        "avg_amount_30d": np.random.normal(loc=450.0, scale=80.0, size=100),
        "max_amount_30d": np.random.normal(loc=1500.0, scale=300.0, size=100),
    })

    res = generate_feature_drift_report(ref_sample, curr_sample)
    logger.info(f"Self-test Result: {res}")
