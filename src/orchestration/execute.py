"""Prefect 3 Pipeline Orchestration & Continuous Training Flow.

Orchestrates 4-step end-to-end pipeline: Batch execution,
Feast materialization, Evidently drift analysis, and Hybrid CT retraining.
"""

from pathlib import Path
import time
import argparse
from datetime import datetime, timezone
import pandas as pd

from prefect import task, flow
from feast import FeatureStore

from src.config import settings
from src.utils.logger import get_logger
from src.offline.batch_feature_job import run_batch_feature_pipeline
from src.quality.feature_monitoring import generate_feature_drift_report
from src.ml.train import train_ensemble_pipeline

logger = get_logger(__name__)


@task(name="Batch Lakehouse & DQ Gate Engine", retries=2, retry_delay_seconds=5)
def task_batch_lakehouse() -> dict:
    """Step 1: Executes DuckDB Batch Feature Pipeline & Data Quality Gate."""
    logger.info("[STEP 1/4] Running Batch Feature Job & Data Quality Gate")
    partition_file = run_batch_feature_pipeline()
    logger.info(f"Step 1 PASSED: Generated Partitioned Parquet Lakehouse at '{partition_file}'")
    return {"status": "SUCCESS", "partition_file": partition_file}


@task(name="Feast Materialization Sync")
def task_feast_materialize() -> dict:
    """Step 2: Materializes latest batch features from Parquet into Redis Online Store via Feast SDK."""
    logger.info("[STEP 2/4] Triggering Feast Materialization into Redis")
    store = FeatureStore(repo_path=settings.feature_repo_dir)
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    store.materialize(start_date=start_date, end_date=end_date, feature_views=["card_batch_features"])
    logger.info("Step 2 PASSED: Feast Materialization synchronized successfully.")
    return {"status": "SUCCESS"}


@task(name="Feature Monitoring & Data Drift (Evidently AI)")
def task_feature_monitoring() -> dict:
    """Step 3: Runs Data Drift Detection using Evidently AI and generates HTML Dashboard report."""
    logger.info("[STEP 3/4] Running Evidently AI Feature Monitoring & Data Drift Analysis")
    try:
        lakehouse_dir = Path(settings.lakehouse_base_dir)
        baseline_path = lakehouse_dir / "baseline" / "reference_baseline.parquet"

        # Load reference baseline dataset
        if baseline_path.exists():
            df_reference = pd.read_parquet(baseline_path)
        elif Path(settings.batch_parquet_path).exists():
            df_reference = pd.read_parquet(settings.batch_parquet_path).head(5000)
        else:
            df_reference = None

        # Find recent current batch feature partitions (sorted by modification time)
        batch_dir = lakehouse_dir / "batch_features"
        batch_files = sorted(
            [p for p in batch_dir.rglob("*.parquet") if "baseline" not in str(p)],
            key=lambda p: p.stat().st_mtime
        ) if batch_dir.exists() else []

        if batch_files and df_reference is not None:
            df_current = pd.read_parquet(batch_files[-1])
            res = generate_feature_drift_report(df_reference, df_current)
            return res
        elif Path(settings.batch_parquet_path).exists() and df_reference is not None:
            df_current = pd.read_parquet(settings.batch_parquet_path)
            res = generate_feature_drift_report(df_reference, df_current)
            return res
        return {"status": "SKIPPED", "reason": "Missing Lakehouse baseline or current batch feature parquet files"}
    except Exception as e:
        logger.warning(f"Could not run Evidently AI Feature Monitoring: {e}")
        return {"status": "SKIPPED", "error": str(e)}


@task(name="Model Ensemble Training & Evaluation (Hybrid CT)")
def task_model_training(trigger_reason: str = "SCHEDULED_WEEKLY_CRON") -> dict:
    """Step 4: Trains or Retrains Model Ensemble (XGBoost + LightGBM Blending)."""
    logger.info(f"[STEP 4/4] Executing Model Ensemble Training (CT Trigger Reason: '{trigger_reason}')")
    try:
        report = train_ensemble_pipeline()
        logger.info(f"Step 4 PASSED: Model Ensemble Trained via '{trigger_reason}' (PR-AUC: {report['metrics']['ensemble']['pr_auc']}, ROC-AUC: {report['metrics']['ensemble']['roc_auc']})")
        return {"status": "SUCCESS", "report": report, "trigger_reason": trigger_reason}
    except Exception as e:
        logger.warning(f"Could not execute Model Ensemble Training: {e}")
        return {"status": "FAILED", "error": str(e)}


@flow(name="Realtime Feature Store Pipeline", log_prints=True)
def realtime_feature_store_flow(
    force_retrain: bool = False,
    scheduled_run: bool = False
) -> dict:
    """Prefect Flow orchestrating DE Hardening & Hybrid CT (Time-based Cron + Data Drift Event-Driven)."""
    logger.info("STARTING PREFECT REALTIME FEATURE STORE & HYBRID CT PIPELINE")

    start_time = time.time()
    pipeline_report = {}

    res_step1 = task_batch_lakehouse()
    pipeline_report["step1_batch_lakehouse"] = res_step1["status"]

    res_step2 = task_feast_materialize()
    pipeline_report["step2_feast_materialize"] = res_step2["status"]

    # Step 3: Data Drift Feature Monitoring
    res_step3 = task_feature_monitoring()
    drift_detected = res_step3.get("dataset_drift", False) if isinstance(res_step3, dict) else False
    pipeline_report["step3_evidently_drift_monitoring"] = res_step3.get("status", "SUCCESS") if isinstance(res_step3, dict) else "SUCCESS"
    pipeline_report["dataset_drift_detected"] = drift_detected

    # Continuous Training (CT) decision logic
    if scheduled_run or force_retrain or drift_detected:
        trigger_reason = "SCHEDULED_TIME_BASED_CRON" if (scheduled_run or force_retrain) else "EVENT_DRIVEN_DATA_DRIFT_DETECTED"
        logger.info(f"[HYBRID CT DECISION: RETRAIN REQUIRED] Reason: '{trigger_reason}'")
        res_step4 = task_model_training(trigger_reason=trigger_reason)
        pipeline_report["step4_model_ensemble_training"] = res_step4.get("status", "SUCCESS")
        pipeline_report["ct_trigger_reason"] = trigger_reason
    else:
        logger.info("[HYBRID CT DECISION: NO RETRAIN NEEDED] Skipping Model Retraining.")
        pipeline_report["step4_model_ensemble_training"] = "SKIPPED (No Drift & Not Scheduled)"
        pipeline_report["ct_trigger_reason"] = "SKIPPED"



    elapsed = time.time() - start_time
    logger.info(f"PREFECT HYBRID CT PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    for k, v in pipeline_report.items():
        logger.info(f"  • {k}: {v}")

    return pipeline_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect Realtime Feature Store Orchestrator")
    parser.add_argument("--serve", action="store_true", help="Serve flow with Prefect Deployment daemon")
    parser.add_argument("--cron", type=str, default="0 0 * * 0", help="Cron schedule for Prefect deployment (Default: Weekly on Sunday 00:00)")
    parser.add_argument("--force-retrain", action="store_true", help="Force time-based CT model retraining regardless of data drift")
    parser.add_argument("--scheduled-run", action="store_true", help="Mark run as scheduled time-based cron CT run")
    args = parser.parse_args()

    if args.serve:
        logger.info(f"Serving Prefect Hybrid CT Flow deployment with Weekly Cron schedule '{args.cron}'...")
        realtime_feature_store_flow.serve(
            name="weekly-hybrid-ct-feature-store-pipeline",
            cron=args.cron
        )
    else:
        realtime_feature_store_flow(
            force_retrain=args.force_retrain,
            scheduled_run=args.scheduled_run
        )
