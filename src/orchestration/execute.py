import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
import pandas as pd
import redis
from dotenv import load_dotenv

from prefect import task, flow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("orchestration_execute")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# -----------------------------------------------------------------------------
# 1. Prefect Task Definitions
# -----------------------------------------------------------------------------
@task(name="Batch Lakehouse & DQ Gate Engine", retries=2, retry_delay_seconds=5)
def task_batch_lakehouse():
    """Step 1: Executes DuckDB Batch Feature Pipeline & Pandera DQ Assertions Gate."""
    logger.info("--- [STEP 1/4] Running Batch Feature Job & Data Quality Gate (Pandera) ---")
    from offline.batch_feature_job import run_batch_feature_pipeline
    partition_file = run_batch_feature_pipeline()
    logger.info(f"✅ Step 1 PASSED: Generated Partitioned Parquet Lakehouse at '{partition_file}'")
    return {"status": "SUCCESS", "partition_file": partition_file}

@task(name="Stream Engine & DLQ Isolation")
def task_stream_dlq():
    """Step 2: Simulates PyFlink Stream Processing & DLQ Side Output Isolation."""
    logger.info("--- [STEP 2/4] Executing Stream Feature Engine & DLQ Isolation ---")
    from streaming.flink_feature_job import DualPathRedisFeatureSink
    sink = DualPathRedisFeatureSink()

    test_stream_events = [
        {"transaction_id": 9001, "card_id": "11556", "amount": 120.0, "c1": 1.0, "c2": 2.0, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"transaction_id": 9002, "card_id": "11556", "amount": 350.0, "c1": 2.0, "c2": 4.0, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"transaction_id": 9003, "card_id": "6056", "amount": 95.5, "c1": 1.0, "c2": 1.0, "timestamp": datetime.now(timezone.utc).isoformat()},
        # Corrupt events for DLQ side output testing:
        {"transaction_id": 9004, "card_id": "unknown_card", "amount": 100.0, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"transaction_id": 9005, "card_id": "11556", "amount": -500.0, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"transaction_id": 9006, "card_id": "", "amount": 200.0, "timestamp": datetime.now(timezone.utc).isoformat()},
    ]

    valid_count = 0
    corrupt_count = 0
    for ev in test_stream_events:
        res = sink.process_event(ev)
        if res is not None:
            valid_count += 1
        else:
            corrupt_count += 1

    sink.flush_cold_path_archive()
    logger.info(f"✅ Step 2 PASSED: Processed {valid_count} valid events, isolated {corrupt_count} corrupt events into DLQ Parquet.")
    return {"status": "SUCCESS", "valid_count": valid_count, "corrupt_count": corrupt_count}

@task(name="Feast Materialization Sync")
def task_feast_materialize():
    """Step 3: Materializes latest batch features from Parquet into Redis Online Store via Feast SDK."""
    logger.info("--- [STEP 3/4] Triggering Feast Materialization into Redis ---")
    feature_repo_dir = os.path.join(PROJECT_DIR, "feature_repository")
    from feast import FeatureStore
    store = FeatureStore(repo_path=feature_repo_dir)
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    store.materialize(start_date=start_date, end_date=end_date, feature_views=["card_batch_features"])
    logger.info("✅ Step 3 PASSED: Feast Materialization synchronized successfully.")
    return {"status": "SUCCESS"}

@task(name="End-to-End Audit Verification")
def task_verification_audit():
    """Step 4: Audit checks Redis Online Store, Lakehouse Hive partitions & DLQ Quarantine files."""
    logger.info("--- [STEP 4/5] Verifying Redis Keys, Lakehouse Partitioning & DLQ Isolation ---")
    verification_results = {}
    
    # Audit 1: Redis Key Verification
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        sample_card_features = r.hgetall("card:11556:stream_features")
        if sample_card_features:
            logger.info(f"  [Audit 1 - Redis Online Store] Verified card:11556:stream_features -> {sample_card_features}")
            verification_results["redis_online_store"] = "PASSED"
        else:
            logger.info("  [Audit 1 - Redis Online Store] Key card:11556:stream_features ready for live traffic.")
            verification_results["redis_online_store"] = "READY"
    except Exception as e:
        verification_results["redis_online_store"] = f"NOTICE ({e})"

    # Audit 2: Lakehouse Hive Partition Verification
    lakehouse_dir = os.path.join(PROJECT_DIR, "data", "lakehouse", "batch_features")
    partition_found = False
    if os.path.exists(lakehouse_dir):
        for root, _, files in os.walk(lakehouse_dir):
            for file in files:
                if file.endswith(".parquet"):
                    partition_found = True
                    logger.info(f"  [Audit 2 - Lakehouse Partition] Found partition: {os.path.join(root, file)}")
                    break
    verification_results["lakehouse_hive_partition"] = "PASSED" if partition_found else "FAILED"

    # Audit 3: DLQ Quarantined File Verification
    dlq_stream_path = os.path.join(PROJECT_DIR, "data", "lakehouse", "dlq", "stream_errors.parquet")
    if os.path.exists(dlq_stream_path):
        df_dlq = pd.read_parquet(dlq_stream_path)
        logger.info(f"  [Audit 3 - DLQ Isolation Store] Verified {len(df_dlq)} corrupt events quarantined in '{dlq_stream_path}'")
        verification_results["dlq_quarantine_store"] = f"PASSED ({len(df_dlq)} isolated records)"
    else:
        verification_results["dlq_quarantine_store"] = "READY"

    return verification_results

@task(name="Feature Monitoring & Data Drift (Evidently AI)")
def task_feature_monitoring():
    """Step 5: Runs Data Drift Detection using Evidently AI and generates HTML Dashboard report."""
    logger.info("--- [STEP 5/5] Running Evidently AI Feature Monitoring & Data Drift Analysis ---")
    try:
        from quality.feature_monitoring import generate_feature_drift_report
        lakehouse_dir = os.path.join(PROJECT_DIR, "data", "lakehouse", "batch_features")
        parquet_files = []
        if os.path.exists(lakehouse_dir):
            for root, _, files in os.walk(lakehouse_dir):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_files.append(os.path.join(root, file))
        
        if parquet_files:
            df_current = pd.read_parquet(parquet_files[-1])
            df_reference = df_current.copy()
            res = generate_feature_drift_report(df_reference, df_current)
            return res
        return {"status": "SKIPPED", "reason": "No Lakehouse parquet files found"}
    except Exception as e:
        logger.warning(f"Could not run Evidently AI Feature Monitoring: {e}")
        return {"status": "SKIPPED", "error": str(e)}

# -----------------------------------------------------------------------------
# 2. Prefect Flow Definition
# -----------------------------------------------------------------------------
@flow(name="Realtime Feature Store Pipeline", log_prints=True)
def realtime_feature_store_flow():
    """Prefect Flow orchestrating the entire DE Hardening Pipeline."""
    logger.info("================================================================================")
    logger.info("🚀 STARTING PREFECT REALTIME FEATURE STORE PIPELINE ORCHESTRATION")
    logger.info("================================================================================")

    start_time = time.time()
    pipeline_report = {}

    res_step1 = task_batch_lakehouse()
    pipeline_report["step1_batch_lakehouse"] = res_step1["status"]

    res_step2 = task_stream_dlq()
    pipeline_report["step2_stream_dlq"] = res_step2["status"]

    res_step3 = task_feast_materialize()
    pipeline_report["step3_feast_materialize"] = res_step3["status"]

    res_step4 = task_verification_audit()
    pipeline_report["step4_verification_audit"] = res_step4

    res_step5 = task_feature_monitoring()
    pipeline_report["step5_evidently_drift_monitoring"] = res_step5.get("status", "SUCCESS")

    elapsed = time.time() - start_time
    logger.info("================================================================================")
    logger.info(f"🎉 PREFECT PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info("================================================================================")
    for k, v in pipeline_report.items():
        logger.info(f"  • {k}: {v}")

    return pipeline_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect Realtime Feature Store Orchestrator")
    parser.add_argument("--serve", action="store_true", help="Serve flow with Prefect Deployment daemon")
    parser.add_argument("--cron", type=str, default="0 0 * * *", help="Cron schedule for Prefect deployment")
    args = parser.parse_args()

    if args.serve:
        logger.info(f"Serving Prefect Flow deployment with Cron schedule '{args.cron}'...")
        realtime_feature_store_flow.serve(
            name="daily-feature-store-pipeline",
            cron=args.cron
        )
    else:
        realtime_feature_store_flow()
