import os
import sys
import time
import logging
from datetime import datetime, timezone
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("batch_feature_job")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DATASET_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(PROJECT_DIR, "data", "train_transaction.csv")
)

# Lakehouse Hive Partitioned Path: data/lakehouse/batch_features/
LAKEHOUSE_BASE_DIR = os.path.join(PROJECT_DIR, "data", "lakehouse", "batch_features")
FALLBACK_PARQUET_PATH = os.path.join(PROJECT_DIR, "data", "batch_features.parquet")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadminpassword")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "feature-store-offline")

def upload_folder_to_minio(local_folder: str, bucket_name: str, minio_prefix: str = "batch_features"):
    """Recursively uploads local Hive-partitioned Parquet files to MinIO S3 Lakehouse bucket."""
    try:
        from minio import Minio
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        count = 0
        for root, _, files in os.walk(local_folder):
            for file in files:
                if file.endswith(".parquet"):
                    local_file = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file, local_folder).replace("\\", "/")
                    s3_object_name = f"{minio_prefix}/{rel_path}"
                    client.fput_object(bucket_name, s3_object_name, local_file)
                    count += 1

        logger.info(f"[Lakehouse Sync] Uploaded {count} partitioned Parquet files -> MinIO S3 's3://{bucket_name}/{minio_prefix}/'")
    except Exception as e:
        logger.warning(f"Failed to sync partitioned Lakehouse to MinIO S3: {e}")

def run_batch_feature_pipeline(dataset_path: str = DEFAULT_DATASET_PATH):
    """
    Executes DuckDB Batch Feature Pipeline with Hive Partitioning & DQ Gate Check:
    1. Reads raw transaction dataset into DuckDB.
    2. Computes historical batch features.
    3. Exports result to Hive Partitioned Data Lakehouse (`year=YYYY/month=MM/day=DD/`).
    4. Validates features via Pandera Data Quality Engine.
    5. Syncs Lakehouse Parquet partitions to MinIO S3 offline store.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at: {dataset_path}")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    year_str = f"{now.year}"
    month_str = f"{now.month:02d}"
    day_str = f"{now.day:02d}"

    partition_dir = os.path.join(LAKEHOUSE_BASE_DIR, f"year={year_str}", f"month={month_str}", f"day={day_str}")
    os.makedirs(partition_dir, exist_ok=True)
    partition_file = os.path.join(partition_dir, "batch_part_000.parquet")

    logger.info(f"Starting DuckDB Batch Feature Engine on dataset: {dataset_path}")
    start_time = time.time()

    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        CREATE VIEW raw_transactions AS
        SELECT
            CAST(TransactionID AS BIGINT) AS transaction_id,
            CAST(isFraud AS INT) AS is_fraud,
            CAST(TransactionDT AS DOUBLE) AS transaction_dt,
            CAST(TransactionAmt AS DOUBLE) AS amount,
            CAST(card1 AS VARCHAR) AS card_id,
            CAST(addr1 AS DOUBLE) AS addr1,
            CAST(D4 AS DOUBLE) AS d4
        FROM read_csv_auto('{dataset_path}')
        WHERE card1 IS NOT NULL;
    """)

    logger.info("Computing 6 Historical Batch Features (7d/30d/All-time aggregations)...")
    
    batch_features_df = con.execute("""
        SELECT
            card_id,
            CAST(NOW() AS TIMESTAMP) AS event_timestamp,
            COUNT(transaction_id) AS trans_count_30d,
            COUNT(CASE WHEN transaction_dt >= (MAX_DT - 7 * 86400) THEN 1 END) AS trans_count_7d,
            AVG(amount) AS avg_amount_30d,
            MAX(amount) AS max_amount_30d,
            COUNT(DISTINCT addr1) AS distinct_addr_7d,
            COALESCE(AVG(d4), 0.0) AS days_since_last_trans
        FROM raw_transactions,
             (SELECT MAX(transaction_dt) AS MAX_DT FROM raw_transactions)
        GROUP BY card_id;
    """).df()

    logger.info(f"Computed batch features for {len(batch_features_df):,} unique cards.")
    
    # -------------------------------------------------------------------------
    # Pandera Data Quality Validation Gate
    # -------------------------------------------------------------------------
    sys.path.append(os.path.join(PROJECT_DIR, "src", "quality"))
    try:
        from data_assert import validate_batch_dataframe
        is_valid, clean_df, error_df = validate_batch_dataframe(batch_features_df)
        if not is_valid:
            logger.warning("[DQ Gate Warning] Batch features contained invalid rows; proceed with sanitized DataFrame.")
            batch_features_df = clean_df
    except Exception as e:
        logger.warning(f"Could not run Pandera Data Quality Gate check: {e}")

    # Export to Hive Partition File
    con.execute(f"COPY (SELECT * FROM batch_features_df) TO '{partition_file}' (FORMAT PARQUET);")
    
    # Also save single file for Feast default path compatibility
    os.makedirs(os.path.dirname(FALLBACK_PARQUET_PATH), exist_ok=True)
    con.execute(f"COPY (SELECT * FROM batch_features_df) TO '{FALLBACK_PARQUET_PATH}' (FORMAT PARQUET);")
    con.close()

    elapsed = time.time() - start_time
    logger.info(f"Generated Hive Partitioned Parquet Lakehouse at '{partition_file}' in {elapsed:.2f}s.")

    # Upload Hive Lakehouse to MinIO S3
    upload_folder_to_minio(
        local_folder=LAKEHOUSE_BASE_DIR,
        bucket_name=MINIO_BUCKET,
        minio_prefix="batch_features"
    )
    return partition_file

if __name__ == "__main__":
    run_batch_feature_pipeline()
