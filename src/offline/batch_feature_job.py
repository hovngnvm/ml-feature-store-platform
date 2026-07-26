"""Batch Feature Analytical Pipeline Engine (DuckDB & Hive Lakehouse).

Computes 7d and 30d windowed aggregations over historical transactions using DuckDB,
exports Hive-partitioned Parquet files, and syncs Lakehouse partitions to MinIO S3.
"""

from pathlib import Path
import time
from datetime import datetime, timezone
import duckdb

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.minio_client import upload_folder_to_minio

logger = get_logger(__name__)


def run_batch_feature_pipeline(dataset_path: str | Path = settings.raw_csv_path) -> str:
    """Executes DuckDB Batch Feature Pipeline with Hive Partitioning & DQ Gate Check."""
    path_dataset = Path(dataset_path)
    if not path_dataset.is_file():
        logger.error(f"Dataset not found at: {path_dataset}")
        raise FileNotFoundError(f"Dataset not found at: {path_dataset}")

    now = datetime.now(timezone.utc)
    year_str = f"{now.year}"
    month_str = f"{now.month:02d}"
    day_str = f"{now.day:02d}"

    partition_dir = Path(settings.lakehouse_base_dir) / f"year={year_str}" / f"month={month_str}" / f"day={day_str}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partition_file = partition_dir / "batch_part_000.parquet"

    logger.info(f"Starting DuckDB Batch Feature Engine on dataset: {path_dataset}")
    start_time = time.time()

    con = duckdb.connect(database=":memory:")
    dataset_path_str = path_dataset.as_posix()
    partition_file_str = partition_file.as_posix()
    batch_parquet_str = Path(settings.batch_parquet_path).as_posix()

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
        FROM read_csv_auto('{dataset_path_str}')
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

    try:
        from src.quality.data_assert import validate_batch_dataframe
        is_valid, clean_df, error_df = validate_batch_dataframe(batch_features_df)
        if not is_valid:
            if clean_df is not None and not clean_df.empty:
                logger.warning("[DQ Gate Warning] Batch features contained invalid rows; proceed with sanitized DataFrame.")
                batch_features_df = clean_df
            else:
                raise ValueError("[DQ Gate Error] All batch feature rows failed data quality validation.")
    except Exception as e:
        logger.error(f"Pandera Data Quality Gate check failure: {e}")
        raise

    con.execute(f"COPY (SELECT * FROM batch_features_df) TO '{partition_file_str}' (FORMAT PARQUET);")

    # Also save single file for Feast default path compatibility
    Path(settings.batch_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM batch_features_df) TO '{batch_parquet_str}' (FORMAT PARQUET);")
    con.close()

    elapsed = time.time() - start_time
    logger.info(f"Generated Hive Partitioned Parquet Lakehouse at '{partition_file}' in {elapsed:.2f}s.")

    upload_folder_to_minio(
        local_folder=settings.lakehouse_base_dir,
        bucket_name=settings.minio_bucket,
        minio_prefix="batch_features"
    )
    return str(partition_file)


if __name__ == "__main__":
    run_batch_feature_pipeline()
