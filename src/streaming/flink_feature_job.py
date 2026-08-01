"""
PyFlink Real-Time Stream Feature Processing Engine & Dual-Path Sink.

Consumes transaction stream, computes sliding window aggregations (1m, 5m, 1h, 24h),
pushes online features to Redis (< 5ms SLA), archives raw stream to Lakehouse Parquet,
and routes corrupt payloads to DLQ side output.
"""

import time
import json
import uuid
import argparse
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from src.config import settings
from src.utils.logger import get_logger
from src.utils.redis_client import get_redis_client
from src.utils.minio_client import upload_file_to_minio

logger = get_logger(__name__)


def register_kafka_source(t_env: Any) -> None:
    """Registers Kafka Source Table using Flink SQL DDL with 5s Event-Time Watermarking."""
    kafka_source_ddl = f"""
        CREATE TABLE raw_transactions_kafka (
            transaction_id BIGINT,
            is_fraud INT,
            transaction_dt DOUBLE,
            `timestamp` STRING,
            card_id STRING,
            amount DOUBLE,
            product_cd STRING,
            card_type STRING,
            card_category STRING,
            p_emaildomain STRING,
            addr1 DOUBLE,
            c1 DOUBLE,
            c2 DOUBLE,
            event_time AS TO_TIMESTAMP(`timestamp`),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{settings.kafka_topic}',
            'properties.bootstrap.servers' = '{settings.kafka_broker}',
            'properties.group.id' = 'pyflink_feature_table_group',
            'scan.startup.mode' = 'earliest-offset',
            'format' = 'json'
        );
    """
    t_env.execute_sql(kafka_source_ddl)
    logger.info("Registered Kafka source table with 5s Watermark.")


def transform_stream(t_env: Any) -> Any:
    """
    Applies Data Quality (DQ) Gateways, Business Rule Assertions, 
    and Schema Normalization using PyFlink Table API.
    """
    from pyflink.table.expressions import col

    transactions = t_env.from_path("raw_transactions_kafka")


    valid_transactions = transactions.filter(
        (col("card_id").is_not_null) &
        (col("card_id") != "") &
        (col("card_id") != "unknown_card") &
        (col("amount").is_not_null) &
        (col("amount") > 0.0) &
        (col("amount") <= 500000.0) &
        (col("transaction_id") > 0)
    )

    sanitized_table = valid_transactions.select(
        col("transaction_id"),
        col("card_id"),
        col("amount"),
        col("c1"),
        col("c2"),
        col("product_cd"),
        col("p_emaildomain"),
        col("timestamp"),
        col("event_time")
    )
    t_env.create_temporary_view("valid_transactions_stream", sanitized_table)
    logger.info("Applied Data Quality Gateways & registered 'valid_transactions_stream' in PyFlink Table API.")
    return sanitized_table


def setup_flink_environment() -> tuple[Any, Any]:
    """Initializes PyFlink StreamTableEnvironment and loads Kafka SQL connector JAR."""
    try:
        from pyflink.datastream import StreamExecutionEnvironment
        from pyflink.table import StreamTableEnvironment, EnvironmentSettings
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "PyFlink is required when running with --engine flink, but is not installed. "
            "Install apache-flink and ensure Java 11 JDK is available."
        ) from exc

    try:
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(1)
        settings_obj = EnvironmentSettings.new_instance().in_streaming_mode().build()
        t_env = StreamTableEnvironment.create(env, environment_settings=settings_obj)

        jar_dir = Path(__file__).resolve().parent / "jars"
        jar_files = list(jar_dir.glob("*.jar"))
        if jar_files:
            jar_uris = [f"file:///{p.as_posix().lstrip('/')}" for p in jar_files]
            jar_config_val = ";".join(jar_uris)
            t_env.get_config().get_configuration().set_string("pipeline.jars", jar_config_val)
            logger.info(f"Loaded {len(jar_files)} Flink SQL Connector JAR(s) into pipeline: {[p.name for p in jar_files]}")

        return env, t_env
    except Exception as e:
        logger.error(f"Failed to initialize PyFlink environment: {e}")
        return None, None


def register_feature_sink_and_execute(t_env: Any) -> None:
    """Defines feature sink and submits streaming sliding window aggregation job on Flink."""
    sink_ddl = """
        CREATE TABLE IF NOT EXISTS card_stream_features_sink (
            card_id STRING,
            window_start TIMESTAMP(3),
            window_end TIMESTAMP(3),
            trans_count_1h BIGINT,
            total_amount_1h DOUBLE,
            avg_amount_1h DOUBLE,
            c2_count_sum_1h DOUBLE,
            PRIMARY KEY (card_id) NOT ENFORCED
        ) WITH (
            'connector' = 'print'
        );
    """
    t_env.execute_sql(sink_ddl)
    logger.info("Registered Flink Feature Sink Table 'card_stream_features_sink'.")

    insert_sql = """
        INSERT INTO card_stream_features_sink
        SELECT
            card_id,
            HOP_START(event_time, INTERVAL '1' MINUTE, INTERVAL '1' HOUR) AS window_start,
            HOP_END(event_time, INTERVAL '1' MINUTE, INTERVAL '1' HOUR) AS window_end,
            COUNT(transaction_id) AS trans_count_1h,
            SUM(amount) AS total_amount_1h,
            AVG(amount) AS avg_amount_1h,
            SUM(c2) AS c2_count_sum_1h
        FROM valid_transactions_stream
        GROUP BY card_id, HOP(event_time, INTERVAL '1' MINUTE, INTERVAL '1' HOUR);
    """
    logger.info("Submitting Flink SQL Hopping Window Job to Execution Engine...")
    t_env.execute_sql(insert_sql)


def upload_to_minio(file_path: str | Path, object_name: str) -> bool:
    """Uploads Parquet archive file to MinIO S3 Cold Path / DLQ Bucket using centralized client."""
    return upload_file_to_minio(file_path, object_name, bucket_name=settings.minio_bucket)


class DualPathRedisFeatureSink:
    """
    Dual-Path Feature Sink with DLQ Side Output Isolation:
    1. Hot Path: Computes rolling sliding window features and pushes valid events to Redis Online Store (< 5ms serving).
    2. Cold Path: Appends raw event stream to Parquet Data Lake & syncs to MinIO S3 for long-term historical retention.
    3. DLQ Routing: Catches invalid/corrupt events, pushes to Kafka Topic 'raw_transactions_dlq' and isolates into S3 DLQ Parquet.
    """
    def __init__(self, redis_client=None, dlq_producer=None):
        self.redis_client = redis_client or get_redis_client()
        self.dlq_producer = dlq_producer
        self.windows = defaultdict(deque)
        self.raw_events_buffer = []
        self.dlq_events_buffer = []

    def is_valid_event(self, event: dict) -> bool:
        card_id = str(event.get("card_id", ""))
        amount = event.get("amount")
        trans_id = event.get("transaction_id")
        
        if not card_id or card_id.strip() == "" or card_id == "unknown_card":
            return False
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0.0 or amount > 500000.0:
            return False
        if trans_id is None or trans_id <= 0:
            return False
        return True

    def route_to_dlq(self, event: dict, reason: str):
        """Pushes invalid event to Kafka DLQ topic & records error buffer."""
        event_with_err = dict(event)
        event_with_err["error_reason"] = reason
        event_with_err["quarantine_timestamp"] = datetime.now(timezone.utc).isoformat()

        self.dlq_events_buffer.append(event_with_err)

        if self.dlq_producer:
            try:
                self.dlq_producer.send(
                    settings.kafka_dlq_topic,
                    key=str(event.get("card_id", "bad_event")),
                    value=event_with_err
                )
            except Exception as e:
                logger.warning(f"Failed to publish event to DLQ Kafka topic: {e}")

        logger.warning(f"[DLQ Isolation] Quarantined corrupt event (Card: '{event.get('card_id')}', Amt: {event.get('amount')}) Reason: {reason}")

    def process_event(self, event: dict) -> dict | None:
        if not self.is_valid_event(event):
            self.route_to_dlq(event, "Failed DQ Gate assertions (Invalid amount, missing or corrupt card_id)")
            return None

        card_id = str(event["card_id"])
        amount = float(event.get("amount", 0.0))
        c2 = float(event.get("c2", 0.0))

        ts_str = str(event.get("timestamp", ""))
        try:
            event_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            event_ts = event_dt.timestamp()
        except Exception:
            event_dt = datetime.now(timezone.utc)
            event_ts = time.time()

        self.raw_events_buffer.append({
            "transaction_id": int(event.get("transaction_id", 0)),
            "card_id": card_id,
            "amount": amount,
            "c1": float(event.get("c1", 0.0)),
            "c2": c2,
            "product_cd": str(event.get("product_cd", "")),
            "p_emaildomain": str(event.get("p_emaildomain", "")),
            "event_timestamp": event_dt
        })

        card_history = self.windows[card_id]
        card_history.append((event_ts, amount, c2))

        # Evict records older than 24h (86400s)
        cutoff_24h = event_ts - 86400
        while card_history and card_history[0][0] < cutoff_24h:
            card_history.popleft()

        cutoff_1m = event_ts - 60
        cutoff_5m = event_ts - 300
        cutoff_1h = event_ts - 3600

        events_1m = [e for e in card_history if e[0] >= cutoff_1m]
        events_5m = [e for e in card_history if e[0] >= cutoff_5m]
        events_1h = [e for e in card_history if e[0] >= cutoff_1h]
        events_24h = list(card_history)

        trans_count_1m = len(events_1m)
        trans_count_5m = len(events_5m)
        trans_count_1h = len(events_1h)

        amounts_1h = [e[1] for e in events_1h]
        total_amount_1h = sum(amounts_1h)
        avg_amount_1h = total_amount_1h / len(amounts_1h) if amounts_1h else 0.0
        c2_count_sum_1h = sum(e[2] for e in events_1h)

        amounts_24h = [e[1] for e in events_24h]
        avg_amount_24h = sum(amounts_24h) / len(amounts_24h) if amounts_24h else 0.0
        max_amount_24h = max(amounts_24h) if amounts_24h else 0.0
        stddev_amount_24h = round(statistics.pstdev(amounts_24h), 2) if len(amounts_24h) > 1 else 0.0

        feature_vector = {
            "card_id": card_id,
            "trans_count_1m": trans_count_1m,
            "trans_count_5m": trans_count_5m,
            "trans_count_1h": trans_count_1h,
            "total_amount_1h": round(total_amount_1h, 2),
            "avg_amount_1h": round(avg_amount_1h, 2),
            "avg_amount_24h": round(avg_amount_24h, 2),
            "stddev_amount_24h": round(stddev_amount_24h, 2),
            "max_amount_24h": round(max_amount_24h, 2),
            "c2_count_sum_1h": round(c2_count_sum_1h, 2),
            "last_updated": event_dt.isoformat()
        }

        redis_key = f"card:{card_id}:stream_features"
        try:
            self.redis_client.hset(redis_key, mapping={k: str(v) for k, v in feature_vector.items()})
        except Exception as e:
            logger.warning(f"Failed to update Redis key {redis_key}: {e}")

        return feature_vector


    def flush_cold_path_archive(self) -> bool:
        """Flushes buffered raw events and DLQ to Partitioned Parquet Lakehouse & MinIO S3 Storage."""
        now_dt = datetime.now(timezone.utc)
        success = True

        if self.raw_events_buffer:
            try:
                batch_id = uuid.uuid4().hex[:8]
                date_part = f"year={now_dt.year}/month={now_dt.month:02d}/day={now_dt.day:02d}"
                raw_dir = Path(settings.lakehouse_base_dir) / "raw_events" / date_part
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_filename = f"events_{now_dt.strftime('%Y%m%d_%H%M%S')}_{batch_id}.parquet"
                raw_path = raw_dir / raw_filename

                df_new = pd.DataFrame(self.raw_events_buffer)
                df_new.to_parquet(raw_path, index=False)

                # Also update latest consolidated snapshot if configured
                base_raw_path = Path(settings.raw_events_parquet_path)
                base_raw_path.parent.mkdir(parents=True, exist_ok=True)
                df_new.to_parquet(base_raw_path, index=False)

                logger.info(f"[Cold Path Archival] Saved {len(df_new):,} raw events to '{raw_path}'")
                minio_key = f"raw_events/{date_part}/{raw_filename}"
                uploaded = upload_to_minio(raw_path, minio_key)
                if uploaded:
                    self.raw_events_buffer.clear()
                else:
                    logger.warning("[Cold Path Archival] MinIO upload failed; clearing buffer to prevent unbounded RAM growth after disk save.")
                    self.raw_events_buffer.clear()
                    success = False
            except Exception as e:
                logger.error(f"[Cold Path Archival Error] Failed to persist raw stream: {e}")
                success = False

        if self.dlq_events_buffer:
            try:
                batch_id = uuid.uuid4().hex[:8]
                date_part = f"year={now_dt.year}/month={now_dt.month:02d}/day={now_dt.day:02d}"
                dlq_dir = Path(settings.lakehouse_base_dir) / "dlq" / date_part
                dlq_dir.mkdir(parents=True, exist_ok=True)
                dlq_filename = f"dlq_{now_dt.strftime('%Y%m%d_%H%M%S')}_{batch_id}.parquet"
                dlq_path = dlq_dir / dlq_filename

                df_dlq = pd.DataFrame(self.dlq_events_buffer)
                df_dlq.to_parquet(dlq_path, index=False)

                base_dlq_path = Path(settings.stream_dlq_parquet_path)
                base_dlq_path.parent.mkdir(parents=True, exist_ok=True)
                df_dlq.to_parquet(base_dlq_path, index=False)

                logger.info(f"[DLQ Archival] Saved {len(df_dlq)} quarantined events to '{dlq_path}'")
                minio_dlq_key = f"dlq/{date_part}/{dlq_filename}"
                uploaded = upload_to_minio(dlq_path, minio_dlq_key)
                if uploaded:
                    self.dlq_events_buffer.clear()
                else:
                    logger.warning("[DLQ Archival] MinIO DLQ upload failed; clearing buffer to prevent unbounded RAM growth after disk save.")
                    self.dlq_events_buffer.clear()
                    success = False
            except Exception as e:
                logger.error(f"[DLQ Archival Error] Failed to persist DLQ stream: {e}")
                success = False

        return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Feature Streaming Engine")
    parser.add_argument("--engine", type=str, choices=["direct", "flink"], default=settings.stream_engine, help="Execution engine mode")
    args = parser.parse_args()

    if args.engine == "flink":
        logger.info("Initializing Real PyFlink Table API Stream Processing Engine...")
        env, t_env = setup_flink_environment()
        if t_env is None:
            logger.error("Cannot run in Flink mode without PyFlink installed and Java JDK environment configured.")
            return

        register_kafka_source(t_env)
        transform_stream(t_env)
        register_feature_sink_and_execute(t_env)
        logger.info("Flink Streaming Job submitted successfully.")
        return

    # Direct Python Stream Processor
    dlq_producer = None
    try:
        dlq_producer = KafkaProducer(
            bootstrap_servers=[settings.kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None
        )
    except Exception as e:
        logger.warning(f"Kafka DLQ Producer init notice: {e}")

    logger.info(f"Starting Stream Processor with DLQ & Dual-Path Archival (Broker: {settings.kafka_broker}, Main: {settings.kafka_topic}, DLQ: {settings.kafka_dlq_topic})...")
    try:
        consumer = KafkaConsumer(
            settings.kafka_topic,
            bootstrap_servers=[settings.kafka_broker],
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=3000,
            group_id="pyflink_table_api_group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8"))
        )
        sink = DualPathRedisFeatureSink(dlq_producer=dlq_producer)
        processed_count = 0
        corrupt_count = 0
        start_time = time.time()

        for message in consumer:
            event = message.value
            features = sink.process_event(event)
            if features is None:
                corrupt_count += 1
            else:
                processed_count += 1

            if (processed_count + corrupt_count) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (processed_count + corrupt_count) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[Stream Engine] Processed: {processed_count} valid, {corrupt_count} corrupt | Rate: {rate:.1f} msg/s"
                )

        sink.flush_cold_path_archive()
        elapsed = time.time() - start_time
        logger.info(f"Stream Job completed. Valid: {processed_count}, Corrupt: {corrupt_count} in {elapsed:.2f}s.")
    except Exception as e:
        logger.error(f"Stream Engine encountered fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
