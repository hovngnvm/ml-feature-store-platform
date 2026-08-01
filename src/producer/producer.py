from pathlib import Path
import time
import json
import random
import argparse
from datetime import datetime, timezone
import pandas as pd
from kafka import KafkaProducer

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SECONDS_PER_DAY: int = 86400


def create_stream_producer(broker_address: str) -> KafkaProducer:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[broker_address],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
            acks="all",
            retries=3,
            linger_ms=10,
        )
        logger.info(f"Connected to Stream Broker (Redpanda) at {broker_address}")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Stream producer at {broker_address}: {e}")
        raise ConnectionError(f"Failed to create Stream producer at {broker_address}: {e}")


def format_event(row: pd.Series, base_time: float | None = None, inject_corrupt: bool = False) -> dict:
    if base_time is None:
        base_time = time.time()
    dt_offset = float(row.get("TransactionDT", 0))
    event_timestamp = datetime.fromtimestamp(base_time + (dt_offset % 86400), tz=timezone.utc).isoformat()

    card_id = str(int(row["card1"])) if pd.notnull(row.get("card1")) else "unknown_card"
    amount = float(row["TransactionAmt"]) if pd.notnull(row.get("TransactionAmt")) else 0.0

    event = {
        "transaction_id": int(row["TransactionID"]),
        "is_fraud": int(row["isFraud"]) if pd.notnull(row.get("isFraud")) else 0,
        "transaction_dt": dt_offset,
        "timestamp": event_timestamp,
        "card_id": card_id,
        "amount": amount,
        "product_cd": str(row.get("ProductCD", "W")),
        "card_type": str(row.get("card4", "unknown")),
        "card_category": str(row.get("card6", "unknown")),
        "p_emaildomain": str(row.get("P_emaildomain", "unknown")) if pd.notnull(row.get("P_emaildomain")) else "unknown",
        "addr1": float(row["addr1"]) if pd.notnull(row.get("addr1")) else 0.0,
        "c1": float(row["C1"]) if pd.notnull(row.get("C1")) else 0.0,
        "c2": float(row["C2"]) if pd.notnull(row.get("C2")) else 0.0,
    }

    if inject_corrupt:
        corrupt_type = random.choice(["invalid_amount", "invalid_card", "null_card"])
        if corrupt_type == "invalid_amount":
            event["amount"] = -999.0
        elif corrupt_type == "invalid_card":
            event["card_id"] = "unknown_card"
        else:
            event["card_id"] = ""
        event["is_fraud"] = 0
        event["card_type"] = "corrupt_test"

    return event


def stream_transactions(
    dataset_path: str = settings.raw_csv_path,
    broker: str = settings.kafka_broker,
    topic: str = settings.kafka_topic,
    limit: int | None = None,
    delay: float = 0.01,
    corrupt_rate: float = 0.0,
) -> None:
    path_obj = Path(dataset_path)
    if not path_obj.is_file():
        logger.error(f"Dataset file not found at: {dataset_path}")
        raise FileNotFoundError(f"Transaction dataset CSV not found at: '{dataset_path}'")

    logger.info(f"Loading dataset from: {dataset_path}")
    producer = create_stream_producer(broker)

    columns = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "ProductCD", "card1", "card4", "card6", "P_emaildomain", "addr1", "C1", "C2"
    ]

    start_time = time.time()
    total_sent = 0
    corrupt_sent = 0
    chunk_size = 5000

    try:
        for chunk in pd.read_csv(dataset_path, usecols=columns, chunksize=chunk_size):
            chunk = chunk.sort_values(by="TransactionDT")

            for _, row in chunk.iterrows():
                is_corrupt = (corrupt_rate > 0.0) and (random.random() < corrupt_rate)
                event = format_event(row, base_time=start_time, inject_corrupt=is_corrupt)
                card_id = event["card_id"]

                if is_corrupt:
                    corrupt_sent += 1

                producer.send(topic, key=card_id, value=event)
                total_sent += 1

                if total_sent % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = total_sent / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[DATASET] Sent {total_sent} events ({corrupt_sent} corrupt) | Rate: {rate:.1f} msg/s | "
                        f"Sample Card ID: {card_id} | Amount: ${event['amount']:.2f} | Fraud: {event['is_fraud']}"
                    )

                if limit and limit > 0 and total_sent >= limit:
                    logger.info(f"Reached event limit ({limit}). Stopping producer.")
                    producer.flush()
                    return

                if delay > 0:
                    time.sleep(delay)

        producer.flush()
        elapsed = time.time() - start_time
        logger.info(f"Completed streaming {total_sent} events ({corrupt_sent} corrupt) in {elapsed:.2f} seconds.")
    finally:
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Transaction Events into Redpanda")
    parser.add_argument("--dataset", type=str, default=settings.raw_csv_path, help="Path to dataset CSV")
    parser.add_argument("--broker", type=str, default=settings.kafka_broker, help="Stream broker address")
    parser.add_argument("--topic", type=str, default=settings.kafka_topic, help="Topic name")
    parser.add_argument("--limit", type=int, default=0, help="Max records to send (0 for all)")
    parser.add_argument("--delay", type=float, default=0.005, help="Delay between events in seconds")
    parser.add_argument("--corrupt-rate", type=float, default=0.0, help="Ratio of corrupt test events (0.0 to 1.0)")

    args = parser.parse_args()
    stream_transactions(
        dataset_path=args.dataset,
        broker=args.broker,
        topic=args.topic,
        limit=args.limit,
        delay=args.delay,
        corrupt_rate=args.corrupt_rate,
    )

