import os
import sys
import time
import json
import random
import logging
import argparse
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("producer")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_transactions")
DEFAULT_DATASET_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(PROJECT_DIR, "data", "train_transaction.csv")
)

def create_stream_producer(broker_address: str):
    try:
        producer = KafkaProducer(
            bootstrap_servers=[broker_address],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
            acks="all",
            retries=3,
            linger_ms=10
        )
        logger.info(f"Connected to Stream Broker (Redpanda) at {broker_address}")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Stream producer at {broker_address}: {e}")
        sys.exit(1)

def format_event(row: pd.Series, base_time: float = None, inject_corrupt: bool = False) -> dict:
    if inject_corrupt:
        # Corrupt test event with invalid amount or card_id
        corrupt_type = random.choice(["invalid_amount", "invalid_card", "null_card"])
        if corrupt_type == "invalid_amount":
            card_id = str(int(row["card1"])) if pd.notnull(row.get("card1")) else "11556"
            amount = -999.0  # Invalid negative amount
        elif corrupt_type == "invalid_card":
            card_id = "unknown_card"
            amount = float(row["TransactionAmt"]) if pd.notnull(row.get("TransactionAmt")) else 100.0
        else:
            card_id = ""  # Empty string card_id
            amount = float(row["TransactionAmt"]) if pd.notnull(row.get("TransactionAmt")) else 100.0
            
        dt_offset = float(row.get("TransactionDT", 0))
        if base_time is None:
            base_time = time.time()
        event_timestamp = datetime.fromtimestamp(base_time + (dt_offset % 86400), tz=timezone.utc).isoformat()
        
        return {
            "transaction_id": int(row["TransactionID"]),
            "is_fraud": 0,
            "transaction_dt": dt_offset,
            "timestamp": event_timestamp,
            "card_id": card_id,
            "amount": amount,
            "product_cd": "W",
            "card_type": "corrupt_test",
            "card_category": "test",
            "p_emaildomain": "test.com",
            "addr1": 0.0,
            "c1": 0.0,
            "c2": 0.0
        }

    card_id = str(int(row["card1"])) if pd.notnull(row.get("card1")) else "unknown_card"
    
    dt_offset = float(row.get("TransactionDT", 0))
    if base_time is None:
        base_time = time.time()
    event_timestamp = datetime.fromtimestamp(base_time + (dt_offset % 86400), tz=timezone.utc).isoformat()

    return {
        "transaction_id": int(row["TransactionID"]),
        "is_fraud": int(row["isFraud"]) if pd.notnull(row.get("isFraud")) else 0,
        "transaction_dt": float(row.get("TransactionDT", 0)),
        "timestamp": event_timestamp,
        "card_id": card_id,
        "amount": float(row["TransactionAmt"]) if pd.notnull(row.get("TransactionAmt")) else 0.0,
        "product_cd": str(row.get("ProductCD", "W")),
        "card_type": str(row.get("card4", "unknown")),
        "card_category": str(row.get("card6", "unknown")),
        "p_emaildomain": str(row.get("P_emaildomain", "unknown")) if pd.notnull(row.get("P_emaildomain")) else "unknown",
        "addr1": float(row["addr1"]) if pd.notnull(row.get("addr1")) else 0.0,
        "c1": float(row["C1"]) if pd.notnull(row.get("C1")) else 0.0,
        "c2": float(row["C2"]) if pd.notnull(row.get("C2")) else 0.0
    }

def generate_synthetic_event(
    transaction_id: int,
    transaction_dt: float,
    base_time: float,
    pools: dict,
    inject_corrupt: bool = False
) -> dict:
    if inject_corrupt:
        corrupt_type = random.choice(["invalid_amount", "invalid_card", "null_card"])
        if corrupt_type == "invalid_amount":
            card_id = random.choice(pools["card_ids"]) if pools.get("card_ids") else "11556"
            amount = -999.0
        elif corrupt_type == "invalid_card":
            card_id = "unknown_card"
            amount = round(random.lognormvariate(3.5, 1.2), 2)
        else:
            card_id = ""
            amount = round(random.lognormvariate(3.5, 1.2), 2)

        event_timestamp = datetime.fromtimestamp(base_time + (transaction_dt % 86400), tz=timezone.utc).isoformat()
        return {
            "transaction_id": transaction_id,
            "is_fraud": 0,
            "transaction_dt": transaction_dt,
            "timestamp": event_timestamp,
            "card_id": card_id,
            "amount": amount,
            "product_cd": "W",
            "card_type": "corrupt_test",
            "card_category": "test",
            "p_emaildomain": "test.com",
            "addr1": 0.0,
            "c1": 0.0,
            "c2": 0.0
        }

    card_id = random.choice(pools["card_ids"]) if pools.get("card_ids") else "13579"
    is_fraud = 1 if random.random() < 0.035 else 0
    amount = round(random.lognormvariate(3.5, 1.2), 2)
    product_cd = random.choice(pools["product_cds"]) if pools.get("product_cds") else "W"
    card_type = random.choice(pools["card_types"]) if pools.get("card_types") else "visa"
    card_category = random.choice(pools["card_categories"]) if pools.get("card_categories") else "debit"
    p_emaildomain = random.choice(pools["p_emaildomains"]) if pools.get("p_emaildomains") else "gmail.com"
    addr1 = float(random.choice(pools["addr1_list"])) if pools.get("addr1_list") else 315.0
    c1 = float(random.choice(pools["c1_list"])) if pools.get("c1_list") else 1.0
    c2 = float(random.choice(pools["c2_list"])) if pools.get("c2_list") else 1.0

    event_timestamp = datetime.fromtimestamp(base_time + (transaction_dt % 86400), tz=timezone.utc).isoformat()

    return {
        "transaction_id": transaction_id,
        "is_fraud": is_fraud,
        "transaction_dt": transaction_dt,
        "timestamp": event_timestamp,
        "card_id": card_id,
        "amount": amount,
        "product_cd": product_cd,
        "card_type": card_type,
        "card_category": card_category,
        "p_emaildomain": p_emaildomain,
        "addr1": addr1,
        "c1": c1,
        "c2": c2
    }

def stream_transactions(
    dataset_path: str,
    broker: str,
    topic: str,
    limit: int = None,
    delay: float = 0.01,
    corrupt_rate: float = 0.0,
    synthetic: bool = True
):
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at: {dataset_path}")
        sys.exit(1)

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

    last_transaction_id = 2987000
    last_transaction_dt = 86400.0

    card_pool = set()
    product_pool = set()
    card_type_pool = set()
    card_cat_pool = set()
    email_pool = set()
    addr1_pool = set()
    c1_pool = set()
    c2_pool = set()

    for chunk in pd.read_csv(dataset_path, usecols=columns, chunksize=chunk_size):
        chunk = chunk.sort_values(by="TransactionDT")

        if len(card_pool) < 5000:
            card_pool.update(chunk["card1"].dropna().astype(int).astype(str).unique())
        if len(product_pool) < 20:
            product_pool.update(chunk["ProductCD"].dropna().astype(str).unique())
        if len(card_type_pool) < 10:
            card_type_pool.update(chunk["card4"].dropna().astype(str).unique())
        if len(card_cat_pool) < 10:
            card_cat_pool.update(chunk["card6"].dropna().astype(str).unique())
        if len(email_pool) < 50:
            email_pool.update(chunk["P_emaildomain"].dropna().astype(str).unique())
        if len(addr1_pool) < 100:
            addr1_pool.update(chunk["addr1"].dropna().astype(float).unique())
        if len(c1_pool) < 50:
            c1_pool.update(chunk["C1"].dropna().astype(float).unique())
        if len(c2_pool) < 50:
            c2_pool.update(chunk["C2"].dropna().astype(float).unique())

        for _, row in chunk.iterrows():
            is_corrupt = (corrupt_rate > 0.0) and (random.random() < corrupt_rate)
            event = format_event(row, base_time=start_time, inject_corrupt=is_corrupt)
            card_id = event["card_id"]

            last_transaction_id = max(last_transaction_id, event["transaction_id"])
            last_transaction_dt = max(last_transaction_dt, event["transaction_dt"])

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

    logger.info(f"Finished dataset file. Total dataset events sent: {total_sent}.")

    pools = {
        "card_ids": list(card_pool),
        "product_cds": list(product_pool),
        "card_types": list(card_type_pool),
        "card_categories": list(card_cat_pool),
        "p_emaildomains": list(email_pool),
        "addr1_list": list(addr1_pool),
        "c1_list": list(c1_pool),
        "c2_list": list(c2_pool),
    }

    if synthetic:
        logger.info("Switching seamlessly to continuous synthetic event generation...")
        while True:
            if limit and limit > 0 and total_sent >= limit:
                logger.info(f"Reached event limit ({limit}). Stopping producer.")
                break

            last_transaction_id += 1
            last_transaction_dt += random.uniform(0.5, 3.0)

            is_corrupt = (corrupt_rate > 0.0) and (random.random() < corrupt_rate)
            event = generate_synthetic_event(
                transaction_id=last_transaction_id,
                transaction_dt=last_transaction_dt,
                base_time=start_time,
                pools=pools,
                inject_corrupt=is_corrupt
            )
            card_id = event["card_id"]

            if is_corrupt:
                corrupt_sent += 1

            producer.send(topic, key=card_id, value=event)
            total_sent += 1

            if total_sent % 500 == 0:
                elapsed = time.time() - start_time
                rate = total_sent / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[SYNTHETIC] Sent {total_sent} events ({corrupt_sent} corrupt) | Rate: {rate:.1f} msg/s | "
                    f"TxID: {last_transaction_id} | Card: {card_id} | Amount: ${event['amount']:.2f} | Fraud: {event['is_fraud']}"
                )

            if delay > 0:
                time.sleep(delay)

    producer.flush()
    elapsed = time.time() - start_time
    logger.info(f"Completed streaming {total_sent} events ({corrupt_sent} corrupt) in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Transaction Events into Redpanda")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset CSV")
    parser.add_argument("--broker", type=str, default=KAFKA_BROKER, help="Stream broker address")
    parser.add_argument("--topic", type=str, default=KAFKA_TOPIC, help="Topic name")
    parser.add_argument("--limit", type=int, default=0, help="Max records to send (0 or negative for unlimited continuous generation)")
    parser.add_argument("--delay", type=float, default=0.005, help="Delay between events in seconds")
    parser.add_argument("--corrupt-rate", type=float, default=0.0, help="Ratio of corrupt test events (0.0 to 1.0)")
    parser.add_argument("--no-synthetic", action="store_true", help="Disable synthetic generation after dataset completion")

    args = parser.parse_args()
    stream_transactions(
        dataset_path=args.dataset,
        broker=args.broker,
        topic=args.topic,
        limit=args.limit,
        delay=args.delay,
        corrupt_rate=args.corrupt_rate,
        synthetic=not args.no_synthetic
    )

