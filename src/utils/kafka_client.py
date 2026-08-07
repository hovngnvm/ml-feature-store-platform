"""
Kafka Client Infrastructure Wrapper.

Provides Kafka producer factory and helper methods for publishing streaming events.
"""

import os
import sys
import json
from typing import Any, Dict, Optional
from kafka import KafkaProducer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("kafka_client")


class KafkaProducerWrapper:
    """Wrapper around KafkaProducer managing serialization and delivery."""

    def __init__(self, broker: Optional[str] = None) -> None:
        """Initializes Kafka Producer with JSON serialization.

        Args:
            broker: Kafka bootstrap server address. Defaults to settings.kafka_broker.
        """
        self.broker = broker or settings.kafka_broker
        self._producer: Optional[KafkaProducer] = None

    def get_producer(self) -> KafkaProducer:
        """Returns active KafkaProducer instance with retry and buffer settings.

        Returns:
            Configured KafkaProducer instance.
        """
        if self._producer is None:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=[self.broker],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
                    acks="all",
                    retries=3,
                    linger_ms=10
                )
                logger.info(f"Connected to Kafka broker at {self.broker}")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka broker at {self.broker}: {e}")
                raise
        return self._producer

    def send_event(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[Any] = None
    ) -> None:
        """Publishes a single event to specified Kafka topic.

        Args:
            topic: Target Kafka topic name.
            value: Event payload dictionary.
            key: Optional message partition key.
        """
        producer = self.get_producer()
        try:
            producer.send(topic, value=value, key=key)
        except Exception as e:
            logger.error(f"Failed to send message to topic '{topic}': {e}")
            raise

    def flush(self) -> None:
        """Flushes buffered records to Kafka broker."""
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        """Flushes and closes active Kafka producer."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Kafka producer closed successfully.")
