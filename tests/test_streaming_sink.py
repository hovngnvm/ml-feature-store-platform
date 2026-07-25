"""Automated Test Suite for DualPathRedisFeatureSink & Stream DLQ Isolation."""

from unittest.mock import MagicMock
from src.streaming.flink_feature_job import DualPathRedisFeatureSink


def test_dual_path_sink_valid_event_processing() -> None:
    mock_redis = MagicMock()
    mock_dlq_producer = MagicMock()
    sink = DualPathRedisFeatureSink(redis_client=mock_redis, dlq_producer=mock_dlq_producer)

    valid_event = {
        "transaction_id": 1001,
        "card_id": "11556",
        "amount": 250.0,
        "c1": 1.0,
        "c2": 2.0,
        "timestamp": "2026-08-21T12:00:00Z"
    }

    assert sink.is_valid_event(valid_event) is True
    processed = sink.process_event(valid_event)
    assert processed is not None
    assert processed["card_id"] == "11556"
    assert len(sink.raw_events_buffer) == 1
    assert len(sink.dlq_events_buffer) == 0


def test_dual_path_sink_corrupt_event_quarantine() -> None:
    mock_redis = MagicMock()
    mock_dlq_producer = MagicMock()
    sink = DualPathRedisFeatureSink(redis_client=mock_redis, dlq_producer=mock_dlq_producer)

    corrupt_event = {
        "transaction_id": 1002,
        "card_id": "unknown_card",
        "amount": -50.0,
        "timestamp": "2026-08-21T12:00:00Z"
    }

    assert sink.is_valid_event(corrupt_event) is False
    processed = sink.process_event(corrupt_event)
    assert processed is None
    assert len(sink.raw_events_buffer) == 0
    assert len(sink.dlq_events_buffer) == 1
