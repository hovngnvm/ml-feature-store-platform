"""
Automated Model Serving API Test Suite.

Verifies FastAPI health probe, online feature inference endpoints, and error handling.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def mock_ensemble_model(monkeypatch):
    mock_model = MagicMock()
    mock_model.feature_names = [
        "trans_count_7d", "trans_count_30d", "avg_amount_30d", "max_amount_30d",
        "distinct_addr_7d", "days_since_last_trans", "TransactionAmt",
        "amount_ratio_30d", "is_amount_gt_30d_max"
    ]
    mock_model.xgb_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_model.lgb_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_model.optimal_threshold = 0.5
    monkeypatch.setattr("src.api.main.ensemble_model", mock_model)
    return mock_model


@pytest.fixture
def mock_redis_healthy(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    monkeypatch.setattr("src.api.main.redis_client", mock_redis)
    monkeypatch.setattr("src.api.main.check_redis_health", lambda client: True)
    return mock_redis


@pytest.fixture
def mock_feast_store(monkeypatch):
    mock_store = MagicMock()
    mock_response = MagicMock()
    mock_response.to_dict.return_value = {
        "card_batch_features:trans_count_7d": [5],
        "card_batch_features:trans_count_30d": [20],
        "card_batch_features:avg_amount_30d": [150.0],
        "card_batch_features:max_amount_30d": [500.0],
        "card_batch_features:distinct_addr_7d": [2],
        "card_batch_features:days_since_last_trans": [1.5],
    }
    mock_store.get_online_features.return_value = mock_response
    monkeypatch.setattr("src.api.main.FeatureStore", lambda repo_path: mock_store)
    monkeypatch.setattr("src.api.main.feast_store", mock_store)
    return mock_store



def test_fastapi_health_endpoint(mock_redis_healthy) -> None:
    """Verifies GET /health returns HTTP 200 status."""

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"].upper() in ["HEALTHY", "DEGRADED"]
        assert "model_loaded" in data


def test_fastapi_readiness_endpoint_ready(mock_ensemble_model, mock_redis_healthy) -> None:
    """Verifies GET /ready returns HTTP 200 when both model and redis are ready."""
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "READY"


def test_fastapi_readiness_endpoint_redis_unreachable(mock_ensemble_model, monkeypatch) -> None:
    """Verifies GET /ready returns HTTP 503 when redis store is unreachable."""
    mock_redis = MagicMock()
    monkeypatch.setattr("src.api.main.redis_client", mock_redis)
    monkeypatch.setattr("src.api.main.check_redis_health", lambda client: False)
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503


def test_fastapi_predict_endpoint(mock_ensemble_model, mock_redis_healthy, mock_feast_store) -> None:
    """Verifies POST /predict returns valid fraud score, decision, and decision_threshold with HTTP 200."""
    with TestClient(app) as client:
        payload = {
            "card_id": "11556",
            "current_amount": 250.0
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "decision" in data
        assert "decision_threshold" in data
        assert data["decision"] in ["APPROVED", "ALERT: FRAUD DETECTED"]
        assert "latency_ms" in data
        assert isinstance(data["fraud_score"], float)
        assert isinstance(data["decision_threshold"], float)

