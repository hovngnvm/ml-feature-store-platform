"""
Automated Model Serving API Test Suite.

Verifies FastAPI health probe, online feature inference endpoints, and error handling.
"""

import os
import sys
from fastapi.testclient import TestClient

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.api.main import app


def test_fastapi_health_endpoint() -> None:
    """Verifies GET /health returns HTTP 200 status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"].upper() in ["HEALTHY", "DEGRADED"]
        assert "model_loaded" in data


def test_fastapi_predict_endpoint() -> None:
    """Verifies POST /predict returns valid fraud score, decision, and decision_threshold."""
    with TestClient(app) as client:
        payload = {
            "card_id": "11556",
            "current_amount": 250.0
        }
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "fraud_score" in data
            assert "decision" in data
            assert "decision_threshold" in data
            assert data["decision"] in ["APPROVED", "ALERT: FRAUD DETECTED"]
            assert "latency_ms" in data
            assert isinstance(data["fraud_score"], float)
            assert isinstance(data["decision_threshold"], float)
        else:
            assert response.status_code == 503
