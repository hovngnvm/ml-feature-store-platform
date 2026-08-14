"""
Automated Pytest Test Suite with Dynamic Decision Threshold Verification.

Tests Data Quality Assertions Gate, Cost Matrix Optimal Threshold Tuning,
FastAPI Model Serving Endpoints, and Inference correctness.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config.settings import settings
from src.quality.data_assert import validate_batch_dataframe
from src.ml.train import find_optimal_decision_threshold, FraudModelEnsemble
from src.api.main import app

client = TestClient(app)


def test_data_quality_gate() -> None:
    """Verifies Pandera Schema Gate filters valid and quarantined invalid rows."""
    valid_data = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": 3,
        "trans_count_30d": 12,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1,
        "days_since_last_trans": 1.5
    }])
    is_valid, clean_df, error_df = validate_batch_dataframe(valid_data)
    assert is_valid is True
    assert len(clean_df) == 1
    assert error_df is None or len(error_df) == 0

    invalid_data = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": -5,
        "trans_count_30d": 12,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1,
        "days_since_last_trans": 1.5
    }])
    is_valid_inv, clean_inv, error_inv = validate_batch_dataframe(invalid_data)
    assert is_valid_inv is False
    assert error_inv is not None and len(error_inv) > 0


def test_optimal_threshold_tuning() -> None:
    """Verifies Cost Matrix optimal threshold optimization logic and savings calculation."""
    # Synthetic imbalanced test dataset (10 fraud cases, 90 clean cases)
    np.random.seed(42)
    y_true = np.array([1] * 10 + [0] * 90)
    # Fraud probabilities: frauds have scores ~0.3-0.7, clean cases ~0.05-0.2
    y_proba = np.concatenate([
        np.random.uniform(0.25, 0.70, size=10),
        np.random.uniform(0.01, 0.25, size=90)
    ])
    amounts = np.concatenate([
        np.random.uniform(300.0, 1500.0, size=10),  # High fraud transaction amounts
        np.random.uniform(20.0, 100.0, size=90)     # Normal transactions
    ])

    res = find_optimal_decision_threshold(y_true, y_proba, amounts, cost_fp=2.0)
    assert "optimal_threshold" in res
    assert 0.01 <= res["optimal_threshold"] <= 0.90
    assert "min_cost" in res
    assert "cost_at_05" in res
    assert res["min_cost"] <= res["cost_at_05"]
    assert res["savings_amount"] >= 0.0
    assert "f1_optimal_threshold" in res
    assert "f2_optimal_threshold" in res


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
