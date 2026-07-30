"""
Unit test suite for Centralized Settings, Logger, and Ensemble utilities.
"""

import numpy as np
import pandas as pd

from src.config import settings, get_settings, SystemSettings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble
from src.producer.producer import SECONDS_PER_DAY, format_event


def test_system_settings_paths() -> None:
    """Verifies core system settings paths and configurations are properly initialized."""
    assert settings.project_dir is not None
    assert settings.data_dir.endswith("data")
    assert settings.model_dir.endswith("models")
    assert settings.redis_port == 6379
    assert settings.default_fp_cost > 0
    assert isinstance(settings.minio_secure, bool)
    assert settings.api_port > 0
    assert settings.api_host is not None
    assert settings.streamlit_port > 0
    assert settings.log_level in ("INFO", "DEBUG", "WARNING", "ERROR")
    assert settings.metrics_port > 0

    s1 = get_settings()
    s2 = get_settings()
    assert isinstance(s1, SystemSettings)
    assert s1 is s2
    assert s1 is settings


def test_producer_module_constants() -> None:
    """Verifies SECONDS_PER_DAY constant and format_event helper in producer."""
    assert SECONDS_PER_DAY == 86400
    sample_row = pd.Series({
        "TransactionID": 1001,
        "isFraud": 0,
        "TransactionDT": 100.0,
        "TransactionAmt": 50.0,
        "card1": 11556,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
        "addr1": 300.0,
        "C1": 1.0,
        "C2": 1.0,
    })
    event = format_event(sample_row)
    assert event["transaction_id"] == 1001
    assert event["card_id"] == "11556"
    assert event["amount"] == 50.0


def test_iso_logger_initialization() -> None:
    """Verifies ISO 8601 logger creation and properties."""
    logger = get_logger(__name__)
    assert logger is not None
    assert logger.name == __name__
    assert logger.propagate is False


def test_fraud_model_ensemble_instantiation() -> None:
    """Verifies FraudModelEnsemble instantiation and scoring methods."""
    class DummyModel:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.ones(n) * 0.2, np.ones(n) * 0.8])

    ensemble = FraudModelEnsemble(
        xgb_model=DummyModel(),
        lgb_model=DummyModel(),
        feature_names=["f1", "f2"],
        optimal_threshold=0.6,
    )
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    probas = ensemble.predict_proba(df)
    assert probas.shape == (2, 2)
    assert np.allclose(probas[:, 1], 0.8)

    preds = ensemble.predict(df)
    assert list(preds) == [1, 1]
