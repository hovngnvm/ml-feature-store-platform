"""
Unit test suite for Centralized Settings, Logger, and Ensemble utilities.
"""

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble
import numpy as np
import pandas as pd


def test_system_settings_paths() -> None:
    """Verifies core system settings paths are properly initialized."""
    assert settings.project_dir is not None
    assert settings.data_dir.endswith("data")
    assert settings.model_dir.endswith("models")
    assert settings.redis_port == 6379
    assert settings.default_fp_cost > 0


def test_iso_logger_initialization() -> None:
    """Verifies ISO 8601 logger creation and properties."""
    logger = get_logger("test_feast_logger")
    assert logger is not None
    assert logger.name == "test_feast_logger"
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
