"""
Standalone Model Ensemble Module for Real-Time Fraud Detection.
Eradicates __main__ serialization monkey-patching anti-patterns.
"""

from typing import Any
import numpy as np
import pandas as pd


class FraudModelEnsemble:
    """Production Model Ensemble combining XGBoost & LightGBM with Weighted Blending & Dynamic Thresholding."""

    def __init__(
        self,
        xgb_model: Any,
        lgb_model: Any,
        feature_names: list[str],
        xgb_weight: float = 0.5,
        lgb_weight: float = 0.5,
        optimal_threshold: float = 0.5,
        threshold_metrics: dict | None = None
    ) -> None:
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.feature_names = feature_names
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.optimal_threshold = optimal_threshold
        self.threshold_metrics = threshold_metrics or {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 2D array of class probabilities [[p0, p1], ...]."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        p1_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p1_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p1_ensemble = (self.xgb_weight * p1_xgb) + (self.lgb_weight * p1_lgb)
        p0_ensemble = 1.0 - p1_ensemble
        return np.column_stack((p0_ensemble, p1_ensemble))

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Returns binary predictions based on decision threshold (defaults to optimal_threshold)."""
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
