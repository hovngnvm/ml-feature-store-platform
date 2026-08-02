"""
Automated Machine Learning Algorithm Test Suite.

Verifies Cost Matrix optimization logic, Dynamic Decision Threshold Tuning, and financial fraud savings.
"""

from pathlib import Path
import pytest
import numpy as np
from src.config import settings
from src.ml.train import find_optimal_decision_threshold
from src.ml.evaluate import evaluate_models


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


def test_evaluate_models_execution() -> None:
    """Verifies offline model evaluation pipeline runs without errors."""
    if not Path(settings.model_artifact_path).exists() or not Path(settings.ml_dataset_path).exists():
        pytest.skip(f"Model artifact ({settings.model_artifact_path}) or dataset ({settings.ml_dataset_path}) missing; skipping offline evaluation test.")

    evaluate_models()
    assert (Path(settings.model_dir) / "roc_curve.png").exists()
    assert (Path(settings.model_dir) / "pr_curve.png").exists()
    assert (Path(settings.model_dir) / "cost_vs_threshold.png").exists()
    assert (Path(settings.model_dir) / "threshold_tradeoffs.png").exists()
    assert (Path(settings.model_dir) / "confusion_matrix.png").exists()
