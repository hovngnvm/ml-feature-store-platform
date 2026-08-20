"""
Automated Machine Learning Algorithm Test Suite.

Verifies Cost Matrix optimization logic, Dynamic Decision Threshold Tuning, and financial fraud savings.
"""

import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.ml.train import find_optimal_decision_threshold, FraudModelEnsemble


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
