"""Real-Time Serving Latency Benchmark & Performance Regression Suite.

Measures nanosecond-precision execution percentiles (p50, p90, p95, p99)
across Online Feature Parsing, On-Demand Feature Engineering, and Model
Ensemble Blended Scoring over warm-path inference iterations.
"""

import sys
from pathlib import Path
import json
import time
import argparse
from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.ml.ensemble import FraudModelEnsemble


def assemble_feature_payload(
    card_id: str,
    current_amount: float,
    online_cache: dict[str, dict[str, Any]],
    feature_names: list[str]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Simulates on-demand feature engineering and builds model input DataFrame."""
    cached = online_cache.get(card_id, {})
    avg_30d = cached.get("avg_amount_30d", 100.0)
    max_30d = cached.get("max_amount_30d", 500.0)

    ratio_30d = current_amount / (avg_30d + 1.0)
    is_gt_max = int(current_amount > max_30d)

    feature_dict = {
        "trans_count_7d": cached.get("trans_count_7d", 1),
        "trans_count_30d": cached.get("trans_count_30d", 5),
        "avg_amount_30d": avg_30d,
        "max_amount_30d": max_30d,
        "distinct_addr_7d": cached.get("distinct_addr_7d", 1),
        "days_since_last_trans": cached.get("days_since_last_trans", 1.0),
        "TransactionAmt": current_amount,
        "amount_ratio_30d": ratio_30d,
        "is_amount_gt_30d_max": is_gt_max,
    }
    df_vector = pd.DataFrame([feature_dict])[feature_names]
    return feature_dict, df_vector


def run_latency_benchmark(num_iterations: int = 1000, save_report: bool = True) -> dict[str, Any]:
    """Executes warm-path serving benchmark requests and computes latency percentiles."""
    model_path = Path(settings.model_artifact_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    model_pipeline: FraudModelEnsemble = joblib.load(model_path)
    feature_names = model_pipeline.feature_names

    online_cache = {
        "11556": {
            "trans_count_7d": 4,
            "trans_count_30d": 18,
            "avg_amount_30d": 120.5,
            "max_amount_30d": 450.0,
            "distinct_addr_7d": 1,
            "days_since_last_trans": 0.5,
        }
    }

    # Warm up JIT and C-extension buffers
    for _ in range(10):
        _, df_warmup = assemble_feature_payload("11556", 750.0, online_cache, feature_names)
        _ = model_pipeline.predict_proba(df_warmup)

    lookup_timings_ns: list[int] = []
    transform_timings_ns: list[int] = []
    predict_timings_ns: list[int] = []
    total_timings_ns: list[int] = []

    for _ in range(num_iterations):
        card_id = "11556"
        current_amount = 750.0

        t_start = time.perf_counter_ns()

        # Step 1: Online Feature Retrieval
        _ = online_cache.get(card_id, {})
        t_lookup = time.perf_counter_ns()

        # Step 2: Dynamic Feature Transformation
        _, df_input = assemble_feature_payload(card_id, current_amount, online_cache, feature_names)
        t_transform = time.perf_counter_ns()

        # Step 3: Model Ensemble Blended Inference
        _ = model_pipeline.predict_proba(df_input)[:, 1][0]
        t_predict = time.perf_counter_ns()

        lookup_timings_ns.append(t_lookup - t_start)
        transform_timings_ns.append(t_transform - t_lookup)
        predict_timings_ns.append(t_predict - t_transform)
        total_timings_ns.append(t_predict - t_start)

    def calculate_percentiles(timings_ns: list[int]) -> dict[str, float]:
        timings_ms = np.array(timings_ns) / 1e6
        return {
            "p50_ms": round(float(np.percentile(timings_ms, 50)), 2),
            "p90_ms": round(float(np.percentile(timings_ms, 90)), 2),
            "p95_ms": round(float(np.percentile(timings_ms, 95)), 2),
            "p99_ms": round(float(np.percentile(timings_ms, 99)), 2),
            "mean_ms": round(float(np.mean(timings_ms)), 2),
            "max_ms": round(float(np.max(timings_ms)), 2),
        }

    results = {
        "status": "success",
        "benchmark_sample_count": num_iterations,
        "phases": {
            "1_online_feature_lookup": calculate_percentiles(lookup_timings_ns),
            "2_dynamic_feature_derivation": calculate_percentiles(transform_timings_ns),
            "3_model_ensemble_prediction": calculate_percentiles(predict_timings_ns),
            "total_warm_path_serving": calculate_percentiles(total_timings_ns),
        }
    }

    if save_report:
        report_file = Path(settings.model_dir) / "serving_latency_benchmark.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


import pytest


def test_serving_warm_path_latency_meets_sla() -> None:
    """Performance regression test: ensures warm-path inference latency remains within SLA."""
    model_path = Path(settings.model_artifact_path)
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}; skipping warm-path latency SLA test.")
    benchmark_report = run_latency_benchmark(num_iterations=20, save_report=False)
    assert benchmark_report["status"] == "success"
    p95_latency = benchmark_report["phases"]["total_warm_path_serving"]["p95_ms"]
    assert p95_latency < 60.0, f"Serving p95 latency exceeded SLA threshold: {p95_latency}ms >= 60.0ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Serving Latency Benchmark Runner")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of warm-path inference benchmark iterations (default: 1000)"
    )
    args = parser.parse_args()

    results = run_latency_benchmark(num_iterations=args.iterations, save_report=True)
    print(f"Latency Benchmark Complete ({args.iterations:,} iterations)")
    for phase_name, metrics in results["phases"].items():
        name = phase_name.replace("_", " ").title()
        print(f"  {name:<32} p50: {metrics['p50_ms']:>5.2f}ms | p95: {metrics['p95_ms']:>5.2f}ms | p99: {metrics['p99_ms']:>5.2f}ms | mean: {metrics['mean_ms']:>5.2f}ms")


if __name__ == "__main__":
    main()
