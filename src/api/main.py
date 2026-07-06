"""
Real-Time Fraud Detection Model Serving REST API with Dynamic Threshold Serving.

Provides sub-5ms low-latency inference endpoint retrieving online features from Redis/Feast,
scoring transactions via Model Ensemble (XGBoost + LightGBM), and evaluating risk
against the dynamically tuned cost-optimal decision threshold.
"""

import os
import sys
import time
import joblib
import redis
import numpy as np
import pandas as pd
from typing import Dict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.utils.logger import get_logger
from src.ml.train import FraudModelEnsemble

sys.modules['__main__'].FraudModelEnsemble = FraudModelEnsemble

load_dotenv()
logger = get_logger("fraud_serving_api")

app = FastAPI(
    title="Real-Time Fraud Detection Model Serving API",
    description="Low-latency REST API (< 5ms SLA) retrieving online features from Redis/Feast and predicting transaction fraud scores via Model Ensemble with Dynamic Threshold Tuning.",
    version="1.0.0"
)

ensemble_model = None
feast_store = None
redis_client = None


class TransactionRequest(BaseModel):
    card_id: str = Field(..., example="11556", description="Unique credit card identifier")
    current_amount: float = Field(..., example=250.0, gt=0, description="Current transaction amount in USD")


class PredictionResponse(BaseModel):
    card_id: str
    is_fraud: int
    fraud_score: float
    decision: str
    decision_threshold: float
    xgb_score: float
    lgb_score: float
    latency_ms: float
    features_used: Dict[str, float]


@app.on_event("startup")
def load_serving_resources() -> None:
    """Loads Ensemble Model Artifact, Feast Registry & Redis Client on API startup."""
    global ensemble_model, feast_store, redis_client
    logger.info("Initializing Model Serving API Resources...")

    # Load Ensemble Model
    if os.path.exists(settings.model_artifact_path):
        try:
            ensemble_model = joblib.load(settings.model_artifact_path)
            th = getattr(ensemble_model, "optimal_threshold", 0.5)
            logger.info(f"Ensemble Model loaded from '{settings.model_artifact_path}' (Optimal Decision Threshold: {th:.4f})")
        except Exception as e:
            logger.error(f"Failed to load Ensemble Model artifact: {e}")
    else:
        logger.warning(f"Ensemble Model artifact not found at '{settings.model_artifact_path}'. Run src/ml/train.py first.")

    # Load Feast Store
    if os.path.exists(settings.feature_repo_dir):
        try:
            from feast import FeatureStore
            feast_store = FeatureStore(repo_path=settings.feature_repo_dir)
            logger.info(f"Feast FeatureStore initialized from '{settings.feature_repo_dir}'")
        except Exception as e:
            logger.warning(f"Could not load Feast FeatureStore: {e}")

    # Load Direct Redis Client
    try:
        redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
        redis_client.ping()
        logger.info(f"Redis direct connection established to {settings.redis_host}:{settings.redis_port}")
    except Exception as e:
        logger.warning(f"Redis direct connection notice: {e}")


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {
        "status": "ONLINE",
        "service": "Real-Time Fraud Detection Model Serving API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Healthcheck endpoint verifying model & infrastructure readiness."""
    model_loaded = ensemble_model is not None
    redis_ok = False
    if redis_client:
        try:
            redis_ok = redis_client.ping()
        except Exception:
            redis_ok = False

    decision_th = getattr(ensemble_model, "optimal_threshold", 0.5) if ensemble_model else None

    return {
        "status": "HEALTHY" if model_loaded else "DEGRADED",
        "model_loaded": model_loaded,
        "optimal_decision_threshold": decision_th,
        "redis_connected": redis_ok,
        "feast_ready": feast_store is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(req: TransactionRequest):
    """
    Sub-5ms Inference Endpoint with Dynamic Decision Threshold:
    1. Fetches online features from Redis/Feast.
    2. Calculates On-Demand features.
    3. Executes Model Ensemble inference (XGBoost + LightGBM).
    4. Evaluates decision using dynamically tuned optimal threshold.
    """
    start_time = time.perf_counter()

    if ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model via src/ml/train.py first."
        )

    card_id = str(req.card_id)
    curr_amount = float(req.current_amount)

    # Retrieve Online Features
    online_features = {
        "trans_count_7d": 1.0,
        "trans_count_30d": 5.0,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1.0,
        "days_since_last_trans": 2.0,
        "TransactionAmt": curr_amount
    }

    # Try Feast online lookup
    if feast_store:
        try:
            res_dict = feast_store.get_online_features(
                features=[
                    "card_batch_features:trans_count_7d",
                    "card_batch_features:trans_count_30d",
                    "card_batch_features:avg_amount_30d",
                    "card_batch_features:max_amount_30d",
                    "card_batch_features:distinct_addr_7d",
                    "card_batch_features:days_since_last_trans"
                ],
                entity_rows=[{"card_id": card_id}]
            ).to_dict()

            for key, val_list in res_dict.items():
                col_name = key.split(":")[-1]
                if val_list and val_list[0] is not None and not np.isnan(val_list[0]):
                    online_features[col_name] = float(val_list[0])
        except Exception as e:
            logger.debug(f"Feast online lookup notice for card '{card_id}': {e}")

    # Compute Derived Features
    avg_30d = online_features.get("avg_amount_30d", 150.0)
    max_30d = online_features.get("max_amount_30d", 500.0)

    online_features["amount_ratio_30d"] = curr_amount / (avg_30d + 1.0)
    online_features["is_amount_gt_30d_max"] = 1.0 if curr_amount > max_30d else 0.0

    feature_cols = ensemble_model.feature_names
    input_df = pd.DataFrame([online_features])[feature_cols].fillna(0.0)

    # Run Model Ensemble Inference
    xgb_score = float(ensemble_model.xgb_model.predict_proba(input_df)[0, 1])
    lgb_score = float(ensemble_model.lgb_model.predict_proba(input_df)[0, 1])
    fraud_score = float(ensemble_model.predict_proba(input_df)[0, 1])

    # Evaluate against Dynamic Optimal Decision Threshold
    decision_th = float(getattr(ensemble_model, "optimal_threshold", 0.5))
    is_fraud = 1 if fraud_score >= decision_th else 0
    decision = "ALERT: FRAUD DETECTED" if is_fraud == 1 else "APPROVED"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logger.info(f"Predict Card '{card_id}' (${curr_amount:.2f}) -> {decision} (Score: {fraud_score:.4f}, Threshold: {decision_th:.4f}, Latency: {elapsed_ms:.2f}ms)")

    return PredictionResponse(
        card_id=card_id,
        is_fraud=is_fraud,
        fraud_score=round(fraud_score, 4),
        decision=decision,
        decision_threshold=round(decision_th, 4),
        xgb_score=round(xgb_score, 4),
        lgb_score=round(lgb_score, 4),
        latency_ms=round(elapsed_ms, 2),
        features_used={k: round(v, 2) for k, v in online_features.items()}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
