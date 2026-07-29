"""Real-Time Fraud Detection Model Serving REST API with Dynamic Threshold Serving.

Provides sub-5ms low-latency inference endpoint retrieving online features from Redis/Feast,
scoring transactions via Model Ensemble (XGBoost + LightGBM), and evaluating risk
against the dynamically tuned cost-optimal decision threshold.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import Any
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from feast import FeatureStore

from src.config import settings
from src.utils.logger import get_logger
from src.ml.ensemble import FraudModelEnsemble
from src.utils.redis_client import get_redis_client, check_redis_health

if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "FraudModelEnsemble"):
    setattr(sys.modules["__main__"], "FraudModelEnsemble", FraudModelEnsemble)

logger = get_logger(__name__)

ensemble_model: FraudModelEnsemble | None = None
feast_store = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads Ensemble Model Artifact, Feast Registry & Redis Client on API startup."""
    global ensemble_model, feast_store, redis_client
    logger.info("Initializing Model Serving API Resources...")

    if ensemble_model is None and Path(settings.model_artifact_path).exists():
        try:
            ensemble_model = joblib.load(settings.model_artifact_path)
            th = getattr(ensemble_model, "optimal_threshold", 0.5)
            logger.info(f"Ensemble Model loaded from '{settings.model_artifact_path}' (Optimal Decision Threshold: {th:.4f})")
        except Exception as e:
            logger.error(f"Failed to load Ensemble Model artifact: {e}")
    else:
        logger.warning(f"Ensemble Model artifact not found at '{settings.model_artifact_path}'. Run src/ml/train.py first.")

    if Path(settings.feature_repo_dir).exists():
        try:
            feast_store = FeatureStore(repo_path=settings.feature_repo_dir)
            logger.info(f"Feast FeatureStore initialized from '{settings.feature_repo_dir}'")
        except Exception as e:
            logger.warning(f"Could not load Feast FeatureStore: {e}")

    try:
        redis_client = get_redis_client()
        if check_redis_health(redis_client):
            logger.info(f"Redis direct connection established to {settings.redis_host}:{settings.redis_port}")
    except Exception as e:
        logger.warning(f"Redis direct connection notice: {e}")

    yield

    logger.info("Shutting down Model Serving API Resources...")


app = FastAPI(
    title="Real-Time Fraud Detection Model Serving API",
    description="Low-latency REST API (< 5ms SLA) retrieving online features from Redis/Feast and predicting transaction fraud scores via Model Ensemble with Dynamic Threshold Tuning.",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    card_id: str = Field(..., description="Unique credit card identifier", json_schema_extra={"example": "11556"})
    current_amount: float = Field(..., gt=0, description="Current transaction amount in USD", json_schema_extra={"example": 250.0})


class PredictionResponse(BaseModel):
    card_id: str
    is_fraud: int
    fraud_score: float
    decision: str
    decision_threshold: float
    xgb_score: float
    lgb_score: float
    latency_ms: float
    features_used: dict[str, float]


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {
        "status": "ONLINE",
        "service": "Real-Time Fraud Detection Model Serving API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Healthcheck endpoint verifying model & infrastructure readiness."""
    model_loaded = ensemble_model is not None
    redis_ok = check_redis_health(redis_client) if redis_client is not None else False
    decision_th = getattr(ensemble_model, "optimal_threshold", 0.5) if ensemble_model else None

    is_healthy = model_loaded and (redis_ok or redis_client is None)

    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "model_loaded": model_loaded,
        "optimal_decision_threshold": decision_th,
        "redis_connected": redis_ok,
        "feast_ready": feast_store is not None,
    }


@app.get("/ready", tags=["Health"])
def readiness_check() -> dict[str, str]:
    """Kubernetes/Container Readiness probe ensuring model and stores are operational."""
    if ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready/loaded."
        )
    redis_ok = check_redis_health(redis_client) if redis_client is not None else True
    if not redis_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis online feature store is unreachable."
        )
    return {"status": "READY"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(req: TransactionRequest) -> PredictionResponse:
    """Sub-50ms Inference Endpoint with Dynamic Decision Threshold."""
    start_time = time.perf_counter()

    if ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model via src/ml/train.py first."
        )

    card_id = str(req.card_id)
    curr_amount = float(req.current_amount)
    is_cold_start = True

    # Baseline Cold-Start Features (Safe Defaults for unseen cards)
    online_features: dict[str, float] = {
        "trans_count_7d": 1.0,
        "trans_count_30d": 1.0,
        "avg_amount_30d": curr_amount,
        "max_amount_30d": curr_amount,
        "distinct_addr_7d": 1.0,
        "days_since_last_trans": 0.0,
        "TransactionAmt": curr_amount,
    }

    # Retrieve Online Features via Feast Online Store
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

            expected_numeric_features = set(ensemble_model.feature_names)
            found_any = False
            for key, val_list in res_dict.items():
                col_name = key.split(":")[-1]
                if col_name in expected_numeric_features and val_list and val_list[0] is not None and not np.isnan(val_list[0]):
                    try:
                        online_features[col_name] = float(val_list[0])
                        found_any = True
                    except (ValueError, TypeError):
                        pass
            if found_any:
                is_cold_start = False
        except Exception as e:
            logger.warning(f"Feast online feature lookup error for card '{card_id}': {e}")
            pass

    # Compute Derived Features
    avg_30d = online_features.get("avg_amount_30d", curr_amount)
    max_30d = online_features.get("max_amount_30d", curr_amount)

    online_features["amount_ratio_30d"] = curr_amount / (avg_30d + 1.0)
    online_features["is_amount_gt_30d_max"] = 1.0 if curr_amount > max_30d else 0.0

    feature_cols = ensemble_model.feature_names
    input_df = pd.DataFrame([online_features])[feature_cols].fillna(0.0)

    # Run Single-Pass Model Inference & Blend
    xgb_score = float(ensemble_model.xgb_model.predict_proba(input_df)[0, 1])
    lgb_score = float(ensemble_model.lgb_model.predict_proba(input_df)[0, 1])
    w_xgb = getattr(ensemble_model, "xgb_weight", 0.5)
    w_lgb = getattr(ensemble_model, "lgb_weight", 0.5)
    fraud_score = float(w_xgb * xgb_score + w_lgb * lgb_score)

    # Evaluate against Dynamic Optimal Decision Threshold
    decision_th = float(getattr(ensemble_model, "optimal_threshold", 0.5))
    is_fraud = 1 if fraud_score >= decision_th else 0
    decision = "ALERT: FRAUD DETECTED" if is_fraud == 1 else "APPROVED"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logger.info(f"Predict Card '{card_id}' (${curr_amount:.2f}) -> {decision} (Score: {fraud_score:.4f}, Threshold: {decision_th:.4f}, Latency: {elapsed_ms:.2f}ms, ColdStart: {is_cold_start})")

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
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
