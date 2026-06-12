"""
Real-Time Fraud Detection Model Serving REST API.

Low-latency REST API (< 5ms SLA) retrieving online features from Redis/Feast 
and predicting transaction fraud scores via Model Ensemble (XGBoost + LightGBM).
"""

import os
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.redis_client import RedisClient
from src.ml.ensemble import FraudModelEnsemble

logger = get_logger("fraud_serving_api")

MODEL_PATH = os.path.join(settings.project_dir, "models", "ensemble_fraud_model.joblib")
REPO_PATH = os.path.join(settings.project_dir, "feature_repository")

# Global variables for cached model & feature store
ensemble_model: Optional[FraudModelEnsemble] = None
feast_store: Optional[Any] = None
redis_wrapper: Optional[RedisClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading serving artifacts on startup and cleaning up on shutdown."""
    global ensemble_model, feast_store, redis_wrapper
    logger.info("Initializing Model Serving API Resources...")

    # 1. Load Ensemble Model
    if os.path.exists(MODEL_PATH):
        try:
            ensemble_model = joblib.load(MODEL_PATH)
            logger.info(f"Ensemble Model successfully loaded from '{MODEL_PATH}'")
        except Exception as e:
            logger.error(f"Failed to load Ensemble Model artifact: {e}")
    else:
        logger.warning(f"Ensemble Model artifact not found at '{MODEL_PATH}'. Run src/ml/train.py first.")

    # 2. Load Feast Store
    if os.path.exists(REPO_PATH):
        try:
            from feast import FeatureStore
            feast_store = FeatureStore(repo_path=REPO_PATH)
            logger.info(f"Feast FeatureStore initialized from '{REPO_PATH}'")
        except Exception as e:
            logger.warning(f"Could not load Feast FeatureStore: {e}")

    # 3. Load Direct Redis Client
    try:
        redis_wrapper = RedisClient()
        if redis_wrapper.ping():
            logger.info(f"Redis direct connection established to {settings.redis_host}:{settings.redis_port}")
    except Exception as e:
        logger.warning(f"Redis direct connection notice: {e}")

    yield

    # Cleanup resources on shutdown
    if redis_wrapper:
        redis_wrapper.close()
    logger.info("Serving API resources released.")


app = FastAPI(
    title="Real-Time Fraud Detection Model Serving API",
    description="Low-latency REST API retrieving online features from Redis/Feast and predicting transaction fraud scores via Model Ensemble.",
    version="1.0.0",
    lifespan=lifespan
)


class TransactionRequest(BaseModel):
    """Schema for incoming online transaction inference request."""
    card_id: str = Field(..., json_schema_extra={"example": "11556"}, description="Unique credit card identifier")
    current_amount: float = Field(..., json_schema_extra={"example": 250.0}, gt=0, description="Current transaction amount in USD")


class PredictionResponse(BaseModel):
    """Schema for fraud score prediction response."""
    card_id: str
    is_fraud: int
    fraud_score: float
    decision: str
    xgb_score: float
    lgb_score: float
    latency_ms: float
    features_used: Dict[str, float]


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Root health check endpoint."""
    return {
        "status": "ONLINE",
        "service": "Real-Time Fraud Detection Model Serving API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint verifying model & infrastructure readiness."""
    model_loaded = ensemble_model is not None
    redis_ok = redis_wrapper.ping() if redis_wrapper else False

    return {
        "status": "HEALTHY" if model_loaded else "DEGRADED",
        "model_loaded": model_loaded,
        "redis_connected": redis_ok,
        "feast_ready": feast_store is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(req: TransactionRequest) -> PredictionResponse:
    """Inference Endpoint:
    1. Fetches online features from Redis/Feast.
    2. Calculates On-Demand derived features.
    3. Executes Model Ensemble inference (XGBoost + LightGBM).
    """
    start_time = time.perf_counter()

    if ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model via src/ml/train.py first."
        )

    card_id = str(req.card_id)
    curr_amount = float(req.current_amount)

    online_features: Dict[str, float] = {
        "trans_count_7d": 1.0,
        "trans_count_30d": 5.0,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1.0,
        "days_since_last_trans": 2.0,
        "TransactionAmt": curr_amount
    }

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

    avg_30d = online_features.get("avg_amount_30d", 150.0)
    max_30d = online_features.get("max_amount_30d", 500.0)

    online_features["amount_ratio_30d"] = curr_amount / (avg_30d + 1.0)
    online_features["is_amount_gt_30d_max"] = 1.0 if curr_amount > max_30d else 0.0

    feature_cols = ensemble_model.feature_names
    input_df = pd.DataFrame([online_features])[feature_cols].fillna(0.0)

    xgb_score = float(ensemble_model.xgb_model.predict_proba(input_df)[0, 1])
    lgb_score = float(ensemble_model.lgb_model.predict_proba(input_df)[0, 1])
    fraud_score = float(ensemble_model.predict_proba(input_df)[0, 1])

    is_fraud = 1 if fraud_score >= 0.5 else 0
    decision = "ALERT: FRAUD DETECTED" if is_fraud == 1 else "APPROVED"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logger.info(
        f"Predict Card '{card_id}' (${curr_amount:.2f}) -> {decision} "
        f"(Score: {fraud_score:.4f}, Latency: {elapsed_ms:.2f}ms)"
    )

    return PredictionResponse(
        card_id=card_id,
        is_fraud=is_fraud,
        fraud_score=round(fraud_score, 4),
        decision=decision,
        xgb_score=round(xgb_score, 4),
        lgb_score=round(lgb_score, 4),
        latency_ms=round(elapsed_ms, 2),
        features_used={k: round(v, 2) for k, v in online_features.items()}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=settings.fastapi_port, reload=True)
