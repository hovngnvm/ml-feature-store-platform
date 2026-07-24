from datetime import timedelta
from feast import (
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64
import pandas as pd

try:
    from entities import card_entity
except ImportError:
    from feature_repository.entities import card_entity

BATCH_FEATURE_TTL_DAYS = 365
STREAM_FEATURE_TTL_DAYS = 7
RATIO_EPSILON = 1.0
ZSCORE_EPSILON = 0.01
HIGH_VELOCITY_THRESHOLD_5M = 3.0

# Batch Feature Store Source & View (Historical 7d/30d/All-time Features)
batch_file_source = FileSource(
    name="batch_features_source",
    path="../data/batch_features.parquet",
    timestamp_field="event_timestamp"
)

card_batch_feature_view = FeatureView(
    name="card_batch_features",
    entities=[card_entity],
    ttl=timedelta(days=BATCH_FEATURE_TTL_DAYS),
    schema=[
        Field(name="trans_count_7d", dtype=Int64),
        Field(name="trans_count_30d", dtype=Int64),
        Field(name="avg_amount_30d", dtype=Float64),
        Field(name="max_amount_30d", dtype=Float64),
        Field(name="distinct_addr_7d", dtype=Int64),
        Field(name="days_since_last_trans", dtype=Float64)
    ],
    online=True,
    source=batch_file_source
)

# Push / Stream Feature Store Source & View (PyFlink Real-Time Window Features)
stream_push_source = PushSource(
    name="stream_features_push_source",
    batch_source=batch_file_source
)

card_stream_feature_view = FeatureView(
    name="card_stream_features",
    entities=[card_entity],
    ttl=timedelta(days=STREAM_FEATURE_TTL_DAYS),
    schema=[
        Field(name="trans_count_1m", dtype=Int64),
        Field(name="trans_count_5m", dtype=Int64),
        Field(name="trans_count_1h", dtype=Int64),
        Field(name="total_amount_1h", dtype=Float64),
        Field(name="avg_amount_1h", dtype=Float64),
        Field(name="avg_amount_24h", dtype=Float64),
        Field(name="stddev_amount_24h", dtype=Float64),
        Field(name="max_amount_24h", dtype=Float64),
        Field(name="c2_count_sum_1h", dtype=Float64)
    ],
    online=True,
    source=stream_push_source
)

# Request Data Source (Payload sent via REST API request)
transaction_request_source = RequestSource(
    name="transaction_request_source",
    schema=[
        Field(name="current_amount", dtype=Float64)
    ]
)

# Standalone On-Demand Transformation UDF
@on_demand_feature_view(
    name="card_on_demand_features",
    sources=[
        card_stream_feature_view,
        card_batch_feature_view,
        transaction_request_source
    ],
    schema=[
        Field(name="amount_ratio_24h", dtype=Float64),
        Field(name="amount_ratio_30d", dtype=Float64),
        Field(name="amount_zscore_24h", dtype=Float64),
        Field(name="is_amount_gt_30d_max", dtype=Float64),
        Field(name="is_high_velocity_5m", dtype=Float64)
    ]
)
def card_on_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes real-time on-demand derived fraud risk indicators."""
    out = pd.DataFrame()
    curr_amt = df["current_amount"]
    avg_24h = df["avg_amount_24h"].fillna(0.0)
    avg_30d = df["avg_amount_30d"].fillna(0.0)
    stddev_24h = df["stddev_amount_24h"].fillna(0.0)
    max_30d = df["max_amount_30d"].fillna(0.0)
    count_5m = df["trans_count_5m"].fillna(0)

    out["amount_ratio_24h"] = curr_amt / (avg_24h + RATIO_EPSILON)
    out["amount_ratio_30d"] = curr_amt / (avg_30d + RATIO_EPSILON)
    out["amount_zscore_24h"] = (curr_amt - avg_24h) / (stddev_24h + ZSCORE_EPSILON)
    out["is_amount_gt_30d_max"] = (curr_amt > max_30d).astype(float)
    out["is_high_velocity_5m"] = (count_5m >= HIGH_VELOCITY_THRESHOLD_5M).astype(float)
    return out
