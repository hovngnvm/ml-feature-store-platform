from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feature_repository.entities import card_entity

BATCH_FEATURE_TTL_DAYS = 365

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

