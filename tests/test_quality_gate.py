"""
Automated Data Quality Gate Test Suite.

Verifies Pandera schema assertion gate filters valid transactions and quarantines invalid records.
"""

import pandas as pd
from src.quality.data_assert import validate_batch_dataframe


def test_data_quality_gate() -> None:
    """Verifies Pandera Schema Gate filters valid and quarantined invalid rows."""
    valid_data = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": 3,
        "trans_count_30d": 12,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1,
        "days_since_last_trans": 1.5
    }])
    is_valid, clean_df, error_df = validate_batch_dataframe(valid_data)
    assert is_valid is True
    assert len(clean_df) == 1
    assert error_df is None or len(error_df) == 0

    invalid_data = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": -5,
        "trans_count_30d": 12,
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "distinct_addr_7d": 1,
        "days_since_last_trans": 1.5
    }])
    is_valid_inv, clean_inv, error_inv = validate_batch_dataframe(invalid_data)
    assert is_valid_inv is False
    assert error_inv is not None and len(error_inv) > 0
