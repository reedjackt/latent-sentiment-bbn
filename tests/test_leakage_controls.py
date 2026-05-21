from datetime import datetime, timedelta

import polars as pl
import pytest

from models.feature_contract import (
    CAPTURE_TIME_TRAINING_COLUMNS,
    EXTENDED_FUNNEL_TRAINING_COLUMNS,
    validate_feature_availability,
)
from models.splits import (
    MISSING_TIMESTAMP_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TemporalSplitConfig,
    assign_temporal_splits,
    validate_temporal_order,
)
from models.train import prepare_temporal_modeling_frames


def test_feature_contract_rejects_post_capture_and_proxy_features() -> None:
    validate_feature_availability(CAPTURE_TIME_TRAINING_COLUMNS)

    with pytest.raises(ValueError, match="replied_within_7d_bool"):
        validate_feature_availability(EXTENDED_FUNNEL_TRAINING_COLUMNS)

    validate_feature_availability(
        EXTENDED_FUNNEL_TRAINING_COLUMNS,
        allow_post_capture_features=True,
        allow_proxy_score_features=True,
    )


def test_temporal_split_orders_known_timestamps_and_excludes_missing() -> None:
    start = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "lead_id_clean": [f"LD-{idx:08d}" for idx in range(6)],
            "captured_at": [
                start + timedelta(days=idx) if idx != 2 else None
                for idx in range(6)
            ],
        }
    )

    split_df = assign_temporal_splits(
        df,
        TemporalSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
    )
    validate_temporal_order(split_df)

    assert split_df.filter(pl.col("captured_at").is_null())[SPLIT_COLUMN].item() == (
        MISSING_TIMESTAMP_SPLIT
    )
    assert set(split_df[SPLIT_COLUMN]) == {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
        MISSING_TIMESTAMP_SPLIT,
    }


def test_temporal_modeling_fits_discretization_on_train_only() -> None:
    cleaned = _cleaned_leads_frame()

    frames = prepare_temporal_modeling_frames(
        cleaned,
        split_config=TemporalSplitConfig(
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        ),
    )

    session_rule = next(
        rule
        for rule in frames.discretization_config.rules
        if rule.source_column == "web_session_seconds_clean"
    )

    assert max(session_rule.upper_bounds) < 1_000.0
    assert tuple(frames.train.columns) == CAPTURE_TIME_TRAINING_COLUMNS
    assert not frames.validation.empty
    assert not frames.test.empty


def _cleaned_leads_frame() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    for idx in range(12):
        high_intent = idx >= 6
        rows.append(
            {
                "captured_at": start + timedelta(days=idx),
                "marketing_channel_clean": "paid_search" if high_intent else "organic",
                "campaign_tier_clean": "tier1_enterprise" if high_intent else "pilot",
                "region_clean": "NA" if idx % 2 else "EMEA",
                "employee_size_bucket": "enterprise" if high_intent else "small",
                "job_title_clean": "vp marketing" if high_intent else "manager",
                "web_session_seconds_clean": 1000.0 + idx if high_intent else 10.0 + idx,
                "replied_within_7d_bool": high_intent,
                "lead_score_clean": 88.0 if high_intent else 35.0 + idx,
                "demo_requested_bool": high_intent,
            }
        )
    return pl.DataFrame(rows)

