"""Temporal splitting utilities for leakage-aware model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from models.feature_contract import SCORING_TIMESTAMP_COLUMN

SPLIT_COLUMN = "temporal_split"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
MISSING_TIMESTAMP_SPLIT = "excluded_missing_timestamp"


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Fractions for forward-looking validation."""

    timestamp_column: str = SCORING_TIMESTAMP_COLUMN
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


def assign_temporal_splits(
    df: pl.DataFrame,
    config: TemporalSplitConfig | None = None,
) -> pl.DataFrame:
    """Add a split column using oldest scored leads for train and newest for test."""
    config = config or TemporalSplitConfig()
    _validate_split_config(df, config)

    indexed = df.with_row_index("_row_nr")
    timestamp = pl.col(config.timestamp_column)
    missing = indexed.filter(timestamp.is_null()).with_columns(
        pl.lit(MISSING_TIMESTAMP_SPLIT).alias(SPLIT_COLUMN)
    )
    non_missing = indexed.filter(timestamp.is_not_null()).sort(
        config.timestamp_column,
        "_row_nr",
    )

    row_count = non_missing.height
    train_cutoff = int(row_count * config.train_fraction)
    validation_cutoff = train_cutoff + int(row_count * config.validation_fraction)

    assigned = (
        non_missing.with_row_index("_temporal_order")
        .with_columns(
            pl.when(pl.col("_temporal_order") < train_cutoff)
            .then(pl.lit(TRAIN_SPLIT))
            .when(pl.col("_temporal_order") < validation_cutoff)
            .then(pl.lit(VALIDATION_SPLIT))
            .otherwise(pl.lit(TEST_SPLIT))
            .alias(SPLIT_COLUMN)
        )
        .drop("_temporal_order")
    )

    return (
        pl.concat([assigned, missing], how="vertical_relaxed")
        .sort("_row_nr")
        .drop("_row_nr")
    )


def split_counts(df: pl.DataFrame, split_column: str = SPLIT_COLUMN) -> dict[str, int]:
    """Return split counts in a stable dictionary for metadata and tests."""
    if split_column not in df.columns:
        raise ValueError(f"Missing split column: {split_column}")
    rows = df.group_by(split_column).len().rows()
    return {str(split_name): int(count) for split_name, count in rows}


def validate_temporal_order(
    df: pl.DataFrame,
    config: TemporalSplitConfig | None = None,
    split_column: str = SPLIT_COLUMN,
) -> None:
    """Ensure validation and test timestamps are later than training timestamps."""
    config = config or TemporalSplitConfig()
    if config.timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {config.timestamp_column}")
    if split_column not in df.columns:
        raise ValueError(f"Missing split column: {split_column}")

    boundaries = (
        df.filter(pl.col(split_column) != MISSING_TIMESTAMP_SPLIT)
        .group_by(split_column)
        .agg(
            pl.col(config.timestamp_column).min().alias("min_timestamp"),
            pl.col(config.timestamp_column).max().alias("max_timestamp"),
        )
    )
    by_split = {row[0]: (row[1], row[2]) for row in boundaries.rows()}
    ordered_splits = [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]

    for earlier, later in zip(ordered_splits, ordered_splits[1:], strict=False):
        if earlier not in by_split or later not in by_split:
            continue
        if by_split[earlier][1] > by_split[later][0]:
            raise ValueError(f"{earlier} split contains rows later than {later}")


def _validate_split_config(df: pl.DataFrame, config: TemporalSplitConfig) -> None:
    if config.timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {config.timestamp_column}")
    fractions = (
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("Temporal split fractions must be non-negative")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Temporal split fractions must sum to 1.0")
    if df.is_empty():
        raise ValueError("Cannot split an empty DataFrame")

