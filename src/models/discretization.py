"""Discretization utilities for Bayesian network model preparation.

The BBN consumes categorical states, so continuous cleaned features must be
converted into stable named bins before structure or parameter learning.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import polars as pl

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DISCRETIZATION_CONFIG_PATH = ARTIFACT_DIR / "discretization_config.json"

BinStrategy = Literal["fixed", "quantile"]


@dataclass(frozen=True)
class BinSpec:
    """Specification for how a source numeric column should be bucketed."""

    source_column: str
    output_column: str
    labels: list[str]
    strategy: BinStrategy
    upper_bounds: list[float] | None = None
    missing_label: str = "missing"


@dataclass(frozen=True)
class BinRule:
    """Fitted bin rule persisted with the model artifacts."""

    source_column: str
    output_column: str
    labels: list[str]
    upper_bounds: list[float]
    missing_label: str = "missing"


@dataclass(frozen=True)
class DiscretizationConfig:
    """Serializable collection of fitted rules."""

    rules: list[BinRule]


DEFAULT_BIN_SPECS = [
    BinSpec(
        source_column="lead_score_clean",
        output_column="lead_score_band",
        labels=["low", "medium", "high", "very_high"],
        strategy="fixed",
        upper_bounds=[40.0, 60.0, 80.0],
    ),
    BinSpec(
        source_column="web_session_seconds_clean",
        output_column="web_session_band",
        labels=["short", "medium", "engaged", "deep"],
        strategy="quantile",
    ),
]


def fit_discretization_config(
    df: pl.DataFrame,
    specs: list[BinSpec] | None = None,
) -> DiscretizationConfig:
    """Fit all bin rules, using only the provided training DataFrame."""
    fitted_rules = [
        _fit_rule(df, spec)
        for spec in (specs or DEFAULT_BIN_SPECS)
        if spec.source_column in df.columns
    ]
    return DiscretizationConfig(rules=fitted_rules)


def apply_discretization(
    df: pl.DataFrame,
    config: DiscretizationConfig,
) -> pl.DataFrame:
    """Append discretized categorical columns according to fitted rules."""
    out = df
    for rule in config.rules:
        if rule.source_column not in out.columns:
            raise ValueError(f"Missing source column for discretization: {rule.source_column}")
        out = out.with_columns(
            pl.col(rule.source_column)
            .map_elements(
                lambda value, bin_rule=rule: _assign_bin(value, bin_rule),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias(rule.output_column)
        )
    return out


def save_discretization_config(
    config: DiscretizationConfig,
    path: Path = DISCRETIZATION_CONFIG_PATH,
) -> Path:
    """Persist fitted discretization rules as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rules": [asdict(rule) for rule in config.rules]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_discretization_config(path: Path = DISCRETIZATION_CONFIG_PATH) -> DiscretizationConfig:
    """Load discretization rules saved during training."""
    payload = json.loads(path.read_text())
    return DiscretizationConfig(
        rules=[BinRule(**rule_payload) for rule_payload in payload["rules"]]
    )


def _fit_rule(df: pl.DataFrame, spec: BinSpec) -> BinRule:
    if len(spec.labels) < 2:
        raise ValueError(f"{spec.output_column} needs at least two non-missing labels")

    if spec.strategy == "fixed":
        if spec.upper_bounds is None:
            raise ValueError(f"{spec.output_column} fixed bins require upper_bounds")
        upper_bounds = spec.upper_bounds
    elif spec.strategy == "quantile":
        upper_bounds = _fit_quantile_bounds(df, spec)
    else:
        raise ValueError(f"Unsupported bin strategy: {spec.strategy}")

    if len(upper_bounds) != len(spec.labels) - 1:
        raise ValueError(
            f"{spec.output_column} expects {len(spec.labels) - 1} upper bounds, "
            f"got {len(upper_bounds)}"
        )

    return BinRule(
        source_column=spec.source_column,
        output_column=spec.output_column,
        labels=spec.labels,
        upper_bounds=[float(bound) for bound in upper_bounds],
        missing_label=spec.missing_label,
    )


def _fit_quantile_bounds(df: pl.DataFrame, spec: BinSpec) -> list[float]:
    values = (
        df.select(
            pl.col(spec.source_column)
            .cast(pl.Float64)
            .filter(pl.col(spec.source_column).cast(pl.Float64).is_finite())
        )
        .to_series()
        .sort()
    )
    if values.is_empty():
        raise ValueError(f"{spec.source_column} has no non-null values to discretize")

    quantiles = [idx / len(spec.labels) for idx in range(1, len(spec.labels))]
    bounds = [
        values.quantile(quantile, interpolation="nearest")
        for quantile in quantiles
    ]
    return _deduplicate_bounds(bounds, spec)


def _deduplicate_bounds(bounds: list[float | None], spec: BinSpec) -> list[float]:
    cleaned_bounds: list[float] = []
    previous: float | None = None
    for bound in bounds:
        if bound is None:
            raise ValueError(f"Could not fit quantile bound for {spec.output_column}")
        bound_float = float(bound)
        if previous is not None and bound_float <= previous:
            bound_float = previous + 1e-9
        cleaned_bounds.append(bound_float)
        previous = bound_float
    return cleaned_bounds


def _assign_bin(value: object, rule: BinRule) -> str:
    if value is None:
        return rule.missing_label

    numeric_value = float(value)
    if math.isnan(numeric_value):
        return rule.missing_label

    for label, upper_bound in zip(rule.labels, rule.upper_bounds, strict=False):
        if numeric_value <= upper_bound:
            return label
    return rule.labels[-1]
