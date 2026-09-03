"""Discriminative baseline classifiers for BBN model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from models.feature_contract import TARGET_COLUMN, validate_feature_availability

RANDOM_FOREST_BASELINE = "random_forest"
XGBOOST_BASELINE = "xgboost"


@dataclass(frozen=True)
class BaselineConfig:
    """Runtime knobs for deterministic discriminative baselines."""

    random_state: int = 42
    random_forest_estimators: int = 200
    xgboost_estimators: int = 200
    xgboost_max_depth: int = 3
    xgboost_learning_rate: float = 0.05
    n_jobs: int = 1


def train_discriminative_baselines(
    train_frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    config: BaselineConfig | None = None,
) -> dict[str, Pipeline]:
    """Fit supervised tree baselines on the same observed frame used by the BBN."""
    config = config or BaselineConfig()
    feature_columns = _feature_columns(train_frame, target_column)
    validate_feature_availability(feature_columns)

    x_train = train_frame.loc[:, feature_columns]
    y_train = [_target_to_int(value) for value in train_frame[target_column]]

    baselines = {
        RANDOM_FOREST_BASELINE: _categorical_pipeline(
            RandomForestClassifier(
                n_estimators=config.random_forest_estimators,
                min_samples_leaf=5,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
        ),
        XGBOOST_BASELINE: _categorical_pipeline(
            XGBClassifier(
                n_estimators=config.xgboost_estimators,
                max_depth=config.xgboost_max_depth,
                learning_rate=config.xgboost_learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
        ),
    }

    for model in baselines.values():
        model.fit(x_train, y_train)
    return baselines


def predict_baseline_probabilities(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> list[float]:
    """Return P(target=true) from a fitted discriminative baseline."""
    feature_columns = _feature_columns(frame, target_column)
    probabilities = model.predict_proba(frame.loc[:, feature_columns])
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Baseline model has no positive target class")
    positive_index = classes.index(1)
    return [float(probability) for probability in probabilities[:, positive_index]]


def predict_all_baselines(
    models: Mapping[str, Pipeline],
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> dict[str, list[float]]:
    """Score every fitted baseline against one held-out frame."""
    return {
        model_name: predict_baseline_probabilities(
            model,
            frame,
            target_column=target_column,
        )
        for model_name, model in models.items()
    }


def _categorical_pipeline(classifier: Any) -> Pipeline:
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        classifier,
    )


def _feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    if target_column not in frame.columns:
        raise ValueError(f"Baseline frame is missing target column: {target_column}")
    return [column for column in frame.columns if column != target_column]


def _target_to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text == "true":
        return 1
    if text == "false":
        return 0
    raise ValueError(f"Unsupported target state: {value!r}")
