"""Structure and parameter learning for the latent sentiment BBN."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import polars as pl
from pgmpy.models import DiscreteBayesianNetwork

from models.discretization import (
    DISCRETIZATION_CONFIG_PATH,
    DEFAULT_BIN_SPECS,
    BinSpec,
    DiscretizationConfig,
    apply_discretization,
    fit_discretization_config,
    save_discretization_config,
)
from models.feature_contract import (
    DEFAULT_FEATURE_CONTRACT,
    FeatureAvailabilityContract,
    validate_feature_availability,
)
from models.splits import (
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TemporalSplitConfig,
    assign_temporal_splits,
    split_counts,
    validate_temporal_order,
)
from models.structure import (
    DEBUG_LATENT_COLUMN,
    LATENT_NODE,
    TRAINING_COLUMNS,
    StructureConstraints,
    StructureLearningConfig,
    learn_structure_and_fit_parameters,
    prepare_observed_training_frame,
)
from models.validation import (
    evaluate_model_on_splits,
    save_validation_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_LEADS_PATH = REPO_ROOT / "data" / "clean_leads.parquet"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@dataclass(frozen=True)
class TrainingResult:
    """Fitted model plus structure-learning metadata."""

    model: DiscreteBayesianNetwork
    bic_score: float
    constraints: StructureConstraints
    observed_columns: tuple[str, ...]
    split_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class ModelingFrames:
    """Leakage-aware observed frames and their shared preprocessing config."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_df: pl.DataFrame
    discretization_config: DiscretizationConfig
    observed_columns: tuple[str, ...]


def load_cleaned_leads(path: Path = DEFAULT_CLEAN_LEADS_PATH) -> pl.DataFrame:
    """Load cleaned lead rows produced by the Polars cleaning pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Generate and clean lead data before training."
        )
    return pl.read_parquet(path)


def prepare_modeling_frame(
    cleaned_df: pl.DataFrame,
    columns: tuple[str, ...] = TRAINING_COLUMNS,
    *,
    discretization_fit_df: pl.DataFrame | None = None,
    feature_contract: FeatureAvailabilityContract = DEFAULT_FEATURE_CONTRACT,
    allow_post_capture_features: bool = False,
    allow_proxy_score_features: bool = False,
) -> tuple[pd.DataFrame, DiscretizationConfig]:
    """Fit discretization rules and return an observed, leakage-checked frame."""
    if DEBUG_LATENT_COLUMN in cleaned_df.columns:
        raise ValueError(f"{DEBUG_LATENT_COLUMN} must not be used for BBN training")
    validate_feature_availability(
        columns,
        feature_contract,
        allow_post_capture_features=allow_post_capture_features,
        allow_proxy_score_features=allow_proxy_score_features,
    )

    fit_df = discretization_fit_df if discretization_fit_df is not None else cleaned_df
    discretization_config = fit_discretization_config(
        fit_df,
        _discretization_specs_for_columns(columns),
    )
    discretized = apply_discretization(cleaned_df, discretization_config)
    pandas_df = discretized.to_pandas()
    observed_df = prepare_observed_training_frame(pandas_df, columns)
    return observed_df, discretization_config


def prepare_temporal_modeling_frames(
    cleaned_df: pl.DataFrame,
    columns: tuple[str, ...] = TRAINING_COLUMNS,
    split_config: TemporalSplitConfig | None = None,
    *,
    feature_contract: FeatureAvailabilityContract = DEFAULT_FEATURE_CONTRACT,
    allow_post_capture_features: bool = False,
    allow_proxy_score_features: bool = False,
) -> ModelingFrames:
    """Create train/validation/test frames with preprocessing fit on train only."""
    validate_feature_availability(
        columns,
        feature_contract,
        allow_post_capture_features=allow_post_capture_features,
        allow_proxy_score_features=allow_proxy_score_features,
    )
    split_config = split_config or TemporalSplitConfig()
    split_df = assign_temporal_splits(cleaned_df, split_config)
    validate_temporal_order(split_df, split_config)

    train_cleaned = split_df.filter(pl.col(SPLIT_COLUMN) == TRAIN_SPLIT)
    if train_cleaned.is_empty():
        raise ValueError("Temporal split produced an empty training frame")

    discretization_config = fit_discretization_config(
        train_cleaned,
        _discretization_specs_for_columns(columns),
    )
    discretized = apply_discretization(split_df, discretization_config)

    frames = {
        split_name: prepare_observed_training_frame(
            discretized.filter(pl.col(SPLIT_COLUMN) == split_name).to_pandas(),
            columns,
        )
        for split_name in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }

    return ModelingFrames(
        train=frames[TRAIN_SPLIT],
        validation=frames[VALIDATION_SPLIT],
        test=frames[TEST_SPLIT],
        split_df=split_df,
        discretization_config=discretization_config,
        observed_columns=columns,
    )


def train_model(
    observed_df: pd.DataFrame,
    config: StructureLearningConfig | None = None,
    split_counts_by_name: dict[str, int] | None = None,
) -> TrainingResult:
    """Learn observed structure, add the hidden latent node, and fit CPTs with EM."""
    model, bic_score, constraints = learn_structure_and_fit_parameters(
        observed_df,
        config=config,
        columns=observed_df.columns,
    )
    return TrainingResult(
        model=model,
        bic_score=bic_score,
        constraints=constraints,
        observed_columns=tuple(observed_df.columns),
        split_counts=split_counts_by_name,
    )


def fit_model(
    df: pd.DataFrame,
    config: StructureLearningConfig | None = None,
) -> DiscreteBayesianNetwork:
    """Fit a latent sentiment BBN from an observed, already-discretized frame."""
    return train_model(df, config=config).model


def save_model(model: DiscreteBayesianNetwork, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def save_training_metadata(
    result: TrainingResult,
    path: Path = METADATA_PATH,
) -> Path:
    """Persist lightweight metadata for reproducibility and inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observed_columns": list(result.observed_columns),
        "learned_observed_edges": _sorted_edges(
            edge
            for edge in result.model.edges()
            if LATENT_NODE not in edge
        ),
        "latent_edges": _sorted_edges(result.constraints.latent_edges),
        "forbidden_edge_count": len(result.constraints.forbidden_edges),
        "search_space_edge_count": len(result.constraints.search_space),
        "bic_score": result.bic_score,
        "pgmpy_version": _package_version("pgmpy"),
        "discretization_config_path": str(DISCRETIZATION_CONFIG_PATH),
        "feature_contract": {
            "scoring_timestamp_column": DEFAULT_FEATURE_CONTRACT.scoring_timestamp_column,
            "target_column": DEFAULT_FEATURE_CONTRACT.target_column,
            "scoring_moment": DEFAULT_FEATURE_CONTRACT.scoring_moment,
        },
    }
    if result.split_counts is not None:
        payload["split_counts"] = result.split_counts
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    cleaned_df = load_cleaned_leads()
    modeling_frames = prepare_temporal_modeling_frames(cleaned_df)
    discretization_path = save_discretization_config(modeling_frames.discretization_config)
    result = train_model(
        modeling_frames.train,
        split_counts_by_name=split_counts(modeling_frames.split_df),
    )
    validation_report = evaluate_model_on_splits(
        result.model,
        {
            VALIDATION_SPLIT: modeling_frames.validation,
            TEST_SPLIT: modeling_frames.test,
        },
        result.constraints,
    )
    model_path = save_model(result.model)
    metadata_path = save_training_metadata(result)
    validation_path = save_validation_report(validation_report)
    print(f"Saved model to {model_path}")
    print(f"Saved discretization config to {discretization_path}")
    print(f"Saved training metadata to {metadata_path}")
    print(f"Saved validation report to {validation_path}")


def _sorted_edges(edges: object) -> list[list[str]]:
    return [list(edge) for edge in sorted(edges)]


def _discretization_specs_for_columns(columns: tuple[str, ...]) -> list[BinSpec]:
    requested = set(columns)
    return [spec for spec in DEFAULT_BIN_SPECS if spec.output_column in requested]


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    main()
