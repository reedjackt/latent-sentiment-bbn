"""Validation utilities for fitted Bayesian network artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from networkx.algorithms.dag import is_directed_acyclic_graph
from pgmpy.inference import VariableElimination

from models.feature_contract import (
    DEFAULT_FEATURE_CONTRACT,
    TARGET_COLUMN,
    validate_feature_availability,
)
from models.structure import LATENT_NODE, StructureConstraints

VALIDATION_REPORT_PATH = Path(__file__).resolve().parent / "artifacts" / "model_validation.json"


@dataclass(frozen=True)
class SplitValidationResult:
    """Metrics and diagnostics for one validation split."""

    row_count: int
    target_prevalence: float | None
    metrics: dict[str, float | None]
    calibration_bins: list[dict[str, float | int | None]]
    warnings: list[str]


@dataclass(frozen=True)
class ValidationReport:
    """Serializable validation report for a fitted BBN."""

    splits: dict[str, SplitValidationResult]
    structure: dict[str, Any]
    cpt_checks: dict[str, Any]


@dataclass(frozen=True)
class ModelComparisonReport:
    """Serializable report comparing the BBN against discriminative baselines."""

    splits: dict[str, dict[str, SplitValidationResult]]
    comparison_summary: dict[str, dict[str, dict[str, float | str]]]
    structure: dict[str, Any]
    cpt_checks: dict[str, Any]


def evaluate_model_on_splits(
    model: Any,
    splits: Mapping[str, pd.DataFrame],
    constraints: StructureConstraints | None = None,
) -> ValidationReport:
    """Evaluate held-out splits and return metrics plus model sanity checks."""
    split_results = {
        split_name: evaluate_split(model, frame)
        for split_name, frame in splits.items()
    }
    return ValidationReport(
        splits=split_results,
        structure=check_structure(model, constraints),
        cpt_checks=check_cpts(model),
    )


def evaluate_model_comparison(
    bbn_model: Any,
    baseline_predictions: Mapping[str, Mapping[str, Sequence[float]]],
    splits: Mapping[str, pd.DataFrame],
    constraints: StructureConstraints | None = None,
    *,
    bbn_model_name: str = "bbn",
) -> ModelComparisonReport:
    """Evaluate BBN and discriminative baselines on shared held-out splits."""
    split_results: dict[str, dict[str, SplitValidationResult]] = {}
    for split_name, frame in splits.items():
        split_results[split_name] = {
            bbn_model_name: evaluate_split(bbn_model, frame),
        }
        for baseline_name, predictions_by_split in baseline_predictions.items():
            if split_name not in predictions_by_split:
                raise ValueError(
                    f"Missing {baseline_name!r} predictions for split {split_name!r}"
                )
            split_results[split_name][baseline_name] = evaluate_predictions(
                frame,
                predictions_by_split[split_name],
            )

    return ModelComparisonReport(
        splits=split_results,
        comparison_summary=comparison_summary(split_results),
        structure=check_structure(bbn_model, constraints),
        cpt_checks=check_cpts(bbn_model),
    )


def evaluate_split(
    model: Any,
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> SplitValidationResult:
    """Score one observed frame and compute probabilistic validation metrics."""
    if target_column not in frame.columns:
        raise ValueError(f"Validation frame is missing target column: {target_column}")

    predictions, warnings = predict_probabilities(model, frame, target_column=target_column)
    return evaluate_predictions(
        frame,
        predictions,
        target_column=target_column,
        warnings=warnings,
    )


def evaluate_predictions(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    *,
    target_column: str = TARGET_COLUMN,
    warnings: Sequence[str] = (),
) -> SplitValidationResult:
    """Evaluate precomputed P(target=true) predictions for one split."""
    if target_column not in frame.columns:
        raise ValueError(f"Validation frame is missing target column: {target_column}")

    y_true = [_target_to_int(value) for value in frame[target_column]]
    if len(y_true) != len(probabilities):
        raise ValueError("Validation labels and predictions must have the same length")
    return SplitValidationResult(
        row_count=len(y_true),
        target_prevalence=_mean(y_true),
        metrics=classification_metrics(y_true, probabilities),
        calibration_bins=calibration_bins(y_true, probabilities),
        warnings=list(warnings),
    )


def predict_probabilities(
    model: Any,
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[list[float], list[str]]:
    """Predict P(target=true) for each row, skipping unsupported evidence states."""
    if target_column not in set(model.nodes()):
        raise ValueError(f"Model is missing target node: {target_column}")

    evidence_columns = _evidence_columns(model, frame.columns, target_column)
    inference = VariableElimination(model)
    predictions: list[float] = []
    warnings: list[str] = []

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        evidence = _supported_evidence(model, row, evidence_columns, row_number, warnings)
        posterior = inference.query(
            variables=[target_column],
            evidence=evidence,
            show_progress=False,
        )
        states = posterior.state_names.get(target_column, [])
        if "true" not in states:
            raise ValueError(f"Target node {target_column} has no 'true' state")
        predictions.append(float(posterior.values[states.index("true")]))

    return predictions, warnings


def classification_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, float | None]:
    """Compute core probabilistic and ranking metrics without extra dependencies."""
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not y_true:
        return {
            "log_loss": None,
            "brier_score": None,
            "roc_auc": None,
            "pr_auc": None,
            "accuracy_at_0_5": None,
            "f1_at_0_5": None,
        }

    clipped = [_clip_probability(probability) for probability in probabilities]
    log_loss = -sum(
        truth * math.log(probability) + (1 - truth) * math.log(1 - probability)
        for truth, probability in zip(y_true, clipped, strict=True)
    ) / len(y_true)
    brier_score = sum(
        (truth - probability) ** 2
        for truth, probability in zip(y_true, probabilities, strict=True)
    ) / len(y_true)
    labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
    true_positives = sum(1 for truth, label in zip(y_true, labels, strict=True) if truth and label)
    false_positives = sum(1 for truth, label in zip(y_true, labels, strict=True) if not truth and label)
    false_negatives = sum(1 for truth, label in zip(y_true, labels, strict=True) if truth and not label)
    accuracy = sum(
        1 for truth, label in zip(y_true, labels, strict=True) if truth == label
    ) / len(y_true)
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "log_loss": log_loss,
        "brier_score": brier_score,
        "roc_auc": roc_auc(y_true, probabilities),
        "pr_auc": average_precision(y_true, probabilities),
        "accuracy_at_0_5": accuracy,
        "f1_at_0_5": f1,
    }


def calibration_bins(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    bin_count: int = 10,
) -> list[dict[str, float | int | None]]:
    """Group predictions into equal-width calibration bins."""
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    bins: list[dict[str, float | int | None]] = []
    for idx in range(bin_count):
        lower = idx / bin_count
        upper = (idx + 1) / bin_count
        members = [
            (truth, probability)
            for truth, probability in zip(y_true, probabilities, strict=True)
            if lower <= probability < upper or (idx == bin_count - 1 and probability == 1.0)
        ]
        truths = [truth for truth, _ in members]
        preds = [probability for _, probability in members]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_predicted": _mean(preds),
                "observed_rate": _mean(truths),
            }
        )
    return bins


def roc_auc(y_true: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Return rank-based ROC AUC, or None when one class is absent."""
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
    )
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        tie_end = idx + 1
        while tie_end < len(ranked) and ranked[tie_end][1] == ranked[idx][1]:
            tie_end += 1
        average_rank = (idx + 1 + tie_end) / 2
        for original_idx, _ in ranked[idx:tie_end]:
            if y_true[original_idx] == 1:
                rank_sum += average_rank
        idx = tie_end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(y_true: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Return average precision, a common summary of the PR curve."""
    positives = sum(y_true)
    if positives == 0:
        return None

    true_positives = 0
    precision_sum = 0.0
    ranked = sorted(
        zip(y_true, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    for rank, (truth, _) in enumerate(ranked, start=1):
        if truth:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def comparison_summary(
    split_results: Mapping[str, Mapping[str, SplitValidationResult]],
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Return the best model per split for each primary validation metric."""
    higher_is_better = {
        "roc_auc",
        "pr_auc",
        "accuracy_at_0_5",
        "f1_at_0_5",
    }
    lower_is_better = {
        "log_loss",
        "brier_score",
    }
    summary: dict[str, dict[str, dict[str, float | str]]] = {}

    for split_name, results_by_model in split_results.items():
        summary[split_name] = {}
        metric_names = higher_is_better | lower_is_better
        for metric_name in sorted(metric_names):
            candidates = [
                (model_name, result.metrics.get(metric_name))
                for model_name, result in results_by_model.items()
                if result.metrics.get(metric_name) is not None
            ]
            if not candidates:
                continue
            best_model, best_value = (
                max(candidates, key=lambda item: item[1])
                if metric_name in higher_is_better
                else min(candidates, key=lambda item: item[1])
            )
            summary[split_name][metric_name] = {
                "model": best_model,
                "value": float(best_value),
            }

    return summary


def check_structure(
    model: Any,
    constraints: StructureConstraints | None = None,
) -> dict[str, Any]:
    """Validate graph-level invariants and learned-edge constraints."""
    edges = {tuple(edge) for edge in model.edges()}
    payload: dict[str, Any] = {
        "is_acyclic": bool(is_directed_acyclic_graph(model)),
        "model_check_passed": _model_check_passed(model),
        "node_count": len(model.nodes()),
        "edge_count": len(edges),
        "latent_node_present": LATENT_NODE in set(model.nodes()),
    }
    if constraints is not None:
        forbidden_edges_present = sorted(edges & set(constraints.forbidden_edges))
        missing_latent_edges = sorted(set(constraints.latent_edges) - edges)
        payload.update(
            {
                "honors_forbidden_edges": not forbidden_edges_present,
                "forbidden_edges_present": [list(edge) for edge in forbidden_edges_present],
                "missing_latent_edges": [list(edge) for edge in missing_latent_edges],
            }
        )
    return payload


def check_cpts(model: Any, *, deterministic_threshold: float = 0.995) -> dict[str, Any]:
    """Inspect fitted CPTs for basic validity and near-deterministic rows."""
    checks: dict[str, Any] = {
        "cpd_count": 0,
        "near_deterministic_cpds": [],
        "invalid_probability_cpds": [],
    }
    for cpd in model.get_cpds():
        checks["cpd_count"] += 1
        values = cpd.get_values()
        if (values < -1e-12).any() or (values > 1 + 1e-12).any():
            checks["invalid_probability_cpds"].append(cpd.variable)
        if (values >= deterministic_threshold).any():
            checks["near_deterministic_cpds"].append(cpd.variable)
    checks["has_invalid_probabilities"] = bool(checks["invalid_probability_cpds"])
    return checks


def save_validation_report(
    report: ValidationReport | ModelComparisonReport,
    path: Path = VALIDATION_REPORT_PATH,
) -> Path:
    """Persist validation metrics and diagnostics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _validation_report_payload(report)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _validation_report_payload(
    report: ValidationReport | ModelComparisonReport,
) -> dict[str, Any]:
    if isinstance(report, ModelComparisonReport):
        return {
            "splits": {
                split_name: {
                    model_name: asdict(result)
                    for model_name, result in results_by_model.items()
                }
                for split_name, results_by_model in report.splits.items()
            },
            "comparison_summary": report.comparison_summary,
            "structure": report.structure,
            "cpt_checks": report.cpt_checks,
        }
    return {
        "splits": {
            split_name: asdict(result)
            for split_name, result in report.splits.items()
        },
        "structure": report.structure,
        "cpt_checks": report.cpt_checks,
    }


def _evidence_columns(
    model: Any,
    frame_columns: Sequence[str],
    target_column: str,
) -> list[str]:
    allowed_columns = DEFAULT_FEATURE_CONTRACT.allowed_evidence_columns
    validate_feature_availability(allowed_columns)
    model_nodes = set(model.nodes())
    return [
        column
        for column in allowed_columns
        if column in frame_columns and column in model_nodes and column != target_column
    ]


def _supported_evidence(
    model: Any,
    row: pd.Series,
    evidence_columns: Sequence[str],
    row_number: int,
    warnings: list[str],
) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for column in evidence_columns:
        state = str(row[column])
        allowed_states = _state_names_for(model, column)
        if allowed_states and state not in allowed_states:
            warnings.append(
                f"row {row_number}: skipped unsupported evidence state {column}={state!r}"
            )
            continue
        evidence[column] = state
    return evidence


def _state_names_for(model: Any, column: str) -> set[str]:
    cpd = model.get_cpds(column)
    if cpd is None:
        return set()
    return set(cpd.state_names.get(column, []))


def _target_to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text == "true":
        return 1
    if text == "false":
        return 0
    raise ValueError(f"Unsupported target state: {value!r}")


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-15), 1 - 1e-15)


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _model_check_passed(model: Any) -> bool:
    try:
        return bool(model.check_model())
    except Exception:
        return False
