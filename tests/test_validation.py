import json

import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from models.feature_contract import TARGET_COLUMN
from models.structure import LATENT_NODE, build_structure_constraints
from models.validation import (
    average_precision,
    calibration_bins,
    check_cpts,
    check_structure,
    classification_metrics,
    evaluate_model_comparison,
    evaluate_model_on_splits,
    roc_auc,
    save_validation_report,
)


def test_classification_metrics_cover_probability_quality_and_ranking() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.35, 0.8]

    metrics = classification_metrics(y_true, probabilities)

    assert metrics["brier_score"] == pytest.approx(0.158125)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["accuracy_at_0_5"] == pytest.approx(0.75)
    assert metrics["f1_at_0_5"] == pytest.approx(2 / 3)
    assert roc_auc([1, 1], [0.2, 0.8]) is None
    assert average_precision([0, 0], [0.2, 0.8]) is None


def test_calibration_bins_report_observed_rate_by_probability_band() -> None:
    bins = calibration_bins([0, 1, 1], [0.1, 0.2, 0.9], bin_count=2)

    assert bins == [
        {
            "lower": 0.0,
            "upper": 0.5,
            "count": 2,
            "mean_predicted": pytest.approx(0.15),
            "observed_rate": pytest.approx(0.5),
        },
        {
            "lower": 0.5,
            "upper": 1.0,
            "count": 1,
            "mean_predicted": pytest.approx(0.9),
            "observed_rate": pytest.approx(1.0),
        },
    ]


def test_validation_report_scores_splits_and_persists_artifact(tmp_path) -> None:
    model = _latent_validation_model()
    frame = pd.DataFrame(
        {
            "web_session_band": ["short", "deep", "deep", "short"],
            TARGET_COLUMN: ["false", "true", "true", "false"],
        }
    )

    report = evaluate_model_on_splits(
        model,
        {"validation": frame.iloc[:2], "test": frame.iloc[2:]},
        build_structure_constraints(("web_session_band", TARGET_COLUMN)),
    )
    report_path = save_validation_report(report, tmp_path / "model_validation.json")
    payload = json.loads(report_path.read_text())

    assert payload["splits"]["validation"]["row_count"] == 2
    assert payload["splits"]["test"]["target_prevalence"] == pytest.approx(0.5)
    assert payload["splits"]["validation"]["metrics"]["brier_score"] < 0.13
    assert payload["structure"]["honors_forbidden_edges"] is True
    assert payload["structure"]["missing_latent_edges"] == []
    assert payload["cpt_checks"]["cpd_count"] == 3


def test_model_comparison_report_includes_baselines_and_summary(tmp_path) -> None:
    model = _latent_validation_model()
    frame = pd.DataFrame(
        {
            "web_session_band": ["short", "deep", "deep", "short"],
            TARGET_COLUMN: ["false", "true", "true", "false"],
        }
    )
    splits = {"validation": frame.iloc[:2], "test": frame.iloc[2:]}

    report = evaluate_model_comparison(
        model,
        {
            "random_forest": {
                "validation": [0.2, 0.8],
                "test": [0.7, 0.3],
            },
            "xgboost": {
                "validation": [0.1, 0.9],
                "test": [0.8, 0.2],
            },
        },
        splits,
        build_structure_constraints(("web_session_band", TARGET_COLUMN)),
    )
    report_path = save_validation_report(report, tmp_path / "model_validation.json")
    payload = json.loads(report_path.read_text())

    assert set(payload["splits"]["validation"]) == {
        "bbn",
        "random_forest",
        "xgboost",
    }
    assert (
        payload["splits"]["validation"]["bbn"]["metrics"].keys()
        == payload["splits"]["validation"]["random_forest"]["metrics"].keys()
    )
    assert payload["comparison_summary"]["validation"]["brier_score"]["model"]
    assert payload["structure"]["honors_forbidden_edges"] is True
    assert payload["cpt_checks"]["cpd_count"] == 3


def test_structure_and_cpt_checks_flag_invalid_edges_and_deterministic_cpds() -> None:
    model = DiscreteBayesianNetwork([("web_session_band", TARGET_COLUMN)])
    model.add_cpds(
        TabularCPD(
            variable="web_session_band",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"web_session_band": ["short", "deep"]},
        ),
        TabularCPD(
            variable=TARGET_COLUMN,
            variable_card=2,
            values=[[1.0, 0.0], [0.0, 1.0]],
            evidence=["web_session_band"],
            evidence_card=[2],
            state_names={
                TARGET_COLUMN: ["false", "true"],
                "web_session_band": ["short", "deep"],
            },
        ),
    )

    structure = check_structure(
        model,
        build_structure_constraints(("web_session_band", TARGET_COLUMN)),
    )
    cpt_checks = check_cpts(model)

    assert structure["latent_node_present"] is False
    assert structure["missing_latent_edges"]
    assert cpt_checks["near_deterministic_cpds"] == [TARGET_COLUMN]


def _latent_validation_model() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork(
        [
            (LATENT_NODE, "web_session_band"),
            (LATENT_NODE, TARGET_COLUMN),
        ],
        latents={LATENT_NODE},
    )
    model.add_cpds(
        TabularCPD(
            variable=LATENT_NODE,
            variable_card=3,
            values=[[0.2], [0.5], [0.3]],
            state_names={LATENT_NODE: ["negative", "neutral", "positive"]},
        ),
        TabularCPD(
            variable="web_session_band",
            variable_card=2,
            values=[
                [0.9, 0.5, 0.1],
                [0.1, 0.5, 0.9],
            ],
            evidence=[LATENT_NODE],
            evidence_card=[3],
            state_names={
                "web_session_band": ["short", "deep"],
                LATENT_NODE: ["negative", "neutral", "positive"],
            },
        ),
        TabularCPD(
            variable=TARGET_COLUMN,
            variable_card=2,
            values=[
                [0.95, 0.65, 0.15],
                [0.05, 0.35, 0.85],
            ],
            evidence=[LATENT_NODE],
            evidence_card=[3],
            state_names={
                TARGET_COLUMN: ["false", "true"],
                LATENT_NODE: ["negative", "neutral", "positive"],
            },
        ),
    )
    model.check_model()
    return model
