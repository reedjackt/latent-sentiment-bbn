import json

import pandas as pd
import polars as pl
import pytest

from models.structure import (
    DEBUG_LATENT_COLUMN,
    EXTENDED_TRAINING_COLUMNS,
    LATENT_NODE,
    OUTCOME_COLUMN,
    PLAUSIBLE_SIGNAL_EDGES,
    TRAINING_COLUMNS,
    StructureLearningConfig,
    build_structure_constraints,
    prepare_observed_training_frame,
)
from models.train import prepare_modeling_frame, save_training_metadata, train_model


def test_structure_constraints_enforce_causal_order() -> None:
    constraints = build_structure_constraints(EXTENDED_TRAINING_COLUMNS)

    assert ("marketing_channel_clean", "web_session_band") in constraints.search_space
    assert ("web_session_band", "marketing_channel_clean") in constraints.forbidden_edges
    assert (OUTCOME_COLUMN, "lead_score_band") in constraints.forbidden_edges
    assert (LATENT_NODE, "lead_score_band") in constraints.latent_edges
    assert (LATENT_NODE, OUTCOME_COLUMN) in constraints.latent_edges


def test_structure_constraints_allow_context_bypasses_and_plausible_signal_edges() -> None:
    constraints = build_structure_constraints(EXTENDED_TRAINING_COLUMNS)
    exogenous_columns = EXTENDED_TRAINING_COLUMNS[:5]
    downstream_columns = EXTENDED_TRAINING_COLUMNS[5:]

    for source in exogenous_columns:
        for target in downstream_columns:
            assert (source, target) in constraints.search_space

    assert set(PLAUSIBLE_SIGNAL_EDGES) <= set(constraints.search_space)
    assert ("lead_score_band", "web_session_band") in constraints.forbidden_edges


def test_prepare_observed_training_frame_rejects_debug_latents() -> None:
    df = pd.DataFrame({column: ["state"] for column in TRAINING_COLUMNS})

    with pytest.raises(ValueError, match=DEBUG_LATENT_COLUMN):
        prepare_observed_training_frame(df.assign(**{DEBUG_LATENT_COLUMN: [0.2]}))

    with pytest.raises(ValueError, match=LATENT_NODE):
        prepare_observed_training_frame(df.assign(**{LATENT_NODE: ["positive"]}))


def test_training_pipeline_adds_hidden_latent_and_writes_metadata(tmp_path) -> None:
    cleaned = _cleaned_leads_frame()
    observed_df, discretization_config = prepare_modeling_frame(cleaned)

    assert tuple(observed_df.columns) == TRAINING_COLUMNS
    assert {rule.output_column for rule in discretization_config.rules} == {
        "web_session_band",
    }

    result = train_model(
        observed_df,
        config=StructureLearningConfig(
            show_progress=False,
            max_iter=100,
            em_max_iter=2,
            em_seed=7,
        ),
    )

    assert LATENT_NODE in result.model.nodes()
    assert result.model.latents == {LATENT_NODE}
    assert result.model.check_model()
    assert set(result.constraints.latent_edges) <= set(result.model.edges())
    assert not (set(result.model.edges()) & set(result.constraints.forbidden_edges))

    metadata_path = save_training_metadata(result, tmp_path / "metadata.json")
    payload = json.loads(metadata_path.read_text())

    assert payload["observed_columns"] == list(TRAINING_COLUMNS)
    assert payload["feature_contract"]["scoring_timestamp_column"] == "captured_at"
    assert payload["latent_edges"]
    assert payload["forbidden_edge_count"] > 0
    assert payload["search_space_edge_count"] > 0
    assert isinstance(payload["bic_score"], float)


def _cleaned_leads_frame() -> pl.DataFrame:
    rows = []
    for idx in range(36):
        high_intent = idx % 3 == 0
        rows.append(
            {
                "marketing_channel_clean": "paid_search" if high_intent else "organic",
                "campaign_tier_clean": "tier1_enterprise" if high_intent else "pilot",
                "region_clean": "NA" if idx % 2 else "EMEA",
                "employee_size_bucket": "enterprise" if high_intent else "small",
                "job_title_clean": "vp marketing" if high_intent else "manager",
                "web_session_seconds_clean": 900.0 if high_intent else 45.0 + idx,
                "replied_within_7d_bool": high_intent,
                "lead_score_clean": 88.0 if high_intent else 35.0 + idx % 5,
                "demo_requested_bool": high_intent,
            }
        )
    return pl.DataFrame(rows)
