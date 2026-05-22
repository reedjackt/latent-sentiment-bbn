import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from models.discretization import BinRule, DiscretizationConfig
from models.feature_contract import TARGET_COLUMN
from models.inference import build_lead_frame, score_lead


def test_score_lead_applies_saved_discretization_and_queries_posterior() -> None:
    model = DiscreteBayesianNetwork([("web_session_band", TARGET_COLUMN)])
    model.add_cpds(
        TabularCPD(
            variable="web_session_band",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"web_session_band": ["short", "long"]},
        ),
        TabularCPD(
            variable=TARGET_COLUMN,
            variable_card=2,
            values=[[0.9, 0.1], [0.1, 0.9]],
            evidence=["web_session_band"],
            evidence_card=[2],
            state_names={
                TARGET_COLUMN: ["false", "true"],
                "web_session_band": ["short", "long"],
            },
        ),
    )
    config = DiscretizationConfig(
        rules=[
            BinRule(
                source_column="web_session_seconds_clean",
                output_column="web_session_band",
                labels=["short", "long"],
                upper_bounds=[60.0],
            )
        ]
    )

    result = score_lead(
        model,
        {
            "domain": "acme.com",
            "marketing_channel": "Paid Search",
            "employees": 250,
            "country": "US",
            "web_session_seconds": 120.0,
        },
        config,
    )

    assert result.evidence_used == {"web_session_band": "long"}
    assert result.posterior_conversion_probability == pytest.approx(0.9)


def test_build_lead_frame_normalizes_training_compatible_names() -> None:
    frame = build_lead_frame(
        {
            "marketing_channel": "Paid Search",
            "campaign_tier": "Pilot",
            "country": "US",
            "employees": 25,
            "job_title": "Dir. Markting (interim)",
            "web_session_seconds": 100_000.0,
        }
    )
    row = frame.row(0, named=True)

    assert row["marketing_channel_clean"] == "paid_search"
    assert row["campaign_tier_clean"] == "pilot"
    assert row["region_clean"] == "NA"
    assert row["employee_size_bucket"] == "small"
    assert row["job_title_clean"] == "director marketing"
    assert row["web_session_seconds_clean"] == 86_400.0


def test_score_lead_rejects_blocked_leakage_fields() -> None:
    model = DiscreteBayesianNetwork()
    config = DiscretizationConfig(rules=[])

    with pytest.raises(ValueError, match="lead_score_clean"):
        score_lead(model, {"lead_score_clean": 99.0}, config)


def test_score_lead_scenarios_order_high_missing_and_low_engagement() -> None:
    model = DiscreteBayesianNetwork([("web_session_band", TARGET_COLUMN)])
    model.add_cpds(
        TabularCPD(
            variable="web_session_band",
            variable_card=3,
            values=[[0.4], [0.4], [0.2]],
            state_names={"web_session_band": ["short", "deep", "missing"]},
        ),
        TabularCPD(
            variable=TARGET_COLUMN,
            variable_card=2,
            values=[[0.9, 0.2, 0.65], [0.1, 0.8, 0.35]],
            evidence=["web_session_band"],
            evidence_card=[3],
            state_names={
                TARGET_COLUMN: ["false", "true"],
                "web_session_band": ["short", "deep", "missing"],
            },
        ),
    )
    config = DiscretizationConfig(
        rules=[
            BinRule(
                source_column="web_session_seconds_clean",
                output_column="web_session_band",
                labels=["short", "deep"],
                upper_bounds=[60.0],
            )
        ]
    )

    low = score_lead(model, {"domain": "low.test", "web_session_seconds": 30.0}, config)
    missing = score_lead(model, {"domain": "missing.test"}, config)
    high = score_lead(model, {"domain": "high.test", "web_session_seconds": 300.0}, config)

    assert high.posterior_conversion_probability > missing.posterior_conversion_probability
    assert missing.posterior_conversion_probability > low.posterior_conversion_probability
    assert low.evidence_used == {"web_session_band": "short"}
    assert missing.evidence_used == {"web_session_band": "missing"}
