"""Scoring-time preprocessing and pgmpy inference helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl
from pgmpy.inference import VariableElimination

from models.discretization import DiscretizationConfig, apply_discretization
from models.feature_contract import (
    DEFAULT_FEATURE_CONTRACT,
    TARGET_COLUMN,
    validate_feature_availability,
)
from models.structure import DEBUG_LATENT_COLUMN, LATENT_NODE

BLOCKED_REQUEST_FIELDS = frozenset(
    {
        DEBUG_LATENT_COLUMN,
        LATENT_NODE,
        "lead_score_band",
        "lead_score_clean",
        "replied_within_7d_bool",
    }
)

COMPANY_SIZE_ALIASES = {
    "small": "small",
    "smb": "small",
    "mid": "mid_market",
    "mid_market": "mid_market",
    "mid-market": "mid_market",
    "mid market": "mid_market",
    "commercial": "commercial",
    "enterprise": "enterprise",
}

COUNTRY_REGION_MAP = {
    "US": "NA",
    "USA": "NA",
    "CA": "NA",
    "CAN": "NA",
    "GB": "EMEA",
    "UK": "EMEA",
    "DE": "EMEA",
    "FR": "EMEA",
    "NL": "EMEA",
    "IN": "APAC",
    "JP": "APAC",
    "AU": "APAC",
    "SG": "APAC",
}


@dataclass(frozen=True)
class LeadInferenceResult:
    posterior_conversion_probability: float
    evidence_used: dict[str, str]


def score_lead(
    model: Any,
    lead: Mapping[str, Any],
    discretization_config: DiscretizationConfig,
) -> LeadInferenceResult:
    """Normalize one lead, select valid evidence, and query conversion posterior."""
    _reject_blocked_fields(lead)
    lead_frame = build_lead_frame(lead)
    evidence = prepare_evidence(model, lead_frame, discretization_config)
    probability = query_conversion_probability(model, evidence)
    return LeadInferenceResult(
        posterior_conversion_probability=probability,
        evidence_used=evidence,
    )


def build_lead_frame(lead: Mapping[str, Any]) -> pl.DataFrame:
    """Build a one-row frame using the same feature names as model training."""
    employees = lead.get("employees")
    return pl.DataFrame(
        {
            "marketing_channel_clean": [
                _normalize_marketing_channel(lead.get("marketing_channel"))
            ],
            "campaign_tier_clean": [_normalize_text_token(lead.get("campaign_tier"))],
            "region_clean": [_normalize_region(lead.get("region"), lead.get("country"))],
            "employee_size_bucket": [
                _normalize_company_size(lead.get("company_size"), employees)
            ],
            "job_title_clean": [_normalize_job_title(lead.get("job_title"))],
            "web_session_seconds_clean": [
                _normalize_session_seconds(lead.get("web_session_seconds"))
            ],
        }
    )


def prepare_evidence(
    model: Any,
    lead_frame: pl.DataFrame,
    discretization_config: DiscretizationConfig,
) -> dict[str, str]:
    """Apply saved bins and return allowed evidence columns present in the model."""
    discretized = apply_discretization(lead_frame, discretization_config)
    allowed_columns = DEFAULT_FEATURE_CONTRACT.allowed_evidence_columns
    validate_feature_availability(allowed_columns)
    model_nodes = set(model.nodes())
    selected_columns = [
        column
        for column in allowed_columns
        if column in discretized.columns and column in model_nodes
    ]
    if not selected_columns:
        raise ValueError("No allowed evidence columns are present in the model")

    row = discretized.select(selected_columns).row(0, named=True)
    evidence = {column: _to_state(value) for column, value in row.items()}
    validate_feature_availability(evidence.keys())
    return evidence


def query_conversion_probability(model: Any, evidence: Mapping[str, str]) -> float:
    """Run pgmpy variable elimination and return P(demo_requested_bool=true)."""
    if TARGET_COLUMN not in set(model.nodes()):
        raise ValueError(f"Model is missing target node: {TARGET_COLUMN}")

    inference = VariableElimination(model)
    posterior = inference.query(
        variables=[TARGET_COLUMN],
        evidence=dict(evidence),
        show_progress=False,
    )
    states = posterior.state_names.get(TARGET_COLUMN, [])
    if "true" not in states:
        raise ValueError(f"Target node {TARGET_COLUMN} has no 'true' state")

    true_index = states.index("true")
    values = posterior.values
    return float(values[true_index])


def _reject_blocked_fields(lead: Mapping[str, Any]) -> None:
    blocked = sorted(field for field in BLOCKED_REQUEST_FIELDS if field in lead)
    if blocked:
        raise ValueError(f"Blocked leakage/proxy fields in request: {blocked}")


def _normalize_marketing_channel(value: Any) -> str | None:
    text = _strip_text(value)
    if text is None:
        return None
    return re.sub(r"[\s-]+", "_", text.lower())


def _normalize_text_token(value: Any) -> str | None:
    text = _strip_text(value)
    if text is None:
        return None
    return text.lower()


def _normalize_job_title(value: Any) -> str | None:
    text = _strip_text(value)
    if text is None:
        return None
    text = text.replace("_", " ")
    text = text.replace("Markting", "Marketing")
    text = text.replace("Dir.", "Director")
    text = re.sub(r"\s*/\s*acting$", "", text)
    text = re.sub(r"\s*\(interim\)$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _normalize_region(region: Any, country: Any) -> str | None:
    region_text = _strip_text(region)
    if region_text:
        normalized = region_text.upper()
        return None if normalized == "UNKNOWN" else normalized

    country_text = _strip_text(country)
    if country_text is None:
        return None
    return COUNTRY_REGION_MAP.get(country_text.upper())


def _normalize_company_size(company_size: Any, employees: Any) -> str | None:
    size_text = _strip_text(company_size)
    if size_text is not None:
        normalized = size_text.lower().replace("_", " ")
        return COMPANY_SIZE_ALIASES.get(normalized, normalized.replace(" ", "_"))

    if employees is None:
        return None
    count = int(employees)
    if count >= 5000:
        return "enterprise"
    if count <= 50:
        return "small"
    if count <= 500:
        return "mid_market"
    return "commercial"


def _normalize_session_seconds(value: Any) -> float | None:
    if value is None:
        return None
    seconds = float(value)
    if math.isnan(seconds):
        return None
    if seconds < 0:
        return 0.0
    if seconds > 86_400:
        return 86_400.0
    return seconds


def _strip_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _to_state(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float) and math.isnan(value):
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
