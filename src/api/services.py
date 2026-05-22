from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from api.schemas import PredictRequest, PredictResponse
from models.discretization import DiscretizationConfig, load_discretization_config
from models.inference import score_lead

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "model.pkl"
DISCRETIZATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "artifacts"
    / "discretization_config.json"
)

_model_cache: Any | None = None
_discretization_config_cache: DiscretizationConfig | None = None


def mock_clearbit_enrich(domain: str) -> dict[str, str | int | None]:
    """Return deterministic mock firmographics for a domain."""
    slug = domain.split(".")[0] or "unknown"
    return {
        "domain": domain,
        "company_name": slug.replace("-", " ").title(),
        "industry": "Software",
        "employees": 250,
        "country": "US",
    }


def predict_lead(body: PredictRequest) -> PredictResponse:
    """Run the end-to-end API prediction flow for a single lead."""
    model = load_bbn_model()
    discretization_config = load_bbn_discretization_config()
    firmographics, enriched = _firmographics_for_request(body)
    lead_payload = body.model_dump()
    lead_payload.update(
        {
            key: value
            for key, value in firmographics.items()
            if value is not None and lead_payload.get(key) is None
        }
    )
    result = score_lead(model, lead_payload, discretization_config)
    return PredictResponse(
        domain=body.domain,
        posterior_conversion_probability=result.posterior_conversion_probability,
        evidence_used=result.evidence_used,
        firmographics=firmographics,
        enriched=enriched,
    )


def load_bbn_model() -> Any:
    """Load the serialized pgmpy model from disk, with in-process caching."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not ARTIFACT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {ARTIFACT_PATH}. Run training first.",
        )
    with ARTIFACT_PATH.open("rb") as handle:
        _model_cache = pickle.load(handle)
    return _model_cache


def load_bbn_discretization_config() -> DiscretizationConfig:
    """Load the saved scoring-time discretization rules, with caching."""
    global _discretization_config_cache
    if _discretization_config_cache is not None:
        return _discretization_config_cache
    if not DISCRETIZATION_CONFIG_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Discretization artifact not found at "
                f"{DISCRETIZATION_CONFIG_PATH}. Run training first."
            ),
        )
    _discretization_config_cache = load_discretization_config(DISCRETIZATION_CONFIG_PATH)
    return _discretization_config_cache


def _firmographics_for_request(
    body: PredictRequest,
) -> tuple[dict[str, str | int | None], bool]:
    provided: dict[str, str | int | None] = {
        "domain": body.domain,
        "employees": body.employees,
        "company_size": body.company_size,
        "country": body.country,
        "region": body.region,
    }
    if not _needs_firmographic_enrichment(body):
        return provided, False

    enriched = mock_clearbit_enrich(body.domain)
    enriched["company_size"] = body.company_size
    enriched["region"] = body.region
    if body.employees is not None:
        enriched["employees"] = body.employees
    if body.country is not None:
        enriched["country"] = body.country
    return enriched, True


def _needs_firmographic_enrichment(body: PredictRequest) -> bool:
    has_company_size = body.company_size is not None or body.employees is not None
    has_geo = body.region is not None or body.country is not None
    return not (has_company_size and has_geo)
