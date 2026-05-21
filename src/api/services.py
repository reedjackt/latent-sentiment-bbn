from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from fastapi import HTTPException

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "model.pkl"

_model_cache: Any | None = None


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
