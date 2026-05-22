from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EnrichRequest(BaseModel):
    domain: str = Field(..., examples=["acme.com"])


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(..., examples=["acme.com"])
    marketing_channel: str | None = None
    campaign_tier: str | None = None
    region: str | None = None
    country: str | None = None
    job_title: str | None = None
    company_size: str | None = None
    employees: int | None = Field(default=None, ge=0)
    web_session_seconds: float | None = Field(default=None, ge=0)
    page_views: int = Field(ge=0, default=0)
    email_opens: int = Field(ge=0, default=0)


class PredictResponse(BaseModel):
    domain: str
    posterior_conversion_probability: float = Field(ge=0.0, le=1.0)
    evidence_used: dict[str, str]
    firmographics: dict[str, str | int | None]
    enriched: bool
