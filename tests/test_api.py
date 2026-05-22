from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import PredictRequest
import api.services as services
from models.discretization import DiscretizationConfig
from models.inference import LeadInferenceResult

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_enrich() -> None:
    response = client.post("/enrich", json={"domain": "acme.com"})
    assert response.status_code == 200
    body = response.json()
    assert body["domain"] == "acme.com"
    assert body["company_name"] == "Acme"


def test_predict_returns_posterior_without_enrichment(monkeypatch) -> None:
    monkeypatch.setattr(services, "load_bbn_model", lambda: object())
    monkeypatch.setattr(
        services,
        "load_bbn_discretization_config",
        lambda: DiscretizationConfig(rules=[]),
    )
    captured: dict[str, object] = {}

    def fake_score_lead(
        model: object,
        lead: dict[str, object],
        config: DiscretizationConfig,
    ) -> LeadInferenceResult:
        captured["lead"] = lead
        captured["config"] = config
        return LeadInferenceResult(
            posterior_conversion_probability=0.82,
            evidence_used={"employee_size_bucket": "mid_market"},
        )

    monkeypatch.setattr(services, "score_lead", fake_score_lead)

    response = client.post(
        "/predict",
        json={
            "domain": "acme.com",
            "marketing_channel": "Paid Search",
            "campaign_tier": "tier1_enterprise",
            "region": "NA",
            "job_title": "VP Marketing",
            "employees": 250,
            "web_session_seconds": 300.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["domain"] == "acme.com"
    assert body["posterior_conversion_probability"] == 0.82
    assert body["evidence_used"] == {"employee_size_bucket": "mid_market"}
    assert body["enriched"] is False
    assert captured["lead"]["employees"] == 250


def test_predict_enriches_missing_firmographics(monkeypatch) -> None:
    monkeypatch.setattr(services, "load_bbn_model", lambda: object())
    monkeypatch.setattr(
        services,
        "load_bbn_discretization_config",
        lambda: DiscretizationConfig(rules=[]),
    )
    enriched_domains: list[str] = []
    captured: dict[str, object] = {}

    def fake_enrich(domain: str) -> dict[str, str | int | None]:
        enriched_domains.append(domain)
        return {
            "domain": domain,
            "company_name": "Acme",
            "industry": "Software",
            "employees": 250,
            "country": "US",
        }

    def fake_score_lead(
        model: object,
        lead: dict[str, object],
        config: DiscretizationConfig,
    ) -> LeadInferenceResult:
        captured["lead"] = lead
        return LeadInferenceResult(
            posterior_conversion_probability=0.64,
            evidence_used={"region_clean": "NA", "employee_size_bucket": "mid_market"},
        )

    monkeypatch.setattr(services, "mock_clearbit_enrich", fake_enrich)
    monkeypatch.setattr(services, "score_lead", fake_score_lead)

    response = client.post(
        "/predict",
        json={
            "domain": "acme.com",
            "marketing_channel": "organic",
            "campaign_tier": "pilot",
            "job_title": "Manager",
            "web_session_seconds": 45.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["enriched"] is True
    assert enriched_domains == ["acme.com"]
    assert body["firmographics"]["employees"] == 250
    assert captured["lead"]["employees"] == 250
    assert captured["lead"]["country"] == "US"


def test_predict_rejects_leakage_fields() -> None:
    response = client.post(
        "/predict",
        json={
            "domain": "acme.com",
            "lead_score_clean": 97.0,
        },
    )

    assert response.status_code == 422


def test_predict_surfaces_missing_artifact(monkeypatch) -> None:
    def fake_load_bbn_model() -> object:
        raise HTTPException(status_code=503, detail="model artifact missing")

    monkeypatch.setattr(services, "load_bbn_model", fake_load_bbn_model)

    response = client.post("/predict", json={"domain": "acme.com"})

    assert response.status_code == 503
    assert response.json()["detail"] == "model artifact missing"


def test_predict_lead_maps_inference_errors_to_unprocessable(monkeypatch) -> None:
    def fake_predict_lead(body: PredictRequest) -> object:
        raise ValueError("bad evidence")

    monkeypatch.setattr("api.main.predict_lead", fake_predict_lead)

    response = client.post("/predict", json={"domain": "acme.com"})

    assert response.status_code == 422
    assert response.json()["detail"] == "bad evidence"
