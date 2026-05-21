from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.services import load_bbn_model, mock_clearbit_enrich

app = FastAPI(
    title="Latent Sentiment BBN",
    description="Infer latent sentiment from firmographics and engagement signals.",
    version="0.1.0",
)


class EnrichRequest(BaseModel):
    domain: str = Field(..., examples=["acme.com"])


class PredictRequest(BaseModel):
    domain: str
    page_views: int = Field(ge=0, default=0)
    email_opens: int = Field(ge=0, default=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/enrich")
def enrich(body: EnrichRequest) -> dict[str, str | int | None]:
    return mock_clearbit_enrich(body.domain)


@app.post("/predict")
def predict(body: PredictRequest) -> dict[str, object]:
    model = load_bbn_model()
    _ = model  # inference wiring lands in a follow-up change
    firmographics = mock_clearbit_enrich(body.domain)
    return {
        "domain": body.domain,
        "firmographics": firmographics,
        "signals": {
            "page_views": body.page_views,
            "email_opens": body.email_opens,
        },
        "latent_sentiment": "neutral",
        "note": "Placeholder response until BBN inference is connected.",
    }


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
