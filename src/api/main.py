from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import EnrichRequest, PredictRequest, PredictResponse
from api.services import mock_clearbit_enrich, predict_lead

app = FastAPI(
    title="Latent Sentiment BBN",
    description="Infer latent sentiment from firmographics and engagement signals.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/enrich")
def enrich(body: EnrichRequest) -> dict[str, str | int | None]:
    return mock_clearbit_enrich(body.domain)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    try:
        return predict_lead(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
