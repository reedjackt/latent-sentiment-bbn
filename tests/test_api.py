from fastapi.testclient import TestClient

from api.main import app

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
