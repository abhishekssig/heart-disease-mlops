import json

import pytest
from fastapi.testclient import TestClient

from heartml import config


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_dir / "model.joblib"))
    from heartml.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert "x-request-id" in r.headers


def test_predict_rejects_invalid_input(client, sample_patient):
    sample_patient["age"] = 250
    assert client.post("/predict", json=sample_patient).status_code == 422
    del sample_patient["age"]
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch_predict(client):
    payload = json.loads((config.ROOT_DIR / "sample_batch.json").read_text())
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()["results"]) == len(payload["patients"])


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text


def test_missing_model_returns_503(tmp_path, monkeypatch, sample_patient):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    from heartml.api import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=sample_patient).status_code == 503
