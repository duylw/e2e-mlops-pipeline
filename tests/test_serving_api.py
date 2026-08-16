import numpy as np
from fastapi.testclient import TestClient

from src.serving import main as serving_main


class ConstantPipeline:
    def predict(self, frame):
        return np.full(len(frame), 12.34)


def portfolio_payload() -> dict:
    return {
        "trips": [
            {
                "VendorID": 2,
                "lpep_pickup_datetime": "2026-01-15 08:30:00",
                "PULocationID": 74,
                "DOLocationID": 42,
                "passenger_count": 1,
                "trip_type": 1,
            }
        ]
    }


def test_api_serves_loaded_pipeline_and_preserves_prediction_parity(monkeypatch):
    pipeline = ConstantPipeline()
    metadata = {"model_name": "green_taxi_duration_model", "model_version": 7, "model_alias": "champion"}
    monkeypatch.setattr(serving_main, "load_champion_model", lambda: (pipeline, metadata))

    with TestClient(serving_main.app) as client:
        assert client.get("/health").json() == {"status": "healthy", "model_loaded": True}
        assert client.get("/metadata").json() == {**metadata, "model_loaded": True}
        response = client.post("/predict", json=portfolio_payload())

    assert response.status_code == 200
    assert response.json()["predictions"] == [12.34]
    assert response.json()["model_version"] == 7


def test_api_uses_optional_input_fallbacks_and_rejects_invalid_datetime(monkeypatch):
    monkeypatch.setattr(
        serving_main,
        "load_champion_model",
        lambda: (ConstantPipeline(), {"model_name": "model", "model_version": 1, "model_alias": "champion"}),
    )
    payload = portfolio_payload()
    payload["trips"][0].pop("passenger_count")
    payload["trips"][0].pop("trip_type")

    with TestClient(serving_main.app) as client:
        assert client.post("/predict", json=payload).status_code == 200
        invalid_payload = portfolio_payload()
        invalid_payload["trips"][0]["lpep_pickup_datetime"] = "15/01/2026"
        assert client.post("/predict", json=invalid_payload).status_code == 422


def test_api_returns_503_when_registered_model_cannot_load(monkeypatch):
    def raise_loading_error():
        raise ValueError("champion model is unavailable")

    monkeypatch.setattr(serving_main, "load_champion_model", raise_loading_error)

    with TestClient(serving_main.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=portfolio_payload()).status_code == 503
