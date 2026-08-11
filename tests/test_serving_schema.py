from src.serving.schemas import PredictRequest, PredictResponse


def test_predict_request_schema_accepts_portfolio_payload():
    payload = {
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

    request = PredictRequest(**payload)

    assert len(request.trips) == 1


def test_predict_response_schema_exposes_model_metadata():
    response = PredictResponse(
        predictions=[12.34],
        model_name="green_taxi_duration_model",
        model_version=1,
        model_alias="champion",
        latency=0.01,
    )

    assert response.model_alias == "champion"
