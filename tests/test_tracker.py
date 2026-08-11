import pandas as pd

from src.training.tracker import MlflowTrainingTracker


def test_signature_input_uses_float64_for_numeric_api_fields():
    raw = pd.DataFrame(
        [
            {
                "VendorID": 2,
                "lpep_pickup_datetime": "2026-01-15 08:30:00",
                "PULocationID": 74,
                "DOLocationID": 42,
                "passenger_count": 1,
                "trip_type": 1,
            }
        ]
    )

    signature_input = MlflowTrainingTracker._signature_input(raw)

    assert str(signature_input["VendorID"].dtype) == "int64"
    assert str(signature_input["PULocationID"].dtype) == "int64"
    assert str(signature_input["trip_type"].dtype) == "float64"
