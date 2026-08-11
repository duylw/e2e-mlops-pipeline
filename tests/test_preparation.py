import pandas as pd

from src.data.preparation import MODEL_INPUT_COLUMNS, prepare_training_data


def test_prepare_training_data_derives_duration_filters_invalid_rows_and_removes_leakage():
    df = pd.DataFrame(
        [
            {
                "VendorID": 1,
                "lpep_pickup_datetime": "2026-02-01 10:00:00",
                "lpep_dropoff_datetime": "2026-02-01 10:15:00",
                "PULocationID": 1,
                "DOLocationID": 2,
                "trip_distance": 2.0,
                "passenger_count": None,
                "trip_type": 1.0,
                "fare_amount": 10.0,
            },
            {
                "VendorID": 1,
                "lpep_pickup_datetime": "2026-02-01 10:00:00",
                "lpep_dropoff_datetime": "2026-02-01 09:59:00",
                "PULocationID": 1,
                "DOLocationID": 2,
                "trip_distance": 2.0,
                "passenger_count": 1,
                "trip_type": 1.0,
                "fare_amount": 10.0,
            },
        ]
    )

    X, y = prepare_training_data(df)

    assert len(X) == 1
    assert y.tolist() == [15.0]
    assert list(X.columns) == MODEL_INPUT_COLUMNS
    assert "fare_amount" not in X.columns
    assert "lpep_dropoff_datetime" not in X.columns
    assert "trip_distance" not in X.columns
