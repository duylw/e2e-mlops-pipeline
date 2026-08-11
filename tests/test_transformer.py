import pandas as pd

from src.data.preparation import prepare_training_data
from src.features.transformer import NYCGreenTaxiFeatureTransformer


def lookup_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"LocationID": 1, "Borough": "Manhattan", "Zone": "A", "latitude": 40.75, "longitude": -73.99},
            {"LocationID": 2, "Borough": "Queens", "Zone": "B", "latitude": 40.76, "longitude": -73.90},
            {"LocationID": 3, "Borough": "Brooklyn", "Zone": "C", "latitude": 40.65, "longitude": -73.95},
        ]
    )


def raw_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "VendorID": vendor,
                "lpep_pickup_datetime": f"2026-02-0{day} 10:00:00",
                "lpep_dropoff_datetime": f"2026-02-0{day} 10:{duration:02}:00",
                "PULocationID": pickup,
                "DOLocationID": dropoff,
                "trip_distance": distance,
                "passenger_count": 1,
                "trip_type": 1.0,
            }
            for vendor, day, duration, pickup, dropoff, distance in [
                (1, 1, 12, 1, 2, 2.0),
                (2, 2, 18, 2, 3, 3.0),
                (1, 3, 15, 3, 1, 2.5),
            ]
        ]
    )


def test_transformer_accepts_serving_schema_and_keeps_stable_columns():
    X_train, y_train = prepare_training_data(raw_training_df())
    transformer = NYCGreenTaxiFeatureTransformer(lookup_df=lookup_df()).fit(X_train, y_train)

    inference = pd.DataFrame(
        [
            {
                "VendorID": 99,
                "lpep_pickup_datetime": "2026-02-04 08:00:00",
                "PULocationID": 1,
                "DOLocationID": 3,
                "passenger_count": None,
                "trip_type": None,
            }
        ]
    )

    transformed_train = transformer.transform(X_train)
    transformed_inference = transformer.transform(inference)

    assert list(transformed_inference.columns) == list(transformer.get_feature_names_out())
    assert list(transformed_train.columns) == list(transformer.get_feature_names_out())
    assert len(transformed_inference) == 1
