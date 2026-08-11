import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.preparation import prepare_training_data
from src.features.transformer import NYCGreenTaxiFeatureTransformer
from src.training.pipeline import TrainingPipeline, create_xgboost_model


def test_full_pipeline_fits_predicts_and_generates_feature_importance_plot(tmp_path):
    lookup = pd.DataFrame(
        [
            {"LocationID": 1, "Borough": "Manhattan", "Zone": "A", "latitude": 40.75, "longitude": -73.99},
            {"LocationID": 2, "Borough": "Queens", "Zone": "B", "latitude": 40.76, "longitude": -73.90},
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "VendorID": 1 + index % 2,
                "lpep_pickup_datetime": f"2026-02-{index + 1:02} 10:00:00",
                "lpep_dropoff_datetime": f"2026-02-{index + 1:02} 10:{10 + index:02}:00",
                "PULocationID": 1,
                "DOLocationID": 2,
                "trip_distance": 2.0,
                "passenger_count": 1,
                "trip_type": 1.0,
            }
            for index in range(6)
        ]
    )
    X_train, y_train = prepare_training_data(raw)
    pipeline = Pipeline(
        [
            ("features", NYCGreenTaxiFeatureTransformer(lookup_df=lookup)),
            ("model", create_xgboost_model({"n_estimators": 3, "max_depth": 2, "learning_rate": 0.1})),
        ]
    )
    pipeline.fit(X_train, y_train)

    request_data = X_train.drop(columns=[]).head(1)
    prediction = pipeline.predict(request_data)
    importance_file = tmp_path / "feature_importance.png"
    importance_table = TrainingPipeline.plot_feature_importance(pipeline, importance_file)

    assert prediction.shape == (1,)
    assert isinstance(float(prediction[0]), float)
    assert not importance_table.empty
    assert importance_file.exists()
