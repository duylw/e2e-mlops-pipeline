import numpy as np
import pandas as pd

from src.monitoring.data import (
    INPUT_MONITORING_COLUMNS,
    MISSING_CATEGORICAL_LABEL,
    load_monitoring_batch,
)
from src.monitoring.reports import build_input_report


class ConstantModel:
    def predict(self, X):
        return np.full(len(X), 12.5)


def labeled_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "VendorID": 1,
                "lpep_pickup_datetime": f"2026-02-0{index} 08:00:00",
                "lpep_dropoff_datetime": f"2026-02-0{index} 08:{10 + index:02}:00",
                "PULocationID": 1,
                "DOLocationID": 2,
                "trip_distance": 2.0,
                "passenger_count": 1,
                "trip_type": 1.0,
            }
            for index in range(1, 5)
        ]
    )


def test_labeled_tlc_batch_builds_input_and_quality_frames(tmp_path):
    path = tmp_path / "labeled.parquet"
    labeled_raw_data().to_parquet(path, index=False)

    batch = load_monitoring_batch(path, ConstantModel(), max_rows=10, random_seed=42)

    assert list(batch.input_frame.columns) == INPUT_MONITORING_COLUMNS
    assert batch.quality_frame is not None
    assert batch.quality_frame["duration"].tolist() == [11.0, 12.0, 13.0, 14.0]
    assert batch.quality_frame["prediction"].tolist() == [12.5, 12.5, 12.5, 12.5]
    assert str(batch.input_frame["trip_type"].dtype) == "string"
    assert build_input_report(batch.input_frame, batch.input_frame, drift_warning_share=0.3).metrics[
        "input_drifted_columns_share"
    ] == 0.0


def test_api_shaped_batch_runs_without_labels(tmp_path):
    path = tmp_path / "unlabeled.parquet"
    raw = labeled_raw_data().drop(columns=["lpep_dropoff_datetime", "trip_distance"])
    raw.to_parquet(path, index=False)

    batch = load_monitoring_batch(path, ConstantModel(), max_rows=10, random_seed=42)

    assert len(batch.input_frame) == 4
    assert batch.quality_frame is None


def test_categorical_missing_values_are_evidently_compatible(tmp_path):
    path = tmp_path / "missing-category.parquet"
    raw = labeled_raw_data()
    raw.loc[0, "trip_type"] = None
    raw.to_parquet(path, index=False)

    batch = load_monitoring_batch(path, ConstantModel(), max_rows=10, random_seed=42)

    assert batch.input_frame.loc[0, "trip_type"] == MISSING_CATEGORICAL_LABEL
    assert build_input_report(batch.input_frame, batch.input_frame, drift_warning_share=0.3).metrics[
        "input_drifted_columns_share"
    ] == 0.0
