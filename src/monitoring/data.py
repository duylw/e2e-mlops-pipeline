from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preparation import MODEL_INPUT_COLUMNS, prepare_training_data

INPUT_MONITORING_COLUMNS = [
    "VendorID",
    "PULocationID",
    "DOLocationID",
    "trip_type",
    "pickup_hour",
    "pickup_day_of_week",
    "passenger_count",
    "lpep_pickup_datetime",
]
CATEGORICAL_MONITORING_COLUMNS = [
    "VendorID",
    "PULocationID",
    "DOLocationID",
    "trip_type",
    "pickup_hour",
    "pickup_day_of_week",
]
MISSING_CATEGORICAL_LABEL = "__MISSING__"
DRIFT_MONITORING_COLUMNS = [column for column in INPUT_MONITORING_COLUMNS if column != "lpep_pickup_datetime"]


@dataclass(frozen=True)
class MonitoringBatch:
    input_frame: pd.DataFrame
    quality_frame: pd.DataFrame | None


def load_monitoring_batch(
    path: str | Path,
    model,
    max_rows: int,
    random_seed: int,
) -> MonitoringBatch:
    raw_df = pd.read_parquet(path)
    X, target = _prepare_model_inputs(raw_df)
    X, target = _sample(X, target, max_rows=max_rows, random_seed=random_seed)
    predictions = np.asarray(model.predict(X), dtype=float).reshape(-1)

    input_frame = _build_input_frame(X)
    if target is None:
        return MonitoringBatch(input_frame=input_frame, quality_frame=None)

    quality_frame = input_frame.assign(duration=target.astype(float).to_numpy(), prediction=predictions)
    return MonitoringBatch(input_frame=input_frame, quality_frame=quality_frame)


def _prepare_model_inputs(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    if "lpep_dropoff_datetime" in raw_df.columns:
        if "trip_distance" not in raw_df.columns:
            raise ValueError("Labeled monitoring data must contain trip_distance for training-equivalent filtering.")
        return prepare_training_data(raw_df)

    missing = set(MODEL_INPUT_COLUMNS).difference(raw_df.columns)
    if missing:
        raise ValueError(f"Current data is missing required serving fields: {sorted(missing)}")
    X = raw_df.loc[:, MODEL_INPUT_COLUMNS].copy()
    X["lpep_pickup_datetime"] = pd.to_datetime(X["lpep_pickup_datetime"], errors="raise").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return X.reset_index(drop=True), None


def _sample(
    X: pd.DataFrame,
    target: pd.Series | None,
    max_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if len(X) <= max_rows:
        return X.reset_index(drop=True), None if target is None else target.reset_index(drop=True)
    sampled_index = X.sample(n=max_rows, random_state=random_seed).index
    sampled_X = X.loc[sampled_index].reset_index(drop=True)
    sampled_target = None if target is None else target.loc[sampled_index].reset_index(drop=True)
    return sampled_X, sampled_target


def _build_input_frame(X: pd.DataFrame) -> pd.DataFrame:
    pickup_datetime = pd.to_datetime(X["lpep_pickup_datetime"], errors="raise")
    input_frame = pd.DataFrame(
        {
            "VendorID": X["VendorID"],
            "PULocationID": X["PULocationID"],
            "DOLocationID": X["DOLocationID"],
            "trip_type": X["trip_type"],
            "pickup_hour": pickup_datetime.dt.hour,
            "pickup_day_of_week": pickup_datetime.dt.dayofweek,
            "passenger_count": X["passenger_count"],
            "lpep_pickup_datetime": pickup_datetime,
        }
    ).loc[:, INPUT_MONITORING_COLUMNS]
    # Evidently 0.7 requires categorical labels to be concrete Python-compatible values.
    # A sentinel preserves missing-category drift without passing pandas.NA into Pydantic.
    input_frame[CATEGORICAL_MONITORING_COLUMNS] = input_frame[CATEGORICAL_MONITORING_COLUMNS].fillna(
        MISSING_CATEGORICAL_LABEL
    ).astype("string")
    return input_frame
