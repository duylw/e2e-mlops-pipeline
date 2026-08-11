import pandas as pd

LEAKAGE_AND_PAYMENT_COLUMNS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "congestion_surcharge",
    "cbd_congestion_fee",
    "ehail_fee",
    "store_and_fwd_flag",
    "RatecodeID",
]

MODEL_INPUT_COLUMNS = [
    "VendorID",
    "lpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_type",
]


def prepare_training_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create a leakage-free training matrix and duration target in minutes."""
    df = df_raw.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    df = df[(df["duration"] > 0) & (df["duration"] <= 300)]
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] >= 0]

    missing = set(MODEL_INPUT_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required model inputs: {sorted(missing)}")

    X = df.drop(columns=[*LEAKAGE_AND_PAYMENT_COLUMNS, "lpep_dropoff_datetime", "duration"], errors="ignore")
    X = X.loc[:, MODEL_INPUT_COLUMNS].copy()
    # The API accepts datetime strings, so use the same raw contract during training and serving.
    X["lpep_pickup_datetime"] = X["lpep_pickup_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return X.reset_index(drop=True), df["duration"].reset_index(drop=True)
