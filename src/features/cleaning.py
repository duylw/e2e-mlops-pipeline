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


def clean_taxi_data(df_raw: pd.DataFrame, target_year: int | None = None, target_month: int | None = None) -> pd.DataFrame:
    """Clean raw green taxi data and derive the duration target in minutes."""
    df = df_raw.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    if target_year is not None and target_month is not None:
        start_date = pd.Timestamp(year=target_year, month=target_month, day=1)
        end_date = start_date + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        df = df[(df["lpep_pickup_datetime"] >= start_date) & (df["lpep_pickup_datetime"] <= end_date)]

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    df = df[(df["duration"] > 0) & (df["duration"] <= 300)]
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] >= 0]

    df["passenger_count"] = df["passenger_count"].where(df["passenger_count"].notna(), 1)
    if "trip_type" in df.columns and not df["trip_type"].mode().empty:
        df["trip_type"] = df["trip_type"].where(df["trip_type"].notna(), df["trip_type"].mode()[0])

    return df.drop(columns=[*LEAKAGE_AND_PAYMENT_COLUMNS, "lpep_dropoff_datetime"], errors="ignore")


def clean_taxi_data_inference(df_raw: pd.DataFrame, trip_type_mode_fallback: float = 1) -> pd.DataFrame:
    """Clean request-time data without relying on target or dropoff fields."""
    df = df_raw.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["passenger_count"] = df["passenger_count"].where(df["passenger_count"].notna(), 1)
    if "trip_type" in df.columns:
        df["trip_type"] = df["trip_type"].where(df["trip_type"].notna(), trip_type_mode_fallback)

    return df.drop(columns=[*LEAKAGE_AND_PAYMENT_COLUMNS, "trip_distance"], errors="ignore")
