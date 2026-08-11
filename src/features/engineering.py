import numpy as np
import pandas as pd

from src.utils.geo import calculate_haversine


def engineer_base_features(df_cleaned: pd.DataFrame, df_lookup: pd.DataFrame) -> pd.DataFrame:
    """Add time and location-derived features before learned feature transforms."""
    df = df_cleaned.copy()
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["lpep_pickup_datetime"].dt.dayofweek

    pickup_lookup = df_lookup[["LocationID", "Borough", "Zone", "latitude", "longitude"]]
    df = df.merge(pickup_lookup, left_on="PULocationID", right_on="LocationID", how="left")
    df = df.rename(
        columns={
            "Borough": "PU_Borough",
            "Zone": "PU_Zone",
            "latitude": "PU_lat",
            "longitude": "PU_long",
        }
    )

    dropoff_lookup = df_lookup[["LocationID", "latitude", "longitude"]]
    df = df.merge(dropoff_lookup, left_on="DOLocationID", right_on="LocationID", how="left")
    df = df.rename(columns={"latitude": "DO_lat", "longitude": "DO_long"})

    df = df.dropna(subset=["PU_lat", "PU_long", "DO_lat", "DO_long"])
    df["estimated_distance_miles"] = calculate_haversine(
        df["PU_lat"], df["PU_long"], df["DO_lat"], df["DO_long"]
    )
    return df[df["estimated_distance_miles"] > 0]


def build_speed_profile(df_base: pd.DataFrame) -> pd.DataFrame:
    """Learn historical average speed by pickup hour and pickup borough."""
    df = df_base.copy()
    df["speed"] = df["trip_distance"] / df["duration"]
    df = df[(df["speed"] > 0) & (df["speed"] < 2)]
    profile = df.groupby(["pickup_hour", "PU_Borough"])["speed"].mean().reset_index()
    return profile.rename(columns={"speed": "historical_avg_speed"})


def fit_feature_artifacts(
    df_base_train: pd.DataFrame,
    speed_profile: pd.DataFrame,
    trip_type_mode: float | None = None,
) -> dict:
    """Learn train-only feature artifacts used during transform and inference."""
    df = df_base_train.copy()
    ohe_categories = {}
    for col in ["VendorID", "trip_type", "PU_Borough"]:
        if col in df.columns:
            ohe_categories[col] = sorted(df[col].astype(str).unique().tolist())

    if trip_type_mode is None and "trip_type" in df.columns and not df["trip_type"].mode().empty:
        trip_type_mode = df["trip_type"].mode()[0]

    return {
        "speed_profile": speed_profile,
        "global_mean_speed": speed_profile["historical_avg_speed"].mean(),
        "zone_freq_map": df["PU_Zone"].value_counts(normalize=True).to_dict(),
        "ohe_categories": ohe_categories,
        "trip_type_mode": trip_type_mode,
        "train_columns": None,
    }


def transform_advanced_features(df_base: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply learned feature transforms with stable training column alignment."""
    df = df_base.copy()
    df = df.merge(artifacts["speed_profile"], on=["pickup_hour", "PU_Borough"], how="left")
    df["historical_avg_speed"] = df["historical_avg_speed"].fillna(artifacts["global_mean_speed"])

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["PU_Zone_frequency"] = df["PU_Zone"].map(artifacts["zone_freq_map"]).fillna(0)

    for col, categories in artifacts["ohe_categories"].items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        for category in categories[1:]:
            df[f"{col}_{category}"] = (df[col] == category).astype(int)
        df = df.drop(columns=[col])

    df = df.drop(
        columns=[
            "PULocationID",
            "DOLocationID",
            "LocationID_x",
            "LocationID_y",
            "PU_lat",
            "PU_long",
            "DO_lat",
            "DO_long",
            "lpep_pickup_datetime",
            "PU_Zone",
            "pickup_hour",
            "day_of_week",
            "trip_distance",
        ],
        errors="ignore",
    )

    if artifacts["train_columns"] is not None:
        keep_target = ["duration"] if "duration" in df.columns else []
        df = df.reindex(columns=artifacts["train_columns"] + keep_target, fill_value=0)
    return df


def prepare_X_y(df: pd.DataFrame, target_col: str = "duration") -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]), df[target_col]
