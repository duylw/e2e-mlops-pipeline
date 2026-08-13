import pandas as pd

from src.utils.geo import calculate_haversine


def engineer_base_features(df_raw: pd.DataFrame, df_lookup: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time and lookup-derived features without dropping rows."""
    df = df_raw.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["lpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["lpep_pickup_datetime"].dt.month
    df["day_of_month"] = df["lpep_pickup_datetime"].dt.day

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

    dropoff_lookup = df_lookup[["LocationID", "Borough", "Zone", "latitude", "longitude"]]
    df = df.merge(dropoff_lookup, left_on="DOLocationID", right_on="LocationID", how="left")
    df = df.rename(
        columns={
            "Borough": "DO_Borough",
            "Zone": "DO_Zone",
            "latitude": "DO_lat",
            "longitude": "DO_long",
        }
    )

    df["estimated_distance_miles"] = calculate_haversine(
        df["PU_lat"], df["PU_long"], df["DO_lat"], df["DO_long"]
    )
    df["same_borough"] = (df["PU_Borough"] == df["DO_Borough"]).astype(int)
    df["PU_DO_route"] = df["PULocationID"].astype("string").fillna("__missing__").str.cat(
        df["DOLocationID"].astype("string").fillna("__missing__"), sep="_"
    )
    return df
