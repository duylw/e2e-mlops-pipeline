import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.engineering import engineer_base_features


class NYCGreenTaxiFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform raw taxi request fields into a stable numeric feature matrix."""

    categorical_columns = ("VendorID", "trip_type", "PU_Borough")

    def __init__(self, lookup_df: pd.DataFrame):
        self.lookup_df = lookup_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("y is required to fit NYCGreenTaxiFeatureTransformer")

        base = engineer_base_features(X, self.lookup_df)
        target = pd.Series(y).reset_index(drop=True)
        base = base.reset_index(drop=True)

        valid_distance = base["estimated_distance_miles"].where(base["estimated_distance_miles"] > 0)
        self.global_mean_distance_ = valid_distance.median()
        if pd.isna(self.global_mean_distance_):
            self.global_mean_distance_ = 1.0
        base["estimated_distance_miles"] = base["estimated_distance_miles"].fillna(self.global_mean_distance_)

        speed = base["estimated_distance_miles"].div(target)
        valid_speed = base.assign(_speed=speed)[lambda df: df["_speed"].between(0, 2)]
        self.speed_profile_ = (
            valid_speed.groupby(["pickup_hour", "PU_Borough"], dropna=False)["_speed"].mean().reset_index(name="historical_avg_speed")
        )
        self.global_mean_speed_ = self.speed_profile_["historical_avg_speed"].mean()
        if pd.isna(self.global_mean_speed_):
            self.global_mean_speed_ = 0.25

        self.zone_frequency_ = base["PU_Zone"].value_counts(normalize=True, dropna=False).to_dict()
        trip_type = X["trip_type"] if "trip_type" in X else pd.Series(dtype=float)
        self.trip_type_fallback_ = trip_type.mode().iloc[0] if not trip_type.mode().empty else 1.0
        self.categories_ = {
            column: sorted(base[column].fillna("__missing__").astype(str).unique().tolist())
            for column in self.categorical_columns
            if column in base
        }
        self.feature_columns_ = list(self._make_features(base).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        raw = X.copy()
        if "passenger_count" not in raw:
            raw["passenger_count"] = 1.0
        raw["passenger_count"] = pd.to_numeric(raw["passenger_count"], errors="coerce").fillna(1.0)
        if "trip_type" not in raw:
            raw["trip_type"] = self.trip_type_fallback_
        raw["trip_type"] = pd.to_numeric(raw["trip_type"], errors="coerce").fillna(self.trip_type_fallback_)

        base = engineer_base_features(raw, self.lookup_df)
        base["estimated_distance_miles"] = base["estimated_distance_miles"].fillna(self.global_mean_distance_)
        features = self._make_features(base)
        return features.reindex(columns=self.feature_columns_, fill_value=0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self.feature_columns_, dtype=object)

    def _make_features(self, base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()
        df = df.merge(self.speed_profile_, on=["pickup_hour", "PU_Borough"], how="left")
        df["historical_avg_speed"] = df["historical_avg_speed"].fillna(self.global_mean_speed_)
        df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df["PU_Zone_frequency"] = df["PU_Zone"].map(self.zone_frequency_).fillna(0.0)

        for column, categories in self.categories_.items():
            values = df[column].fillna("__missing__").astype(str)
            for category in categories[1:]:
                df[f"{column}_{category}"] = (values == category).astype(int)

        return df.drop(
            columns=[
                "VendorID",
                "trip_type",
                "PU_Borough",
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
            ],
            errors="ignore",
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "feature_columns_"):
            raise ValueError("NYCGreenTaxiFeatureTransformer must be fitted before transform")
