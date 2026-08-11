import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.cleaning import clean_taxi_data, clean_taxi_data_inference
from src.features.engineering import (
    build_speed_profile,
    engineer_base_features,
    fit_feature_artifacts,
    transform_advanced_features,
)


class NYCGreenTaxiPreprocessor(BaseEstimator, TransformerMixin):
    """Train-fitted feature pipeline shared by batch training and online inference."""

    def __init__(self, df_lookup: pd.DataFrame, is_inference: bool = False):
        self.df_lookup = df_lookup
        self.is_inference = is_inference
        self.trip_type_mode = 1
        self.speed_profile = None
        self.global_mean_speed = None
        self.zone_freq_map = None
        self.ohe_categories = {}
        self.train_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        df_cleaned = clean_taxi_data(X)
        df_base = engineer_base_features(df_cleaned, self.df_lookup)
        self.speed_profile = build_speed_profile(df_base)

        artifacts = fit_feature_artifacts(df_base, self.speed_profile)
        self.global_mean_speed = artifacts["global_mean_speed"]
        self.zone_freq_map = artifacts["zone_freq_map"]
        self.ohe_categories = artifacts["ohe_categories"]
        self.trip_type_mode = artifacts["trip_type_mode"]

        df_final = transform_advanced_features(df_base, artifacts)
        self.train_columns = [col for col in df_final.columns if col != "duration"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.is_inference:
            df_cleaned = clean_taxi_data_inference(X, trip_type_mode_fallback=self.trip_type_mode)
        else:
            df_cleaned = clean_taxi_data(X)

        df_base = engineer_base_features(df_cleaned, self.df_lookup)
        return transform_advanced_features(df_base, self._artifacts())

    def _artifacts(self) -> dict:
        return {
            "speed_profile": self.speed_profile,
            "global_mean_speed": self.global_mean_speed,
            "zone_freq_map": self.zone_freq_map,
            "ohe_categories": self.ohe_categories,
            "trip_type_mode": self.trip_type_mode,
            "train_columns": self.train_columns,
        }
