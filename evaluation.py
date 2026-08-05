import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.preprocessing import (
    clean_taxi_data,
    engineer_base_features,
    transform_advanced_features,
    split_time_series_data,
    prepare_X_y
)

def evaluate_model(model_path="xgb_model.json", artifacts_path="inference_artifacts.pkl", lookup_csv="nyc_taxi_lookup_with_coords.csv"):
    print("--- STARTING EVALUATION PIPELINE ---")
    
    # Check if artifacts exist
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"Trained model or artifacts not found. Please run 'train.py' first to generate {model_path} and {artifacts_path}."
        )

    # 1. Load coordinate lookup
    if not os.path.exists(lookup_csv):
        raise FileNotFoundError(f"Lookup coordinate file {lookup_csv} not found. Please run 'train.py' first.")
    df_lookup_with_loc = pd.read_csv(lookup_csv)

    # 2. Download and read Green Taxi test data (February 2026)
    test_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2026-02.parquet"
    print(f"Loading raw test data from {test_data_url}...")
    df_test = pd.read_parquet(test_data_url)
    print(f"Loaded raw test data of shape: {df_test.shape}")

    # 3. Chronological Time-Series Split (use first portion for evaluation)
    test_base, _ = split_time_series_data(df_test, '2026-02-15')
    print(f"Selected test base shape: {test_base.shape}")

    # 4. Clean test data
    test_base = clean_taxi_data(test_base, target_year=2026, target_month=2)
    
    # 5. Base feature extraction
    test_base_features = engineer_base_features(test_base, df_lookup_with_loc)

    # 6. Load artifacts and transform advanced features
    print(f"Loading preprocessing artifacts from {artifacts_path}...")
    artifacts_final = joblib.load(artifacts_path)
    test_final_features = transform_advanced_features(test_base_features, artifacts_final)

    # 7. Split features and labels
    X_test, y_test = prepare_X_y(test_final_features)
    print(f"Test feature set shape: {X_test.shape}")

    # 8. Load trained XGBoost model
    print(f"Loading trained XGBoost model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # 9. Predict and calculate metrics
    print("Running predictions on test set...")
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("\nEVALUATION METRICS:")
    print(f"Test RMSE: {rmse:.4f} mins")
    print(f"Test MAE: {mae:.4f} mins")

    return {
        'rmse': rmse,
        'mae': mae
    }

if __name__ == "__main__":
    evaluate_model()