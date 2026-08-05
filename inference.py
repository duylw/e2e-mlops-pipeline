import os
import joblib
import pandas as pd
import xgboost as xgb

from src.preprocessing import clean_taxi_data_inference, engineer_base_features, transform_advanced_features

def run_inference(df_raw, df_lookup_with_loc, artifacts_path="artifacts/preprocessor.pkl", model_path="models/model.pkl", model=None):
    """
    Cleans raw inference payloads, processes advanced features, and returns model predictions.
    """
    # 1. Load model if not passed
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please run DVC pipeline first.")
        model = joblib.load(model_path)

    # 2. Load feature artifacts
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts file {artifacts_path} not found. Please run DVC pipeline first.")
    artifacts = joblib.load(artifacts_path)

    # 3. Clean raw inference input
    df_clean = clean_taxi_data_inference(
        df_raw, trip_type_mode_fallback=artifacts.get('trip_type_mode', 1)
    )

    # 4. Extract base features using coordinates lookup
    df_base = engineer_base_features(df_clean, df_lookup_with_loc)
    if df_base.empty:
        raise ValueError("All records filtered out. Check if PULocationID/DOLocationID exist in lookup coordinates.")

    # 5. Extract advanced features mapping speed profile and categorical list
    df_final = transform_advanced_features(df_base, artifacts)

    # 6. Extract features matrix aligned with training columns
    X_infer = df_final.drop(columns=['duration'], errors='ignore')
    X_infer = X_infer.reindex(columns=artifacts['train_columns'], fill_value=0)

    # 7. Predict duration in minutes
    preds = model.predict(X_infer)
    return preds

def main():
    print("--- STARTING INFERENCE VALIDATION ---")

    # Mock inputs corresponding to the TLC schema
    sample_inference_data = pd.DataFrame([
        {
            "VendorID": 2,
            "lpep_pickup_datetime": "2026-01-15 08:30:00",   # Morning rush hour
            "PULocationID": 74,    # East Harlem North (Manhattan)
            "DOLocationID": 42,    # Central Harlem
            "passenger_count": 1,
            "trip_type": 1
        },
        {
            "VendorID": 1,
            "lpep_pickup_datetime": "2026-01-15 23:45:00",   # Late night
            "PULocationID": 7,     # Astoria (Queens)
            "DOLocationID": 179,   # Steinway
            "passenger_count": 2,
            "trip_type": 1
        },
        {
            "VendorID": 2,
            "lpep_pickup_datetime": "2026-01-16 17:15:00",   # Evening rush hour
            "PULocationID": 41,    # Central Harlem
            "DOLocationID": 244,   # Washington Heights South
            "passenger_count": 1,
            "trip_type": 2
        },
        {
            "VendorID": 1,
            "lpep_pickup_datetime": "2026-01-17 03:00:00",   # Early morning, low traffic
            "PULocationID": 82,    # Elmhurst
            "DOLocationID": 129,   # Jackson Heights
            "passenger_count": None,   # Tests null value fallback
            "trip_type": None
        },
    ])

    lookup_csv = "data/raw/nyc_taxi_lookup_with_coords.csv"
    if not os.path.exists(lookup_csv):
        raise FileNotFoundError(f"Lookup file {lookup_csv} not found. Please run DVC pipeline first.")
    df_lookup = pd.read_csv(lookup_csv)

    print("Running inference on sample data...")
    preds = run_inference(sample_inference_data, df_lookup)
    print("\nPREDICTIONS:")
    for idx, pred in enumerate(preds):
        print(f"Sample {idx + 1}: Estimated Duration = {pred:.2f} minutes")

if __name__ == "__main__":
    main()
