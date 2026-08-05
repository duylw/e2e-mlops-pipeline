import os
import joblib
import pandas as pd

from src.preprocessing import (
    clean_taxi_data,
    engineer_base_features,
    build_speed_profile,
    fit_feature_artifacts,
    transform_advanced_features
)

def main():
    print("--- STAGE 3: FITTING PREPROCESSOR ---")
    
    train_raw_path = "data/split/train_raw.parquet"
    lookup_csv = "data/raw/nyc_taxi_lookup_with_coords.csv"
    artifacts_dir = "artifacts"
    
    if not os.path.exists(train_raw_path):
        raise FileNotFoundError(f"Training file {train_raw_path} not found.")
    if not os.path.exists(lookup_csv):
        raise FileNotFoundError(f"Coordinate lookup file {lookup_csv} not found.")
        
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load data
    print(f"Loading raw train data from {train_raw_path}...")
    df_train = pd.read_parquet(train_raw_path)
    
    print(f"Loading coordinate lookup from {lookup_csv}...")
    df_lookup = pd.read_csv(lookup_csv)
    
    # 2. Baseline Cleaning
    print("Running baseline cleaning...")
    df_cleaned = clean_taxi_data(df_train)
    
    # 3. Base feature engineering
    print("Extracting base features...")
    df_base = engineer_base_features(df_cleaned, df_lookup)
    
    # 4. Build speed profile
    print("Building historical speed profile...")
    speed_profile = build_speed_profile(df_base)
    
    # 5. Fit feature artifacts
    print("Fitting preprocessing artifacts...")
    artifacts = fit_feature_artifacts(df_base, speed_profile)
    
    # Run advanced transformation once to determine the final list of train columns
    print("Mapping features to determine training column alignment...")
    df_final_temp = transform_advanced_features(df_base, artifacts)
    artifacts['train_columns'] = [c for c in df_final_temp.columns if c != 'duration']
    
    # Save coordinate lookup table inside preprocessor artifacts to make it self-contained
    artifacts['df_lookup'] = df_lookup
    
    # 6. Save preprocessor artifacts
    preprocessor_pkl_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    print(f"Saving preprocessor artifacts to {preprocessor_pkl_path}...")
    joblib.dump(artifacts, preprocessor_pkl_path)
    
    print(f"Preprocessor artifacts fitted and saved successfully. Train columns list: {artifacts['train_columns']}")

if __name__ == "__main__":
    main()
