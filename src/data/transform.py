import os
import joblib
import pandas as pd

from src.preprocessing import (
    clean_taxi_data,
    engineer_base_features,
    transform_advanced_features
)

def main():
    print("--- STAGE 4: TRANSFORMATION ---")
    
    split_dir = "data/split"
    preprocessor_pkl = "artifacts/preprocessor.pkl"
    processed_dir = "data/processed"
    
    if not os.path.exists(preprocessor_pkl):
        raise FileNotFoundError(f"Preprocessor artifacts not found at {preprocessor_pkl}")
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load artifacts
    print(f"Loading preprocessor artifacts from {preprocessor_pkl}...")
    artifacts = joblib.load(preprocessor_pkl)
    df_lookup = artifacts['df_lookup']
    
    splits = ["train", "val", "test"]
    for split in splits:
        raw_path = os.path.join(split_dir, f"{split}_raw.parquet")
        if not os.path.exists(raw_path):
            print(f"Warning: {split} raw file not found at {raw_path}. Skipping.")
            continue
            
        print(f"Processing split: {split}...")
        
        # Load raw split
        df_raw = pd.read_parquet(raw_path)
        
        # Stateless cleaning
        df_cleaned = clean_taxi_data(df_raw)
        
        # Base feature extraction
        df_base = engineer_base_features(df_cleaned, df_lookup)
        
        # Advanced feature transformations (mapping speed profile, OHE encoding, column alignment)
        df_transformed = transform_advanced_features(df_base, artifacts)
        
        # Save output parquet
        out_path = os.path.join(processed_dir, f"{split}.parquet")
        print(f"Saving processed {split} dataset to {out_path} (shape: {df_transformed.shape})...")
        df_transformed.to_parquet(out_path, index=False)
        
    print("Transformation stage completed successfully.")

if __name__ == "__main__":
    main()
