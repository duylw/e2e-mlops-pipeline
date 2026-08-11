import os

import joblib
import pandas as pd


def main():
    print("--- STAGE 4: TRANSFORMATION ---")
    from src.config.paths import ProjectPaths

    paths = ProjectPaths()
    split_dir = paths.split_dir
    preprocessor_pkl = paths.preprocessor_file
    processed_dir = paths.processed_dir
    
    if not os.path.exists(preprocessor_pkl):
        raise FileNotFoundError(f"Preprocessor object not found at {preprocessor_pkl}")
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load preprocessor class object
    print(f"Loading preprocessor from {preprocessor_pkl}...")
    preprocessor = joblib.load(preprocessor_pkl)
    
    # Ensure inference mode is off during training transformation
    preprocessor.is_inference = False
    
    splits = ["train", "val", "test"]
    for split in splits:
        raw_path = os.path.join(split_dir, f"{split}_raw.parquet")
        if not os.path.exists(raw_path):
            print(f"Warning: {split} raw file not found at {raw_path}. Skipping.")
            continue
            
        print(f"Processing split: {split}...")
        
        # Load raw split
        df_raw = pd.read_parquet(raw_path)
        
        # Transform using the preprocessor class object
        df_transformed = preprocessor.transform(df_raw)
        
        # Save output parquet
        out_path = os.path.join(processed_dir, f"{split}.parquet")
        print(f"Saving processed {split} dataset to {out_path} (shape: {df_transformed.shape})...")
        df_transformed.to_parquet(out_path, index=False)
        
    print("Transformation stage completed successfully.")

if __name__ == "__main__":
    main()
