import os

import joblib
import pandas as pd

from src.config.paths import ProjectPaths
from src.features.preprocessor import NYCGreenTaxiPreprocessor


def main():
    print("--- STAGE 3: FITTING PREPROCESSOR ---")
    paths = ProjectPaths()
    train_raw_path = paths.split_dir / "train_raw.parquet"
    lookup_csv = paths.lookup_file
    artifacts_dir = paths.artifacts_dir
    
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
    
    # 2. Fit preprocessor class
    print("Fitting NYCGreenTaxiPreprocessor class...")
    preprocessor = NYCGreenTaxiPreprocessor(df_lookup=df_lookup, is_inference=False)
    preprocessor.fit(df_train)
    
    # 3. Save preprocessor class object
    preprocessor_pkl_path = paths.preprocessor_file
    print(f"Saving preprocessor object to {preprocessor_pkl_path}...")
    joblib.dump(preprocessor, preprocessor_pkl_path)
    
    print(f"Preprocessor object fitted and saved successfully. Train columns: {preprocessor.train_columns}")

if __name__ == "__main__":
    main()
