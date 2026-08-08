import os
import pandas as pd

def main():
    print("--- STAGE 2: SPLITTING ---")
    
    raw_parquet = "data/raw/raw_data.parquet"
    if not os.path.exists(raw_parquet):
        raise FileNotFoundError(f"Raw data file {raw_parquet} not found. Did ingest stage run successfully?")

    # 1. Load data
    print(f"Loading raw data from {raw_parquet}...")
    df = pd.read_parquet(raw_parquet)
    
    # 2. Sort chronologically
    pickup_col = 'lpep_pickup_datetime'
    df[pickup_col] = pd.to_datetime(df[pickup_col])
    df = df.sort_values(by=pickup_col).reset_index(drop=True)

    # 3. Calculate chronological split indices
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Split completed: Total = {n} rows")
    print(f" - Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%) | Time: {train_df[pickup_col].min()} to {train_df[pickup_col].max()}")
    print(f" - Val: {len(val_df)} rows ({len(val_df)/n*100:.1f}%) | Time: {val_df[pickup_col].min()} to {val_df[pickup_col].max()}")
    print(f" - Test: {len(test_df)} rows ({len(test_df)/n*100:.1f}%) | Time: {test_df[pickup_col].min()} to {test_df[pickup_col].max()}")

    # 4. Save to parquet
    split_dir = "data/split"
    os.makedirs(split_dir, exist_ok=True)
    
    train_path = os.path.join(split_dir, "train_raw.parquet")
    val_path = os.path.join(split_dir, "val_raw.parquet")
    test_path = os.path.join(split_dir, "test_raw.parquet")
    
    print(f"Saving splits to {split_dir}...")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print("Splitting completed successfully.")

if __name__ == "__main__":
    main()
