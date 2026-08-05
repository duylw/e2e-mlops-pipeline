import os
import yaml
import pandas as pd
import glob
from src.utils import download_taxi_zones, create_coordinate_lookup

def load_params(params_path="params.yaml"):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- STAGE 1: INGESTION ---")
    
    # 1. Load parameters
    params = load_params()
    years = params['ingest']['years']
    months = params['ingest']['months']
    
    # Create directories
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

    # 2. Download and prepare coordinates lookup
    lookup_csv = os.path.join(raw_dir, 'nyc_taxi_lookup_with_coords.csv')
    if not os.path.exists(lookup_csv):
        print("Zone coordinates lookup not found. Processing shapefiles...")
        download_taxi_zones(
            url='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip',
            zip_path='tmp/taxi_zones.zip',
            extract_to='tmp/taxi_zones_data'
        )
        shapefile = './tmp/taxi_zones_data/taxi_zones/taxi_zones.shp'
        if not os.path.exists(shapefile):
            shp_files = glob.glob('./tmp/taxi_zones_data/**/*.shp', recursive=True)
            if shp_files:
                shapefile = shp_files[0]
            else:
                raise FileNotFoundError("Could not find TLC shapefile (.shp) in taxi_zones_data.")
                
        df_lookup = create_coordinate_lookup(shapefile)
        df_lookup.to_csv(lookup_csv, index=False)
        print(f"Lookup file created at {lookup_csv}")
    else:
        print(f"Using existing lookup file at {lookup_csv}")

    # 3. Download monthly Parquet files and combine
    dfs = []
    for year in years:
        for month in months:
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
            temp_parquet = f"tmp/green_tripdata_{year}-{month:02d}.parquet"
            
            print(f"Downloading parquet from {url}...")
            # Use pandas read_parquet directly from URL, or download first
            try:
                df_month = pd.read_parquet(url)
                print(f"Successfully loaded {year}-{month:02d} data: shape {df_month.shape}")
                dfs.append(df_month)
            except Exception as e:
                print(f"Failed to load data for {year}-{month:02d} from {url}: {e}")

    if not dfs:
        raise ValueError("No data files were downloaded successfully.")

    # 4. Merge and sort
    print("Combining datasets...")
    df_combined = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Sort chronologically by pickup datetime
    pickup_col = 'lpep_pickup_datetime'
    if pickup_col in df_combined.columns:
        df_combined[pickup_col] = pd.to_datetime(df_combined[pickup_col])
        df_combined.sort_values(by=pickup_col, inplace=True)
    
    # Save combined csv
    raw_data_path = os.path.join(raw_dir, "raw_data.csv")
    print(f"Saving merged dataset to {raw_data_path}...")
    df_combined.to_csv(raw_data_path, index=False)
    
    # Clean up tmp folder
    print("Cleaning up temporary download folders...")
    import shutil
    if os.path.exists("tmp/taxi_zones_data"):
        shutil.rmtree("tmp/taxi_zones_data", ignore_errors=True)
        
    print(f"Ingestion complete. Total combined rows: {len(df_combined)}")

if __name__ == "__main__":
    main()
