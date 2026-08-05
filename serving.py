import os
import joblib
import pandas as pd
from inference import run_inference

# Define the sample inference data as in the original script
sample_inference_data = pd.DataFrame([
    {
        "VendorID": 2,
        "lpep_pickup_datetime": "2026-01-15 08:30:00",   # giờ cao điểm sáng
        "PULocationID": 74,    # East Harlem North (Manhattan)
        "DOLocationID": 42,    # Central Harlem
        "passenger_count": 1,
        "trip_type": 1
    },
    {
        "VendorID": 1,
        "lpep_pickup_datetime": "2026-01-15 23:45:00",   # đêm khuya
        "PULocationID": 7,     # Astoria (Queens)
        "DOLocationID": 179,   # Steinway
        "passenger_count": 2,
        "trip_type": 1
    },
    {
        "VendorID": 2,
        "lpep_pickup_datetime": "2026-01-16 17:15:00",   # giờ cao điểm chiều
        "PULocationID": 41,    # Central Harlem
        "DOLocationID": 244,   # Washington Heights South
        "passenger_count": 1,
        "trip_type": 2
    },
    {
        "VendorID": 1,
        "lpep_pickup_datetime": "2026-01-17 03:00:00",   # rạng sáng, ít traffic
        "PULocationID": 82,    # Elmhurst
        "DOLocationID": 129,   # Jackson Heights
        "passenger_count": None,   # test trường hợp thiếu dữ liệu
        "trip_type": None
    },
])

def main():
    print("--- RUNNING SERVING LAYER (TEST RUN) ---")
    
    lookup_csv = "data/raw/nyc_taxi_lookup_with_coords.csv"
    model_path = "models/model.pkl"
    artifacts_path = "artifacts/preprocessor.pkl"

    if not os.path.exists(lookup_csv) or not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        print(f"Warning: One of the required files is missing. Please run DVC pipeline first.")
        print(f"Checking files:\n - {lookup_csv}: {os.path.exists(lookup_csv)}\n - {model_path}: {os.path.exists(model_path)}\n - {artifacts_path}: {os.path.exists(artifacts_path)}")
        return

    # Load lookup table
    df_lookup_with_loc = pd.read_csv(lookup_csv)
    
    # Load model
    model = joblib.load(model_path)

    # Run predictions
    print("Running serving inference...")
    preds = run_inference(
        df_raw=sample_inference_data, 
        df_lookup_with_loc=df_lookup_with_loc, 
        artifacts_path=artifacts_path,
        model=model
    )
    print("Predictions (duration in minutes):")
    print(preds)

if __name__ == "__main__":
    main()