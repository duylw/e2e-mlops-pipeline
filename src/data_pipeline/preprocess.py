import pandas as pd
import numpy as np
import parser
import logging

logger = logging.getLogger("Preprocessing Data")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='Preprocess Data',
    description='Preprocesses taxi data from TLC',
    epilog='Text at the bottom of help'
)
parser.add_argument('--month', type=int, default=1)
parser.add_argument('--year', type=int, default=2026)

def calculate_haversine(lat1, lon1, lat2, lon2, earth_radius=3958.8):
    """
    Calculates the haversine distance between two sets of GPS points.
    earth_radius: Default is 3958.8 miles. Use 6371.0 for kilometers.
    """
    # 1. Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # 2. Calculate delta coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 3. Apply Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 4. Multiply by Earth radius to get absolute distance
    distance = earth_radius * c

    return distance

def clean_taxi_data(df_raw, target_year=2026, target_month=1):
    """
    Cleans raw taxi trip data for model training/evaluation.
    Includes target column derivation and outliers filtering.
    """
    df = df_raw.copy()
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

    # Auto-calculate the start and end of target month
    start_date = pd.Timestamp(year=target_year, month=target_month, day=1)
    end_date = start_date + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    df = df[(df['lpep_pickup_datetime'] >= start_date) & (df['lpep_pickup_datetime'] <= end_date)]

    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 100)]
    df = df[(df['duration'] > 0) & (df['duration'] <= 300)]
    if 'fare_amount' in df.columns:
        df = df[df['fare_amount'] >= 0]

    df['passenger_count'] = df['passenger_count'].fillna(1)
    if 'trip_type' in df.columns:
        if df['trip_type'].mode().shape[0] > 0:
            df['trip_type'] = df['trip_type'].fillna(df['trip_type'].mode()[0])

    cols_to_drop = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                    'improvement_surcharge', 'total_amount', 'payment_type',
                    'congestion_surcharge', 'cbd_congestion_fee', 'ehail_fee',
                    'store_and_fwd_flag', 'lpep_dropoff_datetime', 'RatecodeID']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def clean_taxi_data_inference(df_raw, trip_type_mode_fallback=1):
    """
    Cleans raw taxi trip data for online/offline inference.
    Does not depend on target variables (duration, dropoff datetime).
    """
    df = df_raw.copy()
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

    df['passenger_count'] = df['passenger_count'].fillna(1)
    if 'trip_type' in df.columns:
        df['trip_type'] = df['trip_type'].fillna(trip_type_mode_fallback)

    # Do not filter based on duration or trip distance as they are unknown or targets
    cols_to_drop = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                    'improvement_surcharge', 'total_amount', 'payment_type',
                    'congestion_surcharge', 'cbd_congestion_fee', 'ehail_fee',
                    'store_and_fwd_flag', 'RatecodeID', 'trip_distance']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def engineer_base_features(df_cleaned, df_lookup):
    """
    Extracts basic time and space features and filters rows with invalid locations.
    """
    df = df_cleaned.copy()

    # 1. Time
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek

    # 2. Space
    df = df.merge(df_lookup[['LocationID', 'Borough', 'Zone', 'latitude', 'longitude']],
                  left_on='PULocationID', right_on='LocationID', how='left')
    df.rename(columns={'Borough': 'PU_Borough', 'Zone': 'PU_Zone',
                       'latitude': 'PU_lat', 'longitude': 'PU_long'}, inplace=True)

    df = df.merge(df_lookup[['LocationID', 'latitude', 'longitude']],
                  left_on='DOLocationID', right_on='LocationID', how='left')
    df.rename(columns={'latitude': 'DO_lat', 'longitude': 'DO_long'}, inplace=True)

    # 3. Filtering
    df = df.dropna(subset=['PU_lat', 'PU_long', 'DO_lat', 'DO_long'])
    df['estimated_distance_miles'] = calculate_haversine(
        df['PU_lat'], df['PU_long'], df['DO_lat'], df['DO_long']
    )
    df = df[df['estimated_distance_miles'] > 0]

    return df

def build_speed_profile(df_base):
    """
    Builds a historical average speed profile by hour and borough.
    Should only receive clean baseline features.
    """
    df = df_base.copy()

    # Actual speed (Miles / Minute)
    df['speed'] = df['trip_distance'] / df['duration']
    df = df[(df['speed'] > 0) & (df['speed'] < 2)] # Remove extreme speed outliers

    speed_profile = df.groupby(['pickup_hour', 'PU_Borough'])['speed'].mean().reset_index()
    speed_profile.rename(columns={'speed': 'historical_avg_speed'}, inplace=True)

    return speed_profile

def fit_feature_artifacts(df_base_train, speed_profile, trip_type_mode=None):
    """
    Learns feature properties from train split (e.g. OHE categories, mean speed).
    """
    df = df_base_train.copy()

    # a. Zone frequency map
    zone_freq_map = df['PU_Zone'].value_counts(normalize=True).to_dict()

    # b. Global mean speed
    global_mean_speed = speed_profile['historical_avg_speed'].mean()

    # c. OHE categories fixed listing
    ohe_categories = {}
    for col in ['VendorID', 'trip_type', 'PU_Borough']:
        if col in df.columns:
            ohe_categories[col] = sorted(df[col].astype(str).unique().tolist())

    # d. Trip type fallback
    if trip_type_mode is None and 'trip_type' in df.columns:
        trip_type_mode = df['trip_type'].mode()[0]

    artifacts = {
        'speed_profile': speed_profile,
        'global_mean_speed': global_mean_speed,
        'zone_freq_map': zone_freq_map,
        'ohe_categories': ohe_categories,
        'trip_type_mode': trip_type_mode,
        'train_columns': None,
    }
    return artifacts

def transform_advanced_features(df_base, artifacts):
    """
    Applies speed mapping, sinusoidal transformations, OHE and column reindexing.
    Used consistently for training, evaluation, and inference.
    """
    df = df_base.copy()

    # 1. Map speed from historical speed profile
    df = df.merge(artifacts['speed_profile'], on=['pickup_hour', 'PU_Borough'], how='left')
    df['historical_avg_speed'] = df['historical_avg_speed'].fillna(artifacts['global_mean_speed'])

    # 2. Sin/Cos time cyclical representations
    df['hour_sin'] = np.sin(2*np.pi*df['pickup_hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['pickup_hour']/24.0)
    df['day_sin'] = np.sin(2*np.pi*df['day_of_week']/7.0)
    df['day_cos'] = np.cos(2*np.pi*df['day_of_week']/7.0)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # 3. Zone Frequency Mapping
    df['PU_Zone_frequency'] = df['PU_Zone'].map(artifacts['zone_freq_map']).fillna(0)

    # 4. Safe One-Hot Encoding based on categories from training set
    for col, categories in artifacts['ohe_categories'].items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        for cat in categories[1:]:  # drop_first=True
            df[f'{col}_{cat}'] = (df[col] == cat).astype(int)
        df.drop(columns=[col], inplace=True)

    # 5. Drop redundant and leakage columns
    cols_to_drop = [
        'PULocationID', 'DOLocationID', 'LocationID_x', 'LocationID_y',
        'PU_lat', 'PU_long', 'DO_lat', 'DO_long',
        'lpep_pickup_datetime', 'PU_Zone',
        'pickup_hour', 'day_of_week',
        'trip_distance'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 6. Final safe column alignment
    if artifacts['train_columns'] is not None:
        target_col = 'duration'
        keep_extra = [target_col] if target_col in df.columns else []
        df = df.reindex(columns=artifacts['train_columns'] + keep_extra, fill_value=0)

    return df

def split_time_series_data(df_base, cutoff_date):
    """
    Splits data chronologically into train and validation sets.
    """
    df = df_base.sort_values('lpep_pickup_datetime').copy()
    train_base = df[df['lpep_pickup_datetime'] < cutoff_date].copy()
    val_base = df[df['lpep_pickup_datetime'] >= cutoff_date].copy()
    return train_base, val_base

def prepare_X_y(df, target_col='duration'):
    """
    Splits features and target label.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

if __name__ == '__main__':

    args = parser.parse_args()
    month = args.month
    year = args.year

    file_path = f'data/raw/green_tripdata_{year}-{month:02d}.parquet'
    df_raw = pd.read_parquet(file_path)
    logger.info(f"Loaded raw dataset of shape: {df_raw.shape}")

    # 3. Chronological Time-Series Split
    cutoff_date = '2026-01-25'
    train_base, val_base = split_time_series_data(df_raw, cutoff_date)
    logger.info(f"Split data into train: {train_base.shape} and val: {val_base.shape}")

    # 4. Clean train and validation datasets
    train_base = clean_taxi_data(train_base, target_year=2026, target_month=1)
    val_base = clean_taxi_data(val_base, target_year=2026, target_month=1)
    logger.info(f"Cleaned datasets - train: {train_base.shape}, val: {val_base.shape}")

    # 5. Base feature extraction (combining spatial information)
    train_base_features = engineer_base_features(train_base, df_lookup_with_loc)
    val_base_features = engineer_base_features(val_base, df_lookup_with_loc)
    logger.info(f"Engineered base features - train: {train_base_features.shape}, val: {val_base_features.shape}")

    # 6. Build Speed Profile and fit advanced feature mapping artifacts
    speed_profile = build_speed_profile(train_base_features)
    artifacts = fit_feature_artifacts(train_base_features, speed_profile)

    # Transform train to capture exact list of training columns
    train_final_features = transform_advanced_features(train_base_features, artifacts)
    artifacts['train_columns'] = [c for c in train_final_features.columns if c != 'duration']

    # Transform validation using the same training artifacts (no fitting)
    val_final_features = transform_advanced_features(val_base_features, artifacts)

    # 7. Split features and labels
    X_train, y_train = prepare_X_y(train_final_features)
    X_val, y_val = prepare_X_y(val_final_features)
    logger.info(f"Feature set shapes - X_train: {X_train.shape}, X_val: {X_val.shape}")

    logger.info('Preprocessing completed.')
