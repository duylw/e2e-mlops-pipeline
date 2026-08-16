# NYC Green Taxi Trip Duration Model Card

## Problem Statement

Predict NYC Green Taxi trip duration in minutes from pickup time, pickup/dropoff TLC zones, vendor, passenger count, and trip type.

## Dataset

- Source: NYC TLC Green Taxi trip records.
- Current params: February-April 2026.
- Target: `duration = lpep_dropoff_datetime - lpep_pickup_datetime`, measured in minutes.

## Data Leakage Handling

- Drop fare/payment fields such as `fare_amount`, `total_amount`, `tip_amount`, and `payment_type`.
- Drop `lpep_dropoff_datetime` after deriving the target.
- Fit feature-transformer state only on the train split; the fitted transformer is stored with XGBoost in one MLflow pipeline artifact.

## Features

- Time: pickup hour, day of week, cyclical hour/day, weekend flag, rush-hour flag.
- Location: pickup borough/zone, estimated haversine distance from TLC zone centroids.
- Route: pickup/dropoff borough, same-borough indicator, and train-only frequency encodings for pickup zone, dropoff zone, and pickup/dropoff route.
- Encodings: fixed one-hot categories learned from train data.

The final transformer does not use target-derived historical-speed features. This keeps every training-row feature independent of that row's duration target.

## Validation Strategy

The data is sorted chronologically and split into:

- Train: first 60%
- Validation: next 20%
- Test: final 20%

This avoids random future-to-past leakage and better matches real taxi demand forecasting.

## Metrics

Primary metrics are RMSE and MAE in minutes. DVC tracks them in `reports/metrics.json`.

| Model | Validation RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Initial recorded baseline | 8.73 | 9.48 | 4.81 |
| Final champion, 20-trial validation tuning | 8.37 | 9.10 | 4.45 |

The final champion is selected with validation RMSE; the chronological test split is reserved for the final result above.

## Error Analysis

Segment metrics are generated on the test split in `reports/segment_metrics.json` for groups with at least 100 trips.

- Bronx pickup trips: RMSE 12.93 minutes across 644 trips.
- Trips with missing `trip_type`: RMSE 12.89 minutes across 3,376 trips.
- Pickups at 06:00: RMSE 20.10 minutes across 502 trips.

These slices identify data quality and early-morning variability as higher-risk cases than the overall aggregate metric.

## Monitoring

- Evidently batch monitoring compares current data with train input distribution
  and validation quality distribution from the same DVC revision as the champion.
- Input monitoring tracks missingness and feature drift for serving-compatible
  fields and pickup-time features.
- Labeled batches also track regression quality, target drift, and prediction drift.
- Monitoring reports are logged to MLflow; raw reference Parquet remains managed by DVC.

## Known Limitations

- TLC zone centroid distance is an approximation, not route distance.
- Traffic, weather, holidays, and live road events are not included.
- Very short, invalid, and extreme-duration trips are filtered before modeling.
