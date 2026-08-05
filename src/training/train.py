import os
import sys
import yaml
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Fix Windows encoding issue for terminal output (e.g. MLflow runner emoji)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.preprocessing import prepare_X_y

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("green_taxi_duration_prediction")

def load_params(params_path="params.yaml"):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- STAGE 5: MODEL TRAINING ---")
    
    processed_dir = "data/processed"
    models_dir = "models"
    params_path = "params.yaml"
    
    with mlflow.start_run() as run:

        # 1. Load parameters
        params = load_params(params_path)['train']
        print(f"Loaded training parameters: {params}")
        
        mlflow.log_params(params)
        # 2. Load processed datasets
        train_path = os.path.join(processed_dir, "train.parquet")
        val_path = os.path.join(processed_dir, "val.parquet")
        test_path = os.path.join(processed_dir, "test.parquet")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset not found at {train_path}")
            
        print(f"Loading train data from {train_path}...")
        df_train = pd.read_parquet(train_path)
        X_train, y_train = prepare_X_y(df_train)
        mlflow.log_metrics({"train_samples": len(X_train)})

        X_val, y_val = None, None
        if os.path.exists(val_path):
            print(f"Loading validation data from {val_path}...")
            df_val = pd.read_parquet(val_path)
            X_val, y_val = prepare_X_y(df_val)
            mlflow.log_metrics({"val_samples": len(X_val)})
            
        X_test, y_test = None, None
        if os.path.exists(test_path):
            print(f"Loading test data from {test_path}...")
            df_test = pd.read_parquet(test_path)
            X_test, y_test = prepare_X_y(df_test)
            mlflow.log_metrics({"test_samples": len(X_test)})

        # 3. Train XGBoost model
        print("Initializing XGBoost model...")
        n_estimators = params.pop('n_estimators', 1000)
        model = xgb.XGBRegressor(**params, n_estimators=n_estimators, random_state=42)
        
        print("Training XGBoost Regressor...")
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=50
            )
        else:
            model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            model,
            artifact_path=f"xgboost-{run.info.run_id}",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
        )

        # Save model locally for evaluation (not tracked by DVC)
        os.makedirs(models_dir, exist_ok=True)
        model_local_path = os.path.join(models_dir, "model.pkl")
        print(f"Saving trained model locally to {model_local_path}...")
        joblib.dump(model, model_local_path)
    
        # 4. Evaluate model
        print("\n--- EVALUATION METRICS ---")
        
        # Evaluate Train
        preds_train = model.predict(X_train)
        rmse_train = root_mean_squared_error(y_train, preds_train)
        mae_train = mean_absolute_error(y_train, preds_train)
        print(f"Train Dataset: RMSE = {rmse_train:.4f} mins | MAE = {mae_train:.4f} mins")

        mlflow.log_metrics({
            'rmse_train': rmse_train,
            'mae_train': mae_train
        })
        
        # Evaluate Val
        if X_val is not None and y_val is not None:
            preds_val = model.predict(X_val)
            rmse_val = root_mean_squared_error(y_val, preds_val)
            mae_val = mean_absolute_error(y_val, preds_val)
            print(f"Val Dataset:   RMSE = {rmse_val:.4f} mins | MAE = {mae_val:.4f} mins")

            mlflow.log_metrics({
                'rmse_val': rmse_val,
                'mae_val': mae_val
            })            
            
        # Evaluate Test
        if X_test is not None and y_test is not None:
            preds_test = model.predict(X_test)
            rmse_test = root_mean_squared_error(y_test, preds_test)
            mae_test = mean_absolute_error(y_test, preds_test)
            print(f"Test Dataset:  RMSE = {rmse_test:.4f} mins | MAE = {mae_test:.4f} mins")

            mlflow.log_metrics({
                'rmse_test': rmse_test,
                'mae_test': mae_test
            })
        
        # Log Artifacts
        mlflow.log_artifact("artifacts/preprocessor.pkl")
        

        print("Model training stage completed successfully.")

if __name__ == "__main__":
    main()