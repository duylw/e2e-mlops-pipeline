import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

def load_champion_model_and_preprocessor():
    """
    Downloads and loads the champion model and its corresponding preprocessor artifact
    from the MLflow Model Registry and Artifact Store.
    """
    print("--- LOADING CHAMPION MODEL & PREPROCESSOR ---")
    
    # 1. Configure MLflow URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"Connecting to MLflow Server: {mlflow_uri}")
    
    client = MlflowClient()
    model_name = "green_taxi_duration_model"
    alias = "champion"
    
    # 2. Get the model version details using the alias
    try:
        model_version_details = client.get_model_version_by_alias(model_name, alias)
        run_id = model_version_details.run_id
        version = model_version_details.version
        print(f"Found champion model: Version {version}, Run ID: {run_id}")
    except Exception as e:
        raise ValueError(
            f"Failed to find model version for '{model_name}' with alias '{alias}'. "
            f"Ensure the model is registered and tagged with the alias. Error: {e}"
        )
    
    # 3. Load the model from MLflow Registry
    model_uri = f"models:/{model_name}@{alias}"
    print(f"Loading model from URI: {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
    
    # 4. Download preprocessor.pkl from the run artifacts
    print(f"Downloading preprocessor.pkl artifact from run {run_id}...")
    try:
        preprocessor_local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor.pkl"
        )
        print(f"Preprocessor downloaded to: {preprocessor_local_path}")
        
        # 5. Load preprocessor artifact
        preprocessor = joblib.load(preprocessor_local_path)
        print("Preprocessor loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download or load preprocessor artifact: {e}")
        
    return model, preprocessor, version