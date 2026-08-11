import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import load_mlflow_settings


def load_champion_model():
    """Load one registered MLflow model containing features and estimator."""
    print("--- LOADING REGISTERED PIPELINE ---")
    settings = load_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    print(f"Connecting to MLflow Server: {settings.tracking_uri}")

    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(
            settings.model_name,
            settings.serving_model_alias,
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot find model '{settings.model_name}' with alias "
            f"'{settings.serving_model_alias}'. Register a model before serving."
        ) from exc

    model_uri = f"models:/{settings.model_name}@{settings.serving_model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)

    metadata = {
        "model_name": settings.model_name,
        "model_version": int(model_version.version),
        "model_alias": settings.serving_model_alias,
    }
    return model, metadata
