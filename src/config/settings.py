import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.utils.io import load_yaml


@dataclass(frozen=True)
class MlflowSettings:
    tracking_uri: str
    experiment_name: str
    model_name: str
    serving_model_alias: str


def load_params(params_path: str = "params.yaml") -> dict:
    return load_yaml(params_path)


def load_mlflow_settings() -> MlflowSettings:
    load_dotenv()
    return MlflowSettings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "green_taxi_duration_prediction"),
        model_name=os.getenv("MODEL_NAME", "green_taxi_duration_model"),
        serving_model_alias=os.getenv("SERVING_MODEL_ALIAS", "champion"),
    )
