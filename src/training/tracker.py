from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.config.settings import MlflowSettings

INTEGER_INPUT_COLUMNS = ("VendorID", "PULocationID", "DOLocationID")
FLOAT_INPUT_COLUMNS = ("passenger_count", "trip_type")


@dataclass(frozen=True)
class LoggedModel:
    run_id: str
    model_uri: str


class MlflowTrainingTracker:
    """Own the MLflow run that stores one complete fitted pipeline."""

    def __init__(self, settings: MlflowSettings):
        self.settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(settings.experiment_name)

    def log_training_run(
        self,
        model: Pipeline,
        params: dict,
        metrics: dict[str, float],
        input_example: pd.DataFrame,
        report_artifacts: list[Path],
    ) -> LoggedModel:
        signature_input = self._signature_input(input_example)
        predictions = model.predict(signature_input)
        signature = infer_signature(signature_input, predictions)

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for artifact_path in report_artifacts:
                mlflow.log_artifact(str(artifact_path), artifact_path="reports")
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=f"model-{run.info.run_id}",
                input_example=signature_input,
                signature=signature,
                serialization_format="cloudpickle",
            )

        return LoggedModel(run_id=run.info.run_id, model_uri=model_info.model_uri)

    @staticmethod
    def _signature_input(input_example: pd.DataFrame) -> pd.DataFrame:
        """Match MLflow model signature to the validated FastAPI request contract."""
        signature_input = input_example.copy()
        for column in INTEGER_INPUT_COLUMNS:
            signature_input[column] = pd.to_numeric(signature_input[column], errors="raise").astype("int64")
        for column in FLOAT_INPUT_COLUMNS:
            signature_input[column] = pd.to_numeric(signature_input[column], errors="coerce").astype("float64")
        return signature_input
