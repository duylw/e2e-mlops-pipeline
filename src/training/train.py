import sys
import time

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings, load_params
from src.features.engineering import prepare_X_y
from src.utils.io import ensure_dir, save_json

PREPROCESSOR_ARTIFACT_PATH = "preprocessor.pkl"


def configure_console_encoding() -> None:
    """Allow MLflow status messages containing Unicode on Windows terminals."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def configure_mlflow():
    settings = load_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    return settings


def load_processed_split(paths: ProjectPaths, split: str):
    path = paths.processed_dir / f"{split}.parquet"
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    return prepare_X_y(df)


def evaluate_split(model: xgb.XGBRegressor, X: pd.DataFrame | None, y: pd.Series | None, split: str) -> dict:
    if X is None or y is None:
        return {}

    started_at = time.time()
    predictions = model.predict(X)
    latency = (time.time() - started_at) / max(len(X), 1)
    return {
        f"rmse_{split}": root_mean_squared_error(y, predictions),
        f"mae_{split}": mean_absolute_error(y, predictions),
        f"latency_per_row_{split}": latency,
    }


def build_model(params: dict) -> xgb.XGBRegressor:
    model_params = dict(params)
    n_estimators = model_params.pop("n_estimators", 1000)
    return xgb.XGBRegressor(
        **model_params,
        n_estimators=n_estimators,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )


def main():
    configure_console_encoding()
    print("--- STAGE 5: MODEL TRAINING ---")
    paths = ProjectPaths()
    settings = configure_mlflow()
    train_params = load_params(paths.params_file)["train"]

    X_train, y_train = load_processed_split(paths, "train")
    X_val, y_val = load_processed_split(paths, "val")
    X_test, y_test = load_processed_split(paths, "test")
    if X_train is None:
        raise FileNotFoundError(f"Training dataset not found in {paths.processed_dir}")
    if not paths.preprocessor_file.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {paths.preprocessor_file}. "
            "Run the fit_preprocessor DVC stage before training."
        )

    ensure_dir(paths.models_dir)
    ensure_dir(paths.reports_dir)

    with mlflow.start_run() as run:
        print(f"MLflow run_id: {run.info.run_id}")
        mlflow.log_params(train_params)
        mlflow.log_metrics(
            {
                "train_samples": len(X_train),
                "val_samples": 0 if X_val is None else len(X_val),
                "test_samples": 0 if X_test is None else len(X_test),
            }
        )

        model = build_model(train_params)
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        started_at = time.time()
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        train_time = time.time() - started_at

        metrics = {"train_time": train_time}
        for split, X, y in [ 
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics.update(evaluate_split(model, X, y, split))

        save_json(metrics, paths.metrics_file)
        joblib.dump(model, paths.model_file)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(paths.preprocessor_file),
        )

        # Fail the run early if the preprocessor cannot be retrieved from its MLflow artifact URI.
        mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=PREPROCESSOR_ARTIFACT_PATH,
            dst_path=paths.artifacts_dir
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name=f"model-{run.info.run_id}",
            model_format="json",
            registered_model_name=None,
        )

        run_info = {
            "run_id": run.info.run_id,
            "model_name": settings.model_name,
            "model_uri": model_info.model_uri,
            "preprocessor_artifact_path": str(paths.preprocessor_file),
            "metrics_path": str(paths.metrics_file),
        }
        save_json(run_info, paths.run_info_file)

    print(f"Saved model to {paths.model_file}")
    print(f"Saved preprocessor to {paths.preprocessor_file}")
    print(f"Saved metrics to {paths.metrics_file}")
    print(f"Saved MLflow run metadata to {paths.run_info_file}")


if __name__ == "__main__":
    main()
