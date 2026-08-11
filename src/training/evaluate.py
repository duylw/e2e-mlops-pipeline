import time

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings
from src.features.engineering import prepare_X_y
from src.utils.io import load_yaml, save_json


def evaluate_split(model, paths: ProjectPaths, split: str) -> dict:
    path = paths.processed_dir / f"{split}.parquet"
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    X, y = prepare_X_y(df)
    started_at = time.time()
    predictions = model.predict(X)
    latency = (time.time() - started_at) / max(len(X), 1)
    return {
        f"rmse_{split}": root_mean_squared_error(y, predictions),
        f"mae_{split}": mean_absolute_error(y, predictions),
        f"latency_per_row_{split}": latency,
    }


def main():
    print("--- STAGE 6: MODEL EVALUATION ---")
    paths = ProjectPaths()
    settings = load_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)

    if not paths.model_file.exists():
        raise FileNotFoundError(f"Model file not found: {paths.model_file}")

    model = joblib.load(paths.model_file)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics.update(evaluate_split(model, paths, split))

    save_json(metrics, paths.metrics_file)
    print(f"Saved metrics to {paths.metrics_file}: {metrics}")

    if paths.run_info_file.exists():
        run_info = load_yaml(paths.run_info_file)
        run_id = run_info.get("run_id")
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
