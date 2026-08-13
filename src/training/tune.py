import argparse
import sys

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings
from src.training.pipeline import build_model_pipeline, load_split
from src.utils.io import save_json


def configure_console_encoding() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def suggest_xgboost_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 250, 800, step=50),
        "tree_method": "hist",
    }


def tune(n_trials: int = 20) -> dict:
    """Select XGBoost parameters on raw train/validation splits without touching test data."""
    paths = ProjectPaths()
    train = load_split(paths, "train")
    validation = load_split(paths, "val")
    lookup_df = pd.read_csv(paths.lookup_file)
    settings = load_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment("green_taxi_duration_tuning")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgboost_params(trial)
        model = build_model_pipeline(lookup_df, params)
        model.fit(train.X, train.y)
        validation_predictions = model.predict(validation.X)
        rmse = root_mean_squared_error(validation.y, validation_predictions)
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params({**params, "trial_number": trial.number, "data_train_rows": len(train.X)})
            mlflow.log_metric("rmse_val", rmse)
        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="green_taxi_duration_xgboost")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    result = {
        "selection_metric": "rmse_val",
        "n_trials": n_trials,
        "best_rmse_val": study.best_value,
        "best_params": suggest_serializable_params(study.best_params),
    }
    save_json(result, paths.tuning_file)
    return result


def suggest_serializable_params(params: dict) -> dict:
    """Restore static parameters omitted from Optuna's sampled parameter dictionary."""
    return {**params, "tree_method": "hist"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the full raw-input XGBoost pipeline on validation RMSE.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials; default: 20.")
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    result = tune(n_trials=args.n_trials)
    print(f"Best validation RMSE: {result['best_rmse_val']:.4f}")
    print(f"Saved tuning result to {ProjectPaths().tuning_file}")
    print("Copy best_params into params.yaml, then run dvc repro to evaluate the final test split.")


if __name__ == "__main__":
    main()
