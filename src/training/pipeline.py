import time
from dataclasses import dataclass

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.config.paths import ProjectPaths
from src.data.preparation import prepare_training_data
from src.features.transformer import NYCGreenTaxiFeatureTransformer
from src.training.tracker import MlflowTrainingTracker
from src.utils.io import ensure_dir, save_json


@dataclass(frozen=True)
class SplitData:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class SplitMetrics:
    rmse: float
    mae: float
    latency_per_row: float


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_uri: str
    metrics: dict[str, float]


def create_xgboost_model(params: dict) -> xgb.XGBRegressor:
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


class TrainingPipeline:
    """Fit, evaluate, and log one sklearn Pipeline from raw chronological splits."""

    def __init__(self, paths: ProjectPaths, train_params: dict, tracker: MlflowTrainingTracker):
        self.paths = paths
        self.train_params = train_params
        self.tracker = tracker

    def run(self) -> TrainingResult:
        train = self._load_split("train")
        validation = self._load_split("val")
        test = self._load_split("test")
        lookup_df = pd.read_csv(self.paths.lookup_file)
        model = self._build_model(lookup_df)

        started_at = time.perf_counter()
        model.fit(train.X, train.y)
        train_time = time.perf_counter() - started_at

        metrics = {
            "train_samples": float(len(train.X)),
            "val_samples": float(len(validation.X)),
            "test_samples": float(len(test.X)),
            "train_time": train_time,
        }
        for split_name, split_data in (("train", train), ("val", validation), ("test", test)):
            split_metrics = self._evaluate(model, split_data)
            metrics.update(
                {
                    f"rmse_{split_name}": split_metrics.rmse,
                    f"mae_{split_name}": split_metrics.mae,
                    f"latency_per_row_{split_name}": split_metrics.latency_per_row,
                }
            )

        logged_model = self.tracker.log_training_run(
            model=model,
            params=self.train_params,
            metrics=metrics,
            input_example=train.X.head(3),
        )
        ensure_dir(self.paths.reports_dir)
        save_json(metrics, self.paths.metrics_file)
        save_json(
            {
                "run_id": logged_model.run_id,
                "model_name": self.tracker.settings.model_name,
                "model_uri": logged_model.model_uri,
                "metrics_path": str(self.paths.metrics_file),
            },
            self.paths.run_info_file,
        )
        return TrainingResult(
            run_id=logged_model.run_id,
            model_uri=logged_model.model_uri,
            metrics=metrics,
        )

    def _load_split(self, split: str) -> SplitData:
        split_path = self.paths.split_dir / f"{split}_raw.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Required {split} split not found at {split_path}")
        X, y = prepare_training_data(pd.read_parquet(split_path))
        if X.empty:
            raise ValueError(f"No usable rows remain in the {split} split after preparation")
        return SplitData(X=X, y=y)

    def _build_model(self, lookup_df: pd.DataFrame) -> Pipeline:
        return Pipeline(
            steps=[
                ("features", NYCGreenTaxiFeatureTransformer(lookup_df=lookup_df)),
                ("model", create_xgboost_model(self.train_params)),
            ]
        )

    @staticmethod
    def _evaluate(model: Pipeline, split: SplitData) -> SplitMetrics:
        started_at = time.perf_counter()
        predictions = model.predict(split.X)
        latency_per_row = (time.perf_counter() - started_at) / len(split.X)
        return SplitMetrics(
            rmse=root_mean_squared_error(split.y, predictions),
            mae=mean_absolute_error(split.y, predictions),
            latency_per_row=latency_per_row,
        )
