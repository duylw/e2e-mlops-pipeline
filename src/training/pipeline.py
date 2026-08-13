import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.config.paths import ProjectPaths
from src.data.preparation import prepare_training_data
from src.features.transformer import NYCGreenTaxiFeatureTransformer
from src.monitoring.lineage import build_training_lineage
from src.training.tracker import MlflowTrainingTracker
from src.utils.io import ensure_dir, save_json


@dataclass(frozen=True)
class SplitData:
    X: pd.DataFrame
    y: pd.Series
    source_path: Path


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


def build_training_params(train: SplitData, validation: SplitData, test: SplitData, model_params: dict) -> dict:
    """Combine fixed model configuration and split sizes as MLflow params."""
    return {
        **model_params,
        "data_train_rows": len(train.X),
        "data_validation_rows": len(validation.X),
        "data_test_rows": len(test.X),
    }


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

        ensure_dir(self.paths.reports_dir)
        save_json(metrics, self.paths.metrics_file)
        self.plot_feature_importance(model, self.paths.feature_importance_file)

        logged_model = self.tracker.log_training_run(
            model=model,
            params=build_training_params(train, validation, test, self.train_params),
            metrics=metrics,
            input_example=train.X.head(3),
            report_artifacts=[self.paths.metrics_file, self.paths.feature_importance_file],
            lineage_tags=build_training_lineage(self.paths, train, validation),
        )
        save_json(
            {
                "run_id": logged_model.run_id,
                "model_name": self.tracker.settings.model_name,
                "model_uri": logged_model.model_uri,
                "metrics_path": str(self.paths.metrics_file),
                "feature_importance_path": str(self.paths.feature_importance_file),
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
        return SplitData(X=X, y=y, source_path=split_path)

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

    @staticmethod
    def plot_feature_importance(model: Pipeline, output_path: Path, top_n: int = 20) -> pd.DataFrame:
        """Save a ranked feature-importance plot and return the full importance table."""
        if not hasattr(model.named_steps["model"], "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")
        feature_names = model.named_steps["features"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        importance_table = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            by="importance", ascending=False
        )
        top_features = importance_table.head(top_n).sort_values("importance")

        figure = Figure(figsize=(10, max(4, len(top_features) * 0.35)))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.barh(top_features["feature"], top_features["importance"], color="#2563eb")
        axis.set_xlabel("XGBoost feature importance")
        axis.set_title(f"Top {len(top_features)} Feature Importances")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        return importance_table.reset_index(drop=True)
