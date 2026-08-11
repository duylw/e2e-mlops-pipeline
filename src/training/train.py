import sys

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings, load_params
from src.training.pipeline import TrainingPipeline
from src.training.tracker import MlflowTrainingTracker


def configure_console_encoding() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def main() -> None:
    configure_console_encoding()
    paths = ProjectPaths()
    train_params = load_params(paths.params_file)["train"]
    tracker = MlflowTrainingTracker(load_mlflow_settings())
    result = TrainingPipeline(paths, train_params, tracker).run()
    print(f"Logged full pipeline to MLflow: {result.model_uri}")
    print(f"MLflow run id: {result.run_id}")
    print(f"Saved metrics to {paths.metrics_file}")


if __name__ == "__main__":
    main()
