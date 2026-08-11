import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings
from src.utils.io import load_yaml, save_json


def promote_if_better(client: MlflowClient, model_name: str, new_version, new_rmse: float | None) -> str:
    if new_rmse is None:
        client.set_registered_model_alias(model_name, "challenger", new_version.version)
        return "challenger"

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_rmse = champion_run.data.metrics.get("rmse_test")
    except Exception:
        client.set_registered_model_alias(model_name, "champion", new_version.version)
        return "champion"

    if champion_rmse is None or new_rmse <= champion_rmse:
        client.set_registered_model_alias(model_name, "champion", new_version.version)
        return "champion"

    client.set_registered_model_alias(model_name, "challenger", new_version.version)
    return "challenger"


def register_model(run_id: str | None = None) -> dict:
    paths = ProjectPaths()
    settings = load_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()

    run_info = load_yaml(paths.run_info_file) if paths.run_info_file.exists() else {}
    selected_run_id = run_id or run_info.get("run_id")
    if not selected_run_id:
        raise ValueError("run_id is required. Pass --run-id or run training first.")

    run = mlflow.get_run(selected_run_id)
    model_uri = run_info.get("model_uri") or run_info.get("model_artifact_uri")
    if not model_uri:
        model_uri = f"runs:/{selected_run_id}/model"
    new_version = mlflow.register_model(model_uri=model_uri, name=settings.model_name)
    alias = promote_if_better(
        client=client,
        model_name=settings.model_name,
        new_version=new_version,
        new_rmse=run.data.metrics.get("rmse_test"),
    )

    registry_info = {
        "run_id": selected_run_id,
        "model_name": settings.model_name,
        "model_version": int(new_version.version),
        "model_alias": alias,
        "model_uri": model_uri,
    }
    save_json(registry_info, paths.reports_dir / "registry.json")
    return registry_info


def parse_args():
    parser = argparse.ArgumentParser(description="Register and promote an MLflow model.")
    parser.add_argument("--run-id", default=None, help="MLflow run id. Defaults to reports/mlflow_run.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    registry_info = register_model(run_id=args.run_id)
    print(f"Registered model: {registry_info}")


if __name__ == "__main__":
    main()
