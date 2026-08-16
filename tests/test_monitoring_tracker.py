import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import MlflowSettings
from src.monitoring.run import log_monitoring_run
from src.utils.io import save_json


def test_monitoring_run_logs_metrics_and_reports_to_mlflow(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    settings = MlflowSettings(
        tracking_uri=tracking_uri,
        experiment_name="unused",
        model_name="green_taxi_duration_model",
        serving_model_alias="champion",
    )
    report_path = tmp_path / "input_monitoring.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary = {
        "model_name": "green_taxi_duration_model",
        "model_version": 1,
        "model_alias": "champion",
        "model_run_id": "model-run-id",
        "labels_available": True,
        "reference_git_revision": "git-revision",
        "reference_dvc_lock_sha256_at_train": "dvc-lock-hash",
        "reference_train_sha256": "train-hash",
        "reference_validation_sha256": "validation-hash",
    }
    save_json(summary, summary_path)

    try:
        run_id = log_monitoring_run(
            settings=settings,
            current_name="test-batch",
            summary=summary,
            metrics={"input_drifted_columns_share": 0.0},
            artifacts=[report_path, summary_path],
            summary_path=summary_path,
        )

        client = MlflowClient(tracking_uri)
        run = client.get_run(run_id)
        artifacts = client.list_artifacts(run_id, "reports")
        assert run.data.metrics["input_drifted_columns_share"] == 0.0
        assert run.data.tags["model_alias"] == "champion"
        assert {artifact.path for artifact in artifacts} == {"reports/input_monitoring.html", "reports/summary.json"}
        assert summary["monitoring_run_id"] == run_id
    finally:
        mlflow.set_tracking_uri("http://localhost:5000")
        try:
            from mlflow.store.db.utils import _EngineRegistry

            _EngineRegistry._registry.clear()
        except Exception:
            pass

