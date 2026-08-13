import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from src.config.paths import ProjectPaths
from src.config.settings import load_mlflow_settings, load_params
from src.monitoring.data import load_monitoring_batch
from src.monitoring.lineage import resolve_champion_model, verify_reference_lineage
from src.monitoring.reports import build_input_report, build_quality_report, save_report
from src.utils.io import ensure_dir, save_json

MONITORING_EXPERIMENT_NAME = "green_taxi_duration_monitoring"


def configure_console_encoding() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def run_monitoring(current_path: str | Path, current_name: str | None = None) -> dict:
    paths = ProjectPaths()
    settings = load_mlflow_settings()
    monitoring_params = load_params(paths.params_file)["monitoring"]
    champion = resolve_champion_model(settings)
    verify_reference_lineage(paths, champion)

    train_path = paths.root / champion.run_tags["data.train.path"]
    validation_path = paths.root / champion.run_tags["data.validation.path"]
    reference_input = load_monitoring_batch(
        path=train_path,
        model=champion.model,
        max_rows=monitoring_params["reference_max_rows"],
        random_seed=monitoring_params["random_seed"],
    ).input_frame
    reference_quality = load_monitoring_batch(
        path=validation_path,
        model=champion.model,
        max_rows=monitoring_params["reference_max_rows"],
        random_seed=monitoring_params["random_seed"],
    ).quality_frame
    if reference_quality is None:
        raise ValueError("Validation reference must contain labels.")

    current_path = Path(current_path)
    current_batch = load_monitoring_batch(
        path=current_path,
        model=champion.model,
        max_rows=monitoring_params["current_max_rows"],
        random_seed=monitoring_params["random_seed"],
    )
    input_result = build_input_report(
        reference_frame=reference_input,
        current_frame=current_batch.input_frame,
        drift_warning_share=monitoring_params["drift_warning_share"],
    )
    quality_result = (
        build_quality_report(reference_quality, current_batch.quality_frame)
        if current_batch.quality_frame is not None
        else None
    )

    output_dir = _output_dir(paths)
    input_html = output_dir / "input_monitoring.html"
    input_json = output_dir / "input_monitoring.json"
    save_report(input_result, input_html, input_json)
    artifacts = [input_html, input_json]
    metrics = dict(input_result.metrics)
    if quality_result is not None:
        quality_html = output_dir / "quality_monitoring.html"
        quality_json = output_dir / "quality_monitoring.json"
        save_report(quality_result, quality_html, quality_json)
        artifacts.extend([quality_html, quality_json])
        metrics.update(quality_result.metrics)

    monitoring_name = current_name or current_path.stem
    summary = {
        "current_name": monitoring_name,
        "current_path": str(current_path),
        "current_rows": len(current_batch.input_frame),
        "labels_available": current_batch.quality_frame is not None,
        "model_name": champion.name,
        "model_version": champion.version,
        "model_alias": champion.alias,
        "model_run_id": champion.run_id,
        "reference_git_revision": champion.run_tags["code.git_revision"],
        "reference_dvc_lock_sha256_at_train": champion.run_tags.get("data.dvc_lock_sha256_at_train", "unavailable"),
        "reference_train_sha256": champion.run_tags["data.train.sha256"],
        "reference_validation_sha256": champion.run_tags["data.validation.sha256"],
        "output_dir": str(output_dir),
        "metrics": metrics,
    }
    summary_path = output_dir / "summary.json"
    save_json(summary, summary_path)
    artifacts.append(summary_path)

    log_monitoring_run(settings, monitoring_name, summary, metrics, artifacts, summary_path)
    return summary


def log_monitoring_run(
    settings,
    current_name: str,
    summary: dict,
    metrics: dict[str, float],
    artifacts: list[Path],
    summary_path: Path,
) -> str:
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(MONITORING_EXPERIMENT_NAME)
    tags = {
        "model_name": str(summary["model_name"]),
        "model_version": str(summary["model_version"]),
        "model_alias": str(summary["model_alias"]),
        "model_run_id": str(summary["model_run_id"]),
        "current_name": current_name,
        "labels_available": str(summary["labels_available"]).lower(),
        "reference_git_revision": str(summary["reference_git_revision"]),
        "reference_dvc_lock_sha256_at_train": str(summary["reference_dvc_lock_sha256_at_train"]),
        "reference_train_sha256": str(summary["reference_train_sha256"]),
        "reference_validation_sha256": str(summary["reference_validation_sha256"]),
    }
    with mlflow.start_run(run_name=f"monitor-{current_name}") as run:
        summary["monitoring_run_id"] = run.info.run_id
        save_json(summary, summary_path)
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="reports")
    return run.info.run_id


def _output_dir(paths: ProjectPaths) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_dir(paths.monitoring_dir / timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evidently batch monitoring for the champion model.")
    parser.add_argument("--current-path", required=True, help="Parquet file containing current TLC or API-shaped data.")
    parser.add_argument("--current-name", default=None, help="Human-readable label for this current batch.")
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    summary = run_monitoring(args.current_path, args.current_name)
    print(f"Monitoring complete: {summary['monitoring_run_id']}")
    print(f"Reports saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
