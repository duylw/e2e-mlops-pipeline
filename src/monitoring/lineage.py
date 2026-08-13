import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config.paths import ProjectPaths
from src.config.settings import MlflowSettings


@dataclass(frozen=True)
class ChampionModel:
    model: object
    name: str
    version: int
    alias: str
    run_id: str
    run_tags: dict[str, str]


def git_revision(root: Path = Path(".")) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_range(X) -> tuple[str, str]:
    pickup_times = X["lpep_pickup_datetime"]
    return str(pickup_times.min()), str(pickup_times.max())


def build_training_lineage(paths: ProjectPaths, train, validation) -> dict[str, str]:
    train_start, train_end = _time_range(train.X)
    validation_start, validation_end = _time_range(validation.X)
    return {
        "code.git_revision": git_revision(paths.root),
        # DVC updates this file after the train stage completes, so it is audit
        # metadata only. Reference-file hashes below are the verification gate.
        "data.dvc_lock_sha256_at_train": file_sha256(paths.root / "dvc.lock"),
        "data.split_strategy": "chronological_60_20_20",
        "data.train.path": train.source_path.as_posix(),
        "data.train.sha256": file_sha256(_resolve_path(paths.root, train.source_path)),
        "data.train.start": train_start,
        "data.train.end": train_end,
        "data.validation.path": validation.source_path.as_posix(),
        "data.validation.sha256": file_sha256(_resolve_path(paths.root, validation.source_path)),
        "data.validation.start": validation_start,
        "data.validation.end": validation_end,
    }


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_champion_model(settings: MlflowSettings) -> ChampionModel:
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(settings.model_name, settings.serving_model_alias)
    run = client.get_run(model_version.run_id)
    model_uri = f"models:/{settings.model_name}@{settings.serving_model_alias}"
    return ChampionModel(
        model=mlflow.pyfunc.load_model(model_uri),
        name=settings.model_name,
        version=int(model_version.version),
        alias=settings.serving_model_alias,
        run_id=model_version.run_id,
        run_tags=dict(run.data.tags),
    )


def verify_reference_lineage(paths: ProjectPaths, champion: ChampionModel) -> None:
    required_tags = (
        "data.train.path",
        "data.train.sha256",
        "data.validation.path",
        "data.validation.sha256",
    )
    missing_tags = [tag for tag in required_tags if not champion.run_tags.get(tag)]
    if missing_tags:
        raise ValueError(
            "Champion model has no DVC lineage tags. Retrain and register it before running monitoring. "
            f"Missing tags: {missing_tags}"
        )

    mismatches = []
    for split_name in ("train", "validation"):
        path = paths.root / champion.run_tags[f"data.{split_name}.path"]
        expected_hash = champion.run_tags[f"data.{split_name}.sha256"]
        if file_sha256(path) != expected_hash:
            mismatches.append(f"{split_name} split hash")
    if mismatches:
        raise ValueError(
            "Current reference data does not match the champion model "
            f"({', '.join(mismatches)}). Run dvc pull for this model's DVC revision before monitoring."
        )
