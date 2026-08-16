from pathlib import Path

import pandas as pd
import pytest

from src.config.paths import ProjectPaths
from src.monitoring.lineage import ChampionModel, build_training_lineage, verify_reference_lineage
from src.training.pipeline import SplitData, build_training_params


def champion_with_tags() -> ChampionModel:
    return ChampionModel(
        model=object(),
        name="green_taxi_duration_model",
        version=1,
        alias="champion",
        run_id="run-id",
        run_tags={
            "code.git_revision": "expected-git",
            "data.dvc_lock_sha256_at_train": "expected-lock",
            "data.train.path": "data/split/train_raw.parquet",
            "data.train.sha256": "expected-train-hash",
            "data.validation.path": "data/split/val_raw.parquet",
            "data.validation.sha256": "expected-validation-hash",
        },
    )


def test_reference_hash_mismatch_stops_monitoring(tmp_path):
    paths = ProjectPaths(root=Path(tmp_path))
    split_dir = tmp_path / "data" / "split"
    split_dir.mkdir(parents=True)
    (split_dir / "train_raw.parquet").write_text("different train", encoding="utf-8")
    (split_dir / "val_raw.parquet").write_text("different validation", encoding="utf-8")

    with pytest.raises(ValueError, match="train split hash"):
        verify_reference_lineage(paths, champion_with_tags())


def test_training_metadata_uses_params_for_rows_and_tags_for_time_coverage(tmp_path, monkeypatch):
    paths = ProjectPaths(root=Path(tmp_path))
    (tmp_path / "dvc.lock").write_text("lock", encoding="utf-8")
    split_dir = tmp_path / "data" / "split"
    split_dir.mkdir(parents=True)
    (split_dir / "train_raw.parquet").write_text("train", encoding="utf-8")
    (split_dir / "val_raw.parquet").write_text("validation", encoding="utf-8")
    monkeypatch.setattr("src.monitoring.lineage.git_revision", lambda root: "git-revision")
    train = SplitData(
        X=pd.DataFrame({"lpep_pickup_datetime": ["2026-02-01 08:00:00", "2026-02-02 08:00:00"]}),
        y=pd.Series([10.0, 12.0]),
        source_path=Path("data/split/train_raw.parquet"),
    )
    validation = SplitData(
        X=pd.DataFrame({"lpep_pickup_datetime": ["2026-02-03 08:00:00"]}),
        y=pd.Series([14.0]),
        source_path=Path("data/split/val_raw.parquet"),
    )
    test = SplitData(
        X=pd.DataFrame({"lpep_pickup_datetime": ["2026-02-04 08:00:00"]}),
        y=pd.Series([16.0]),
        source_path=Path("data/split/test_raw.parquet"),
    )

    lineage_tags = build_training_lineage(paths, train, validation)
    params = build_training_params(train, validation, test, {"max_depth": 7})

    assert lineage_tags["data.train.start"] == "2026-02-01 08:00:00"
    assert lineage_tags["data.validation.end"] == "2026-02-03 08:00:00"
    assert lineage_tags["data.train.sha256"] != "unavailable"
    assert "train_rows" not in lineage_tags
    assert params["data_train_rows"] == 2
    assert params["data_validation_rows"] == 1
    assert params["data_test_rows"] == 1
