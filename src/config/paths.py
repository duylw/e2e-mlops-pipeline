from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(".")

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def split_dir(self) -> Path:
        return self.root / "data" / "split"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def params_file(self) -> Path:
        return self.root / "params.yaml"

    @property
    def lookup_file(self) -> Path:
        return self.raw_dir / "nyc_taxi_lookup_with_coords.csv"

    @property
    def preprocessor_file(self) -> Path:
        return self.artifacts_dir / "preprocessor.pkl"

    @property
    def model_file(self) -> Path:
        return self.models_dir / "model.pkl"

    @property
    def metrics_file(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def run_info_file(self) -> Path:
        return self.reports_dir / "mlflow_run.json"
