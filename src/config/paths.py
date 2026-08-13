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
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def params_file(self) -> Path:
        return self.root / "params.yaml"

    @property
    def lookup_file(self) -> Path:
        return self.raw_dir / "nyc_taxi_lookup_with_coords.csv"

    @property
    def metrics_file(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def feature_importance_file(self) -> Path:
        return self.reports_dir / "feature_importance.png"

    @property
    def segment_metrics_file(self) -> Path:
        return self.reports_dir / "segment_metrics.json"

    @property
    def error_analysis_file(self) -> Path:
        return self.reports_dir / "error_analysis.png"

    @property
    def tuning_file(self) -> Path:
        return self.reports_dir / "tuning.json"

    @property
    def run_info_file(self) -> Path:
        return self.reports_dir / "mlflow_run.json"

    @property
    def monitoring_dir(self) -> Path:
        return self.reports_dir / "monitoring"
