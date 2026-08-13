from dataclasses import dataclass
from pathlib import Path

from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

from src.monitoring.data import CATEGORICAL_MONITORING_COLUMNS, DRIFT_MONITORING_COLUMNS


@dataclass(frozen=True)
class ReportResult:
    report: Report
    metrics: dict[str, float]


def input_data_definition() -> DataDefinition:
    return DataDefinition(
        categorical_columns=CATEGORICAL_MONITORING_COLUMNS,
        numerical_columns=["passenger_count"],
        datetime_columns=["lpep_pickup_datetime"],
        timestamp="lpep_pickup_datetime",
    )


def quality_data_definition() -> DataDefinition:
    definition = input_data_definition()
    definition.numerical_columns = ["passenger_count", "duration", "prediction"]
    definition.regression = [Regression(target="duration", prediction="prediction")]
    return definition


def build_input_report(reference_frame, current_frame, drift_warning_share: float) -> ReportResult:
    definition = input_data_definition()
    reference = Dataset.from_pandas(reference_frame, data_definition=definition)
    current = Dataset.from_pandas(current_frame, data_definition=definition)
    report = Report(
        [DataSummaryPreset(), DataDriftPreset(columns=DRIFT_MONITORING_COLUMNS, drift_share=drift_warning_share)],
        include_tests=True,
    ).run(current_data=current, reference_data=reference)
    return ReportResult(report=report, metrics=extract_input_metrics(report))


def build_quality_report(reference_frame, current_frame) -> ReportResult:
    definition = quality_data_definition()
    reference = Dataset.from_pandas(reference_frame, data_definition=definition)
    current = Dataset.from_pandas(current_frame, data_definition=definition)
    report = Report(
        [RegressionPreset(), ValueDrift(column="duration"), ValueDrift(column="prediction")],
        include_tests=True,
    ).run(current_data=current, reference_data=reference)
    return ReportResult(report=report, metrics=extract_quality_metrics(report))


def save_report(result: ReportResult, html_path: Path, json_path: Path) -> None:
    result.report.save_html(str(html_path))
    result.report.save_json(str(json_path))


def extract_input_metrics(report: Report) -> dict[str, float]:
    metrics = _metrics_by_type(report)
    drift_summary = _first_value(metrics, "DriftedColumnsCount")
    missing_summary = _first_value(metrics, "DatasetMissingValueCount")
    result = {
        "input_drifted_columns_count": float(drift_summary["count"]),
        "input_drifted_columns_share": float(drift_summary["share"]),
        "input_missing_value_share": float(missing_summary["share"]),
    }
    result.update(_drift_test_metrics(report, prefix="input"))
    return result


def extract_quality_metrics(report: Report) -> dict[str, float]:
    metrics = _metrics_by_type(report)
    rmse = float(_first_value(metrics, "RMSE"))
    mae = float(_first_value(metrics, "MAE")["mean"])
    result = {
        "quality_rmse": rmse,
        "quality_mae": mae,
        "quality_mse_from_rmse": rmse**2,
        "quality_failed_tests": float(_failed_test_count(report)),
    }
    for metric in _all_values(metrics, "ValueDrift"):
        column = metric["config"].get("column")
        if column in {"duration", "prediction"}:
            result[f"quality_{column}_drift_score"] = float(metric["value"])
    return result


def _metrics_by_type(report: Report) -> dict[str, list[dict]]:
    metrics: dict[str, list[dict]] = {}
    for metric in report.dict()["metrics"]:
        metric_type = metric["config"]["type"].rsplit(":", maxsplit=1)[-1]
        metrics.setdefault(metric_type, []).append(metric)
    return metrics


def _first_value(metrics: dict[str, list[dict]], metric_type: str):
    values = metrics.get(metric_type, [])
    if not values:
        raise ValueError(f"Evidently report is missing {metric_type}")
    return values[0]["value"]


def _all_values(metrics: dict[str, list[dict]], metric_type: str) -> list[dict]:
    return metrics.get(metric_type, [])


def _drift_test_metrics(report: Report, prefix: str) -> dict[str, float]:
    result = {}
    for test in report.dict().get("tests", []):
        params = test.get("metric_config", {}).get("params", {})
        column = params.get("column")
        if test.get("id") == "drift" and column in DRIFT_MONITORING_COLUMNS:
            result[f"{prefix}_{column}_drift_detected"] = float(test["status"] != "SUCCESS")
    return result


def _failed_test_count(report: Report) -> int:
    return sum(test["status"] != "SUCCESS" for test in report.dict().get("tests", []))
