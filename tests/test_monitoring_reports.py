import pandas as pd

from src.monitoring.reports import build_input_report, build_quality_report, save_report


def monitoring_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [1, 2, 1, 2, 1, 2],
            "PULocationID": [1, 1, 2, 2, 1, 2],
            "DOLocationID": [2, 2, 1, 1, 2, 1],
            "trip_type": [1, 1, 1, 1, 1, 1],
            "pickup_hour": [8, 9, 10, 8, 9, 10],
            "pickup_day_of_week": [1, 2, 3, 1, 2, 3],
            "passenger_count": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "lpep_pickup_datetime": pd.date_range("2026-01-01", periods=6, freq="h"),
        }
    )


def test_evidently_reports_export_metrics_and_artifacts(tmp_path):
    input_frame = monitoring_frame()
    quality_frame = input_frame.assign(
        duration=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        prediction=[10.1, 11.1, 12.1, 13.1, 14.1, 15.1],
    )

    input_result = build_input_report(input_frame, input_frame, drift_warning_share=0.3)
    quality_result = build_quality_report(quality_frame, quality_frame)
    save_report(input_result, tmp_path / "input.html", tmp_path / "input.json")
    save_report(quality_result, tmp_path / "quality.html", tmp_path / "quality.json")

    assert input_result.metrics["input_drifted_columns_share"] == 0.0
    assert quality_result.metrics["quality_rmse"] > 0
    assert quality_result.metrics["quality_mae"] > 0
    assert (tmp_path / "input.html").exists()
    assert (tmp_path / "quality.json").exists()
