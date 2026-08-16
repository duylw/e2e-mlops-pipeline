import json

import httpx

from scripts.benchmark_serving import build_batch, run_benchmark


def test_build_batch_reuses_small_sample_without_changing_input():
    samples = [{"VendorID": 1}, {"VendorID": 2}]

    batch = build_batch(samples, 5)

    assert len(batch) == 5
    assert batch[0] == samples[0]
    assert batch[2] == samples[0]


def test_benchmark_aggregates_latency_without_logging_payload(tmp_path, monkeypatch):
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps([{"VendorID": 1}]), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/metadata":
            return httpx.Response(200, json={"model_version": 1})
        return httpx.Response(200, json={"predictions": [12.0]})

    transport = httpx.MockTransport(handler)

    class MockClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=transport, *args, **kwargs)

    monkeypatch.setattr("scripts.benchmark_serving.httpx.Client", MockClient)
    result = run_benchmark("http://serving.test", sample_path, request_count=1)

    assert len(result["cases"]) == 9
    assert all(case["success_rate"] == 1.0 for case in result["cases"])
    assert "trips" not in str(result)
