import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.paths import ProjectPaths
from src.utils.io import save_json


def benchmark_case(
    client: httpx.Client,
    url: str,
    trips: list[dict],
    request_count: int,
    concurrency: int,
) -> dict:
    payload = {"trips": trips}

    def send_request() -> tuple[int, float]:
        started_at = time.perf_counter()
        response = client.post(url, json=payload)
        return response.status_code, time.perf_counter() - started_at

    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: send_request(), range(request_count)))
    elapsed = time.perf_counter() - started_at
    statuses, latencies = zip(*results, strict=True)
    successful = sum(status == 200 for status in statuses)
    return {
        "batch_size": len(trips),
        "concurrency": concurrency,
        "request_count": request_count,
        "success_rate": successful / request_count,
        "requests_per_second": request_count / elapsed,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50) * 1000),
            "p95": float(np.percentile(latencies, 95) * 1000),
            "p99": float(np.percentile(latencies, 99) * 1000),
        },
    }


def build_batch(samples: list[dict], batch_size: int) -> list[dict]:
    return [samples[index % len(samples)] for index in range(batch_size)]


def run_benchmark(base_url: str, sample_path: Path, request_count: int) -> dict:
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("Sample payload must contain at least one trip.")

    with httpx.Client(timeout=10.0) as client:
        metadata = client.get(f"{base_url}/metadata").json()
        client.get(f"{base_url}/health").raise_for_status()
        results = [
            benchmark_case(client, f"{base_url}/predict", build_batch(samples, batch_size), request_count, concurrency)
            for batch_size in (1, 10, 100)
            for concurrency in (1, 5, 10)
        ]
    return {"model": metadata, "request_count_per_case": request_count, "cases": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local FastAPI serving endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--sample-path", type=Path, default=Path("data_sample.json"))
    parser.add_argument("--request-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_count < 1:
        raise ValueError("--request-count must be at least 1")
    result = run_benchmark(args.base_url.rstrip("/"), args.sample_path, args.request_count)
    output_path = ProjectPaths().reports_dir / "serving_benchmark.json"
    save_json(result, output_path)
    print(f"Saved aggregate benchmark results to {output_path}")


if __name__ == "__main__":
    main()
