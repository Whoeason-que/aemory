import pytest
import tempfile
import shutil
import os
import time
from pathlib import Path

import aemory

# Use lightweight model for benchmarking
BENCH_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def large_md_dataset():
    temp_dir = tempfile.mkdtemp()
    md_path = Path(temp_dir) / "large_docs"
    md_path.mkdir()

    for i in range(50):
        content = f"""---
id: {i}
category: benchmark
---

# Document {i}

{" ".join(["This is test content for benchmarking." for _ in range(20)])}

## Section A

{" ".join(["Additional content here." for _ in range(10)])}

## Section B

{" ".join(["More text for the document." for _ in range(10)])}
"""
        (md_path / f"doc_{i}.md").write_text(content)

    yield str(md_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def built_large_dataset(benchmark, large_md_dataset):
    output_path = os.path.join(tempfile.gettempdir(), "bench_large.lance")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    benchmark(lambda: aemory.build(large_md_dataset, output_path, BENCH_MODEL))
    yield output_path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)


class TestBuildBenchmark:
    def test_benchmark_build_small(self, benchmark, tmp_path):
        md_path = tmp_path / "small"
        md_path.mkdir()

        for i in range(5):
            (md_path / f"doc_{i}.md").write_text(f"# Doc {i}\n\nContent " * 10)

        output = str(tmp_path / "small.lance")
        benchmark(lambda: aemory.build(str(md_path), output, BENCH_MODEL))

    def test_benchmark_build_medium(self, benchmark, tmp_path):
        md_path = tmp_path / "medium"
        md_path.mkdir()

        for i in range(20):
            (md_path / f"doc_{i}.md").write_text(f"# Doc {i}\n\nContent " * 50)

        output = str(tmp_path / "medium.lance")
        benchmark(lambda: aemory.build(str(md_path), output, BENCH_MODEL))


class TestSearchBenchmark:
    def test_benchmark_search_single(self, benchmark, built_large_dataset):
        benchmark(
            lambda: aemory.search(
                built_large_dataset, "benchmark content", 5, BENCH_MODEL
            )
        )

    def test_benchmark_search_multiple_queries(self, benchmark, built_large_dataset):
        queries = ["document", "section", "content", "test", "benchmark"]

        def search_all():
            return [
                aemory.search(built_large_dataset, q, 3, BENCH_MODEL) for q in queries
            ]

        benchmark(search_all)

    def test_benchmark_search_large_limit(self, benchmark, built_large_dataset):
        benchmark(
            lambda: aemory.search(built_large_dataset, "content", 50, BENCH_MODEL)
        )


class TestLatencyBenchmark:
    def test_search_latency_p50(self, built_large_dataset):
        latencies = []
        query = "benchmark test query"

        for _ in range(100):
            start = time.perf_counter()
            aemory.search(built_large_dataset, query, 10, BENCH_MODEL)
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]

        print("\nSearch Latency Stats:")
        print(f"  P50: {p50 * 1000:.2f}ms")
        print(f"  P95: {p95 * 1000:.2f}ms")
        print(f"  P99: {p99 * 1000:.2f}ms")
        print(f"  Min: {min(latencies) * 1000:.2f}ms")
        print(f"  Max: {max(latencies) * 1000:.2f}ms")

        assert p50 < 1.0, f"P50 latency {p50}s exceeds 1s threshold"

    def test_throughput(self, built_large_dataset):
        duration = 5.0
        query = "throughput test query"
        count = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            aemory.search(built_large_dataset, query, 5, BENCH_MODEL)
            count += 1

        elapsed = time.perf_counter() - start
        qps = count / elapsed

        print(f"\nThroughput: {qps:.2f} queries/second")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-sort=mean"])
