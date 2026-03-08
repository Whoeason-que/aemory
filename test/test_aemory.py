import pytest
import tempfile
import os
import shutil
from pathlib import Path

import aemory

# Use lightweight model for testing
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sample_md_files(temp_dir):
    md_path = Path(temp_dir) / "docs"
    md_path.mkdir()

    (md_path / "doc1.md").write_text(
        """---
title: First Document
author: Test
---

# Introduction

This is the first test document with some content.

## Section One

More content here for testing purposes.
"""
    )

    (md_path / "doc2.md").write_text(
        """# Second Document

This document has no frontmatter.

### Subsection

Additional content for semantic search testing.
"""
    )

    (md_path / "doc3.md").write_text(
        "\n".join([f"# Section {i}\n\nContent for section {i}." for i in range(10)])
    )

    return str(md_path)


@pytest.fixture
def built_dataset(sample_md_files, temp_dir):
    output_path = os.path.join(temp_dir, "test.lance")
    aemory.build(sample_md_files, output_path, TEST_MODEL)
    return output_path


class TestBuild:
    def test_build_creates_dataset(self, sample_md_files, temp_dir):
        output_path = os.path.join(temp_dir, "output.lance")
        aemory.build(sample_md_files, output_path, TEST_MODEL)
        assert os.path.exists(output_path)

    def test_build_empty_directory(self, temp_dir):
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)
        output_path = os.path.join(temp_dir, "empty.lance")
        aemory.build(empty_dir, output_path, TEST_MODEL)
        # Empty directories don't create output files
        assert not os.path.exists(output_path)

    def test_build_nonexistent_path_handles_gracefully(self, temp_dir):
        """Non-existent paths should be handled gracefully (no crash)"""
        output_path = os.path.join(temp_dir, "output.lance")
        # Should not raise - just processes 0 files
        aemory.build("/nonexistent/path", output_path, TEST_MODEL)
        # No output file created when input is empty
        assert not os.path.exists(output_path)


class TestSearch:
    def test_search_returns_results(self, built_dataset):
        results = aemory.search(built_dataset, "introduction", 5, TEST_MODEL)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_structure(self, built_dataset):
        results = aemory.search(built_dataset, "content", 1, TEST_MODEL)
        assert len(results) > 0
        result = results[0]
        assert "content" in result
        assert "source_path" in result
        assert "metadata" in result
        assert "score" in result

    def test_search_limit(self, built_dataset):
        results = aemory.search(built_dataset, "document", 3, TEST_MODEL)
        assert len(results) <= 3

    def test_search_empty_query(self, built_dataset):
        results = aemory.search(built_dataset, "", 5, TEST_MODEL)
        assert isinstance(results, list)

    def test_search_nonexistent_dataset_raises(self, temp_dir):
        with pytest.raises(Exception):
            aemory.search(
                os.path.join(temp_dir, "nonexistent.lance"),
                "query",
                5,
                TEST_MODEL,
            )


class TestIntegration:
    def test_build_and_search_workflow(self, sample_md_files, temp_dir):
        output_path = os.path.join(temp_dir, "integration.lance")

        aemory.build(sample_md_files, output_path, TEST_MODEL)

        results = aemory.search(output_path, "section", 5, TEST_MODEL)

        assert len(results) > 0
        for r in results:
            assert r["content"]
            assert r["source_path"]

    def test_multiple_searches(self, built_dataset):
        queries = ["introduction", "document", "section", "content"]

        for query in queries:
            results = aemory.search(built_dataset, query, 3, TEST_MODEL)
            assert isinstance(results, list)
