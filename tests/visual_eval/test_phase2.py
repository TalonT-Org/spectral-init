"""Unit tests for Phase 2 comparison metrics and flow control."""
import json
import subprocess
import numpy as np
import pytest
from pathlib import Path
import tempfile
import sys


class TestComputeMetrics:
    """Tests for _compute_metrics() correctness and structure."""

    def test_returns_expected_keys(self, small_embeddings):
        from generate_umap_comparisons import _compute_metrics
        X, labels, embed_py, embed_rust, embed_rand = small_embeddings
        result = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)
        assert "python_spectral" in result
        assert "rust_spectral" in result
        assert "random" in result
        assert "pass_fail" in result

    def test_python_spectral_has_no_comparison_metrics(self, small_embeddings):
        from generate_umap_comparisons import _compute_metrics
        X, labels, embed_py, embed_rust, embed_rand = small_embeddings
        result = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)
        py = result["python_spectral"]
        assert "trustworthiness" in py
        assert "silhouette" in py
        assert "procrustes_vs_python" not in py
        assert "pairwise_corr_vs_python" not in py

    def test_rust_spectral_has_comparison_metrics(self, small_embeddings):
        from generate_umap_comparisons import _compute_metrics
        X, labels, embed_py, embed_rust, embed_rand = small_embeddings
        result = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)
        rust = result["rust_spectral"]
        assert "trustworthiness" in rust
        assert "silhouette" in rust
        assert "procrustes_vs_python" in rust
        assert "pairwise_corr_vs_python" in rust

    def test_pass_fail_has_overall(self, small_embeddings):
        from generate_umap_comparisons import _compute_metrics
        X, labels, embed_py, embed_rust, embed_rand = small_embeddings
        result = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)
        pf = result["pass_fail"]
        assert "procrustes" in pf
        assert "pairwise_corr" in pf
        assert "trustworthiness" in pf
        assert "silhouette" in pf
        assert "overall" in pf
        assert pf["overall"] in ("PASS", "FAIL")

    def test_identical_embeddings_pass_all(self):
        """When rust init produces identical embedding to python, all metrics pass."""
        from generate_umap_comparisons import _compute_metrics
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 10)
        labels = (rng.rand(n) * 5).astype(int)
        embed_py = rng.randn(n, 2)
        embed_rust = embed_py.copy()  # identical
        embed_rand = rng.randn(n, 2) * 5  # very different
        result = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)
        pf = result["pass_fail"]
        assert pf["procrustes"] == "PASS"
        assert pf["pairwise_corr"] == "PASS"
        assert pf["trustworthiness"] == "PASS"
        assert pf["silhouette"] == "PASS"
        assert pf["overall"] == "PASS"

    def test_large_dataset_subsamples_pairwise(self):
        """n > 2000 must not OOM — subsampling must engage."""
        from generate_umap_comparisons import _compute_metrics
        rng = np.random.RandomState(0)
        n = 2500
        X = rng.randn(n, 10).astype(np.float64)
        labels = (rng.rand(n) * 5).astype(int)
        embed = rng.randn(n, 2)
        # Should complete without MemoryError
        result = _compute_metrics(X, labels, embed, embed, rng.randn(n, 2))
        corr = result["rust_spectral"]["pairwise_corr_vs_python"]
        assert corr is not None
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0
        assert np.isfinite(corr)


class TestRunCompareFileMissing:
    """Tests for graceful handling of missing rust_init.npy."""

    def test_skips_dataset_when_rust_init_missing(self, tmp_path, capsys):
        from generate_umap_comparisons import run_compare
        # Only py artifacts exist; no rust_init.npy
        name = "blobs_1000"
        rng = np.random.RandomState(42)
        n = 50
        np.save(tmp_path / f"{name}_py_spectral.npy", rng.randn(n, 2))
        np.save(tmp_path / f"{name}_py_final.npy", rng.randn(n, 2).astype(np.float32))
        np.save(tmp_path / f"{name}_labels.npy", np.zeros(n, dtype=np.int32))
        # Must not raise; should print warning
        run_compare(name, tmp_path)
        captured = capsys.readouterr()
        assert "rust_init.npy not found" in captured.out.lower()
        # No output files produced
        assert not (tmp_path / f"{name}_comparison.png").exists()


class TestOutputFilesProduced:
    """Integration-style: verify all output files are created."""

    @pytest.mark.slow
    def test_comparison_png_created(self, tmp_path, small_phase1_artifacts):
        """run_compare() produces {name}_comparison.png."""
        from generate_umap_comparisons import run_compare
        name = "blobs_1000"
        _write_small_artifacts(tmp_path, name, small_phase1_artifacts)
        run_compare(name, tmp_path)
        assert (tmp_path / f"{name}_comparison.png").exists()

    @pytest.mark.slow
    def test_overlay_png_created(self, tmp_path, small_phase1_artifacts):
        from generate_umap_comparisons import run_compare
        name = "blobs_1000"
        _write_small_artifacts(tmp_path, name, small_phase1_artifacts)
        run_compare(name, tmp_path)
        assert (tmp_path / f"{name}_overlay.png").exists()

    @pytest.mark.slow
    def test_metrics_json_created_and_valid(self, tmp_path, small_phase1_artifacts):
        from generate_umap_comparisons import run_compare
        name = "blobs_1000"
        _write_small_artifacts(tmp_path, name, small_phase1_artifacts)
        run_compare(name, tmp_path)
        json_path = tmp_path / f"{name}_metrics.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["dataset"] == name
        assert "pass_fail" in data
        assert data["pass_fail"]["overall"] in ("PASS", "FAIL")


class TestCLIPhaseCompare:
    """Test that --phase compare is accepted by argparse."""

    def test_compare_choice_accepted(self, monkeypatch):
        script = Path(__file__).parent / "generate_umap_comparisons.py"
        result = subprocess.run(
            [sys.executable, str(script), "--phase", "compare", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_embeddings():
    rng = np.random.RandomState(0)
    n = 150
    X = rng.randn(n, 10).astype(np.float64)
    labels = (rng.rand(n) * 3).astype(int)
    embed_py = rng.randn(n, 2)
    embed_rust = embed_py + rng.randn(n, 2) * 0.01  # nearly identical
    embed_rand = rng.randn(n, 2) * 5
    return X, labels, embed_py, embed_rust, embed_rand

@pytest.fixture
def small_phase1_artifacts():
    rng = np.random.RandomState(7)
    n = 1000  # Must match blobs_1000 dataset size: run_compare() calls load_dataset("blobs_1000") which returns 1000 samples
    return {
        "X": rng.randn(n, 10).astype(np.float64),
        "labels": (rng.rand(n) * 3).astype(int),
        "py_spectral": rng.randn(n, 2),
        "py_final": rng.randn(n, 2).astype(np.float32),
        "rust_init": rng.randn(n, 2),
    }

def _write_small_artifacts(tmp_path, name, arts):
    np.save(tmp_path / f"{name}_py_spectral.npy", arts["py_spectral"])
    np.save(tmp_path / f"{name}_py_final.npy", arts["py_final"])
    np.save(tmp_path / f"{name}_labels.npy", arts["labels"].astype(np.int32))
    np.save(tmp_path / f"{name}_rust_init.npy", arts["rust_init"])
