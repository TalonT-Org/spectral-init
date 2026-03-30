"""Unit tests for generate_merfish_comparisons.py."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_load_merfish_data_shapes():
    from generate_merfish_comparisons import load_merfish_data

    data_dir = Path(__file__).parent / "merfish_data"
    expression, spatial, labels, section_ids = load_merfish_data(data_dir)
    assert expression.shape == (10000, 1122) and expression.dtype == np.float32
    assert spatial.shape == (10000, 2) and spatial.dtype == np.float32
    assert labels.shape == (10000,) and labels.dtype == np.int32
    assert section_ids.shape == (10000,) and section_ids.dtype == np.int32


def test_export_graph_format(tmp_path):
    import scipy.sparse
    from generate_merfish_comparisons import export_graph

    rng = np.random.RandomState(0)
    G = scipy.sparse.random(50, 50, density=0.1, format="csr", random_state=rng).astype(
        np.float32
    )
    out = tmp_path / "test_graph.npz"
    export_graph(G, out)
    loaded = np.load(out)
    assert loaded["data"].dtype == np.float32
    assert loaded["indices"].dtype == np.int32
    assert loaded["indptr"].dtype == np.int32
    assert loaded["shape"].dtype == np.int32
    assert tuple(loaded["shape"]) == (50, 50)


def test_run_compare_skips_when_rust_init_missing(tmp_path, capsys):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(0)
    n = 50
    np.save(tmp_path / "merfish_10k_py_spectral.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_py_final.npy", rng.randn(n, 2).astype(np.float32))
    np.save(tmp_path / "merfish_10k_labels.npy", np.zeros(n, dtype=np.int32))
    np.save(tmp_path / "merfish_10k_pca.npy", rng.randn(n, 30))
    result = run_compare(tmp_path)
    assert result is None
    out = capsys.readouterr().out
    assert "rust_init.npy not found" in out.lower()


@pytest.mark.slow
def test_run_compare_produces_output_files(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(42)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    run_compare(tmp_path)
    assert (tmp_path / "merfish_10k_comparison.png").exists()
    assert (tmp_path / "merfish_10k_overlay.png").exists()
    assert (tmp_path / "merfish_10k_three_way_overlay.png").exists()
    assert (tmp_path / "merfish_10k_metrics.json").exists()


@pytest.mark.slow
def test_metrics_json_has_required_keys(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(7)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    run_compare(tmp_path)
    data = json.loads((tmp_path / "merfish_10k_metrics.json").read_text())
    assert data["dataset"] == "merfish_10k"
    assert "pass_fail" in data
    assert data["pass_fail"]["overall"] in ("PASS", "FAIL")
    assert "python_spectral" in data
    assert "rust_spectral" in data


def test_cli_accepts_phase_baseline_and_compare():
    script = Path(__file__).parent / "generate_merfish_comparisons.py"
    for phase in ["baseline", "compare"]:
        r = subprocess.run(
            [sys.executable, str(script), "--phase", phase, "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0


def _write_compare_artifacts(tmp_path, rng, n):
    np.save(tmp_path / "merfish_10k_py_spectral.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_py_final.npy", rng.randn(n, 2).astype(np.float32))
    np.save(tmp_path / "merfish_10k_rust_init.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_labels.npy", (rng.rand(n) * 3).astype(np.int32))
    np.save(tmp_path / "merfish_10k_pca.npy", rng.randn(n, 30).astype(np.float64))
