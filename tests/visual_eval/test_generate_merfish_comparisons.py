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
    if not data_dir.exists():
        pytest.skip("merfish_data/ not present — skipping data shape test")
    expression, spatial, labels, section_ids = load_merfish_data(data_dir)
    assert expression.shape == (10000, 1122) and expression.dtype == np.float32
    assert spatial.shape == (10000, 2) and spatial.dtype == np.float32
    assert labels.shape == (10000,) and labels.dtype == np.int32
    assert section_ids.shape == (10000,) and section_ids.dtype == np.int32


def test_export_graph_format(tmp_path):
    import scipy.sparse
    from generate_umap_comparisons import export_graph

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


def test_run_compare_skips_when_rust_init_missing(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(0)
    n = 50
    np.save(tmp_path / "merfish_10k_py_spectral.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_py_final.npy", rng.randn(n, 2).astype(np.float32))
    np.save(tmp_path / "merfish_10k_labels.npy", np.zeros(n, dtype=np.int32))
    np.save(tmp_path / "merfish_10k_pca.npy", rng.randn(n, 30))
    result = run_compare(tmp_path)
    assert result is None


def _write_compare_artifacts(tmp_path, rng, n):
    np.save(tmp_path / "merfish_10k_py_spectral.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_py_final.npy", rng.randn(n, 2).astype(np.float32))
    np.save(tmp_path / "merfish_10k_rust_init.npy", rng.randn(n, 2))
    np.save(tmp_path / "merfish_10k_labels.npy", (rng.rand(n) * 5).astype(np.int32))
    np.save(tmp_path / "merfish_10k_pca.npy", rng.randn(n, 30).astype(np.float64))
    np.savez_compressed(
        tmp_path / "merfish_10k_spatial.npz",
        arr_0=rng.randn(n, 2).astype(np.float32),
    )
    (tmp_path / "merfish_10k_rust_perf.txt").write_text("0.5 100000\n")


@pytest.mark.slow
def test_run_compare_produces_output_files(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(42)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    result = run_compare(tmp_path)
    assert result is not None
    for png in ["merfish_10k_comparison.png", "merfish_10k_overlay.png", "merfish_10k_three_way_overlay.png"]:
        p = tmp_path / png
        assert p.exists() and p.stat().st_size > 0
    metrics_path = tmp_path / "merfish_10k_metrics.json"
    assert metrics_path.exists()
    json.loads(metrics_path.read_text())  # validates parseable JSON


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


def test_sna_gate_logic():
    from generate_merfish_comparisons import _check_sna_gate

    # Rust SNA exactly at threshold: PASS
    assert _check_sna_gate(rust_sna=0.30, python_sna=0.32) == "PASS"
    # Rust SNA above threshold: PASS
    assert _check_sna_gate(rust_sna=0.35, python_sna=0.32) == "PASS"
    # Rust SNA just below threshold: FAIL (strict boundary)
    assert _check_sna_gate(rust_sna=0.2999, python_sna=0.32) == "FAIL"
    # Rust SNA below threshold by small margin: FAIL
    assert _check_sna_gate(rust_sna=0.29, python_sna=0.32) == "FAIL"
    # Custom threshold
    assert _check_sna_gate(rust_sna=0.20, python_sna=0.30, threshold=0.05) == "FAIL"
    assert _check_sna_gate(rust_sna=0.26, python_sna=0.30, threshold=0.05) == "PASS"


@pytest.mark.slow
def test_metrics_json_bcd_keys(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(13)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    run_compare(tmp_path)
    data = json.loads((tmp_path / "merfish_10k_metrics.json").read_text())

    cat_b_keys = ("sna", "spatial_dist_corr", "morans_i_max", "morans_i_dim0",
                  "morans_i_dim1", "chaos", "pas")
    cat_c_keys = ("ari", "nmi", "celltype_purity")
    cat_d_keys = ("triplet_accuracy", "shepard_pearson", "shepard_spearman",
                  "centroid_dist_corr", "knn_preservation")

    for emb_key in ("python_spectral", "rust_spectral", "random"):
        emb = data[emb_key]
        for k in cat_b_keys + cat_c_keys + cat_d_keys:
            assert k in emb, f"{emb_key} missing key '{k}'"


@pytest.mark.slow
def test_sna_gate_in_pass_fail(tmp_path):
    from generate_merfish_comparisons import run_compare

    rng = np.random.RandomState(99)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    run_compare(tmp_path)
    data = json.loads((tmp_path / "merfish_10k_metrics.json").read_text())
    assert "sna" in data["pass_fail"], "SNA gate missing from pass_fail"
    assert data["pass_fail"]["sna"] in ("PASS", "FAIL")


def test_cli_accepts_phase_baseline_and_compare():
    script = Path(__file__).parent / "generate_merfish_comparisons.py"
    # Verify valid choices are accepted: argparse prints usage and exits 0 for --help,
    # but we need to confirm --phase is actually validated. Test that an invalid phase
    # is rejected (non-zero exit) to confirm the choices constraint is enforced.
    r_invalid = subprocess.run(
        [sys.executable, str(script), "--phase", "invalid_phase"],
        capture_output=True,
        text=True,
    )
    assert r_invalid.returncode != 0
    # Verify omitting --phase is also rejected (it is required).
    r_missing = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert r_missing.returncode != 0


def test_run_baseline_returns_tuple(tmp_path):
    """run_baseline() must return (dict, float) — not None."""
    from generate_merfish_comparisons import run_baseline
    data_dir = Path(__file__).parent / "merfish_data"
    if not data_dir.exists():
        pytest.skip("merfish_data/ not present")
    result = run_baseline(tmp_path, data_dir=data_dir)
    assert isinstance(result, tuple) and len(result) == 2
    timings, rss_mb = result
    assert isinstance(timings, dict)
    assert isinstance(rss_mb, float) and rss_mb > 0


def test_run_baseline_timing_keys(tmp_path):
    """run_baseline() timing dict must have all 6 required keys, all float."""
    from generate_merfish_comparisons import run_baseline
    data_dir = Path(__file__).parent / "merfish_data"
    if not data_dir.exists():
        pytest.skip("merfish_data/ not present")
    timings, _ = run_baseline(tmp_path, data_dir=data_dir)
    required = {
        "data_loading_s", "preprocessing_s", "python_spectral_init_s",
        "python_sgd_s", "graph_export_s", "total_baseline_s",
    }
    assert required <= set(timings), f"Missing keys: {required - set(timings)}"
    for k in required:
        assert isinstance(timings[k], float) and timings[k] >= 0, \
            f"{k}={timings[k]} is not a non-negative float"


def test_run_compare_returns_tuple(tmp_path):
    """run_compare() must return a 3-tuple when rust_init exists."""
    from generate_merfish_comparisons import run_compare
    rng = np.random.RandomState(42)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    result = run_compare(tmp_path)
    assert result is not None
    assert isinstance(result, tuple) and len(result) == 3
    timings, rss_mb, rust_peak = result
    assert isinstance(timings, dict)
    assert isinstance(rss_mb, float) and rss_mb > 0
    assert isinstance(rust_peak, float) and rust_peak > 0


@pytest.mark.slow
def test_run_compare_timing_keys(tmp_path):
    """run_compare() timing dict must contain all 6 required keys."""
    from generate_merfish_comparisons import run_compare
    rng = np.random.RandomState(42)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    timings, _, _ = run_compare(tmp_path)
    required = {
        "rust_spectral_init_s", "rust_init_sgd_s", "random_sgd_s",
        "metrics_s", "plots_s", "total_compare_s",
    }
    assert required <= set(timings), f"Missing keys: {required - set(timings)}"
    for k in required:
        assert isinstance(timings[k], float) and timings[k] >= 0


@pytest.mark.slow
def test_main_writes_timing_json_after_baseline(tmp_path, monkeypatch):
    """Running main --phase baseline writes merfish_10k_timing.json with baseline keys."""
    from generate_merfish_comparisons import main
    data_dir = Path(__file__).parent / "merfish_data"
    if not data_dir.exists():
        pytest.skip("merfish_data/ not present")
    monkeypatch.setattr("sys.argv", [
        "generate_merfish_comparisons.py",
        "--phase", "baseline",
        "--output-dir", str(tmp_path),
    ])
    main()
    timing_path = tmp_path / "merfish_10k_timing.json"
    assert timing_path.exists()
    data = json.loads(timing_path.read_text())
    for k in ("data_loading_s", "preprocessing_s", "python_spectral_init_s",
               "python_sgd_s", "graph_export_s", "total_baseline_s"):
        assert k in data, f"merfish_10k_timing.json missing '{k}'"


@pytest.mark.slow
def test_main_writes_memory_json_after_compare(tmp_path):
    """Running run_compare() and writing JSON produces all 3 memory keys."""
    from generate_merfish_comparisons import run_compare, _write_memory_json
    rng = np.random.RandomState(42)
    n = 200
    _write_compare_artifacts(tmp_path, rng, n)
    (tmp_path / "merfish_10k_rust_perf.txt").write_text("0.5 102400\n")
    timings, rss_compare_mb, rust_peak = run_compare(tmp_path)
    _write_memory_json(tmp_path, {
        "peak_rss_baseline_mb": 0.0,
        "peak_rss_compare_mb": rss_compare_mb,
        "rust_peak_rss_mb": rust_peak,
    })
    loaded = json.loads((tmp_path / "merfish_10k_memory.json").read_text())
    for k in ("peak_rss_baseline_mb", "peak_rss_compare_mb", "rust_peak_rss_mb"):
        assert k in loaded, f"merfish_10k_memory.json missing '{k}'"
        assert isinstance(loaded[k], float)


def test_run_compare_skips_when_rust_init_missing_still_returns_none(tmp_path):
    """run_compare() still returns None (not a tuple) when rust_init.npy is absent."""
    from generate_merfish_comparisons import run_compare
    result = run_compare(tmp_path)
    assert result is None
