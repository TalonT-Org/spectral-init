"""Tests for analyze_sweep.py — CV computation, plots, and solver_levels.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analyze_sweep  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: compute_cv
# ---------------------------------------------------------------------------

def test_compute_cv_basic():
    df = pd.DataFrame({
        "param_swept": ["n_neighbors"] * 4,
        "param_value": [5, 10, 15, 20],
        "init_method": ["rust_spectral"] * 4,
        "trustworthiness": [0.90, 0.92, 0.94, 0.95],
    })
    result = analyze_sweep.compute_cv(df, "trustworthiness", "n_neighbors", ["rust_spectral"])
    assert set(result.columns) == {"init_method", "metric", "cv"}
    assert len(result) == 1
    assert result["metric"].iloc[0] == "trustworthiness"
    series = pd.Series([0.90, 0.92, 0.94, 0.95])
    expected_cv = series.std() / series.mean()
    assert abs(result["cv"].iloc[0] - expected_cv) < 1e-10


def test_compute_cv_zero_mean_is_nan():
    df = pd.DataFrame({
        "param_swept": ["n_neighbors"] * 4,
        "param_value": [5, 10, 15, 20],
        "init_method": ["rust_spectral"] * 4,
        "trustworthiness": [0.0, 0.0, 0.0, 0.0],
    })
    result = analyze_sweep.compute_cv(df, "trustworthiness", "n_neighbors", ["rust_spectral"])
    assert pd.isna(result["cv"].iloc[0])


def test_compute_cv_multiple_init_methods():
    df = pd.DataFrame({
        "param_swept": ["n_neighbors"] * 8,
        "param_value": [5, 10, 15, 20] * 2,
        "init_method": ["rust_spectral"] * 4 + ["random"] * 4,
        "trustworthiness": [0.90, 0.92, 0.94, 0.95, 0.80, 0.82, 0.84, 0.86],
    })
    result = analyze_sweep.compute_cv(df, "trustworthiness", "n_neighbors", ["rust_spectral", "random"])
    assert len(result) == 2
    assert set(result["init_method"]) == {"rust_spectral", "random"}


# ---------------------------------------------------------------------------
# Integration tests: main()
# ---------------------------------------------------------------------------

def _make_sweep_csv(tmp_path: Path) -> Path:
    """Build a minimal results_sweep.csv covering all plot branches."""
    rows = []
    # n_neighbors rows for all 4 init_methods
    for pv in [5, 10, 15]:
        for method in ["rust_spectral", "python_spectral", "pca", "random"]:
            sl = 1.0 if method == "rust_spectral" else float("nan")
            proc = 0.05 if method == "rust_spectral" else float("nan")
            rows.append({
                "param_swept": "n_neighbors",
                "param_value": pv,
                "init_method": method,
                "trustworthiness": 0.90 + pv * 0.001,
                "triplet_accuracy": 0.70 + pv * 0.001,
                "knn_preservation": 0.5,
                "sna": 0.002,
                "morans_i_max": 0.06,
                "procrustes_rust_vs_python": proc,
                "procrustes_vs_default": 0.1,
                "solver_level": sl,
                "wall_time_s": 5.0,
            })
    # min_dist rows for all 4 init_methods
    for pv in [0.0, 0.1, 0.5]:
        for method in ["rust_spectral", "python_spectral", "pca", "random"]:
            sl = 1.0 if method == "rust_spectral" else float("nan")
            proc = 0.05 if method == "rust_spectral" else float("nan")
            rows.append({
                "param_swept": "min_dist",
                "param_value": pv,
                "init_method": method,
                "trustworthiness": 0.91 + pv * 0.01,
                "triplet_accuracy": 0.71 + pv * 0.01,
                "knn_preservation": 0.5,
                "sna": 0.002,
                "morans_i_max": 0.06,
                "procrustes_rust_vs_python": proc,
                "procrustes_vs_default": 0.1,
                "solver_level": sl,
                "wall_time_s": 5.0,
            })
    path = tmp_path / "results_sweep.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_tsne_csv(tmp_path: Path) -> Path:
    path = tmp_path / "results_tsne.csv"
    pd.DataFrame({
        "perplexity": [5, 15, 30],
        "trustworthiness": [0.88, 0.89, 0.90],
        "triplet_accuracy": [0.65, 0.66, 0.67],
        "knn_preservation": [0.4, 0.41, 0.42],
        "wall_time_s": [3.0, 3.1, 3.2],
    }).to_csv(path, index=False)
    return path


def test_analyze_produces_six_plots(tmp_path):
    _make_sweep_csv(tmp_path)
    _make_tsne_csv(tmp_path)

    analyze_sweep.main(["--results-dir", str(tmp_path)])

    plots_dir = tmp_path / "plots"
    assert (plots_dir / "trustworthiness_vs_n_neighbors.png").exists()
    assert (plots_dir / "triplet_accuracy_vs_n_neighbors.png").exists()
    assert (plots_dir / "trustworthiness_vs_min_dist.png").exists()
    assert (plots_dir / "cv_comparison_bar.png").exists()
    assert (plots_dir / "procrustes_rust_vs_python_heatmap.png").exists()
    assert (plots_dir / "tsne_reference.png").exists()
    assert (tmp_path / "solver_levels.json").exists()


def test_solver_levels_json_structure(tmp_path):
    rows = []
    for n, sl in [(5, 1), (10, 2)]:
        rows.append({
            "param_swept": "n_neighbors",
            "param_value": n,
            "init_method": "rust_spectral",
            "trustworthiness": 0.90,
            "triplet_accuracy": 0.70,
            "knn_preservation": 0.5,
            "sna": 0.002,
            "morans_i_max": 0.06,
            "procrustes_rust_vs_python": 0.05,
            "procrustes_vs_default": 0.1,
            "solver_level": float(sl),
            "wall_time_s": 5.0,
        })
    # Add min_dist rows so line chart doesn't crash on missing param_swept
    for pv in [0.0, 0.1]:
        rows.append({
            "param_swept": "min_dist",
            "param_value": pv,
            "init_method": "rust_spectral",
            "trustworthiness": 0.91,
            "triplet_accuracy": 0.71,
            "knn_preservation": 0.5,
            "sna": 0.002,
            "morans_i_max": 0.06,
            "procrustes_rust_vs_python": 0.05,
            "procrustes_vs_default": 0.1,
            "solver_level": 1.0,
            "wall_time_s": 5.0,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "results_sweep.csv", index=False)

    analyze_sweep.main(["--results-dir", str(tmp_path)])

    data = json.loads((tmp_path / "solver_levels.json").read_text())
    assert "n_neighbors_5_euclidean" in data and data["n_neighbors_5_euclidean"] == 1
    assert "n_neighbors_10_euclidean" in data and data["n_neighbors_10_euclidean"] == 2


def test_tsne_overlay_skips_silently_if_missing(tmp_path):
    _make_sweep_csv(tmp_path)
    # No results_tsne.csv

    analyze_sweep.main(["--results-dir", str(tmp_path)])

    plots_dir = tmp_path / "plots"
    assert (plots_dir / "trustworthiness_vs_n_neighbors.png").exists()
    assert not (plots_dir / "tsne_reference.png").exists()
