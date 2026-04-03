"""Tests for write_sweep_report.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import analyze_sweep  # noqa: E402
from analyze_sweep import compute_cv  # noqa: E402
from write_sweep_report import (  # noqa: E402
    _procrustes_status,
    compute_h1_verdict,
    compute_h5_verdict,
    main as report_main,
)

import analyze_sweep as _analyze_sweep_mod  # noqa: E402


def _make_n_neighbors_df() -> pd.DataFrame:
    """Minimal n_neighbors sweep for integration tests."""
    rows = []
    for n in [5, 10, 15]:
        for method in ["rust_spectral", "python_spectral", "pca", "random"]:
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": 0.99 + n * 0.0001,
                    "triplet_accuracy": 0.70,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": 0.04
                    if method == "rust_spectral"
                    else float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": 1.0 if method == "rust_spectral" else float("nan"),
                    "wall_time_s": 7.0,
                }
            )
    return pd.DataFrame(rows)


def _make_min_dist_df() -> pd.DataFrame:
    """Minimal min_dist sweep for integration tests."""
    rows = []
    for d in [0.0, 0.1, 0.5]:
        for method in ["rust_spectral", "python_spectral", "pca", "random"]:
            rows.append(
                {
                    "param_swept": "min_dist",
                    "param_value": d,
                    "init_method": method,
                    "trustworthiness": 0.98,
                    "triplet_accuracy": 0.69,
                    "knn_preservation": 0.34,
                    "sna": 0.002,
                    "morans_i_max": 0.06,
                    "procrustes_rust_vs_python": 0.04
                    if method == "rust_spectral"
                    else float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": 1.0 if method == "rust_spectral" else float("nan"),
                    "wall_time_s": 7.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_h1_verdict_supported
# ---------------------------------------------------------------------------

def test_h1_verdict_supported():
    rows = []
    for n, trust_rust, trust_random in zip(
        [5, 10, 15, 30],
        [0.990, 0.991, 0.992, 0.993],
        [0.90, 0.94, 0.97, 0.99],
    ):
        for method, trust in [("rust_spectral", trust_rust), ("random", trust_random)]:
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": trust,
                    "triplet_accuracy": trust,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": float("nan"),
                    "wall_time_s": 7.0,
                }
            )
        # pad out python_spectral and pca so compute_cv doesn't crash
        for method in ("python_spectral", "pca"):
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": 0.95,
                    "triplet_accuracy": 0.70,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": float("nan"),
                    "wall_time_s": 7.0,
                }
            )
    df = pd.DataFrame(rows)
    result = compute_h1_verdict(df, compute_cv)
    assert result["verdict"] == "SUPPORTED"
    assert result["cv_ratio_trustworthiness"] < 0.8


# ---------------------------------------------------------------------------
# test_h1_verdict_refuted
# ---------------------------------------------------------------------------

def test_h1_verdict_refuted():
    # rust_spectral has higher variance than random → cv_ratio > 1.2
    rows = []
    for n, trust_rust, trust_random in zip(
        [5, 10, 15, 30],
        [0.80, 0.90, 0.95, 0.99],   # wide spread → high CV
        [0.98, 0.981, 0.982, 0.983],  # tight → low CV
    ):
        for method, trust in [("rust_spectral", trust_rust), ("random", trust_random)]:
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": trust,
                    "triplet_accuracy": trust,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": float("nan"),
                    "wall_time_s": 7.0,
                }
            )
        for method in ("python_spectral", "pca"):
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": 0.95,
                    "triplet_accuracy": 0.70,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": float("nan"),
                    "wall_time_s": 7.0,
                }
            )
    df = pd.DataFrame(rows)
    result = compute_h1_verdict(df, compute_cv)
    assert result["verdict"] == "REFUTED"
    assert result["cv_ratio_trustworthiness"] > 1.2


# ---------------------------------------------------------------------------
# test_h1_verdict_inconclusive
# ---------------------------------------------------------------------------

def test_h1_verdict_inconclusive():
    # cv_ratio between 0.8 and 1.2 → INCONCLUSIVE
    rows = []
    # Both methods have similar CVs
    for n, trust in zip([5, 10, 15, 30], [0.97, 0.98, 0.99, 0.995]):
        for method in ("rust_spectral", "random", "python_spectral", "pca"):
            rows.append(
                {
                    "param_swept": "n_neighbors",
                    "param_value": n,
                    "init_method": method,
                    "trustworthiness": trust,
                    "triplet_accuracy": trust,
                    "knn_preservation": 0.35,
                    "sna": 0.002,
                    "morans_i_max": 0.07,
                    "procrustes_rust_vs_python": float("nan"),
                    "procrustes_vs_default": 0.1,
                    "solver_level": float("nan"),
                    "wall_time_s": 7.0,
                }
            )
    df = pd.DataFrame(rows)
    result = compute_h1_verdict(df, compute_cv)
    assert result["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# test_h2_pass_warning_fail_thresholds
# ---------------------------------------------------------------------------

def test_h2_pass_warning_fail_thresholds():
    assert _procrustes_status(0.03) == "PASS"
    assert _procrustes_status(0.07) == "WARNING"
    assert _procrustes_status(0.15) == "FAIL"
    assert _procrustes_status(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# test_h5_verdict_std
# ---------------------------------------------------------------------------

def test_h5_verdict_std():
    # std ≈ 0.005 → SUPPORTED
    rows_low = []
    for d, proc in zip([0.0, 0.1, 0.5], [0.040, 0.044, 0.046]):
        rows_low.append(
            {
                "param_swept": "min_dist",
                "param_value": d,
                "init_method": "rust_spectral",
                "trustworthiness": 0.98,
                "triplet_accuracy": 0.70,
                "knn_preservation": 0.35,
                "sna": 0.002,
                "morans_i_max": 0.07,
                "procrustes_rust_vs_python": proc,
                "procrustes_vs_default": 0.1,
                "solver_level": 1.0,
                "wall_time_s": 7.0,
            }
        )
    df_low = pd.DataFrame(rows_low)
    r_low = compute_h5_verdict(df_low)
    assert r_low["verdict"] == "SUPPORTED"
    assert r_low["procrustes_std"] < 0.01

    # std ≈ 0.02 → REFUTED
    rows_high = []
    for d, proc in zip([0.0, 0.1, 0.5], [0.03, 0.05, 0.07]):
        rows_high.append(
            {
                "param_swept": "min_dist",
                "param_value": d,
                "init_method": "rust_spectral",
                "trustworthiness": 0.98,
                "triplet_accuracy": 0.70,
                "knn_preservation": 0.35,
                "sna": 0.002,
                "morans_i_max": 0.07,
                "procrustes_rust_vs_python": proc,
                "procrustes_vs_default": 0.1,
                "solver_level": 1.0,
                "wall_time_s": 7.0,
            }
        )
    df_high = pd.DataFrame(rows_high)
    r_high = compute_h5_verdict(df_high)
    assert r_high["verdict"] == "REFUTED"
    assert r_high["procrustes_std"] >= 0.01


# ---------------------------------------------------------------------------
# test_report_contains_required_sections  (integration)
# ---------------------------------------------------------------------------

def test_report_contains_required_sections(tmp_path):
    # Build minimal results_sweep.csv
    nn_df = _make_n_neighbors_df()
    md_df = _make_min_dist_df()
    sweep_df = pd.concat([nn_df, md_df], ignore_index=True)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sweep_df.to_csv(results_dir / "results_sweep.csv", index=False)

    # Build minimal results_tsne.csv (include all columns plot_tsne_reference needs)
    tsne_rows = [
        {"perplexity": 10, "trustworthiness": 0.96, "triplet_accuracy": 0.68, "run": 1},
        {"perplexity": 30, "trustworthiness": 0.97, "triplet_accuracy": 0.70, "run": 1},
        {"perplexity": 50, "trustworthiness": 0.975, "triplet_accuracy": 0.71, "run": 1},
    ]
    pd.DataFrame(tsne_rows).to_csv(results_dir / "results_tsne.csv", index=False)

    # Run analyze_sweep.main to create solver_levels.json (plots go to tmp_path)
    _analyze_sweep_mod.main(["--results-dir", str(results_dir)])

    # Run write_sweep_report.main
    output = tmp_path / "report.md"
    report_main(
        [
            "--results-dir",
            str(results_dir),
            "--output",
            str(output),
        ]
    )

    text = output.read_text()
    required_sections = [
        "## Hypothesis Verdicts",
        "## Quantitative CV Table",
        "## Procrustes Alignment (H2)",
        "## t-SNE Reference Comparison",
        "## Solver Level Diagnostics",
        "## Threats to Validity",
        "## Success Criteria Checklist",
    ]
    for section in required_sections:
        assert section in text, f"Missing section: {section!r}"
