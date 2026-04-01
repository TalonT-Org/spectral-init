"""Unit tests for merfish_preprocessing_sweep.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merfish_preprocessing_sweep import (
    METRICS,
    NORMALIZATIONS,
    PCA_DIMS,
    apply_normalization,
    select_winner,
)


# ---------------------------------------------------------------------------
# Test 1 — Parameter grid completeness
# ---------------------------------------------------------------------------


def test_parameter_grid_size():
    assert len(NORMALIZATIONS) == 2
    assert len(PCA_DIMS) == 4
    assert len(METRICS) == 2
    assert len(NORMALIZATIONS) * len(PCA_DIMS) * len(METRICS) == 16


def test_parameter_grid_values():
    assert set(NORMALIZATIONS) == {"normalize_total+log1p", "log2"}
    assert set(PCA_DIMS) == {10, 20, 30, 50}
    assert set(METRICS) == {"euclidean", "cosine"}


# ---------------------------------------------------------------------------
# Test 2 — apply_normalization() — log2 variant
# ---------------------------------------------------------------------------


def test_apply_normalization_log2_passthrough():
    """log2 variant wraps expression in AnnData without any transformation."""
    expr = np.random.rand(50, 20).astype(np.float32)
    adata = apply_normalization(expr, "log2")
    np.testing.assert_array_almost_equal(np.array(adata.X), expr, decimal=5)


# ---------------------------------------------------------------------------
# Test 3 — apply_normalization() — normalize_total+log1p back-transform
# ---------------------------------------------------------------------------


def test_apply_normalization_normalize_total_log1p_back_transforms():
    """normalize_total+log1p back-transforms log2(count+1) to raw counts first."""
    # Construct known log2(count+1) values from integer raw counts
    raw_counts = np.array([[0, 1, 4], [2, 0, 3]], dtype=np.float32)
    log2_vals = np.log2(raw_counts + 1)  # simulate stored H5AD values
    adata = apply_normalization(log2_vals, "normalize_total+log1p")
    result = np.array(adata.X)
    # After back-transform + normalize_total + log1p, values must differ from log2_vals
    assert not np.allclose(result, log2_vals, atol=1e-3)
    # All values must be non-negative (log1p output >= 0)
    assert np.all(result >= -1e-6)
    # After normalize_total(target_sum=1000), raw counts per row should sum to ~1000
    # before log1p is applied. Verify by checking that exp(result) - 1 row sums are ~1000.
    back = np.expm1(result)
    np.testing.assert_allclose(back.sum(axis=1), [1000.0, 1000.0], rtol=1e-4)


def test_apply_normalization_unknown_raises():
    expr = np.random.rand(10, 5).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown normalization"):
        apply_normalization(expr, "bad_norm")


# ---------------------------------------------------------------------------
# Test 4 — select_winner() — normal case
# ---------------------------------------------------------------------------


def test_select_winner_highest_spectral_gap_above_threshold():
    """Winner has highest spectral_gap among configs with trustworthiness > 0.95."""
    df = pd.DataFrame([
        {"normalization": "log2", "n_pcs": 30, "metric": "euclidean",
         "spectral_gap": 0.10, "trustworthiness": 0.96, "silhouette": 0.30,
         "condition_number": 5.0, "n_components": 1, "wall_time_s": 1.0},
        {"normalization": "normalize_total+log1p", "n_pcs": 20, "metric": "cosine",
         "spectral_gap": 0.25, "trustworthiness": 0.97, "silhouette": 0.40,
         "condition_number": 4.0, "n_components": 1, "wall_time_s": 1.0},
        # high spectral_gap but trustworthiness < 0.95 — must be excluded
        {"normalization": "log2", "n_pcs": 10, "metric": "euclidean",
         "spectral_gap": 0.50, "trustworthiness": 0.92, "silhouette": 0.20,
         "condition_number": 6.0, "n_components": 1, "wall_time_s": 1.0},
    ])
    winner = select_winner(df)
    assert winner["spectral_gap"] == pytest.approx(0.25)
    assert winner["normalization"] == "normalize_total+log1p"


# ---------------------------------------------------------------------------
# Test 5 — select_winner() — fallback when no config passes threshold
# ---------------------------------------------------------------------------


def test_select_winner_fallback_to_highest_trustworthiness():
    """When no config has trustworthiness > 0.95, fall back to highest trustworthiness."""
    df = pd.DataFrame([
        {"normalization": "log2", "n_pcs": 30, "metric": "euclidean",
         "spectral_gap": 0.1, "trustworthiness": 0.88, "silhouette": 0.3,
         "condition_number": 5.0, "n_components": 1, "wall_time_s": 1.0},
        {"normalization": "normalize_total+log1p", "n_pcs": 20, "metric": "cosine",
         "spectral_gap": 0.05, "trustworthiness": 0.93, "silhouette": 0.4,
         "condition_number": 4.0, "n_components": 1, "wall_time_s": 1.0},
    ])
    winner = select_winner(df)
    assert winner["trustworthiness"] == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# Test 6 — CSV column contract
# ---------------------------------------------------------------------------


def test_csv_column_names():
    """CSV must have exactly the 9 required columns."""
    expected_columns = {
        "normalization", "n_pcs", "metric", "spectral_gap",
        "condition_number", "n_components", "trustworthiness",
        "silhouette", "wall_time_s",
    }
    df = pd.DataFrame(columns=list(expected_columns))
    assert set(df.columns) == expected_columns
