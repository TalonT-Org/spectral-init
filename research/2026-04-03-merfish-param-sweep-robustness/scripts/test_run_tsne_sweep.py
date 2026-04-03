"""Tests for run_tsne_sweep.py output — fail before script runs, pass after."""
import pandas as pd
import pytest
from pathlib import Path

RESULTS_DIR = Path(__file__).parents[1] / "results"
CSV_PATH = RESULTS_DIR / "results_tsne.csv"
EXPECTED_PERPLEXITIES = [5, 15, 30, 50, 100]
EXPECTED_COLUMNS = {"perplexity", "trustworthiness", "triplet_accuracy",
                    "knn_preservation", "wall_time_s"}


def _df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def test_csv_exists():
    assert CSV_PATH.exists(), f"results_tsne.csv not found at {CSV_PATH}"


def test_csv_shape():
    df = _df()
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    assert set(df.columns) == EXPECTED_COLUMNS, \
        f"Column mismatch: {set(df.columns)} != {EXPECTED_COLUMNS}"


def test_perplexity_values():
    df = _df()
    assert sorted(df["perplexity"].tolist()) == EXPECTED_PERPLEXITIES


def test_metric_ranges():
    df = _df()
    for col in ("trustworthiness", "triplet_accuracy", "knn_preservation"):
        assert df[col].between(0.0, 1.0).all(), \
            f"{col} contains out-of-range values: {df[col].tolist()}"
    assert (df["wall_time_s"] >= 0.0).all()


def test_no_nans():
    df = _df()
    assert not df.isnull().any().any(), \
        f"NaN values found:\n{df[df.isnull().any(axis=1)]}"
