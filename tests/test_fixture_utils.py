# tests/test_fixture_utils.py
"""Tests for fixture_utils.py — run from repo root with:
   python -m pytest tests/test_fixture_utils.py -v
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist

from fixture_utils import (
    DATASETS,
    make_blobs_dataset,
    make_circles_dataset,
    make_disconnected_dataset,
    make_moons_dataset,
    make_near_dupes_dataset,
    save_dense,
    save_metadata,
    save_sparse,
)


def test_registry_has_9_datasets():
    """DATASETS registry must contain exactly the 9 required entries."""
    assert len(DATASETS) == 9
    names = [name for name, _, _ in DATASETS]
    for expected in [
        "blobs_50", "blobs_500", "moons_200", "blobs_5000",
        "circles_300", "near_dupes_100", "disconnected_200",
        "blobs_connected_200", "blobs_connected_2000",
    ]:
        assert expected in names, f"Missing dataset: {expected}"


def test_blobs_connected_shapes():
    """blobs_connected datasets return correct (n, n_features) shapes."""
    X, params = make_blobs_dataset(n=200, n_features=2, n_centers=3, cluster_std=2.0, seed=42)
    assert X.shape == (200, 2)
    assert params["cluster_std"] == 2.0

    X, params = make_blobs_dataset(n=2000, n_features=10, n_centers=5, cluster_std=3.0, seed=42)
    assert X.shape == (2000, 10)
    assert params["n_features"] == 10


def test_blobs_connected_deterministic():
    """blobs_connected generators are deterministic with same seed."""
    X1, _ = make_blobs_dataset(n=200, n_features=2, n_centers=3, cluster_std=2.0, seed=42)
    X2, _ = make_blobs_dataset(n=200, n_features=2, n_centers=3, cluster_std=2.0, seed=42)
    np.testing.assert_array_equal(X1, X2)


def test_generators_are_deterministic():
    """Same seed must produce identical output; different seeds must differ."""
    for gen, kwargs in [
        (make_blobs_dataset,       {"n": 50,  "n_features": 2, "n_centers": 3, "seed": 42}),
        (make_moons_dataset,       {"n": 200, "noise": 0.1,   "seed": 42}),
        (make_circles_dataset,     {"n": 300, "noise": 0.05,  "factor": 0.5, "seed": 42}),
        (make_near_dupes_dataset,  {"n": 100, "n_dupes": 10,  "jitter": 3e-7, "seed": 42}),
        (make_disconnected_dataset,{"n": 200, "n_groups": 4,  "separation": 100.0, "seed": 42}),
    ]:
        X1, _ = gen(**kwargs)
        X2, _ = gen(**kwargs)
        np.testing.assert_array_equal(X1, X2)
        # Different seed gives different result
        alt_kwargs = {**kwargs, "seed": 99}
        X3, _ = gen(**alt_kwargs)
        assert not np.array_equal(X1, X3)


def test_generator_shapes():
    """Each generator returns correct (n, n_features) shape."""
    X, _ = make_blobs_dataset(n=50, n_features=2, n_centers=3, seed=42)
    assert X.shape == (50, 2)

    X, _ = make_moons_dataset(n=200, noise=0.1, seed=42)
    assert X.shape == (200, 2)

    X, _ = make_circles_dataset(n=300, noise=0.05, factor=0.5, seed=42)
    assert X.shape == (300, 2)

    X, _ = make_near_dupes_dataset(n=100, n_dupes=10, jitter=3e-7, seed=42)
    assert X.shape == (100, 2)

    X, _ = make_disconnected_dataset(n=200, n_groups=4, separation=100.0, seed=42)
    assert X.shape[1] == 2 and X.shape[0] >= 196  # may vary by remainder


def test_near_dupes_max_pairwise_distance():
    """near_dupes_100: max pairwise distance among duplicate points < 1e-6."""
    n_dupes = 10
    X, params = make_near_dupes_dataset(n=100, n_dupes=n_dupes, jitter=3e-7, seed=42)
    dupes = X[-n_dupes:]
    dists = pdist(dupes)
    assert dists.max() < 1e-6, (
        f"Max pairwise dupe distance {dists.max():.3e} >= 1e-6"
    )


def test_disconnected_cluster_separation():
    """disconnected_200: inter-cluster min distance > 50× cluster radius."""
    n_groups = 4
    X, params = make_disconnected_dataset(n=200, n_groups=n_groups, separation=100.0, seed=42)
    n = len(X)
    pts_per = n // n_groups
    cluster_radius = 1.0  # unit-std Gaussian
    min_sep = float("inf")
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            gi = X[i * pts_per : (i + 1) * pts_per]
            gj = X[j * pts_per : (j + 1) * pts_per]
            d = cdist(gi, gj).min()
            min_sep = min(min_sep, d)
    assert min_sep > 50 * cluster_radius, (
        f"Min inter-cluster separation {min_sep:.2f} <= 50×radius={50*cluster_radius}"
    )


def test_save_dense_roundtrip():
    """save_dense() writes a valid .npz that loads back correctly."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.npz"
        save_dense(path, X=X)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded["X"], X.astype(np.float64))


def test_save_metadata_roundtrip():
    """save_metadata() writes valid JSON that loads back identical."""
    meta = {"dataset": "test", "n": 50, "seed": 42, "env": {"python": "3.11.0"}}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "meta.json"
        save_metadata(path, meta)
        with open(path) as f:
            loaded = json.load(f)
    assert loaded == meta


def test_generate_fixtures_cli_creates_outputs():
    """generate_fixtures.py end-to-end: creates subdirs, meta.json, manifest.json."""
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "generate_fixtures.py"),
                "--output-dir", td,
                "--datasets", "blobs_50",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        outdir = Path(td)
        assert (outdir / "blobs_50" / "meta.json").exists()
        assert (outdir / "manifest.json").exists()
        manifest = json.loads((outdir / "manifest.json").read_text())
        assert any(d["name"] == "blobs_50" for d in manifest["datasets"])
