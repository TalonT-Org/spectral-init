"""
fixture_utils.py — Dataset generators and save/load helpers for spectral-init fixtures.
"""
from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
from sklearn.datasets import make_blobs, make_circles, make_moons


# ---------------------------------------------------------------------------
# Dataset generators — each returns (X: ndarray[n, d], params: dict)
# ---------------------------------------------------------------------------

def make_blobs_dataset(
    n: int, n_features: int = 2, n_centers: int = 5,
    cluster_std: float = 1.0, seed: int = 42
) -> tuple[np.ndarray, dict]:
    X, _ = make_blobs(
        n_samples=n, n_features=n_features, centers=n_centers,
        cluster_std=cluster_std, random_state=seed
    )
    return X.astype(np.float64), {
        "type": "blobs", "n": n, "n_features": n_features,
        "n_centers": n_centers, "cluster_std": cluster_std, "seed": seed,
    }


def make_moons_dataset(
    n: int, noise: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, dict]:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float64), {
        "type": "moons", "n": n, "noise": noise, "seed": seed,
    }


def make_circles_dataset(
    n: int, noise: float = 0.05, factor: float = 0.5, seed: int = 42
) -> tuple[np.ndarray, dict]:
    X, _ = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
    return X.astype(np.float64), {
        "type": "circles", "n": n, "noise": noise, "factor": factor, "seed": seed,
    }


def make_near_dupes_dataset(
    n: int, n_dupes: int = 10, jitter: float = 3e-7, seed: int = 42
) -> tuple[np.ndarray, dict]:
    """
    n - n_dupes normal random points + n_dupes near-duplicate points.

    Dupes are all within jitter of a shared center. Max pairwise Euclidean
    distance among dupes = 2 * jitter * sqrt(2) ≈ 8.49e-7 < 1e-6 for jitter=3e-7.
    """
    if n_dupes >= n:
        raise ValueError(f"n_dupes ({n_dupes}) must be less than n ({n})")
    rng = np.random.default_rng(seed)
    n_normal = n - n_dupes
    X_normal = rng.standard_normal((n_normal, 2))
    # Dupe cluster well away from normal points
    dupe_center = np.array([50.0, 50.0])
    X_dupes = dupe_center + rng.uniform(-jitter, jitter, (n_dupes, 2))
    X = np.vstack([X_normal, X_dupes])
    return X.astype(np.float64), {
        "type": "near_dupes", "n": n, "n_dupes": n_dupes,
        "jitter": float(jitter), "dupe_center": list(dupe_center), "seed": seed,
    }


def make_disconnected_dataset(
    n: int, n_groups: int = 4, separation: float = 100.0, seed: int = 42
) -> tuple[np.ndarray, dict]:
    """
    n_groups clusters of unit-std Gaussian points separated by `separation`.

    Cluster radius ≈ 1.0 (std of unit normal). Min inter-cluster center
    distance = separation. With separation=100: 100 >> 50 * 1.0.
    """
    rng = np.random.default_rng(seed)
    pts_per = n // n_groups
    groups = []
    for i in range(n_groups):
        pts = rng.standard_normal((pts_per, 2))
        pts[:, 0] += i * separation
        groups.append(pts)
    X = np.vstack(groups)
    return X.astype(np.float64), {
        "type": "disconnected", "n": len(X), "n_groups": n_groups,
        "separation": float(separation), "seed": seed,
    }


# ---------------------------------------------------------------------------
# DATASETS registry: (name, generator_fn, kwargs)
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, Any, dict]] = [
    ("blobs_50",         make_blobs_dataset,        {"n": 50,   "n_features": 2, "n_centers": 3,  "seed": 42}),
    ("blobs_500",        make_blobs_dataset,        {"n": 500,  "n_features": 2, "n_centers": 5,  "seed": 42}),
    ("moons_200",        make_moons_dataset,        {"n": 200,  "noise": 0.1,                     "seed": 42}),
    ("blobs_5000",       make_blobs_dataset,        {"n": 5000, "n_features": 2, "n_centers": 10, "seed": 42}),
    ("circles_300",      make_circles_dataset,      {"n": 300,  "noise": 0.05,   "factor": 0.5,   "seed": 42}),
    ("near_dupes_100",   make_near_dupes_dataset,   {"n": 100,  "n_dupes": 10,   "jitter": 3e-7,  "seed": 42}),
    ("disconnected_200",      make_disconnected_dataset, {"n": 200,  "n_groups": 4,   "separation": 100.0, "seed": 42}),
    ("blobs_connected_200",   make_blobs_dataset,        {"n": 200,  "n_features": 2,  "n_centers": 3, "cluster_std": 3.0, "seed": 42}),
    ("blobs_connected_2000",  make_blobs_dataset,        {"n": 2000, "n_features": 10, "n_centers": 5, "cluster_std": 5.0, "seed": 42}),
]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_dense(path: Path | str, dtype: np.dtype = np.float64, **arrays: np.ndarray) -> None:
    """Save one or more dense arrays to a .npz file, casting to `dtype`."""
    cast = {k: np.asarray(v, dtype=dtype) for k, v in arrays.items()}
    np.savez(path, **cast)


def save_sparse(path: Path | str, matrix: scipy.sparse.sparray | scipy.sparse.spmatrix) -> None:
    """Save a sparse matrix to a .npz file via scipy.sparse.save_npz."""
    scipy.sparse.save_npz(str(path), matrix)


def save_metadata(path: Path | str, metadata: dict) -> None:
    """Write a metadata dictionary as indented JSON."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def get_env_metadata() -> dict:
    """Collect Python and key package versions for reproducibility records."""
    env: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg, attr in [
        ("numpy",   "__version__"),
        ("scipy",   "__version__"),
        ("sklearn", "__version__"),
        ("umap",    "__version__"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            env[pkg] = getattr(mod, attr, "unknown")
        except ImportError as e:
            env[pkg] = f"import-error: {e}"
    return env
