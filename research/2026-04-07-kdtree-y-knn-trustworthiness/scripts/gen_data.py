#!/usr/bin/env python3
"""Generate synthetic benchmark data for kdtree-y-knn-trustworthiness experiment.

Produces:
  uniform_n{n}_x.npy  (n×10, float64) — uniform [0,1), seed 42
  uniform_n{n}_y.npy  (n×2,  float64) — uniform [0,1), seed 42
  gauss_n{n}_x.npy    (n×10, float64) — uniform [0,1), seed 99
  gauss_n{n}_y.npy    (n×2,  float64) — 8-cluster Gaussian mixture, seed 99

for n ∈ {1000, 5000, 10000, 50000, 100000}.

Verifies each file immediately after writing; aborts on any failure.
Writes data/manifest.json with per-file records.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_VALUES = [1000, 5000, 10000, 50000, 100000]
D_X = 10
D_Y = 2
N_CLUSTERS = 8
SIGMA = 0.3


def _gauss_centers() -> np.ndarray:
    """8 centers on a 2×4 grid within [0,3]²."""
    xs = np.linspace(0, 3, 4)   # [0, 1, 2, 3]
    ys = np.linspace(0, 3, 2)   # [0, 3]
    centers = np.array([(x, y_val) for y_val in ys for x in xs])
    return centers  # shape (8, 2)


def _gauss_mixture_y(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n points from a balanced 8-cluster isotropic Gaussian mixture."""
    centers = _gauss_centers()
    n_clusters = len(centers)
    counts = np.full(n_clusters, n // n_clusters, dtype=int)
    counts[: n % n_clusters] += 1  # distribute remainder to first clusters

    parts = []
    for center, count in zip(centers, counts):
        noise = rng.standard_normal((count, D_Y)) * SIGMA
        parts.append(center + noise)

    y = np.concatenate(parts, axis=0)   # shape (n, 2)
    rng.shuffle(y)                       # mix cluster order in-place
    return y


def _verify(path: Path, expected_shape: tuple) -> None:
    """Reload file and assert shape, dtype, finite values. Aborts on failure."""
    arr = np.load(path)
    if arr.shape != expected_shape:
        sys.exit(f"FAIL {path.name}: shape {arr.shape} != {expected_shape}")
    if arr.dtype != np.float64:
        sys.exit(f"FAIL {path.name}: dtype {arr.dtype} != float64")
    if not np.all(np.isfinite(arr)):
        sys.exit(f"FAIL {path.name}: NaN or Inf detected")


def _save_and_verify(
    out_dir: Path,
    filename: str,
    arr: np.ndarray,
    records: list,
    seed: int,
) -> None:
    path = out_dir / filename
    np.save(path, arr)
    expected_shape = arr.shape
    _verify(path, expected_shape)

    loaded = np.load(path)
    records.append(
        {
            "filename": filename,
            "shape": list(loaded.shape),
            "source_type": "generated",
            "seed": seed,
            "verified": True,
        }
    )
    print(
        f"{filename}  shape={loaded.shape}  dtype={loaded.dtype}"
        f"  min={loaded.min():.6f}  max={loaded.max():.6f}"
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data/",
        type=Path,
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--n-max",
        default=None,
        type=int,
        help="Only generate for n ≤ n_max (dry-run support)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_values = [n for n in N_VALUES if args.n_max is None or n <= args.n_max]
    if not n_values:
        sys.exit(f"--n-max {args.n_max} excludes all n values {N_VALUES}")

    rng_uniform = np.random.default_rng(42)
    rng_gauss = np.random.default_rng(99)
    records: list = []

    for n in n_values:
        # Uniform condition (seed 42)
        _save_and_verify(
            out_dir, f"uniform_n{n}_x.npy",
            rng_uniform.random((n, D_X)), records, seed=42,
        )
        _save_and_verify(
            out_dir, f"uniform_n{n}_y.npy",
            rng_uniform.random((n, D_Y)), records, seed=42,
        )
        # Gaussian condition (seed 99)
        _save_and_verify(
            out_dir, f"gauss_n{n}_x.npy",
            rng_gauss.random((n, D_X)), records, seed=99,
        )
        _save_and_verify(
            out_dir, f"gauss_n{n}_y.npy",
            _gauss_mixture_y(rng_gauss, n), records, seed=99,
        )

    manifest = {"files": records}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest.json written: {len(records)} files, all verified=true")


if __name__ == "__main__":
    main()
