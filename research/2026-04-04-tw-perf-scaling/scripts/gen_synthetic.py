"""Generate synthetic Gaussian and blobs datasets for trustworthiness scaling experiments.

Run from research/2026-04-04-tw-perf-scaling/:
    python scripts/gen_synthetic.py
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

NS = [1000, 5000, 10000, 25000, 50000, 100000]

DATA_DIR = Path("data")


def generate_gaussian(n: int) -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(n, 10)
    y = np.random.RandomState(0).randn(n, 2)
    out_dir = DATA_DIR / "gaussian"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"gaussian_n{n}_x.npy", x.astype(np.float64))
    np.save(out_dir / f"gaussian_n{n}_y.npy", y.astype(np.float64))
    print(f"  [gaussian] n={n}: x{x.shape} y{y.shape}")


def generate_blobs(n: int) -> None:
    x, _ = make_blobs(n_samples=n, centers=8, cluster_std=2.0, random_state=1)
    y = np.random.RandomState(1).randn(n, 2)
    out_dir = DATA_DIR / "blobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"blobs_n{n}_x.npy", x.astype(np.float64))
    np.save(out_dir / f"blobs_n{n}_y.npy", y.astype(np.float64))
    print(f"  [blobs]    n={n}: x{x.shape} y{y.shape}")


def main() -> None:
    print("Generating synthetic datasets...")
    for n in NS:
        generate_gaussian(n)
        generate_blobs(n)
    print("Done.")


if __name__ == "__main__":
    main()
