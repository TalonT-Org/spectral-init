#!/usr/bin/env python3
"""Generate synthetic benchmark data for y_heap bottleneck experiment.

Produces gaussian_n{n}_x.npy (shape n×10, float64) and gaussian_n{n}_y.npy
(shape n×2, float64) for n in {1000, 5000, 10000}. Values drawn from uniform[0,1]
using numpy.random.default_rng(seed=42).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def generate_and_verify(out_dir: Path, n: int, rng: np.random.Generator) -> None:
    for tag, shape in [("x", (n, 10)), ("y", (n, 2))]:
        fname = out_dir / f"gaussian_n{n}_{tag}.npy"
        arr = rng.uniform(0.0, 1.0, size=shape)
        np.save(fname, arr)

        # Reload to verify what was written
        loaded = np.load(fname)
        assert loaded.shape == shape, f"{fname}: shape {loaded.shape} != {shape}"
        assert loaded.dtype == np.float64, f"{fname}: dtype {loaded.dtype} != float64"
        finite_ok = np.all(np.isfinite(loaded))
        col_ranges = loaded.max(axis=0) - loaded.min(axis=0)
        range_ok = np.all(col_ranges > 0.01)
        assert finite_ok, f"{fname}: NaN or Inf detected"
        assert range_ok, (
            f"{fname}: column max-min <= 0.01 in at least one column "
            f"(min range: {col_ranges.min():.6f})"
        )

        has_nan_inf = not finite_ok  # always False after assert
        print(
            f"{fname.name}  shape={loaded.shape}  dtype={loaded.dtype}"
            f"  min={loaded.min():.6f}  max={loaded.max():.6f}"
            f"  NaN/Inf={has_nan_inf}"
        )
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data/",
        type=Path,
        help="Output directory for .npy files (default: data/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for n in [1000, 5000, 10000]:
        generate_and_verify(out_dir, n, rng)


if __name__ == "__main__":
    main()
