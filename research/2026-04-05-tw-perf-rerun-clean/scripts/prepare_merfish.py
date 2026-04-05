"""Prepare MERFISH PCA-50 data for tw-perf-rerun-clean experiment.

Usage:
    python scripts/prepare_merfish.py \
        --npz-dir /home/talon/projects/spectral-init/temp/merfish_100k \
        --output-dir research/2026-04-05-tw-perf-rerun-clean/data/merfish \
        --n 10000
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MERFISH PCA-50 data")
    parser.add_argument("--npz-dir", type=Path, required=True,
                        help="Directory containing merfish_100k_*.npz files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research/2026-04-05-tw-perf-rerun-clean/data/merfish"))
    parser.add_argument("--n", type=int, default=10000,
                        help="Number of cells to use (sliced from first N rows)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading expression from {args.npz_dir}/merfish_100k_expression.npz ...")
    expr_npz = np.load(args.npz_dir / "merfish_100k_expression.npz")
    expression = expr_npz[list(expr_npz.files)[0]]

    print(f"Loading spatial from {args.npz_dir}/merfish_100k_spatial.npz ...")
    spat_npz = np.load(args.npz_dir / "merfish_100k_spatial.npz")
    spatial = spat_npz[list(spat_npz.files)[0]]

    n = args.n
    tag = f"n{n // 1000}k" if n % 1000 == 0 else str(n)

    print(f"Running PCA(50) on first {n} cells ...")
    x = PCA(n_components=50, random_state=42).fit_transform(expression[:n])
    x = x.astype(np.float64)
    y = spatial[:n].astype(np.float64)

    x_path = args.output_dir / f"merfish_{tag}_x.npy"
    y_path = args.output_dir / f"merfish_{tag}_y.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    print(f"  x{x.shape} → {x_path}")
    print(f"  y{y.shape} → {y_path}")
    print("Done.")


if __name__ == "__main__":
    main()
