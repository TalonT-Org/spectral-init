"""Generate synthetic Gaussian datasets for tw-perf-rerun-clean experiment.

Usage:
    python scripts/gen_synthetic.py \
        --seed 2026 \
        --output-dir research/2026-04-05-tw-perf-rerun-clean/data/gaussian \
        --sizes 1000 5000 10000 25000 50000 100000 \
        --d 10
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gaussian benchmark data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research/2026-04-05-tw-perf-rerun-clean/data/gaussian"))
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1000, 5000, 10000, 25000, 50000, 100000])
    parser.add_argument("--d", type=int, default=10,
                        help="Input dimensionality for x arrays")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for n in args.sizes:
        x = rng.standard_normal((n, args.d)).astype(np.float64)
        y = rng.standard_normal((n, 2)).astype(np.float64)
        np.save(args.output_dir / f"gaussian_n{n}_x.npy", x)
        np.save(args.output_dir / f"gaussian_n{n}_y.npy", y)
        print(f"  [gaussian] n={n}: x{x.shape} y{y.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
