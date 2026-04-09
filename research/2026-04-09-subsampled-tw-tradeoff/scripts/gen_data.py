"""Generate Gaussian d=50 datasets for the subsampled-tw-tradeoff experiment.

Usage (from experiment directory):
    micromamba run -n subsampled-tw-tradeoff python scripts/gen_data.py
    micromamba run -n subsampled-tw-tradeoff python scripts/gen_data.py --sizes 10000 50000
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gaussian benchmark data (d=50)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000])
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "gaussian"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    for n in args.sizes:
        x = rng.randn(n, 50).astype(np.float64)
        y = rng.randn(n, 2).astype(np.float64)
        np.save(output_dir / f"gaussian_n{n}_x.npy", x)
        np.save(output_dir / f"gaussian_n{n}_y.npy", y)
        print(f"  [gaussian] n={n}: x{x.shape} y{y.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
