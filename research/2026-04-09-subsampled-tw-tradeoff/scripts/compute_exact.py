"""Compute exact trustworthiness baselines for all (dataset, n) combinations.

Saves results/raw/exact_{dataset}_{n}.json with fields:
  dataset, n, k, T_exact, wall_median_s, wall_runs

Run from experiment root:
    micromamba run -n subsampled-tw-tradeoff python scripts/compute_exact.py
    micromamba run -n subsampled-tw-tradeoff python scripts/compute_exact.py --dry-run
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_tw

sys.path.insert(0, str(Path(__file__).parent))
from utils import K, load_npy_pair, save_result_json

EXPROOT = Path(__file__).parent.parent

DATASETS = [
    ("merfish",  10_000, EXPROOT / "data" / "merfish"),
    ("merfish",  50_000, EXPROOT / "data" / "merfish"),
    ("gaussian", 10_000, EXPROOT / "data" / "gaussian"),
    ("gaussian", 50_000, EXPROOT / "data" / "gaussian"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Only process MERFISH n=10K (skip all other datasets).")
    args = parser.parse_args()

    datasets = (
        [("merfish", 10_000, EXPROOT / "data" / "merfish")]
        if args.dry_run
        else DATASETS
    )

    for dataset, n, data_dir in datasets:
        try:
            X, Y = load_npy_pair(data_dir, dataset, n)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping {dataset} n={n}", file=sys.stderr)
            continue

        # 1 warmup run
        sklearn_tw(X, Y, n_neighbors=K)

        # 3 timed runs; record median
        wall_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            T_exact = sklearn_tw(X, Y, n_neighbors=K)
            wall_runs.append(time.perf_counter() - t0)

        result = {
            "dataset": dataset,
            "n": n,
            "k": K,
            "T_exact": float(T_exact),
            "wall_median_s": float(np.median(wall_runs)),
            "wall_runs": [float(w) for w in wall_runs],
        }
        out_path = EXPROOT / "results" / "raw" / f"exact_{dataset}_{n}.json"
        save_result_json(out_path, result)
        print(
            f"[exact] {dataset} n={n}: T_exact={T_exact:.6f} "
            f"wall_median={np.median(wall_runs):.3f}s"
        )


if __name__ == "__main__":
    main()
