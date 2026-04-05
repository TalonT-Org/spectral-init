"""Compute sklearn trustworthiness reference scores for parity validation.

Default (no flags): iterate all data/*/ subdirectories, compute
sklearn.manifold.trustworthiness for each X/Y pair, write JSON to
results/parity/sklearn_{dataset_name}.json.

With --n and --output: run on a specific dataset size and write to the
given output path (used for dry-run invocations).

Run from research/2026-04-04-tw-perf-scaling/:
    python scripts/sklearn_reference.py
    python scripts/sklearn_reference.py --n 1000 --output results/parity/dry_run.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.manifold import trustworthiness

DATA_DIR = Path("data")
PARITY_DIR = Path("results/parity")
K_NEIGHBORS = 15


def score_pair(x_path: Path, y_path: Path) -> float:
    x = np.load(x_path)
    y = np.load(y_path)
    return float(trustworthiness(x, y, n_neighbors=K_NEIGHBORS))


def run_all() -> None:
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_dir in sorted(DATA_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        x_files = sorted(dataset_dir.glob("*_x.npy"))
        for x_path in x_files:
            stem = x_path.stem  # e.g. "gaussian_n1000_x"
            y_path = x_path.with_name(stem[:-2] + "_y.npy")
            if not y_path.exists():
                print(f"  [skip] no matching y file for {x_path.name}")
                continue
            # Extract n from filename (e.g. "gaussian_n1000_x" → 1000)
            m = re.search(r"_n(\d+)_", stem)
            n = int(m.group(1)) if m else -1
            print(f"  [{dataset_dir.name}] n={n} ...", end=" ", flush=True)
            score = score_pair(x_path, y_path)
            dataset_name = stem[:-2]  # strip trailing "_x"
            out_path = PARITY_DIR / f"sklearn_{dataset_name}.json"
            out_path.write_text(json.dumps({"n": n, "score": score}, indent=2))
            print(f"score={score:.6f} → {out_path}")


def run_single(n: int, output: Path) -> None:
    # Find any dataset of the requested size
    for dataset_dir in sorted(DATA_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        x_files = list(dataset_dir.glob(f"*_n{n}_x.npy"))
        if x_files:
            x_path = x_files[0]
            stem = x_path.stem
            y_path = x_path.with_name(stem[:-2] + "_y.npy")
            if y_path.exists():
                print(f"Using {x_path.name} and {y_path.name}")
                score = score_pair(x_path, y_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"n": n, "score": score}, indent=2))
                print(f"score={score:.6f} → {output}")
                return
    raise FileNotFoundError(f"No dataset of size n={n} found under {DATA_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sklearn trustworthiness reference scores")
    parser.add_argument("--n", type=int, default=None, help="Dataset size (for single-run mode)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (for single-run mode)")
    args = parser.parse_args()

    if args.n is not None and args.output is not None:
        run_single(args.n, args.output)
    else:
        run_all()


if __name__ == "__main__":
    main()
