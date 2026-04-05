"""Prepare MERFISH PCA-50 fixtures for trustworthiness scaling experiments.

Run from research/2026-04-04-tw-perf-scaling/:
    python scripts/prepare_merfish.py
"""

import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Three levels up from scripts/ → repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPR_PATH = REPO_ROOT / "temp/merfish_100k/merfish_100k_expression.npz"
SPAT_PATH = REPO_ROOT / "temp/merfish_100k/merfish_100k_spatial.npz"

FIXTURE_X = REPO_ROOT / "tests/fixtures/merfish/merfish_n10k_x.npy"
FIXTURE_Y = REPO_ROOT / "tests/fixtures/merfish/merfish_n10k_y.npy"

DATA_DIR = Path("data/merfish")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if FIXTURE_X.exists() and FIXTURE_Y.exists():
        print("Fixtures already exist — copying to data/merfish/")
        shutil.copy(FIXTURE_X, DATA_DIR / FIXTURE_X.name)
        shutil.copy(FIXTURE_Y, DATA_DIR / FIXTURE_Y.name)
        print("Done.")
        return

    print(f"Loading expression from {EXPR_PATH} ...")
    expr_npz = np.load(EXPR_PATH)
    expression = expr_npz[list(expr_npz.files)[0]]

    print(f"Loading spatial from {SPAT_PATH} ...")
    spat_npz = np.load(SPAT_PATH)
    spatial = spat_npz[list(spat_npz.files)[0]]

    print("Running PCA(50) on first 10 000 cells ...")
    x = PCA(n_components=50, random_state=42).fit_transform(expression[:10000])
    x = x.astype(np.float64)
    y = spatial[:10000].astype(np.float64)

    FIXTURE_X.parent.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_X, x)
    np.save(FIXTURE_Y, y)
    print(f"Saved fixtures: {FIXTURE_X} {FIXTURE_Y}")

    shutil.copy(FIXTURE_X, DATA_DIR / FIXTURE_X.name)
    shutil.copy(FIXTURE_Y, DATA_DIR / FIXTURE_Y.name)
    print("Copied to data/merfish/")
    print("Done.")


if __name__ == "__main__":
    main()
