"""Prepare MERFISH PCA-50 fixtures for trustworthiness scaling experiments.

Run from research/2026-04-04-tw-perf-scaling/:
    python scripts/prepare_merfish.py
    python scripts/prepare_merfish.py --expr-path /path/to/merfish_expression.npz \
        --spat-path /path/to/merfish_spatial.npz

Data acquisition:
    The MERFISH dataset used here is Allen Brain Cell Atlas MERFISH whole-brain
    mouse data (Yao et al. 2023, https://doi.org/10.1038/s41586-023-06812-z).
    Download via the Allen Brain Cell Atlas data portal:
        https://alleninstitute.org/division/brain-science/allen-brain-cell-atlas/
    Extract expression (.npz, cells × genes float32) and spatial (.npz, cells × 2 float64)
    arrays for ~100K cells and place them at:
        temp/merfish_100k/merfish_100k_expression.npz
        temp/merfish_100k/merfish_100k_spatial.npz
    or pass --expr-path and --spat-path to this script.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Three levels up from scripts/ → repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

FIXTURE_X = REPO_ROOT / "tests/fixtures/merfish/merfish_n10k_x.npy"
FIXTURE_Y = REPO_ROOT / "tests/fixtures/merfish/merfish_n10k_y.npy"

DATA_DIR = Path("data/merfish")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MERFISH PCA-50 fixtures")
    parser.add_argument(
        "--expr-path",
        type=Path,
        default=REPO_ROOT / "temp/merfish_100k/merfish_100k_expression.npz",
        help="Path to MERFISH expression .npz (cells × genes float32)",
    )
    parser.add_argument(
        "--spat-path",
        type=Path,
        default=REPO_ROOT / "temp/merfish_100k/merfish_100k_spatial.npz",
        help="Path to MERFISH spatial .npz (cells × 2 float64)",
    )
    args = parser.parse_args()
    expr_path = args.expr_path
    spat_path = args.spat_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if FIXTURE_X.exists() and FIXTURE_Y.exists():
        print("Fixtures already exist — copying to data/merfish/")
        shutil.copy(FIXTURE_X, DATA_DIR / FIXTURE_X.name)
        shutil.copy(FIXTURE_Y, DATA_DIR / FIXTURE_Y.name)
        print("Done.")
        return

    print(f"Loading expression from {expr_path} ...")
    expr_npz = np.load(expr_path)
    expression = expr_npz[list(expr_npz.files)[0]]

    print(f"Loading spatial from {spat_path} ...")
    spat_npz = np.load(spat_path)
    spatial = spat_npz[list(spat_npz.files)[0]]

    # Known limitation: row ordering of the source .npz file is not validated.
    # If re-downloaded with different cell ordering, PCA output will differ
    # even with random_state=42, producing a different fixture. Users
    # regenerating from source should verify row count matches the original
    # (expression.shape[0] should equal the expected cell count for the
    # downloaded dataset version) and compare fixture checksums.
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
