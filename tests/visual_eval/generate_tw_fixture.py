#!/usr/bin/env python3
"""Generate trustworthiness parity fixture for Rust vs sklearn validation.

Output: tests/fixtures/tw_parity/tw_parity.npz
  - X: (200, 10) float64 — synthetic high-dimensional data
  - Y: (200, 2) float64 — synthetic 2D embedding
  - k: int64 — number of neighbors (15)
  - sklearn_score: float64 — trustworthiness computed by sklearn

Run from the repository root:
  python tests/visual_eval/generate_tw_fixture.py
"""

import pathlib
import sys
import numpy as np
from sklearn.manifold import trustworthiness

FIXTURE_DIR = pathlib.Path(__file__).parents[2] / "tests" / "fixtures" / "tw_parity"

def main() -> None:
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    Y = rng.randn(200, 2)
    k = 15

    sklearn_score = trustworthiness(X, Y, n_neighbors=k)
    print(f"sklearn trustworthiness(n=200, d=10, k={k}) = {sklearn_score:.15f}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURE_DIR / "tw_parity.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float64),
        Y=Y.astype(np.float64),
        k=np.int64(k),
        sklearn_score=np.float64(sklearn_score),
    )
    print(f"Wrote fixture: {out_path}")

if __name__ == "__main__":
    main()
