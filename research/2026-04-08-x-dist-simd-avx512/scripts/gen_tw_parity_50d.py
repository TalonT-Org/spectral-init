#!/usr/bin/env python3
"""Generate trustworthiness parity fixture at d_x=50 for the x-dist SIMD experiment.

Output: research/2026-04-08-x-dist-simd-avx512/data/tw_parity_50d.npz
  - X: (200, 50) float64 — synthetic high-dimensional data
  - Y: (200, 2) float64 — synthetic 2D embedding
  - k: int64 — number of neighbors (15)
  - sklearn_score: float64 — trustworthiness computed by sklearn

Run from the repository root with the spectral-test env active:
  source envs/spectral-test/bin/activate
  python research/2026-04-08-x-dist-simd-avx512/scripts/gen_tw_parity_50d.py
"""

import pathlib
import numpy as np
from sklearn.manifold import trustworthiness

FIXTURE_DIR = pathlib.Path(__file__).parents[3] / "research" / "2026-04-08-x-dist-simd-avx512" / "data"

def main() -> None:
    rng = np.random.RandomState(42)
    X = rng.randn(200, 50)
    Y = rng.randn(200, 2)
    k = 15

    sklearn_score = trustworthiness(X, Y, n_neighbors=k)
    print(f"sklearn trustworthiness(n=200, d=50, k={k}) = {sklearn_score:.15f}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURE_DIR / "tw_parity_50d.npz"
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
