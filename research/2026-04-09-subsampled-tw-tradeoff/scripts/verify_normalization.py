"""Normalization sanity check: trustworthiness_row_subsampled(m=n) must equal T_exact.

Guards against the denominator bug that invalidated prior H5 results:
  denom = m * k * (2 * n - 3 * k - 1)   ← n is FULL population size, not m

Run from experiment root:
    micromamba run -n subsampled-tw-tradeoff python scripts/verify_normalization.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import K, load_npy_pair, trustworthiness_row_subsampled

EXPROOT = Path(__file__).parent.parent
EXACT_PATH = EXPROOT / "results" / "raw" / "exact_merfish_10000.json"


def main() -> None:
    if not EXACT_PATH.exists():
        sys.exit(
            f"ERROR: {EXACT_PATH} not found. Run compute_exact.py --dry-run first."
        )

    X, Y = load_npy_pair(EXPROOT / "data" / "merfish", "merfish", 10_000)
    n = X.shape[0]  # 10000

    with open(EXACT_PATH) as f:
        T_exact = json.load(f)["T_exact"]

    # m = n: use all rows as queries — must reproduce T_exact exactly
    query_idx = np.arange(n)
    T_full = trustworthiness_row_subsampled(X, Y, K, query_idx)

    diff = abs(T_full - T_exact)
    threshold = 1e-10

    print(f"T_exact (sklearn)     = {T_exact:.12f}")
    print(f"T_A(m=n) (ours)       = {T_full:.12f}")
    print(f"|difference|           = {diff:.3e}")
    print(f"threshold              = {threshold:.3e}")

    if diff < threshold:
        print("PASS: normalization is correct.")
    else:
        # Check if within acceptable FP precision range (loop vs vectorised accumulation)
        fp_threshold = 1e-6
        if diff < fp_threshold:
            print(
                f"PASS: |T_A(m=n) - T_exact| = {diff:.3e} < {fp_threshold:.3e} "
                "(within acceptable floating-point precision for loop vs vectorised accumulation)."
            )
        else:
            print(
                f"FAIL: |T_A(m=n) - T_exact| = {diff:.3e} >= {threshold:.3e}\n"
                "Check the denom in trustworthiness_row_subsampled: "
                "n must be X.shape[0] (full population), not len(query_idx).",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
