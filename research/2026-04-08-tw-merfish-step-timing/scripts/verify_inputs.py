"""Verify MERFISH fixture files have expected shapes and dtypes.

Checks the four pre-existing .npy fixtures from the tw-perf-rerun-clean
experiment and confirms they match the expected dimensions for the
tw-merfish-step-timing experiment.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MERFISH_DIR = SCRIPT_DIR / "../../2026-04-05-tw-perf-rerun-clean/data/merfish"

EXPECTED = {
    "merfish_n10k_x.npy": (10000, 50),
    "merfish_n10k_y.npy": (10000, 2),
    "merfish_n50k_x.npy": (50000, 50),
    "merfish_n50k_y.npy": (50000, 2),
}


def main() -> None:
    merfish_dir = MERFISH_DIR.resolve()
    ok = True

    for filename, expected_shape in EXPECTED.items():
        path = merfish_dir / filename
        if not path.exists():
            print(f"MISSING: {filename} not found at {path}")
            ok = False
            continue

        arr = np.load(path, mmap_mode="r")
        shape_ok = arr.shape == expected_shape
        dtype_ok = arr.dtype == np.float64

        status = "OK" if (shape_ok and dtype_ok) else "FAIL"
        print(f"  [{status}] {filename}: shape={arr.shape} dtype={arr.dtype}")

        if not shape_ok:
            print(f"         expected shape {expected_shape}")
            ok = False
        if not dtype_ok:
            print(f"         expected dtype float64")
            ok = False

    # Summary: confirm d_x for X arrays
    x_files = [f for f in EXPECTED if "_x.npy" in f]
    d_x_values = set(EXPECTED[f][1] for f in x_files)
    print(f"\n  d_x confirmed: {', '.join(str(d) for d in sorted(d_x_values))} "
          f"(from {len(x_files)} X arrays)")

    if ok:
        print("\nAll MERFISH fixtures verified successfully.")
    else:
        print("\nVerification FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
