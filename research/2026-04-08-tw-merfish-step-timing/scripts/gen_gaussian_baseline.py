"""Generate Gaussian baseline dataset for tw-merfish-step-timing experiment.

Produces:
    data/gaussian/gaussian_n10k_x.npy  (10000, 10) float64  N(0,1)
    data/gaussian/gaussian_n10k_y.npy  (10000, 2)  float64  U(0,1)
"""

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "../data/gaussian"

SEED = 2026
N = 10000
D_X = 10
D_Y = 2


def main() -> None:
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    x = rng.standard_normal((N, D_X))
    y = rng.uniform(0.0, 1.0, (N, D_Y))

    np.save(output_dir / "gaussian_n10k_x.npy", x)
    np.save(output_dir / "gaussian_n10k_y.npy", y)

    print(f"  [gaussian] n={N}: x{x.shape} y{y.shape}")
    print(f"  x dtype={x.dtype}, y dtype={y.dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
