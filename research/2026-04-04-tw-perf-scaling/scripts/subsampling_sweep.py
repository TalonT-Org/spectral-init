"""Row-subsampling sweep: call tw_approx_runner for each trial at a given m.

Usage:
    python subsampling_sweep.py --x X.npy --y Y.npy --m 5000 --seed 99 \
        --output results/subsampling/sweep_m5000.json
    python subsampling_sweep.py --x X.npy --y Y.npy --m 5000 --n-trials 3 --seed 99 \
        --output results/subsampling/sweep_m5000.json
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent.parent
BINARY = PROJECT_ROOT / "target" / "release" / "tw_approx_runner"


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-subsampling sweep for a single m value")
    parser.add_argument("--x", required=True, type=Path, help="Path to X embedding .npy")
    parser.add_argument("--y", required=True, type=Path, help="Path to Y embedding .npy")
    parser.add_argument("--m", required=True, type=int, help="Subsample size")
    parser.add_argument("--n-trials", default=1, type=int, dest="n_trials",
                        help="Number of trials to run (default: 1)")
    parser.add_argument("--seed", required=True, type=int, help="Master RNG seed")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    trials = []
    tmp_files: list[Path] = []

    for _ in range(args.n_trials):
        trial_seed = int(rng.randint(0, 2**31))
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        tmp_files.append(tmp_path)

        subprocess.run(
            [
                str(BINARY),
                "--x", str(args.x),
                "--y", str(args.y),
                "--k", "15",
                "--sample", str(args.m),
                "--seed", str(trial_seed),
                "--output", str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        trial_data = json.loads(tmp_path.read_text())
        trials.append(trial_data)

    for p in tmp_files:
        try:
            p.unlink()
        except OSError:
            pass

    deltas = [t["delta"] for t in trials]
    output_data = {
        "m": args.m,
        "seed": args.seed,
        "mean_delta": float(np.mean(deltas)),
        "max_delta": float(np.max(np.abs(deltas))),
        "trials": trials,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
