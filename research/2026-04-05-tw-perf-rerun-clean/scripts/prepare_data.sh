#!/usr/bin/env bash
# Orchestrates all data generation for tw-perf-rerun-clean experiment.
# Run from the worktree root:
#   bash research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NPZ_DIR="${MERFISH_NPZ_DIR:-/home/talon/projects/spectral-init/temp/merfish_100k}"
GAUSSIAN_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean/data/gaussian"
MERFISH_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean/data/merfish"
SCRIPTS_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean/scripts"

echo "=== Generating synthetic Gaussian data (seed=2026) ==="
python "$SCRIPTS_DIR/gen_synthetic.py" \
    --seed 2026 \
    --output-dir "$GAUSSIAN_DIR" \
    --sizes 1000 5000 10000 25000 50000 100000 \
    --d 10

echo ""
echo "=== Preparing MERFISH 10k (PCA-50) ==="
python "$SCRIPTS_DIR/prepare_merfish.py" \
    --npz-dir "$NPZ_DIR" \
    --output-dir "$MERFISH_DIR" \
    --n 10000

echo ""
echo "=== Preparing MERFISH 50k (PCA-50) ==="
python "$SCRIPTS_DIR/prepare_merfish_50k.py" \
    --npz-dir "$NPZ_DIR" \
    --output-dir "$MERFISH_DIR" \
    --n 50000

echo ""
echo "=== Verifying output shapes ==="
python - "$GAUSSIAN_DIR" "$MERFISH_DIR" <<'PYEOF'
import numpy as np, sys

gaussian_dir, merfish_dir = sys.argv[1], sys.argv[2]
failures = []

for n in [1000, 5000, 10000, 25000, 50000, 100000]:
    for suffix, expected in [("x", (n, 10)), ("y", (n, 2))]:
        path = f"{gaussian_dir}/gaussian_n{n}_{suffix}.npy"
        arr = np.load(path)
        if arr.shape != expected:
            failures.append(f"FAIL {path}: expected {expected}, got {arr.shape}")
        elif arr.dtype != np.float64:
            failures.append(f"FAIL {path}: expected float64, got {arr.dtype}")
        else:
            print(f"  OK  gaussian_n{n}_{suffix}.npy {arr.shape}")

for tag, n in [("n10k", 10000), ("n50k", 50000)]:
    for suffix, expected in [("x", (n, 50)), ("y", (n, 2))]:
        path = f"{merfish_dir}/merfish_{tag}_{suffix}.npy"
        arr = np.load(path)
        if arr.shape != expected:
            failures.append(f"FAIL {path}: expected {expected}, got {arr.shape}")
        elif arr.dtype != np.float64:
            failures.append(f"FAIL {path}: expected float64, got {arr.dtype}")
        else:
            print(f"  OK  merfish_{tag}_{suffix}.npy {arr.shape}")

if failures:
    for f in failures:
        print(f, file=sys.stderr)
    sys.exit(1)

print("")
print("All shape checks passed.")
PYEOF

echo ""
echo "=== Data generation complete ==="
