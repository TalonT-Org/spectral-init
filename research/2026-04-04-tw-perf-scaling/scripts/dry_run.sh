#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

cd "$RESEARCH_DIR"

echo "============================================"
echo "  tw-perf-scaling: Dry Run"
echo "============================================"

# Check 1 — Build
echo ""
echo "[1/4] Building release binaries (features: testing, cli)..."
cargo build --features "testing cli" --release --manifest-path "$PROJECT_ROOT/Cargo.toml"
echo "  ✓ Build succeeded"

# Check 2 — tw_profiler dry run
echo ""
echo "[2/4] Running tw_profiler on n=1K gaussian data..."
mkdir -p results/step_timing
"$PROJECT_ROOT/target/release/tw_profiler" \
  --x data/gaussian/gaussian_n1000_x.npy \
  --y data/gaussian/gaussian_n1000_y.npy \
  --k 15 --iters 1 --warmup 0 \
  --variant baseline \
  --output results/step_timing/dry_run.json
python -c "import json; json.load(open('results/step_timing/dry_run.json'))"
echo "  ✓ tw_profiler produced valid JSON"

# Check 3 — sklearn reference
echo ""
echo "[3/4] Running sklearn_reference.py on n=1K..."
mkdir -p results/parity
python scripts/sklearn_reference.py --n 1000 --output results/parity/dry_run.json
python -c "import json; json.load(open('results/parity/dry_run.json'))"
echo "  ✓ sklearn_reference produced valid JSON"

# Check 4 — Criterion benchmark (single quick run)
echo ""
echo "[4/4] Running Criterion benchmark (baseline/1000 only)..."
cargo criterion --bench trustworthiness_bench --manifest-path "$PROJECT_ROOT/Cargo.toml" -- "baseline/1000"
echo "  ✓ Criterion benchmark completed"

echo ""
echo "============================================"
echo "  Dry run passed"
echo "============================================"
