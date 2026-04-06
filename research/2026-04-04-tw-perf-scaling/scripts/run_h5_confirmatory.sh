#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

cd "$RESEARCH_DIR"

OUTPUT="results/subsampling/h5_confirmatory_result.json"

echo "============================================"
echo "  tw-perf-scaling: H5 Confirmatory Gate"
echo "============================================"

# Sealed gate check
if [ -f "$OUTPUT" ]; then
  echo "ERROR: Sealed result already exists at $OUTPUT"
  echo "Confirmatory result must not be overwritten after sealing."
  echo "Existing delta value:"
  python -c "import json; print(json.load(open('$OUTPUT'))['delta'])"
  exit 1
fi

mkdir -p results/subsampling

# Run tw_approx_runner with fixed seed
echo ""
echo "Running tw_approx_runner (seed=42, sample=5000)..."
"$PROJECT_ROOT/target/release/tw_approx_runner" \
  --x data/merfish/merfish_n10k_x.npy \
  --y data/merfish/merfish_n10k_y.npy \
  --k 15 \
  --sample 5000 \
  --seed 42 \
  --output "$OUTPUT"

# Verify and report
if [ ! -f "$OUTPUT" ]; then
  echo "ERROR: Output file was not created"
  exit 1
fi

DELTA=$(python -c "import json; print(json.load(open('$OUTPUT'))['delta'])")
echo ""
echo "============================================"
echo "  H5 confirmatory result sealed: delta = $DELTA"
echo "============================================"
