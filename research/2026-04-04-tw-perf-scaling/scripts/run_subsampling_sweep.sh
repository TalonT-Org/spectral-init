#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"

cd "$RESEARCH_DIR"

echo "============================================"
echo "  tw-perf-scaling: Subsampling Sweep"
echo "============================================"

# Sealed gate presence check
SEALED="results/subsampling/h5_confirmatory_result.json"
if [ ! -f "$SEALED" ]; then
  echo "ERROR: H5 confirmatory result not found at $SEALED"
  echo "Run scripts/run_h5_confirmatory.sh first to seal the confirmatory result."
  exit 1
fi

MS=(500 1000 2000 5000 10000)
for m in "${MS[@]}"; do
  echo ""
  echo "Running sweep m=$m ..."
  python scripts/subsampling_sweep.py \
    --x data/merfish/merfish_n10k_x.npy \
    --y data/merfish/merfish_n10k_y.npy \
    --m "$m" \
    --seed 99 \
    --output "results/subsampling/sweep_m${m}.json"
done

echo ""
echo "============================================"
echo "  Subsampling sweep complete. Files written:"
echo "============================================"
ls -1 results/subsampling/sweep_m*.json 2>/dev/null || echo "  (none)"
