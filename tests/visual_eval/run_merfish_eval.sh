#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Accept subset scale as first positional argument
SUBSET="${1:-10k}"

# Determine data-dir arg (empty string for 10K = use default)
DATA_DIR_ARG=""
if [ "$SUBSET" != "10k" ]; then
    DATA_DIR_ARG="--data-dir $PROJECT_ROOT/temp/merfish_${SUBSET}"
fi

# Map subset label to cell count
declare -A SUBSET_CELLS=( [20k]=20000 [50k]=50000 [100k]=100000 [250k]=250000 [500k]=500000 )

# Phase 0: generate subset (non-10k only, idempotent)
if [ "$SUBSET" != "10k" ]; then
    N_CELLS="${SUBSET_CELLS[$SUBSET]}"
    DATA_DIR="$PROJECT_ROOT/temp/merfish_${SUBSET}"
    META_FILE="$DATA_DIR/merfish_${SUBSET}_meta.json"
    if [ ! -f "$META_FILE" ]; then
        echo "=== Phase 0: Generate MERFISH ${SUBSET} subset ==="
        python "$SCRIPT_DIR/generate_merfish_subset.py" \
            --n-cells "$N_CELLS" \
            --output-dir "$DATA_DIR"
    else
        echo "=== Phase 0: MERFISH ${SUBSET} subset already present, skipping ==="
    fi
fi

echo "=== Phase 1: Generate MERFISH baselines ==="
python "$SCRIPT_DIR/generate_merfish_comparisons.py" \
    --phase baseline \
    --subset "$SUBSET" \
    $DATA_DIR_ARG

echo ""
echo "=== Phase 2: Rust Export — MERFISH spectral init ==="
cd "$PROJECT_ROOT"
RUST_PERF_FILE="$SCRIPT_DIR/output/merfish_${SUBSET}_rust_perf.txt"
/usr/bin/time -o "$RUST_PERF_FILE" -f "%e %M" \
    cargo nextest run \
        --profile merfish-eval \
        --run-ignored all \
        --features testing \
        -E "test(export_merfish_init_${SUBSET})"

echo ""
echo "=== Phase 3: Python comparison (Python vs Rust vs Random) ==="
python "$SCRIPT_DIR/generate_merfish_comparisons.py" \
    --phase compare \
    --subset "$SUBSET" \
    $DATA_DIR_ARG

echo ""
echo "=== Summary ==="
python -c "
import json, glob, sys
script_dir = sys.argv[1]
results = sorted(glob.glob(script_dir + '/output/merfish_*_metrics.json'))
if not results:
    print('  No metrics files found in ' + script_dir + '/output/')
else:
    for f in results:
        with open(f) as fh:
            m = json.load(fh)
        status = m.get('pass_fail', {}).get('overall', 'UNKNOWN')
        print(f\"  {m['dataset']:30s} {status}\")
" "$SCRIPT_DIR"
