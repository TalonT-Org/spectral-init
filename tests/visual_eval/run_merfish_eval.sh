#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Accept subset scale as first positional argument
SUBSET="${1:-10k}"

# Determine data-dir arg for 100K (empty string for 10K = use default)
DATA_DIR_ARG=""
if [ "$SUBSET" = "100k" ]; then
    DATA_DIR_ARG="--data-dir $PROJECT_ROOT/temp/merfish_100k"
fi

# Phase 0: generate 100K subset (100K only, idempotent)
if [ "$SUBSET" = "100k" ] && [ ! -f "$PROJECT_ROOT/temp/merfish_100k/merfish_100k_meta.json" ]; then
    echo "=== Phase 0: Generate MERFISH 100K subset ==="
    python "$SCRIPT_DIR/generate_merfish_subset.py" \
        --n-cells 100000 \
        --output-dir "$PROJECT_ROOT/temp/merfish_100k"
elif [ "$SUBSET" = "100k" ]; then
    echo "=== Phase 0: MERFISH 100K subset already present, skipping ==="
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
