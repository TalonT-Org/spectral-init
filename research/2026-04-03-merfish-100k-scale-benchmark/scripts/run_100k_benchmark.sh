#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_DIR="$PROJECT_ROOT/temp/merfish_100k"
OUTPUT_DIR="$PROJECT_ROOT/tests/visual_eval/output"
RESULTS_DIR="$SCRIPT_DIR/../results"

EVAL_SCRIPT="$PROJECT_ROOT/tests/visual_eval/run_merfish_eval.sh"

echo "=== MERFISH 100K Scale Benchmark ==="
echo "  PROJECT_ROOT : $PROJECT_ROOT"
echo "  DATA_DIR     : $DATA_DIR"
echo "  OUTPUT_DIR   : $OUTPUT_DIR"
echo "  RESULTS_DIR  : $RESULTS_DIR"
echo ""

# Phase 0: conditionally generate 100K subset
if [ ! -f "$DATA_DIR/merfish_100k_meta.json" ]; then
    echo "=== Phase 0: Generating MERFISH 100K subset ==="
    mkdir -p "$DATA_DIR"
    python "$PROJECT_ROOT/tests/visual_eval/generate_merfish_subset.py" \
        --n-cells 100000 \
        --output-dir "$DATA_DIR"
else
    echo "=== Phase 0: MERFISH 100K subset already present, skipping generation ==="
fi

echo ""

# Phases 1–3: baseline, Rust export, Python comparison
echo "=== Phases 1–3: MERFISH 100K pipeline ==="
bash "$EVAL_SCRIPT" 100k

echo ""

# Collect results
echo "=== Collecting results ==="
mkdir -p "$RESULTS_DIR"
cp "$OUTPUT_DIR/merfish_100k_metrics.json"  "$RESULTS_DIR/"
cp "$OUTPUT_DIR/merfish_100k_timing.json"   "$RESULTS_DIR/"
cp "$OUTPUT_DIR/merfish_100k_memory.json"   "$RESULTS_DIR/"
cp "$OUTPUT_DIR/merfish_100k_rust_perf.txt" "$RESULTS_DIR/"

echo ""
echo "=== Results saved to $RESULTS_DIR ==="
ls -lh "$RESULTS_DIR/"
