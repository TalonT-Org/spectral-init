#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS_DIR="research/2026-03-22-lobpcg-warm-restart-benefit/results"
mkdir -p "$RESULTS_DIR"

echo "=== Running warm restart benefit experiment ===" | tee "$RESULTS_DIR/raw.txt"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS_DIR/raw.txt"
echo "" | tee -a "$RESULTS_DIR/raw.txt"

cargo test --features testing --release --test test_warm_restart_benefit \
    -- warm_restart --nocapture 2>&1 \
    | tee -a "$RESULTS_DIR/raw.txt"

echo "" | tee -a "$RESULTS_DIR/raw.txt"
echo "=== Done. Results in $RESULTS_DIR/raw.txt ===" | tee -a "$RESULTS_DIR/raw.txt"
