#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
RESULTS_DIR="$REPO_ROOT/research/2026-03-27-computemode-simd-spmv-parity/results"
SCRIPTS_DIR="$REPO_ROOT/research/2026-03-27-computemode-simd-spmv-parity/scripts"

cd "$REPO_ROOT"

echo "=== Phase 1: Rust experiment tests ==="
RESULTS_DIR="$RESULTS_DIR" cargo test --features testing \
    --test test_simd_parity -- --nocapture 2>&1 | tee "$RESULTS_DIR/rust_test_output.txt"

echo "=== Phase 2: Subspace angle analysis ==="
micromamba run -n spectral-test python "$SCRIPTS_DIR/analyze_subspace.py"

echo "=== Phase 3: RQ4 scipy backend documentation ==="
micromamba run -n spectral-test python "$SCRIPTS_DIR/collect_rq4.py"

echo "=== Done. Results in $RESULTS_DIR ==="
