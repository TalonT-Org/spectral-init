#!/usr/bin/env bash
# Usage: bash scripts/run_benchmarks.sh
# Runs the full spmv_avx2 Criterion group (scalar, avx2_gather, avx512_gather).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-avx512-gather-zen5-spmv/results"
mkdir -p "$RESULTS"

export RUSTFLAGS="-C target-cpu=native"

echo "=== Running Criterion group: spmv_avx2 ==="
cargo bench --features testing --bench simd_spmv_exp -- spmv_avx2 \
    2>&1 | tee "$RESULTS/criterion_avx2.txt"

echo "=== run_benchmarks.sh complete. Results in $RESULTS/ ==="
