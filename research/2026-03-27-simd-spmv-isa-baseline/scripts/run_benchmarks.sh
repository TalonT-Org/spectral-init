#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-simd-spmv-isa-baseline/results"
mkdir -p "$RESULTS"

export RUSTFLAGS="-C target-cpu=native"

run_group() {
    local group="$1"
    local outfile="$RESULTS/${group}.txt"
    echo "=== Running Criterion group: $group ==="
    cargo bench --features testing --bench simd_spmv_exp -- "$group" \
        2>&1 | tee "$outfile"
    echo "Results written to $outfile"
}

run_group spmv_csr_scaling
run_group spmv_sell_c
run_group spmv_avx2
run_group spmv_sell_c_conversion
run_group pipeline_blobs5000

echo "=== run_benchmarks.sh complete. Results in $RESULTS/ ==="
