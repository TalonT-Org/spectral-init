#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RESULTS_DIR="$SCRIPT_DIR/../results/profiler"

mkdir -p "$RESULTS_DIR"

# Build profiler binary with profiling instrumentation
echo "=== Building tw_profiler (cli,profiling) ==="
cargo build --release \
    --features cli,profiling \
    --manifest-path "$REPO_ROOT/Cargo.toml"

PROFILER="$REPO_ROOT/target/release/tw_profiler"

run_variant() {
    local variant="$1"
    echo "=== Profiling variant: $variant (n=10000) ==="
    "$PROFILER" \
        --x "$DATA_DIR/gaussian_n10000_x.npy" \
        --y "$DATA_DIR/gaussian_n10000_y.npy" \
        --k 15 \
        --iters 30 \
        --warmup 5 \
        --variant "$variant" \
        --stderr-capture "$RESULTS_DIR/stderr_${variant}.txt" \
        --output "$RESULTS_DIR/profiler_${variant}_n10000.json"
    echo "  wrote: $RESULTS_DIR/profiler_${variant}_n10000.json"
}

run_variant baseline
run_variant heap_reuse
run_variant flat_partial
run_variant flat_simd

echo "=== run_profiler.sh complete ==="
