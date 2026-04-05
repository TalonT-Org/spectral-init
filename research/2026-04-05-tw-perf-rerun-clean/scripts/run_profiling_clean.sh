#!/usr/bin/env bash
# Profiling Runner (clean)
# Drives tw_profiler with step-timing instrumentation for 5 Gaussian variants.
# Feature flags: cli,profiling — enables step_timing atomics in the library.
#
# Note on CLI flags: tw_profiler (src/bin/tw_profiler.rs) accepts --warmup and --iters.
# Use --warmup and --iters as defined in the compiled binary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
DATA_DIR="$EXP_DIR/data/gaussian"
RESULTS_DIR="$EXP_DIR/results/step_timing"
BINARY="$REPO_ROOT/target/release/tw_profiler"
X="$DATA_DIR/gaussian_n100000_x.npy"
Y="$DATA_DIR/gaussian_n100000_y.npy"
K=15
WARMUP=5
ITERS=30

echo "=== Building tw_profiler (features: cli,profiling) ==="
(cd "$REPO_ROOT" && \
    cargo build --release --features cli,profiling --no-default-features --bin tw_profiler)
echo "Build complete: $BINARY"

echo ""
echo "=== Step-Timing: 5 Gaussian Variants at n=100K ==="
VARIANTS=(baseline thread_local partial_rank avx2_kernel combined)
for VARIANT in "${VARIANTS[@]}"; do
    OUT="$RESULTS_DIR/gaussian_n100000_${VARIANT}.json"
    echo "  Running variant: $VARIANT"
    "$BINARY" \
        --x "$X" \
        --y "$Y" \
        --k $K \
        --warmup $WARMUP \
        --iters $ITERS \
        --variant "$VARIANT" \
        --output "$OUT"
    echo "  Done: $VARIANT -> $OUT"
done

echo ""
echo "=== run_profiling_clean.sh complete ==="
