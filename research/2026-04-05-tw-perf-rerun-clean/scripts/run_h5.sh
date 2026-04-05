#!/usr/bin/env bash
# H5 Hypothesis Runner
# Drives tw_approx_runner for the approximate trustworthiness experiment.
# Timing protocol: one warm-up binary invocation before each timed invocation.
# No file I/O inside timing window (tw_approx_runner loads npy before Instant::now()).
# RT-4 re-run policy: re-run individual seeds; results are deterministic per seed+m.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
DATA_DIR="$EXP_DIR/data/gaussian"
RESULTS_DIR="$EXP_DIR/results/h5"
BINARY="$REPO_ROOT/target/release/tw_approx_runner"
WARMUP_TMP="$REPO_ROOT/temp/h5_warmup_discard.json"
X="$DATA_DIR/gaussian_n100000_x.npy"
Y="$DATA_DIR/gaussian_n100000_y.npy"
K=15
M_TRIAL=5000

mkdir -p "$REPO_ROOT/temp"

echo "=== Building tw_approx_runner ==="
(cd "$REPO_ROOT" && cargo build --release --features cli --no-default-features --bin tw_approx_runner)
echo "Build complete: $BINARY"

echo ""
echo "=== H5 Seed Trials (m=$M_TRIAL, seeds 42-51) ==="
for SEED in $(seq 42 51); do
    OUT="$RESULTS_DIR/h5_trial_seed${SEED}.json"
    echo "  Warm-up: seed=$SEED"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample $M_TRIAL --seed "$SEED" \
        --output "$WARMUP_TMP" 2>/dev/null
    echo "  Timed:   seed=$SEED -> $OUT"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample $M_TRIAL --seed "$SEED" --output "$OUT"
done

echo ""
echo "=== H5 M-Sweep (seed=42) ==="
for M in 500 1000 2000 10000; do
    OUT="$RESULTS_DIR/h5_sweep_m${M}.json"
    echo "  Warm-up: m=$M"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample "$M" --seed 42 \
        --output "$WARMUP_TMP" 2>/dev/null
    echo "  Timed:   m=$M -> $OUT"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample "$M" --seed 42 --output "$OUT"
done

rm -f "$WARMUP_TMP"
echo ""
echo "=== run_h5.sh complete ==="
