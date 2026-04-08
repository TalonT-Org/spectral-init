#!/usr/bin/env bash
set -euo pipefail

# ── Constants ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
MERFISH_DIR="$EXPERIMENT_DIR/../2026-04-05-tw-perf-rerun-clean/data/merfish"

RESULTS_DIR="$EXPERIMENT_DIR/results/profiler"
K=15

# ── Overridable via environment ─────────────────────────────────────
ITERS="${PROFILER_ITERS:-5}"
WARMUP="${PROFILER_WARMUP:-2}"
DATASETS="${PROFILER_DATASETS:-gaussian_10k merfish_10k merfish_50k}"
PREFIX="${PROFILER_PREFIX:-}"

export RAYON_NUM_THREADS=16

# ── Step 1: Build ───────────────────────────────────────────────────
echo "=== Building tw_profiler ==="
(cd "$PROJECT_ROOT" && cargo build --release --features cli,profiling --bin tw_profiler)
PROFILER="$PROJECT_ROOT/target/release/tw_profiler"

# ── Step 2: Hardware profile ────────────────────────────────────────
echo "=== Recording hardware profile ==="
mkdir -p "$EXPERIMENT_DIR/results"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: $(hostname)"
    uname -a
    lscpu 2>/dev/null || echo "lscpu not available"
    grep MemTotal /proc/meminfo 2>/dev/null || echo "meminfo not available"
    echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
} > "$EXPERIMENT_DIR/results/hardware_profile.txt"

# ── Step 3: Dataset configurations ──────────────────────────────────
resolve_dataset() {
    local dataset="$1"
    case "$dataset" in
        gaussian_10k)
            X_PATH="$EXPERIMENT_DIR/data/gaussian/gaussian_n10k_x.npy"
            Y_PATH="$EXPERIMENT_DIR/data/gaussian/gaussian_n10k_y.npy"
            OUTPUT_NAME="gaussian_n10k"
            ;;
        merfish_10k)
            X_PATH="$MERFISH_DIR/merfish_n10k_x.npy"
            Y_PATH="$MERFISH_DIR/merfish_n10k_y.npy"
            OUTPUT_NAME="merfish_n10k"
            ;;
        merfish_50k)
            X_PATH="$MERFISH_DIR/merfish_n50k_x.npy"
            Y_PATH="$MERFISH_DIR/merfish_n50k_y.npy"
            OUTPUT_NAME="merfish_n50k"
            ;;
        *)
            echo "ERROR: Unknown dataset '$dataset'" >&2
            exit 1
            ;;
    esac
}

# ── Helper: check if dataset is in DATASETS list ───────────────────
should_run() { echo " $DATASETS " | grep -q " $1 "; }

# ── Step 4: Run profiler sequentially ───────────────────────────────
mkdir -p "$RESULTS_DIR"

if should_run gaussian_10k; then
    echo "=== Profiling gaussian_10k ==="
    "$PROFILER" \
        --x "$EXPERIMENT_DIR/data/gaussian/gaussian_n10k_x.npy" \
        --y "$EXPERIMENT_DIR/data/gaussian/gaussian_n10k_y.npy" \
        --output "$RESULTS_DIR/${PREFIX}gaussian_n10k.json" \
        --k "$K" --iters "$ITERS" --warmup "$WARMUP" \
        --stderr-capture "$RESULTS_DIR/${PREFIX}stderr_gaussian_10k.txt"
    echo "  -> Saved ${PREFIX}gaussian_n10k.json"
fi

if should_run merfish_10k; then
    echo "=== Profiling merfish_10k ==="
    "$PROFILER" \
        --x "$MERFISH_DIR/merfish_n10k_x.npy" \
        --y "$MERFISH_DIR/merfish_n10k_y.npy" \
        --output "$RESULTS_DIR/${PREFIX}merfish_n10k.json" \
        --k "$K" --iters "$ITERS" --warmup "$WARMUP" \
        --stderr-capture "$RESULTS_DIR/${PREFIX}stderr_merfish_10k.txt"
    echo "  -> Saved ${PREFIX}merfish_n10k.json"
fi

if should_run merfish_50k; then
    echo "=== Profiling merfish_50k ==="
    "$PROFILER" \
        --x "$MERFISH_DIR/merfish_n50k_x.npy" \
        --y "$MERFISH_DIR/merfish_n50k_y.npy" \
        --output "$RESULTS_DIR/${PREFIX}merfish_n50k.json" \
        --k "$K" --iters "$ITERS" --warmup "$WARMUP" \
        --stderr-capture "$RESULTS_DIR/${PREFIX}stderr_merfish_50k.txt"
    echo "  -> Saved ${PREFIX}merfish_n50k.json"
fi

echo "=== Profiling complete ==="
