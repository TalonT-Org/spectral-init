#!/usr/bin/env bash
# Usage: bash scripts/run_with_turbostat.sh
# Runs the AVX-512-only benchmark with CPU frequency monitoring.
# Uses turbostat if available and sudo-accessible; falls back to /sys polling.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-avx512-gather-zen5-spmv/results"
mkdir -p "$RESULTS"

export RUSTFLAGS="-C target-cpu=native"

# ── Decide frequency-monitoring method ───────────────────────────────────────
MONITOR_PID=""

if command -v turbostat >/dev/null 2>&1 && sudo -n turbostat --help >/dev/null 2>&1; then
    echo "Using turbostat for CPU frequency monitoring."
    sudo turbostat \
        --interval 1 \
        --quiet \
        --show Avg_MHz,Busy%,Bzy_MHz \
        > "$RESULTS/turbostat_avx512.txt" 2>&1 &
    MONITOR_PID=$!
else
    echo "turbostat unavailable or sudo denied; falling back to /sys polling."
    (
        while true; do
            echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
            cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null \
                | awk '{sum+=$1; n++} END {if(n>0) printf "avg_kHz=%d n_cpu=%d\n", sum/n, n}'
            sleep 1
        done
    ) > "$RESULTS/turbostat_avx512.txt" &
    MONITOR_PID=$!
fi

# ── Run AVX-512-only benchmark ────────────────────────────────────────────────
echo "=== Running Criterion filter: avx512_gather ==="
cargo bench --features testing --bench simd_spmv_exp -- avx512_gather \
    2>&1 | tee "$RESULTS/criterion_avx512_only.txt"

# ── Stop frequency monitor ────────────────────────────────────────────────────
if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

echo "=== run_with_turbostat.sh complete. Results in $RESULTS/ ==="
