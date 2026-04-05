#!/usr/bin/env bash
# Criterion Benchmark Runner (clean)
#
# RT-4 Re-run Policy:
#   If a benchmark variant produces anomalous results (e.g., >10% CV or a point
#   estimate that contradicts prior runs), re-run that single variant only.
#   Always APPEND to the existing JSON-lines file; never overwrite. The analysis
#   script uses the final record for each benchmark ID in the file.
#
# W4 Cache Warm-State Check:
#   After all forward runs, re-runs tw_combined_bench then tw_baseline_bench in
#   reversed order to detect systematic cache-warm bias. Results appended to
#   criterion_reversed_output.json. The analysis script flags if forward vs
#   reversed point estimates differ by >5%.
#
# W8 Check:
#   Criterion builds must NOT enable the profiling feature. That feature activates
#   step_timing atomics and adds overhead to every trustworthiness call, corrupting
#   wall-clock measurements.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
RESULTS_DIR="$EXP_DIR/results/criterion"

# --- W8 Check ---
if [[ -n "${CARGO_FEATURE_PROFILING:-}" ]]; then
    echo "W8 FAIL: CARGO_FEATURE_PROFILING is set in environment." >&2
    echo "Criterion builds must not enable CARGO_FEATURE_PROFILING (W8 violation)." >&2
    exit 1
fi
echo "W8 PASS: CARGO_FEATURE_PROFILING is not set in environment"

# --- Verify cargo-criterion is on PATH ---
if ! command -v cargo-criterion &>/dev/null; then
    echo "ERROR: cargo-criterion not found on PATH." >&2
    echo "Install with: cargo install cargo-criterion" >&2
    exit 1
fi
echo "cargo-criterion: $(cargo-criterion --version 2>&1 | head -1)"

echo ""
echo "=== Criterion: Five Gaussian Variants (forward order) ==="
FORWARD_BENCHES=(
    tw_baseline_bench
    tw_thread_local_bench
    tw_partial_rank_bench
    tw_avx2_bench
    tw_combined_bench
)
for BENCH in "${FORWARD_BENCHES[@]}"; do
    echo "  Running: $BENCH"
    (cd "$REPO_ROOT" && \
        cargo criterion --bench "$BENCH" --message-format=json --features cli) \
        >> "$RESULTS_DIR/criterion_output.json"
    echo "  Done: $BENCH. Sleeping 60s for thermal/cache isolation..."
    sleep 60
done

echo ""
echo "=== Criterion: MERFISH Partial-Rank Bench ==="
echo "  Running: tw_partial_rank_merfish_bench"
(cd "$REPO_ROOT" && \
    cargo criterion --bench tw_partial_rank_merfish_bench --message-format=json --features cli) \
    >> "$RESULTS_DIR/criterion_merfish_output.json"
echo "  Done: tw_partial_rank_merfish_bench"

echo ""
echo "=== W4 Cache Warm-State Check (reversed order) ==="
echo "  Sleeping 60s before reversed run..."
sleep 60
echo "  Running: tw_combined_bench (reversed 1/2)"
(cd "$REPO_ROOT" && \
    cargo criterion --bench tw_combined_bench --message-format=json --features cli) \
    >> "$RESULTS_DIR/criterion_reversed_output.json"
echo "  Sleeping 60s..."
sleep 60
echo "  Running: tw_baseline_bench (reversed 2/2)"
(cd "$REPO_ROOT" && \
    cargo criterion --bench tw_baseline_bench --message-format=json --features cli) \
    >> "$RESULTS_DIR/criterion_reversed_output.json"
echo "  W4 reversed run complete."

echo ""
echo "=== run_criterion_clean.sh complete ==="
