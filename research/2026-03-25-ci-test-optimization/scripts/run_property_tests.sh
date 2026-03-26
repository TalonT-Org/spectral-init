#!/usr/bin/env bash
# RQ4: Property test runner.
# Runs the 9 proptest-based property tests, records wall time.
#
# Sets SPECTRAL_DENSE_N_THRESHOLD=50 to force LOBPCG path for n in [50,200],
# matching the ci-fast profile intent.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq4_property_tests"
mkdir -p "$RESULTS_DIR"

export SPECTRAL_DENSE_N_THRESHOLD=50

echo "=== RQ4: Running property tests (DENSE_N_THRESHOLD=${SPECTRAL_DENSE_N_THRESHOLD}) ==="

START_TIME=$(date +%s%N)

cargo nextest run \
    --profile ci-fast \
    --features testing \
    -E 'test(proptest_)' \
    2>&1 | tee /tmp/proptest_output.txt

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
ELAPSED_S=$(awk "BEGIN {printf \"%.3f\", ${ELAPSED_MS}/1000}")

echo "wall_time_ms=${ELAPSED_MS}" > "${RESULTS_DIR}/property_timings.txt"
echo "wall_time_s=${ELAPSED_S}" >> "${RESULTS_DIR}/property_timings.txt"
echo "threshold=${SPECTRAL_DENSE_N_THRESHOLD}" >> "${RESULTS_DIR}/property_timings.txt"
echo "date=$(date -Iseconds)" >> "${RESULTS_DIR}/property_timings.txt"

echo "=== Property tests complete: ${ELAPSED_S}s → ${RESULTS_DIR}/property_timings.txt ==="
