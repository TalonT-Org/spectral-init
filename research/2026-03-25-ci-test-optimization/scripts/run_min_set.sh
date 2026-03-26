#!/usr/bin/env bash
# Run the minimum covering set of tests (RQ2).
# Covers Levels 0, 1, 2-direct, 3-direct, and 4 of the solver escalation chain.
# See results/rq2_min_set/coverage_map.md for rationale.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq2_min_set"
EXTRACT="research/2026-03-25-ci-test-optimization/scripts/extract_timings.py"
mkdir -p "$RESULTS_DIR"

# Minimum covering set (Levels 0, 1, 2-direct, 3-direct, 4)
# Level 0: test_level_0_dense_evd_for_small_n (asserts level==0)
# Level 1: test_level_1_lobpcg_for_large_well_conditioned_n (asserts level==1)
# Level 2: binary(test_comp_g_sinv) — all sinv solver tests
# Level 3: binary(test_comp_f_lobpcg) — all lobpcg tests (regularize=true path)
# Level 4: comp_d_rsvd_blobs_connected_200 — smallest rSVD fixture test
FILTER="test(test_level_0_dense_evd_for_small_n) \
  + test(test_level_1_lobpcg_for_large_well_conditioned_n) \
  + binary(test_comp_g_sinv) \
  + binary(test_comp_f_lobpcg) \
  + test(comp_d_rsvd_blobs_connected_200)"

cargo nextest run \
    --profile ci \
    --features testing \
    --no-fail-fast \
    -E "$FILTER"

cp target/nextest/ci/junit.xml "${RESULTS_DIR}/min_set_junit.xml"

python3 "$EXTRACT" \
    "${RESULTS_DIR}/min_set_junit.xml" \
    "${RESULTS_DIR}/min_set_timings.csv"

echo "min_set run complete — results in ${RESULTS_DIR}/"
