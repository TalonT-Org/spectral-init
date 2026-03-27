#!/usr/bin/env bash
# run_rq1_nbiter_check.sh — Secondary script for nbiter sensitivity check.
#
# Run this ONLY if any fixture shows passes=false at p=10 in
# rq1_oversampling_sweep.csv. Fill in FAILING_FIXTURES below with the
# fixture names that failed at p=10 before running.
#
# nbiter=3 mechanism options (choose one before running):
#   Option A — env-var seam (preferred if SPECTRAL_RSVD_NBITER seam exists):
#     Set SPECTRAL_RSVD_NBITER=3 alongside SPECTRAL_RSVD_OVERSAMPLING=10.
#   Option B — local source patch (fallback if seam does not exist):
#     Temporarily change `let nbiter = 2;` to `let nbiter = 3;` in
#     src/solvers/rsvd.rs::rsvd_solve, run the script, then revert.
#   The nbiter seam does not exist as of groupB. Add it in a follow-up
#   if Option A is needed.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$REPO_ROOT/research/2026-03-26-rsvd-oversampling-coverage/results"
OUTPUT_CSV="$RESULTS_DIR/rq1_nbiter_check.csv"

mkdir -p "$RESULTS_DIR"

# FILL IN: fixture names that failed at p=10 in the production sweep.
FAILING_FIXTURES=(
    # "near_dupes_100"
    # "blobs_connected_200"
    # "moons_200"
    # "circles_300"
    # "blobs_500"
    # "blobs_connected_2000"
)

if [[ ${#FAILING_FIXTURES[@]} -eq 0 ]]; then
    echo "[nbiter_check] No failing fixtures specified. Edit FAILING_FIXTURES in this script."
    exit 1
fi

echo "fixture,n,p,k,residual,wall_time_s,passes" > "$OUTPUT_CSV"

echo "[nbiter_check] Running with SPECTRAL_RSVD_OVERSAMPLING=10 and nbiter=3 mechanism..."
# If Option A (env-var seam) is available:
#   SPECTRAL_RSVD_NBITER=3 SPECTRAL_RSVD_OVERSAMPLING=10 \
#     cargo nextest run --features testing \
#       --test test_exp_rsvd_sweep production_sweep \
#       --no-capture 2>&1 \
#     | grep '^SWEEP_ROW,' | sed 's/^SWEEP_ROW,//' >> "$OUTPUT_CSV"
#
# If Option B (local patch) is needed, apply the patch to src/solvers/rsvd.rs
# first, then uncomment:
# SPECTRAL_RSVD_OVERSAMPLING=10 \
#     cargo nextest run --features testing \
#       --test test_exp_rsvd_sweep production_sweep \
#       --no-capture 2>&1 \
#     | grep '^SWEEP_ROW,' | sed 's/^SWEEP_ROW,//' >> "$OUTPUT_CSV"

echo "[nbiter_check] Uncomment the appropriate command above after applying the nbiter mechanism."
echo "[nbiter_check] Results will go to $OUTPUT_CSV"
