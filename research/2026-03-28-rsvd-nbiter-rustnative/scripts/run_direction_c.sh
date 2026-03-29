#!/usr/bin/env bash
# run_direction_c.sh — Direction C subspace quality for passing pairs.
# Reference: rsvd_solve_accurate at nbiter=10, p=100.
# Output: results/direction_c_subspace.csv
set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
RESULTS_DIR="$RESEARCH_DIR/results"

mkdir -p "$RESULTS_DIR"

OUTPUT_CSV="$RESULTS_DIR/direction_c_subspace.csv"

echo "fixture,method,nbiter,p,gram_det,sign_error" > "$OUTPUT_CSV"

echo "[direction_c] Running Direction C subspace quality (--release, --test-threads=1) ..."
cargo nextest run \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --features testing \
    --release \
    --test test_exp_rsvd_nbiter_sweep \
    direction_c_subspace \
    --test-threads=1 \
    --no-capture \
    2>/dev/null \
  | grep '^QUALITY_ROW_C,' \
  | sed 's/^QUALITY_ROW_C,//' \
  >> "$OUTPUT_CSV"

DATA_ROWS=$(tail -n +2 "$OUTPUT_CSV" | wc -l | tr -d ' ')
echo "[direction_c] Done. $DATA_ROWS rows written to $OUTPUT_CSV"
