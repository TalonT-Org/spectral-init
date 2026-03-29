#!/usr/bin/env bash
# run_direction_b.sh — Full Direction B 2D sweep.
# Both fixtures, nbiter in {2,3,4,5,6,8,10}, p in {20,30,50,100}.
# Output: results/direction_b_sweep.csv
set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
RESULTS_DIR="$RESEARCH_DIR/results"

mkdir -p "$RESULTS_DIR"

OUTPUT_CSV="$RESULTS_DIR/direction_b_sweep.csv"

echo "fixture,method,nbiter,p,residual,ortho_error,wall_time_us,spectral_gap,passes" > "$OUTPUT_CSV"

echo "[direction_b] Running full Direction B sweep (--release, --test-threads=1) ..."
cargo nextest run \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --features testing \
    --release \
    --test test_exp_rsvd_nbiter_sweep \
    direction_b_sweep \
    --test-threads=1 \
    --no-capture \
    2>/dev/null \
  | grep '^SWEEP_ROW_B,' \
  | sed 's/^SWEEP_ROW_B,//' \
  >> "$OUTPUT_CSV"

DATA_ROWS=$(tail -n +2 "$OUTPUT_CSV" | wc -l | tr -d ' ')
echo "[direction_b] Done. $DATA_ROWS rows written to $OUTPUT_CSV"
