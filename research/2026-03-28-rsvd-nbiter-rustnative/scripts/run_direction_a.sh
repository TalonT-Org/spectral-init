#!/usr/bin/env bash
# run_direction_a.sh — Full Direction A sweep: 2I-L vs direct-L projection.
# Fixture: blobs_connected_2000. nbiter in {2,4,6}, p in {30,100}.
# Output: results/direction_a_2il_vs_direct.csv
set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
RESULTS_DIR="$RESEARCH_DIR/results"

mkdir -p "$RESULTS_DIR"

OUTPUT_CSV="$RESULTS_DIR/direction_a_2il_vs_direct.csv"

echo "fixture,method,nbiter,p,residual,wall_time_us" > "$OUTPUT_CSV"

echo "[direction_a] Running full Direction A sweep (--release, --test-threads=1) ..."
cargo nextest run \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --features testing \
    --release \
    --test test_exp_rsvd_nbiter_sweep \
    direction_a_2il_vs_direct_l \
    --test-threads=1 \
    --no-capture \
    2>/dev/null \
  | grep '^SWEEP_ROW_A,' \
  | sed 's/^SWEEP_ROW_A,//' \
  >> "$OUTPUT_CSV"

DATA_ROWS=$(tail -n +2 "$OUTPUT_CSV" | wc -l | tr -d ' ')
echo "[direction_a] Done. $DATA_ROWS rows written to $OUTPUT_CSV"
