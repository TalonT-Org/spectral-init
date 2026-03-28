#!/usr/bin/env bash
# run_dry_run.sh — Dry run for the rSVD nbiter experiment.
# Runs direction_a_2il_vs_direct_l (blobs_connected_2000 only), writes to
# results/dry_run.csv, and verifies >= 4 data rows are present.
set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
RESULTS_DIR="$RESEARCH_DIR/results"

mkdir -p "$RESULTS_DIR"

OUTPUT_CSV="$RESULTS_DIR/dry_run.csv"

echo "fixture,method,nbiter,p,residual,wall_time_us" > "$OUTPUT_CSV"

echo "[dry_run] Running direction_a_2il_vs_direct_l (--release, --test-threads=1) ..."
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

echo "[dry_run] Wrote $DATA_ROWS data rows to $OUTPUT_CSV"

if [ "$DATA_ROWS" -lt 4 ]; then
    echo "[dry_run] FAIL: expected >= 4 data rows, got $DATA_ROWS"
    exit 1
fi

echo "[dry_run] PASS: $DATA_ROWS rows (expected 12: 2 methods x 3 nbiter x 2 p)"
echo "[dry_run] Sample output:"
tail -n +2 "$OUTPUT_CSV" | head -4
