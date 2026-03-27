#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$REPO_ROOT/research/2026-03-26-rsvd-oversampling-coverage/results"
OUTPUT_CSV="$RESULTS_DIR/rq1_oversampling_sweep.csv"

mkdir -p "$RESULTS_DIR"

# Write CSV header
echo "fixture,n,p,k,residual,wall_time_s,passes" > "$OUTPUT_CSV"

for P in 5 10 15 20 25 30 50 100 200; do
    echo "[sweep] p=$P ..."
    SPECTRAL_RSVD_OVERSAMPLING="$P" \
        cargo nextest run \
            --features testing \
            --test test_exp_rsvd_sweep \
            production_sweep \
            --no-capture \
            2>&1 \
        | grep '^SWEEP_ROW,' \
        | sed 's/^SWEEP_ROW,//' \
        >> "$OUTPUT_CSV"
done

echo "[sweep] Done. Results in $OUTPUT_CSV"
