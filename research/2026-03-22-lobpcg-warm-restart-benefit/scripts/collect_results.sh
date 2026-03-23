#!/usr/bin/env bash
# Extract METRIC lines from raw.txt and format as a markdown table.
# Usage: bash collect_results.sh  (run from the research directory)
# Output: writes results/metrics.md

set -euo pipefail
cd "$(git rev-parse --show-toplevel)/research/2026-03-22-lobpcg-warm-restart-benefit"

RAW="results/raw.txt"
OUT="results/metrics.md"

if [ ! -f "$RAW" ]; then
    echo "ERROR: $RAW not found. Run run_experiment.sh first." >&2
    exit 1
fi

{
echo "# Warm Restart Benefit — Metrics"
echo ""
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "| graph | seed | restart_count | max_residual_with_restart | max_residual_no_restart | converged | improvement_ratio |"
echo "|-------|------|---------------|--------------------------|------------------------|-----------|------------------|"

grep '^METRIC ' "$RAW" | while read -r line; do
    graph=$(echo "$line"   | sed 's/.*graph=\([^ ]*\).*/\1/')
    seed=$(echo "$line"    | sed 's/.*seed=\([^ ]*\).*/\1/')
    rc=$(echo "$line"      | sed 's/.*restart_count=\([^ ]*\).*/\1/')
    res_wr=$(echo "$line"  | sed 's/.*max_residual_with_restart=\([^ ]*\).*/\1/')
    res_nr=$(echo "$line"  | sed 's/.*max_residual_no_restart=\([^ ]*\).*/\1/')
    conv=$(echo "$line"    | sed 's/.*converged=\([^ ]*\).*/\1/')
    ratio=$(echo "$line"   | sed 's/.*improvement_ratio=\([^ ]*\).*/\1/')
    echo "| $graph | $seed | $rc | $res_wr | $res_nr | $conv | $ratio |"
done
} > "$OUT"

echo "Metrics written to $OUT"
cat "$OUT"
