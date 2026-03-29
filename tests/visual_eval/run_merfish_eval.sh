#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Phase 1: Generate MERFISH baselines ==="
python "$SCRIPT_DIR/generate_merfish_comparisons.py" --phase baseline

echo ""
echo "=== Phase 2: Rust Export — MERFISH spectral init ==="
cd "$PROJECT_ROOT"
cargo nextest run --profile merfish-eval --run-ignored all --features testing

echo ""
echo "=== Phase 3: Python comparison (Python vs Rust vs Random) ==="
python "$SCRIPT_DIR/generate_merfish_comparisons.py" --phase compare

echo ""
echo "=== Summary ==="
python -c "
import json, glob, sys
script_dir = sys.argv[1]
results = sorted(glob.glob(script_dir + '/output/merfish_*_metrics.json'))
if not results:
    print('  No metrics files found in ' + script_dir + '/output/')
else:
    for f in results:
        with open(f) as fh:
            m = json.load(fh)
        status = m.get('pass_fail', {}).get('overall', 'UNKNOWN')
        print(f\"  {m['dataset']:30s} {status}\")
" "$SCRIPT_DIR"
