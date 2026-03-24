#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Phase 1: Generate baselines ==="
python "$SCRIPT_DIR/generate_umap_comparisons.py" --phase baseline

echo ""
echo "=== Rust Export: Generate spectral init coordinates ==="
cd "$PROJECT_ROOT"
cargo test --test export_rust_init --features testing -- --ignored --nocapture

echo ""
echo "=== Phase 2: Compare and evaluate ==="
python "$SCRIPT_DIR/generate_umap_comparisons.py" --phase compare

echo ""
echo "=== Summary ==="
python -c "
import json, glob
for f in sorted(glob.glob('$SCRIPT_DIR/output/*_metrics.json')):
    m = json.load(open(f))
    status = m['pass_fail']['overall']
    print(f\"  {m['dataset']:25s} {status}\")
"
