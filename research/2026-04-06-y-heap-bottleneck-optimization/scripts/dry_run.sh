#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RESULTS_DIR="$SCRIPT_DIR/../results"

mkdir -p "$RESULTS_DIR/profiler"

echo "=== DRY RUN: y-heap bottleneck optimization ==="

# ── Step 1: Data gate ─────────────────────────────────────────────────────────
if [[ -f "$DATA_DIR/gaussian_n1000_x.npy" ]]; then
    echo "[Step 1] n=1000 data already present, skipping generation."
else
    echo "[Step 1] Generating data (all sizes)..."
    python3 "$SCRIPT_DIR/gen_data.py" --out-dir "$DATA_DIR"
fi

# ── Step 2: Criterion fast run (all 4 variants, n=1000 only) ──────────────────
echo "[Step 2] Running Criterion (n=1000, fast config)..."
cargo bench \
    --bench y_heap_variants_bench \
    --features testing \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    -- n/1000 --sample-size 3 --warm-up-time 2 --measurement-time 5

# ── Step 3: Build profiler and run baseline at n=1000 ─────────────────────────
echo "[Step 3] Building profiler (features: cli)..."
cargo build --release \
    --features cli \
    --manifest-path "$REPO_ROOT/Cargo.toml"

echo "[Step 3] Running profiler baseline at n=1000..."
"$REPO_ROOT/target/release/tw_profiler" \
    --x "$DATA_DIR/gaussian_n1000_x.npy" \
    --y "$DATA_DIR/gaussian_n1000_y.npy" \
    --k 15 \
    --iters 2 \
    --warmup 1 \
    --variant baseline \
    --output "$RESULTS_DIR/profiler/profiler_baseline_n1000.json"

# ── Step 4: Verification gates ────────────────────────────────────────────────
FAIL=0

echo "[Step 4] Checking Criterion JSON files..."
for variant in baseline heap_reuse flat_partial flat_simd; do
    JSON="$REPO_ROOT/target/criterion/y_heap_${variant}/n/1000/estimates.json"
    if [[ -f "$JSON" ]]; then
        echo "  OK: $JSON"
    else
        echo "  FAIL: missing $JSON" >&2
        FAIL=1
    fi
done

echo "[Step 4] Checking for NaN in profiler output..."
SCORE="$(python3 -c "
import json, math, sys
with open('$RESULTS_DIR/profiler/profiler_baseline_n1000.json') as f:
    d = json.load(f)
score = d.get('score', None)
if score is None or (isinstance(score, float) and math.isnan(score)):
    print('NaN', file=sys.stderr)
    sys.exit(1)
print(score)
")" || { echo "  FAIL: NaN or missing score in profiler output" >&2; FAIL=1; }
[[ "$FAIL" -eq 0 ]] && echo "  OK: score=$SCORE"

echo "[Step 4] Running correctness tests..."
if cargo test \
    --features testing \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    -- t_tw_heap_reuse 2>&1 | tail -5; then
    echo "  OK: correctness tests passed"
else
    echo "  FAIL: correctness tests failed" >&2
    FAIL=1
fi

echo "[Step 4] Checking profiler JSON output..."
if [[ -f "$RESULTS_DIR/profiler/profiler_baseline_n1000.json" ]]; then
    echo "  OK: profiler JSON present"
else
    echo "  FAIL: profiler JSON missing" >&2
    FAIL=1
fi

# ── Exit gate ─────────────────────────────────────────────────────────────────
if [[ "$FAIL" -ne 0 ]]; then
    echo "DRY RUN FAILED" >&2
    exit 1
fi

echo "DRY RUN PASSED"
