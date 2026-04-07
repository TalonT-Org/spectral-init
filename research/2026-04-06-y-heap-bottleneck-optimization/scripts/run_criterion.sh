#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results/criterion"

# W8 guard: abort if profiling feature is active — would contaminate timing
# Check both explicit env var and cargo feature metadata
if [[ -n "${CARGO_FEATURE_PROFILING:-}" ]]; then
    echo "ERROR: CARGO_FEATURE_PROFILING is set. Benchmark must run without profiling instrumentation." >&2
    exit 1
fi
if cargo metadata --manifest-path "$REPO_ROOT/Cargo.toml" --format-version 1 --no-deps 2>/dev/null \
   | python3 -c "import sys,json; feats=json.load(sys.stdin)['packages'][0].get('features',{}); sys.exit(0 if 'profiling' not in feats.get('default',[]) else 1)" 2>/dev/null; then
    :
else
    echo "ERROR: 'profiling' is in default features. Benchmark must run without profiling instrumentation." >&2
    exit 1
fi

export RAYON_NUM_THREADS
RAYON_NUM_THREADS="$(nproc)"
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

mkdir -p "$RESULTS_DIR"

run_variant() {
    local variant="$1"
    local group="y_heap_${variant}"

    echo "=== Running variant: $variant ==="
    cargo bench \
        --bench y_heap_variants_bench \
        --features testing \
        --manifest-path "$REPO_ROOT/Cargo.toml" \
        -- "$group"

    # Harvest Criterion JSON for each n
    for n in 1000 5000 10000; do
        local src="$REPO_ROOT/target/criterion/${group}/n/${n}/new/estimates.json"
        local dst="$RESULTS_DIR/y_heap_${variant}_n${n}.json"
        if [[ -f "$src" ]]; then
            cp "$src" "$dst"
            echo "  copied: $dst"
        else
            echo "  WARNING: expected JSON not found: $src" >&2
        fi
    done
}

run_variant baseline
sleep 60

run_variant heap_reuse
sleep 60

run_variant flat_partial
sleep 60

run_variant flat_simd

# Snapshot Cargo.lock
cp "$REPO_ROOT/Cargo.lock" "$SCRIPT_DIR/../results/Cargo.lock.snapshot"
echo "Cargo.lock snapshot saved."
echo "=== run_criterion.sh complete ==="
