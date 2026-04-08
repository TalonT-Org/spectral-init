#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# ---------------------------------------------------------------------------
# Thread count (inherit from environment or detect)
# ---------------------------------------------------------------------------
export RAYON_NUM_THREADS
if [[ -z "${RAYON_NUM_THREADS:-}" ]]; then
    RAYON_NUM_THREADS="$(nproc)"
fi

# ---------------------------------------------------------------------------
# Mode-dependent values
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    N_VALUES=(1000)
    ITERS=3
else
    N_VALUES=(1000 5000 10000 50000 75000 100000)
    ITERS=30
fi
VARIANTS=(flat_simd kdtree)
DISTRIBUTIONS=(uniform gauss)
WARMUP=2

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
mkdir -p "$RESEARCH_DIR/results/profiler"
mkdir -p "$PROJECT_ROOT/temp"

# ---------------------------------------------------------------------------
# Main loop — one fresh cargo run per variant × dist × n
# ---------------------------------------------------------------------------
LOG_ENTRIES=()

for variant in "${VARIANTS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            stderr_file="$(mktemp "$PROJECT_ROOT/temp/tw_profiler_stderr_XXXXXX.txt")"
            out="$RESEARCH_DIR/results/profiler/${variant}_n${n}_${dist}.json"

            echo "[run_profiler] variant=$variant dist=$dist n=$n iters=$ITERS"

            status="completed"
            if (cd "$PROJECT_ROOT" && \
                    RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
                    cargo run --bin tw_profiler \
                        --features profiling,cli --release -- \
                        --n "$n" --dist "$dist" --variant "$variant" \
                        --iters "$ITERS" --warmup "$WARMUP" \
                        --stderr-capture "$stderr_file" \
                        --output "$out"); then
                echo "[run_profiler]   -> written: $out"
            else
                echo "[run_profiler]   ERROR: tw_profiler failed for variant=$variant dist=$dist n=$n" >&2
                status="failed"
            fi

            rm -f "$stderr_file"

            LOG_ENTRIES+=("{\"variant\": \"$variant\", \"dist\": \"$dist\", \"n\": $n, \"status\": \"$status\"}")
        done
    done
done

# ---------------------------------------------------------------------------
# Append profiler section to results/run_log.json
# ---------------------------------------------------------------------------
PROFILER_JSON_ARRAY="["
for i in "${!LOG_ENTRIES[@]}"; do
    if [[ $i -gt 0 ]]; then
        PROFILER_JSON_ARRAY+=","
    fi
    PROFILER_JSON_ARRAY+="${LOG_ENTRIES[$i]}"
done
PROFILER_JSON_ARRAY+="]"

LOG_PATH="$RESEARCH_DIR/results/run_log.json"

if command -v python3 &>/dev/null; then
    python3 -c "
import json, sys
path = '$LOG_PATH'
try:
    log = json.loads(open(path).read())
except Exception:
    log = {}
log['profiler'] = json.loads(sys.argv[1])
open(path, 'w').write(json.dumps(log, indent=2))
" "$PROFILER_JSON_ARRAY"
else
    # Fallback: write separate file
    echo "[run_profiler] WARNING: python3 not found; writing run_log_profiler.json separately" >&2
    cat > "$RESEARCH_DIR/results/run_log_profiler.json" <<EOF
{
  "profiler": $PROFILER_JSON_ARRAY
}
EOF
fi

echo "[run_profiler] log updated with ${#LOG_ENTRIES[@]} entries"
echo "[run_profiler] DONE"
