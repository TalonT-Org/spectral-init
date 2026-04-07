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
# Thread count
# ---------------------------------------------------------------------------
export RAYON_NUM_THREADS
RAYON_NUM_THREADS="$(nproc)"

# ---------------------------------------------------------------------------
# Mode-dependent values
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    N_VALUES=(1000)
    EXTRA_FLAGS=(--sample-size 2 --warm-up-time 1 --measurement-time 2)
else
    N_VALUES=(1000 5000 10000 50000 75000 100000)
    EXTRA_FLAGS=(--measurement-time 10)
fi
VARIANTS=(flat_simd kdtree)
DISTRIBUTIONS=(uniform gauss)
REPS=3

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
mkdir -p "$RESEARCH_DIR/results/criterion"

# ---------------------------------------------------------------------------
# Write run_metadata.json
# ---------------------------------------------------------------------------
RUST_CHANNEL="$(cd "$PROJECT_ROOT" && rustup show active-toolchain 2>/dev/null | awk '{print $1}' || echo "unknown")"
TIMESTAMP="$(date -Iseconds)"

cat > "$RESEARCH_DIR/results/run_metadata.json" <<EOF
{
  "experiment": "kdtree-y-knn-trustworthiness",
  "kiddo_version": "5.3.0",
  "rust_channel": "$RUST_CHANNEL",
  "rayon_num_threads": $RAYON_NUM_THREADS,
  "timestamp": "$TIMESTAMP",
  "dry_run": $DRY_RUN
}
EOF

echo "[run_criterion] metadata written (rust_channel=$RUST_CHANNEL, threads=$RAYON_NUM_THREADS, dry_run=$DRY_RUN)"

# ---------------------------------------------------------------------------
# Main loop — run cargo criterion for each variant × dist × n × rep
# ---------------------------------------------------------------------------
LOG_ENTRIES=()

for variant in "${VARIANTS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            group="${variant}_${dist}_n${n}"
            for rep in $(seq 1 "$REPS"); do
                echo "[run_criterion] variant=$variant dist=$dist n=$n rep=$rep group=$group"

                status="completed"
                if (cd "$PROJECT_ROOT" && cargo criterion \
                        --bench trustworthiness_bench \
                        --features testing -- "$group" "${EXTRA_FLAGS[@]}"); then
                    src="$PROJECT_ROOT/target/criterion/$group/$n/estimates.json"
                    dst="$RESEARCH_DIR/results/criterion/${group}_rep${rep}.json"
                    if [[ -f "$src" ]]; then
                        cp "$src" "$dst"
                        echo "[run_criterion]   -> copied estimates.json to $dst"
                    else
                        echo "[run_criterion]   WARNING: estimates.json not found at $src" >&2
                        status="missing_estimates"
                    fi
                else
                    echo "[run_criterion]   ERROR: cargo criterion failed for $group rep$rep" >&2
                    status="failed"
                fi

                LOG_ENTRIES+=("{\"variant\": \"$variant\", \"dist\": \"$dist\", \"n\": $n, \"rep\": $rep, \"status\": \"$status\"}")
            done
        done
    done
done

# ---------------------------------------------------------------------------
# Write results/run_log.json (criterion section)
# ---------------------------------------------------------------------------
CRITERION_ARRAY="["
for i in "${!LOG_ENTRIES[@]}"; do
    if [[ $i -gt 0 ]]; then
        CRITERION_ARRAY+=","
    fi
    CRITERION_ARRAY+="${LOG_ENTRIES[$i]}"
done
CRITERION_ARRAY+="]"

cat > "$RESEARCH_DIR/results/run_log.json" <<EOF
{
  "criterion": $CRITERION_ARRAY
}
EOF

echo "[run_criterion] run_log.json written with ${#LOG_ENTRIES[@]} entries"
echo "[run_criterion] DONE"
