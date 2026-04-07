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
    EXTRA_FLAGS=(--sample-size 10 --warm-up-time 1 --measurement-time 2)
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
mkdir -p "$PROJECT_ROOT/temp"

# ---------------------------------------------------------------------------
# Write run_metadata.json
# ---------------------------------------------------------------------------
RUST_CHANNEL="$(cd "$PROJECT_ROOT" && rustup show active-toolchain 2>/dev/null | awk '{print $1}' || echo "unknown")"
TIMESTAMP="$(date -Iseconds)"
KIDDO_VERSION="$(grep -A1 '^name = "kiddo"' "$PROJECT_ROOT/Cargo.lock" 2>/dev/null | grep '^version' | head -1 | sed 's/version = "\(.*\)"/\1/' || echo "unknown")"

cat > "$RESEARCH_DIR/results/run_metadata.json" <<EOF
{
  "experiment": "kdtree-y-knn-trustworthiness",
  "kiddo_version": "$KIDDO_VERSION",
  "rust_channel": "$RUST_CHANNEL",
  "rayon_num_threads": $RAYON_NUM_THREADS,
  "timestamp": "$TIMESTAMP",
  "dry_run": $DRY_RUN
}
EOF

echo "[run_criterion] metadata written (rust_channel=$RUST_CHANNEL, threads=$RAYON_NUM_THREADS, dry_run=$DRY_RUN)"

# ---------------------------------------------------------------------------
# Main loop — run cargo criterion for each variant × dist × n × rep
#
# cargo-criterion 1.1.0 with criterion 0.5 uses CBOR storage, not JSON.
# We use --message-format json to get machine-readable output on stdout.
# The filter "${group}/${n}" is precise enough to avoid substring matches
# (e.g. "flat_simd_uniform_n1000/1000" does NOT match "n10000/10000").
# ---------------------------------------------------------------------------
LOG_ENTRIES=()

for variant in "${VARIANTS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            group="${variant}_${dist}_n${n}"
            bench_id="${group}/${n}"
            for rep in $(seq 1 "$REPS"); do
                echo "[run_criterion] variant=$variant dist=$dist n=$n rep=$rep group=$group"

                json_tmp="$(mktemp "$PROJECT_ROOT/temp/criterion_json_XXXXXX.jsonl")"
                status="completed"

                # stdout → JSON capture; stderr → terminal (progress/warnings)
                if (cd "$PROJECT_ROOT" && cargo criterion \
                        --bench trustworthiness_bench \
                        --features testing \
                        --message-format json \
                        -- "$bench_id" "${EXTRA_FLAGS[@]}" > "$json_tmp"); then

                    dst="$RESEARCH_DIR/results/criterion/${group}_rep${rep}.json"

                    # Extract the benchmark-complete line for our exact bench_id
                    result_line=$(python3 -c "
import json, sys
target = sys.argv[1]
jsonl = sys.argv[2]
with open(jsonl) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get('reason') == 'benchmark-complete' and d.get('id') == target:
                print(line)
                break
        except Exception:
            pass
" "$bench_id" "$json_tmp" 2>/dev/null || true)

                    if [[ -n "$result_line" ]]; then
                        echo "$result_line" > "$dst"
                        echo "[run_criterion]   -> saved estimates to $dst"
                    else
                        echo "[run_criterion]   WARNING: no benchmark-complete for '$bench_id' in JSON output" >&2
                        status="missing_estimates"
                    fi
                else
                    echo "[run_criterion]   ERROR: cargo criterion failed for $group rep$rep" >&2
                    status="failed"
                fi

                rm -f "$json_tmp"
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
