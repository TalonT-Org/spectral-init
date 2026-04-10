#!/usr/bin/env bash
set -euo pipefail

# -- Path anchoring -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
DATA_DIR="$RESEARCH_DIR/data/merfish"
RESULTS_RAW="$RESEARCH_DIR/results/raw"
RESULTS_ANALYSIS="$RESEARCH_DIR/results/analysis"

# -- Constants (matching utils.py) --------------------------------------------
K=15
SEEDS_MAX=9          # seeds 0..9
REPS=5
WARMUP=1
M_VALUES_10K=(500 1000 2000 3000 5000 7500 10000)
M_VALUES_50K=(1000 2000 5000 10000 20000 35000 50000)

# -- Build --------------------------------------------------------------------
echo "=== [1/7] Building tw_subsample_experiment ==="
(cd "$PROJECT_ROOT" && cargo build --release --example tw_subsample_experiment --features cli)
BIN="$PROJECT_ROOT/target/release/examples/tw_subsample_experiment"

# -- Preflight ----------------------------------------------------------------
echo "=== [2/7] Preflight check ==="
"$BIN" --mode preflight --data-dir "$DATA_DIR"
# Binary exits 1 on failure with "PREFLIGHT FAILED: ..." -> set -e aborts

# -- Rayon determinism gate ---------------------------------------------------
echo "=== [3/7] Rayon determinism check ==="
# Run exact mode twice on n=10K, compare T values
DETERM_1=$(mktemp)
DETERM_2=$(mktemp)
"$BIN" --mode exact \
    --x "$DATA_DIR/merfish_n10k_x.npy" --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --reps 1 --warmup 0 --output "$DETERM_1"
"$BIN" --mode exact \
    --x "$DATA_DIR/merfish_n10k_x.npy" --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --reps 1 --warmup 0 --output "$DETERM_2"
# Extract t_exact values and compare via Python one-liner
python3 -c "
import json, sys
t1 = json.load(open(sys.argv[1]))['t_exact']
t2 = json.load(open(sys.argv[2]))['t_exact']
delta = abs(t1 - t2)
print(f'Determinism check: |T1-T2| = {delta:.2e}')
if delta > 1e-6:
    print(f'FATAL: Rayon non-determinism detected: T1={t1}, T2={t2}', file=sys.stderr)
    sys.exit(1)
" "$DETERM_1" "$DETERM_2"
rm -f "$DETERM_1" "$DETERM_2"

# -- Sanity checks ------------------------------------------------------------
echo "=== [4/7] Sanity checks ==="
mkdir -p "$RESULTS_RAW"
"$BIN" --mode sanity \
    --x "$DATA_DIR/merfish_n10k_x.npy" --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --m 10000 --output "$RESULTS_RAW/sanity_n10000.json"
echo "  -> sanity_n10000.json"

"$BIN" --mode sanity \
    --x "$DATA_DIR/merfish_n50k_x.npy" --y "$DATA_DIR/merfish_n50k_y.npy" \
    --k "$K" --m 50000 --output "$RESULTS_RAW/sanity_n50000.json"
echo "  -> sanity_n50000.json"

# -- Exact baselines ----------------------------------------------------------
echo "=== [5/7] Exact baselines ==="
"$BIN" --mode exact \
    --x "$DATA_DIR/merfish_n10k_x.npy" --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --reps "$REPS" --warmup "$WARMUP" --output "$RESULTS_RAW/exact_n10000.json"
echo "  -> exact_n10000.json (n=10K)"

"$BIN" --mode exact \
    --x "$DATA_DIR/merfish_n50k_x.npy" --y "$DATA_DIR/merfish_n50k_y.npy" \
    --k "$K" --reps "$REPS" --warmup "$WARMUP" --output "$RESULTS_RAW/exact_n50000.json"
echo "  -> exact_n50000.json (n=50K)"

# -- Subsample trials ---------------------------------------------------------
echo "=== [6/7] Subsample trials ==="
trial_count=0

# n=10K first (faster), then n=50K; m ascending within each n
for n in 10000 50000; do
    if [[ "$n" == "10000" ]]; then
        label="n10k"
        m_values=("${M_VALUES_10K[@]}")
    else
        label="n50k"
        m_values=("${M_VALUES_50K[@]}")
    fi

    x_path="$DATA_DIR/merfish_${label}_x.npy"
    y_path="$DATA_DIR/merfish_${label}_y.npy"

    for m in "${m_values[@]}"; do
        for seed in $(seq 0 "$SEEDS_MAX"); do
            out="$RESULTS_RAW/trial_n${n}_m${m}_s${seed}.json"
            "$BIN" --mode subsample \
                --x "$x_path" --y "$y_path" \
                --k "$K" --m "$m" --seed "$seed" \
                --reps "$REPS" --warmup "$WARMUP" \
                --output "$out"
            trial_count=$((trial_count + 1))
            echo "  [$trial_count/140] trial_n${n}_m${m}_s${seed}.json"
        done
    done
done

# -- Analysis -----------------------------------------------------------------
echo "=== [7/7] Running analysis ==="
micromamba run -n subsampled-tw-rust \
    python "$RESEARCH_DIR/scripts/analyze_results.py"

echo "=== Experiment complete ==="
echo "Verdicts: $(python3 -c 'import json,sys; print(json.load(sys.stdin)["overall"])' < "$RESULTS_ANALYSIS/verdicts.json")"
