#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# groupD: Dry-run validation — 3-trial subset to verify end-to-end pipeline
#
# Runs 3 carefully chosen trials (2 subsample + 1 sanity), then validates
# the 4 acceptance criteria that confirm binary, orchestration, and analysis
# scripts interoperate correctly.
#
# Acceptance criteria:
#   AC-1: JSON field completeness (all 3 JSONs have 17 expected fields)
#   AC-2: Sanity trial precision (abs_delta_t < 1e-10)
#   AC-3: Analysis pipeline produces verdicts.json with H1-H6
#   AC-4: Plots (3 PNGs) and summary.md exist and are non-empty
# =============================================================================

# -- Path anchoring -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
DATA_DIR="$RESEARCH_DIR/data/merfish"
RESULTS_RAW="$RESEARCH_DIR/results/raw"
RESULTS_ANALYSIS="$RESEARCH_DIR/results/analysis"
BIN="$PROJECT_ROOT/target/release/examples/tw_subsample_experiment"

# -- Constants ----------------------------------------------------------------
K=15
REPS=5
WARMUP=1

# -- Track acceptance results -------------------------------------------------
AC1="PENDING"
AC2="PENDING"
AC3="PENDING"
AC4="PENDING"

# =============================================================================
# Step 1: Build the Binary
# =============================================================================
echo "=== [Step 1/11] Building tw_subsample_experiment ==="
(cd "$PROJECT_ROOT" && cargo build --release --example tw_subsample_experiment --features cli)

if [[ ! -x "$BIN" ]]; then
    echo "FATAL: Binary not found at $BIN"
    exit 1
fi
echo "  -> Binary built: $BIN"

# =============================================================================
# Step 2: Preflight Check
# =============================================================================
echo ""
echo "=== [Step 2/11] Preflight check ==="
"$BIN" --mode preflight --data-dir "$DATA_DIR"
echo "  -> Preflight passed"

# =============================================================================
# Step 3: Trial 1 — Subsample (n=10K, m=2000, seed=0)
# =============================================================================
echo ""
echo "=== [Step 3/11] Trial 1: subsample n=10K m=2000 seed=0 ==="
mkdir -p "$RESULTS_RAW"
"$BIN" --mode subsample \
    --x "$DATA_DIR/merfish_n10k_x.npy" \
    --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --m 2000 --seed 0 --reps "$REPS" --warmup "$WARMUP" \
    --output "$RESULTS_RAW/trial_n10000_m2000_s0.json"
echo "  -> trial_n10000_m2000_s0.json"

# =============================================================================
# Step 4: Trial 2 — Sanity (n=10K, m=10000)
# =============================================================================
echo ""
echo "=== [Step 4/11] Trial 2: sanity n=10K m=10000 ==="
"$BIN" --mode sanity \
    --x "$DATA_DIR/merfish_n10k_x.npy" \
    --y "$DATA_DIR/merfish_n10k_y.npy" \
    --k "$K" --m 10000 \
    --output "$RESULTS_RAW/sanity_n10000.json"
echo "  -> sanity_n10000.json"

# =============================================================================
# Step 5: Trial 3 — Subsample (n=50K, m=2000, seed=0)
# =============================================================================
echo ""
echo "=== [Step 5/11] Trial 3: subsample n=50K m=2000 seed=0 ==="
"$BIN" --mode subsample \
    --x "$DATA_DIR/merfish_n50k_x.npy" \
    --y "$DATA_DIR/merfish_n50k_y.npy" \
    --k "$K" --m 2000 --seed 0 --reps "$REPS" --warmup "$WARMUP" \
    --output "$RESULTS_RAW/trial_n50000_m2000_s0.json"
echo "  -> trial_n50000_m2000_s0.json"

# =============================================================================
# Step 6: Verify AC-1 — JSON Field Completeness
# =============================================================================
echo ""
echo "=== [Step 6/11] Verifying AC-1: JSON field completeness ==="
python3 -c "
import json, sys

REQUIRED = {'n','m','k','seed','mode','t_exact','t_sub','abs_delta_t',
            'wall_exact_ms','wall_sub_ms','warmup_exact_ms','warmup_sub_ms',
            'cpu_model','core_count','rust_version','git_commit'}

ok = True
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    missing = REQUIRED - set(d.keys())
    extra = set(d.keys()) - REQUIRED
    if missing:
        print(f'FAIL: {path}: missing keys {missing}')
        ok = False
    else:
        print(f'OK: {path} (mode={d[\"mode\"]}, n={d[\"n\"]}, m={d[\"m\"]})')
    if extra:
        print(f'  note: extra keys {extra}')
if not ok:
    sys.exit(1)
print('AC-1 PASS')
" "$RESULTS_RAW/trial_n10000_m2000_s0.json" \
  "$RESULTS_RAW/sanity_n10000.json" \
  "$RESULTS_RAW/trial_n50000_m2000_s0.json"
AC1="PASS"

# =============================================================================
# Step 7: Verify AC-2 — Sanity Precision
# =============================================================================
echo ""
echo "=== [Step 7/11] Verifying AC-2: sanity precision ==="
python3 -c "
import json
d = json.load(open('$RESULTS_RAW/sanity_n10000.json'))
adt = d['abs_delta_t']
print(f'abs_delta_t = {adt:.2e}')
if adt >= 1e-10:
    print(f'FAIL: abs_delta_t={adt} >= 1e-10')
    exit(1)
print('AC-2 PASS')
"
AC2="PASS"

# =============================================================================
# Step 8: Run Analysis Pipeline
# =============================================================================
echo ""
echo "=== [Step 8/11] Running analysis pipeline ==="
micromamba run -n subsampled-tw-rust \
    python "$RESEARCH_DIR/scripts/analyze_results.py"
echo "  -> Analysis complete"

# =============================================================================
# Step 9: Verify AC-3 — verdicts.json Structure
# =============================================================================
echo ""
echo "=== [Step 9/11] Verifying AC-3: verdicts.json with H1-H6 ==="
python3 -c "
import json
v = json.load(open('$RESULTS_ANALYSIS/verdicts.json'))
hyps = v['hypotheses']
expected = {'H1','H2','H3','H4','H5','H6'}
actual = set(hyps.keys())
missing = expected - actual
if missing:
    print(f'FAIL: Missing hypothesis keys: {missing}')
    exit(1)
for k in sorted(expected):
    print(f'  {k}: {hyps[k][\"verdict\"]}')
print('AC-3 PASS')
"
AC3="PASS"

# =============================================================================
# Step 10: Verify AC-4 — Plots and Summary Exist
# =============================================================================
echo ""
echo "=== [Step 10/11] Verifying AC-4: plots and summary exist ==="
ac4_ok=true
for f in error_vs_m.png speedup_vs_m.png variance_decay.png summary.md; do
    if [ -s "$RESULTS_ANALYSIS/$f" ]; then
        echo "  OK: $f ($(wc -c < "$RESULTS_ANALYSIS/$f") bytes)"
    else
        echo "  FAIL: $f missing or empty"
        ac4_ok=false
    fi
done
if [[ "$ac4_ok" != "true" ]]; then
    exit 1
fi
AC4="PASS"

# =============================================================================
# Step 11: Final Acceptance Summary
# =============================================================================
echo ""
echo "==========================================="
echo "  groupD Dry Run Validation Summary"
echo "==========================================="
echo "  AC-1 (JSON completeness):     $AC1"
echo "  AC-2 (sanity precision):      $AC2"
echo "  AC-3 (verdicts.json H1-H6):   $AC3"
echo "  AC-4 (plots + summary.md):    $AC4"
echo "==========================================="
echo ""

if [[ "$AC1" == "PASS" && "$AC2" == "PASS" && "$AC3" == "PASS" && "$AC4" == "PASS" ]]; then
    echo "ALL 4 ACCEPTANCE CRITERIA PASS"
    echo "Ready for full 144-trial run via run_experiment.sh"
    exit 0
else
    echo "SOME CRITERIA FAILED — diagnose root cause before proceeding"
    exit 1
fi
