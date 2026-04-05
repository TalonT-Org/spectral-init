#!/usr/bin/env bash
# Dry Run Validation (groupF)
# Validates H5 runner, Criterion bench, profiling binary, and analysis script
# with minimal computation. All 4 checks must PASS before proceeding to groupG.
# Exits 0 if all pass, 1 if any fail.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
DATA_GAUSSIAN="$EXP_DIR/data/gaussian"
DATA_MERFISH="$EXP_DIR/data/merfish"
RESULTS_H5="$EXP_DIR/results/h5"
RESULTS_CRIT="$EXP_DIR/results/criterion"
RESULTS_STEP="$EXP_DIR/results/step_timing"
BENCH_FILE="$REPO_ROOT/benches/tw_baseline_bench.rs"
PASS_COUNT=0
FAIL_COUNT=0

# Cleanup trap: restore bench file if the script aborts mid-patch
BENCH_PATCHED=0
cleanup() {
    if [[ "$BENCH_PATCHED" -eq 1 ]]; then
        echo "  [trap] Restoring bench file..."
        (cd "$REPO_ROOT" && git checkout -- benches/tw_baseline_bench.rs) || true
    fi
}
trap cleanup EXIT

pass() { echo "  PASS: $1"; ((PASS_COUNT++)) || true; }
fail() { echo "  FAIL: $1"; ((FAIL_COUNT++)) || true; }

# ── Prerequisites ──────────────────────────────────────────────────────────────
echo "=== Prerequisites ==="

if ! command -v cargo-criterion &>/dev/null; then
    echo "ERROR: cargo-criterion not found. Install: cargo install cargo-criterion" >&2
    exit 1
fi
echo "  cargo-criterion: $(cargo-criterion --version 2>&1 | head -1)"

if ! python3 -c "import numpy, scipy, statsmodels" 2>/dev/null; then
    echo "ERROR: Python environment missing numpy/scipy/statsmodels" >&2
    exit 1
fi
echo "  Python env OK"

for f in \
    "$DATA_MERFISH/merfish_n10k_x.npy" \
    "$DATA_MERFISH/merfish_n10k_y.npy" \
    "$DATA_GAUSSIAN/gaussian_n1000_x.npy" \
    "$DATA_GAUSSIAN/gaussian_n1000_y.npy"; do
    [[ -f "$f" ]] || { echo "ERROR: missing data file: $f" >&2; exit 1; }
done
echo "  Data files OK"

# ── Build ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Build: tw_approx_runner ==="
(cd "$REPO_ROOT" && cargo build --release --features cli --no-default-features --bin tw_approx_runner)

echo ""
echo "=== Build: tw_profiler (features: cli,profiling) ==="
(cd "$REPO_ROOT" && cargo build --release --features cli,profiling --no-default-features --bin tw_profiler)

# ── REQ-P5-001: H5 Dry Run ─────────────────────────────────────────────────────
echo ""
echo "=== REQ-P5-001: H5 Dry Run (MERFISH n=10K, seed=42, m=5000) ==="
OUT_H5="$RESULTS_H5/h5_dry_run.json"

if "$REPO_ROOT/target/release/tw_approx_runner" \
    --x "$DATA_MERFISH/merfish_n10k_x.npy" \
    --y "$DATA_MERFISH/merfish_n10k_y.npy" \
    --k 15 --sample 5000 --seed 42 \
    --output "$OUT_H5"; then
    if python3 -c "
import json, sys
d = json.load(open('${OUT_H5}'))
fields = ('delta', 'wall_exact_s', 'wall_approx_s')
ok = all(isinstance(d.get(f), (int, float)) for f in fields)
if ok:
    print('  delta={:.6f}, wall_exact={:.4f}s, wall_approx={:.4f}s'.format(
        d['delta'], d['wall_exact_s'], d['wall_approx_s']))
else:
    missing = [f for f in fields if not isinstance(d.get(f), (int, float))]
    print('  Missing or non-numeric fields:', missing)
sys.exit(0 if ok else 1)
"; then
        pass "REQ-P5-001 h5_dry_run.json has delta, wall_exact_s, wall_approx_s"
    else
        fail "REQ-P5-001 h5_dry_run.json missing or non-numeric field(s)"
    fi
else
    fail "REQ-P5-001 tw_approx_runner exited non-zero"
fi

# ── REQ-P5-002: Criterion Dry Run ─────────────────────────────────────────────
echo ""
echo "=== REQ-P5-002: Criterion Dry Run (tw_baseline_bench, n=1K, sample_size=5) ==="

BENCH_PATCHED=1
python3 - "$BENCH_FILE" <<'PYEOF'
import sys
path = sys.argv[1]
content = open(path).read()
replacements = [
    ('group.sample_size(100)',                           'group.sample_size(10)'),
    ('group.warm_up_time(Duration::from_secs(10))',      'group.warm_up_time(Duration::from_secs(1))'),
    ('group.measurement_time(Duration::from_secs(60))',  'group.measurement_time(Duration::from_secs(5))'),
    # n=100K overrides
    ('group.sample_size(63)',                            'group.sample_size(10)'),
    ('group.warm_up_time(Duration::from_secs(30))',      'group.warm_up_time(Duration::from_secs(1))'),
    ('group.measurement_time(Duration::from_secs(1500))','group.measurement_time(Duration::from_secs(5))'),
]
for old, new in replacements:
    content = content.replace(old, new)
open(path, 'w').write(content)
print('  tw_baseline_bench.rs patched for dry run.')
PYEOF

DRY_CRIT_OUT="$RESULTS_CRIT/criterion_dry_run_output.json"
> "$DRY_CRIT_OUT"   # truncate / create
CRIT_OK=0
if (cd "$REPO_ROOT" && \
    cargo criterion --bench tw_baseline_bench --message-format=json --features cli \
        -- "baseline/1000") >> "$DRY_CRIT_OUT"; then
    CRIT_OK=1
fi

# Restore bench file immediately (trap is backup only)
(cd "$REPO_ROOT" && git checkout -- benches/tw_baseline_bench.rs)
BENCH_PATCHED=0
echo "  tw_baseline_bench.rs restored."

if [[ "$CRIT_OK" -eq 1 ]]; then
    if python3 -c "
import json, sys
lines = [l.strip() for l in open('${DRY_CRIT_OUT}') if l.strip()]
records = []
for l in lines:
    try: records.append(json.loads(l))
    except json.JSONDecodeError: pass
bcs = [r for r in records if r.get('reason') == 'benchmark-complete' and 'typical' in r]
sys.exit(0 if bcs else 1)
"; then
        pass "REQ-P5-002 Criterion JSON-lines has benchmark-complete with typical field"
    else
        fail "REQ-P5-002 Criterion output missing benchmark-complete or typical field"
    fi
else
    fail "REQ-P5-002 cargo criterion exited non-zero"
fi

# ── REQ-P5-003: Profiling Dry Run ─────────────────────────────────────────────
echo ""
echo "=== REQ-P5-003: Profiling Dry Run (baseline, --warmup 1 --iters 2, gaussian n=1K) ==="
OUT_PROF="$RESULTS_STEP/dry_run.json"

if "$REPO_ROOT/target/release/tw_profiler" \
    --x "$DATA_GAUSSIAN/gaussian_n1000_x.npy" \
    --y "$DATA_GAUSSIAN/gaussian_n1000_y.npy" \
    --k 15 \
    --warmup 1 \
    --iters 2 \
    --variant baseline \
    --output "$OUT_PROF"; then
    if python3 -c "
import json, sys
d = json.load(open('${OUT_PROF}'))
keys = ('x_dist', 'x_sort', 'rank_scatter', 'x_knn_set', 'y_heap', 'penalty')
recs = d.get('step_times_ns', [])
if not recs:
    print('  step_times_ns is empty (profiling feature not active?)')
    sys.exit(1)
zero_recs = [r for r in recs if sum(r.get(k, 0) for k in keys) == 0]
if zero_recs:
    print(f'  {len(zero_recs)} iteration(s) have all-zero step times (R6 fix not active?)')
    sys.exit(1)
totals = [sum(r.get(k, 0) for k in keys) for r in recs]
print(f'  {len(recs)} iter(s), total ns: {totals}')
sys.exit(0)
"; then
        pass "REQ-P5-003 dry_run.json has non-zero step_times_ns for all iterations"
    else
        fail "REQ-P5-003 dry_run.json step_times_ns missing or all-zero (check R6 fix)"
    fi
else
    fail "REQ-P5-003 tw_profiler exited non-zero"
fi

# ── REQ-P5-004: Analysis Dry Run ──────────────────────────────────────────────
echo ""
echo "=== REQ-P5-004: Analysis Dry Run ==="
REPORT="$EXP_DIR/results/analysis/analysis_report.md"

if (cd "$EXP_DIR" && python3 scripts/analyze_clean.py --dry-run); then
    if [[ -f "$REPORT" ]]; then
        pass "REQ-P5-004 analysis_report.md produced and analyze_clean.py exited 0"
    else
        fail "REQ-P5-004 analyze_clean.py exited 0 but analysis_report.md not found"
    fi
else
    fail "REQ-P5-004 analyze_clean.py --dry-run exited non-zero"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Dry Run Summary: $PASS_COUNT PASS, $FAIL_COUNT FAIL ==="
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "ABORT: Resolve all failures before proceeding to groupG." >&2
fi
[[ "$FAIL_COUNT" -eq 0 ]]
