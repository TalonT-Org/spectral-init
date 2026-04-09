#!/usr/bin/env bash
# Run baseline benchmarks (current dist_sq_avx2 kernel at d_x=50).
# Must be run from repository root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/../results"

cd "${REPO_ROOT}"

# ── 1. Criterion benchmark (trustworthiness_d50 group) ─────────────────────────
echo "[run_baseline] Running Criterion trustworthiness_d50..."
cargo bench --bench trustworthiness_bench --features testing \
  -- trustworthiness_d50 --save-baseline baseline 2>&1 | tee "${RESULTS}/baseline_criterion.txt"

# ── 1b. Extract Criterion estimates into baseline_criterion.json ────────────────
echo "[run_baseline] Extracting Criterion baseline JSON..."
python3 - <<'PYEOF'
import json, pathlib, sys

criterion_base = pathlib.Path("target/criterion/trustworthiness_d50")
if not criterion_base.exists():
    print("ERROR: target/criterion/trustworthiness_d50 not found — did --save-baseline work?", file=sys.stderr)
    sys.exit(1)

results = {}
for est_file in sorted(criterion_base.glob("n/*/baseline/estimates.json")):
    n_str = est_file.parent.parent.name
    data = json.loads(est_file.read_text())
    # Criterion stores time in nanoseconds (float)
    median_ns = data["median"]["point_estimate"]
    ci_low_ns  = data["median"]["confidence_interval"]["lower_bound"]
    ci_high_ns = data["median"]["confidence_interval"]["upper_bound"]
    results[n_str] = {
        "median_ms": median_ns / 1e6,
        "ci_low_ms": ci_low_ns  / 1e6,
        "ci_high_ms": ci_high_ns / 1e6,
    }

if not results:
    print("ERROR: no n/*/baseline/estimates.json files found", file=sys.stderr)
    sys.exit(1)

out = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/results/baseline_criterion.json")
out.write_text(json.dumps({"trustworthiness_d50": results}, indent=2))
print(f"[extractor] Wrote {out} ({len(results)} n-values: {sorted(int(k) for k in results)})")
PYEOF

# ── 2. Generate temporary .npy inputs for tw_profiler (n=10000, d_x=50) ────────
echo "[run_baseline] Generating profiler .npy inputs..."
python3 - <<'PYEOF'
import numpy as np, pathlib
rng = np.random.RandomState(42)
x = rng.randn(10000, 50).astype(np.float64)
y = rng.randn(10000, 2).astype(np.float64)
tmp = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/data")
tmp.mkdir(parents=True, exist_ok=True)
np.save(str(tmp / "profiler_x_tmp.npy"), x)
np.save(str(tmp / "profiler_y_tmp.npy"), y)
print(f"Wrote {tmp}/profiler_x_tmp.npy and profiler_y_tmp.npy")
PYEOF

# ── 3. Build and run tw_profiler ────────────────────────────────────────────────
echo "[run_baseline] Running tw_profiler (n=10000, d_x=50)..."
cargo run --release --bin tw_profiler --features "cli profiling" -- \
  --x  "research/2026-04-08-x-dist-simd-avx512/data/profiler_x_tmp.npy" \
  --y  "research/2026-04-08-x-dist-simd-avx512/data/profiler_y_tmp.npy" \
  --output "${RESULTS}/baseline_profiler.json" \
  --stderr-capture "${RESULTS}/baseline_stderr.txt" \
  --k 15 --iters 10 --warmup 3

# ── 4. Compute and record x_dist fraction of total profiled runtime ────────────
echo "[run_baseline] Computing x_dist fraction..."
python3 - <<'PYEOF'
import json, pathlib, sys

prof_file = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/results/baseline_profiler.json")
if not prof_file.exists():
    print("ERROR: baseline_profiler.json missing", file=sys.stderr)
    sys.exit(1)

prof = json.loads(prof_file.read_text())
st = prof.get("step_timing", {})

if not st:
    # step_timing absent — profiling feature may not have emitted to stderr
    # Inspect baseline_stderr.txt for [timing:x_dist] lines
    stderr_file = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/results/baseline_stderr.txt")
    if stderr_file.exists():
        content = stderr_file.read_text()
        timing_lines = [l for l in content.splitlines() if l.startswith("[timing:")]
        note = (
            f"step_timing absent from baseline_profiler.json.\n"
            f"Found {len(timing_lines)} [timing:*] lines in baseline_stderr.txt.\n"
            f"First 10:\n" + "\n".join(timing_lines[:10])
        )
    else:
        note = "step_timing absent and baseline_stderr.txt does not exist. --stderr-capture may have failed."
    notes_file = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/results/baseline_timing_notes.txt")
    notes_file.write_text(note)
    print(f"[fraction] WARNING: step_timing absent. Wrote diagnostic to {notes_file}")
    sys.exit(0)

def total(key):
    return sum(st.get(key, [0]))

x_dist  = total("x_dist")
x_sort  = total("x_sort")
y_dist  = total("y_dist")
penalty = total("penalty")
grand   = x_dist + x_sort + y_dist + penalty

def frac(ns):
    return ns / grand if grand > 0 else None

summary = {
    "x_dist_ns_total":  x_dist,
    "x_sort_ns_total":  x_sort,
    "y_dist_ns_total":  y_dist,
    "penalty_ns_total": penalty,
    "x_dist_fraction":  frac(x_dist),
    "step_fractions": {
        "x_dist":  frac(x_dist),
        "x_sort":  frac(x_sort),
        "y_dist":  frac(y_dist),
        "penalty": frac(penalty),
    },
}

out = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/results/baseline_timing_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"[fraction] x_dist fraction: {frac(x_dist):.3f} ({frac(x_dist)*100:.1f}%)")
print(f"[fraction] Wrote {out}")
PYEOF

echo "[run_baseline] Done. Results in ${RESULTS}/"
