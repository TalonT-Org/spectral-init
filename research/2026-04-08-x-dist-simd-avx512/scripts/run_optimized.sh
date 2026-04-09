#!/usr/bin/env bash
# Run benchmarks for a named kernel variant.
# Usage: ./run_optimized.sh <variant_name>
# Example: ./run_optimized.sh avx2_looped
# Must be run from repository root.
set -euo pipefail

VARIANT="${1:?Usage: run_optimized.sh <variant_name>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/../results"

cd "${REPO_ROOT}"

# ── 1. Criterion benchmark ───────────────────────────────────────────────────────
echo "[run_optimized:${VARIANT}] Running Criterion trustworthiness_d50..."
cargo bench --bench trustworthiness_bench --features testing \
  -- trustworthiness_d50 2>&1 | tee "${RESULTS}/${VARIANT}_criterion.txt"

# ── 1b. Extract Criterion estimates into ${VARIANT}_criterion.json ───────────────
echo "[run_optimized:${VARIANT}] Extracting Criterion JSON..."
export VARIANT="${VARIANT}"
export RESULTS="${RESULTS}"
python3 - <<'PYEOF'
import json, os, pathlib, sys

variant = os.environ["VARIANT"]
results = pathlib.Path(os.environ["RESULTS"])
criterion_base = pathlib.Path("target/criterion/trustworthiness_d50")
if not criterion_base.exists():
    print(f"ERROR: {criterion_base} not found", file=sys.stderr)
    sys.exit(1)

data_out = {}
for est_file in sorted(criterion_base.glob("n/*/new/estimates.json")):
    n_str = est_file.parent.parent.name
    data = json.loads(est_file.read_text())
    median_ns  = data["median"]["point_estimate"]
    ci_low_ns  = data["median"]["confidence_interval"]["lower_bound"]
    ci_high_ns = data["median"]["confidence_interval"]["upper_bound"]
    data_out[n_str] = {
        "median_ms":  median_ns  / 1e6,
        "ci_low_ms":  ci_low_ns  / 1e6,
        "ci_high_ms": ci_high_ns / 1e6,
    }

if not data_out:
    print("ERROR: no n/*/new/estimates.json files found", file=sys.stderr)
    sys.exit(1)

out = results / f"{variant}_criterion.json"
out.write_text(json.dumps({"trustworthiness_d50": data_out}, indent=2))
print(f"[extractor] Wrote {out} ({len(data_out)} n-values)")
PYEOF

# ── 2. Generate temporary .npy inputs if not already present ────────────────────
if [[ ! -f "research/2026-04-08-x-dist-simd-avx512/data/profiler_x_tmp.npy" ]]; then
  echo "[run_optimized:${VARIANT}] Generating profiler .npy inputs..."
  python3 - <<'PYEOF'
import numpy as np, pathlib
rng = np.random.RandomState(42)
x = rng.randn(10000, 50).astype(np.float64)
y = rng.randn(10000, 2).astype(np.float64)
tmp = pathlib.Path("research/2026-04-08-x-dist-simd-avx512/data")
tmp.mkdir(parents=True, exist_ok=True)
np.save(str(tmp / "profiler_x_tmp.npy"), x)
np.save(str(tmp / "profiler_y_tmp.npy"), y)
PYEOF
fi

# ── 3. Run tw_profiler ───────────────────────────────────────────────────────────
echo "[run_optimized:${VARIANT}] Running tw_profiler (n=10000, d_x=50)..."
cargo run --release --bin tw_profiler --features "cli profiling" -- \
  --x  "research/2026-04-08-x-dist-simd-avx512/data/profiler_x_tmp.npy" \
  --y  "research/2026-04-08-x-dist-simd-avx512/data/profiler_y_tmp.npy" \
  --output "${RESULTS}/${VARIANT}_profiler.json" \
  --stderr-capture "${RESULTS}/${VARIANT}_stderr.txt" \
  --k 15 --iters 5 --warmup 2

echo "[run_optimized:${VARIANT}] Done. Results in ${RESULTS}/"
