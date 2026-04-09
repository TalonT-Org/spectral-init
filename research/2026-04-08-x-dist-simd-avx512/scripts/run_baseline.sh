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
  -- trustworthiness_d50 2>&1 | tee "${RESULTS}/baseline_criterion.txt"

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
  --k 15 --iters 5 --warmup 2

echo "[run_baseline] Done. Results in ${RESULTS}/"
