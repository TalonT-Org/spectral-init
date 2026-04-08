# Experiment Plan: Trustworthiness Performance — Scaling Analysis and Optimization Evaluation

## Motivation

The `trustworthiness()` function in `src/metrics.rs` is the dominant runtime
bottleneck beyond n=100K cells. At n=250K it was killed after 7+ minutes at
805% CPU. Without a profiled breakdown of where time is actually spent, any
optimization is a guess. This experiment characterizes the per-step scaling
profile, evaluates six specific algorithmic and SIMD alternatives, and produces
a ranked GO/NO-GO recommendation grounded in measured wall-clock speedups at
the practical target scale of 100K–250K cells.

The results directly inform the implementation decision: which optimizations
to ship in `ComputeMode::RustNative`, which to mark NO-GO, and what approximate
parity threshold to document for the subsampling code path.

This plan supersedes the prior experiment plan
`.autoskillit/temp/plan-experiment/experiment_plan_trustworthiness_perf_scaling_2026-04-04_123526.md`,
which received a STOP verdict from review-design on three critical grounds
(cold-start allocation confound, post-hoc sampling calibration, vacuous parity
gate on Gaussian data). All three findings are addressed below.

---

## Hypothesis

**Null hypothesis (H0):** No single optimization achieves a measured speedup
greater than 2x at n=100K under warm-start conditions with ≥3 iterations, and
row subsampling at m=5000 produces |T_approx − T_exact| > 0.001 on
MERFISH-structured data.

**Alternative hypothesis (H1_combined):** At least one exact optimization
achieves ≥2x speedup at n=100K (warm), AND row subsampling at the
pre-registered m=5000 produces |T_approx − T_exact| ≤ 0.001 on MERFISH-derived
structured data.

Individual falsifiable sub-hypotheses:

- **H1** X-sort is the dominant wall-clock step (≥40% of total) at n≥10K, and
  partial-rank computation achieves the largest single-step speedup.
- **H2** Thread-local Vec reuse provides 5–15% throughput improvement at
  n=5K–50K (Criterion ratio, 95% CI).
- **H3** Manual AVX2 for the X-distance inner loop yields < 2x over the current
  auto-vectorized baseline at d=10 f64 (confirmed by `cargo asm` + benchmark).
- **H4** AVX-512 for the distance kernel provides < 20% improvement over AVX2
  at d=10 due to 62.5% register utilization and potential frequency throttling.
- **H5** Row subsampling at the pre-registered m=5000 produces
  |T_approx − T_exact| ≤ 0.001 on MERFISH-derived structured data (n=10K
  slice, PCA-reduced to 50 dims).
- **H6** Combined exact optimizations (thread-local + partial-rank + distance
  kernel, if applicable) achieve 3–6x end-to-end speedup at n=100K (warm).

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Dataset size n | 1K, 5K, 10K, 50K, 100K, (250K with timeout guard) | Span the pre-bottleneck to bottleneck regime |
| Optimization variant | baseline, thread_local, partial_rank, avx2_kernel, combined, approx_m5000 | Each addresses a distinct hypothesis |
| Sample size m (subsampling only) | 500, 1000, 2000, 5000, 10000 | Sweep to characterize error bound; m=5000 pre-registered |
| Data type | Gaussian synthetic (d=10), MERFISH-derived structured (d=50) | Gaussian = throughput characterization; structured = parity validation |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Per-step wall-clock fraction | % of total row time | `#[cfg(feature="testing")]` timing in `trustworthiness()`, stderr JSON parse | NEW — no entry in `src/metrics.rs` |
| Criterion throughput ratio | dimensionless (speedup factor) | `criterion::BenchmarkGroup`, `BenchmarkId`, `throughput` | NEW |
| Criterion mean ± CI | ms | Criterion HTML report + extracted JSON | NEW |
| Wall-clock at large n | seconds (mean ± std, ≥3 warm iters) | `tw_profiler` binary with `--iters 5 --warmup 2` | NEW |
| Trustworthiness parity (exact) | absolute | `|T_rust − T_sklearn|`, verified via Python reference | `trustworthiness` (in `src/metrics.rs`); threshold `1e-6` in integration test |
| Subsampling deviation | absolute | `|T_approx(m) − T_exact(n)|` on MERFISH-derived data | NEW — no canonical entry; approximate threshold `< 0.001` proposed |
| AVX2 auto-vectorization status | boolean | `cargo asm` instruction inspection | NEW — qualitative, not in metrics.rs |

**NEW metrics requiring definition before finalization:**
- `tw_step_fraction`: formula = `step_wall_us / total_row_wall_us`; unit = %; no threshold (diagnostic)
- `tw_speedup_ratio`: formula = `baseline_mean_ms / variant_mean_ms`; unit = ×; GO threshold ≥ 1.5×
- `tw_approx_deviation`: formula = `|T_approx − T_exact|`; unit = absolute; GO threshold ≤ 0.001 on structured data

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighborhood size) | 15 | Matches existing sklearn parity fixture and production default |
| X dimensionality (synthetic) | d_x = 10 | Matches complexity analysis in scope report; controls per-step FLOP count |
| Y dimensionality | d_y = 2 | Standard UMAP 2D embedding output |
| RNG seed | 42 | Reproducibility for synthetic datasets |
| Rayon thread count | system default (≥8) | Not pinned; document actual count in results |
| Rust toolchain | stable (current) | No nightly features needed |
| target-cpu | native (from `.cargo/config.toml`) | AVX2+FMA available; document in results |
| Criterion measurement time | 10s per benchmark | Sufficient for stable estimates at n≤50K |
| Warm iterations (large n) | 2 discarded warmup + 5 measured | Addresses STOP-1 cold-start allocation confound |

---

## Inputs and Data

The experiment requires both synthetic datasets (to control n and d, and characterize
throughput without structured-data confounds) and MERFISH-derived structured
data (to validate subsampling parity on realistic biological data, addressing
STOP-3). Gaussian synthetic results are labeled **throughput characterization
only** and cannot be used as the parity GO gate.

**Pre-registration (STOP-2 fix):** The production subsampling value m=5000 is
fixed here before any measurement is run. The fraction scan over
m={500, 1000, 2000, 5000, 10000} is conducted as an exploratory secondary
analysis only. The GO/NO-GO criterion for H5 is evaluated at m=5000 exclusively.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| `gaussian_n{N}_d10.npy` (X), `gaussian_n{N}_d2.npy` (Y) | Generated: `np.random.RandomState(42).randn(n, d)` | Gaussian i.i.d., n ∈ {1K,5K,10K,50K,100K}, d_x=10, d_y=2 | Throughput benchmarks (H1–H4, H6) — NOT parity gate |
| `structured_n{N}_d10.npy` (X), `structured_n{N}_d2.npy` (Y) | Generated: `make_blobs(n, centers=8, n_features=10, random_state=42)` + 2D PCA for Y | 8-cluster structure, n ∈ {1K,5K,10K}, d_x=10, d_y=2 | Subsampling parity validation on non-Gaussian data |
| `merfish_n10k_d50.npy` (X), `merfish_n10k_d2.npy` (Y) | `temp/merfish_100k/` first 10K rows; X PCA-reduced to 50 dims | Biological structure, n=10K, d_x=50, d_y=2 | Primary MERFISH parity gate for H5 (STOP-3 fix) |
| `tw_parity.npz` | Existing: `tests/fixtures/tw_parity/tw_parity.npz` | n=200, d_x=10, d_y=2, k=15, sklearn reference score | Exact parity regression check (n=200) |

**MERFISH preparation rationale:** The full 100K expression matrix is 1122-dimensional.
Computing T_exact at n=100K, d_x=1122 is prohibitively expensive (≈112× more than d=10).
Using a 10K slice PCA-reduced to 50 dims (standard UMAP preprocessing) preserves the
biological cluster structure — the property that makes STOP-3 relevant — while making
T_exact tractable. This scope limitation must be documented in the report.

---

## Experiment Directory Layout

```
research/2026-04-04-tw-perf-scaling/
├── environment.yml               # Not needed — reuse existing spectral-test env
├── scripts/
│   ├── gen_synthetic.py          # Generate Gaussian + structured synthetic .npy datasets
│   ├── prepare_merfish.py        # Extract + PCA-reduce MERFISH 10K slice to .npy
│   ├── sklearn_reference.py      # Compute sklearn T(k) scores for all datasets → results/sklearn_scores.json
│   ├── run_profiling.sh          # Drive tw_profiler binary across n values, capture JSON
│   ├── run_criterion.sh          # Drive cargo bench for trustworthiness_bench
│   ├── run_subsampling_sweep.sh  # Drive subsampling sweep binary on MERFISH data
│   └── analyze_results.py        # Parse all results/ JSON → recommendation table, plots
├── data/
│   ├── gaussian/                 # gaussian_n1k_d10.npy, ..., gaussian_n100k_d10.npy + Y variants
│   ├── structured/               # structured_n1k_d10.npy, ..., structured_n10k_d10.npy + Y variants
│   └── merfish/                  # merfish_n10k_d50.npy, merfish_n10k_d2.npy
├── results/
│   ├── step_timing/              # JSON per n: {n, step, mean_us, std_us, fraction}
│   ├── criterion/                # Criterion output (JSON or extracted summary)
│   ├── subsampling/              # JSON: {m, T_approx, T_exact, delta, data_type}
│   ├── sklearn_scores.json       # sklearn T(k) reference values for all datasets
│   ├── asm_inspection.txt        # cargo asm output for distance kernel (H3 evidence)
│   └── recommendation_table.md  # Final ranked GO/NO-GO table
└── report.md                     # Written by write-report skill
```

**Source-tree artifacts** (in canonical Cargo locations — to be added by the implementer):

| File | Location | Purpose |
|------|----------|---------|
| `trustworthiness_bench.rs` | `benches/` | Criterion baseline + variant benchmarks at n=1K–50K |
| `tw_profiler.rs` | `src/bin/` | Warm-start per-step timing binary; JSON output to stdout |
| New `[[bench]]` entry | `Cargo.toml` | Register `trustworthiness_bench`, `required-features = ["testing"]` |
| New `[[bin]]` entry | `Cargo.toml` | Register `tw_profiler`, `required-features = ["testing"]` |
| Per-step timing in `trustworthiness()` | `src/metrics.rs` | `#[cfg(feature="testing")]`-gated `Instant` + `eprintln!("[timing:tw_step_N]")` at each of the 6 steps |

---

## Environment

**No custom environment needed.**

The `spectral-test` conda environment (at `tests/environment.yml`) already
provides all required Python packages: `python=3.11`, `numpy=2.2`, `scikit-learn=1.8`,
`scipy=1.15`, and `ndarray-npy` I/O compatibility. The Rust toolchain with
`cargo`, `criterion=0.5`, and `target-cpu=native` (from `.cargo/config.toml`)
covers all benchmark needs.

Activate with: `micromamba activate spectral-test`

No `environment.yml` will be created for this experiment.

---

## Implementation Phases

### Phase 0: Crate Instrumentation (prerequisite — must land before any measurement)

Modifies `src/metrics.rs`, `src/bin/` (new file), and `Cargo.toml`.
**This phase must be complete before any Phase 1+ work runs.**

**0a. Per-step timing in `trustworthiness()`** (`src/metrics.rs`):

Add `#[cfg(feature = "testing")]` timing guards around each of the 6 inner-loop
steps, matching the exact pattern from `src/solvers/mod.rs`. The six steps:

```
[timing:tw_step_1] X pairwise distances (us)
[timing:tw_step_2] X full sort (us)
[timing:tw_step_3] rank scatter (us)
[timing:tw_step_4] X k-NN set build (us)
[timing:tw_step_5] Y streaming heap (us)
[timing:tw_step_6] penalty accumulation (us)
[timing:tw_total]  total per-call (us)
```

Use `std::time::Instant` captures before/after each step inside the parallel
closure. Emit via `eprintln!` as structured tags parseable by the profiler binary.

**0b. `tw_profiler` binary** (`src/bin/tw_profiler.rs`):

CLI interface: `tw_profiler --x X.npy --y Y.npy [--k 15] [--iters 5] [--warmup 2]`

Behavior:
- Load X, Y from `.npy` via `ndarray_npy::read_npy`
- Run `warmup` iterations (discard timing — STOP-1 fix)
- Run `iters` measured iterations, capturing stderr timing lines
- Compute mean ± std per step across measured iterations
- Output JSON to stdout:
  ```json
  {
    "n": 10000, "d_x": 10, "d_y": 2, "k": 15, "iters": 5,
    "steps": [
      {"step": 1, "name": "x_dist", "mean_us": ..., "std_us": ..., "fraction": ...},
      ...
    ],
    "total_mean_us": ..., "total_std_us": ...
  }
  ```
- Write JSON to a `--output path.json` argument; also print to stdout

**0c. `Cargo.toml` additions:**

```toml
[[bench]]
name = "trustworthiness_bench"
harness = false
required-features = ["testing"]

[[bin]]
name = "tw_profiler"
path = "src/bin/tw_profiler.rs"
required-features = ["testing", "cli"]
```

**Verification:** `cargo build --features testing,cli --bin tw_profiler` succeeds.

---

### Phase 1: Directory Structure and Data Generation

Create the research directory tree:

```bash
mkdir -p research/2026-04-04-tw-perf-scaling/{scripts,data/{gaussian,structured,merfish},results/{step_timing,criterion,subsampling}}
```

**1a. `scripts/gen_synthetic.py`:**

Generates Gaussian and `make_blobs` structured datasets. For each n in
{1000, 5000, 10000, 50000, 100000}:
- `X = np.random.RandomState(42).randn(n, 10).astype(np.float64)` → `data/gaussian/gaussian_n{n}_x.npy`
- `Y = np.random.RandomState(42).randn(n, 2).astype(np.float64)` → `data/gaussian/gaussian_n{n}_y.npy`

For n in {1000, 5000, 10000}:
- `X, _ = make_blobs(n_samples=n, centers=8, n_features=10, random_state=42)` → `data/structured/structured_n{n}_x.npy`
- `Y = PCA(2).fit_transform(X).astype(np.float64)` → `data/structured/structured_n{n}_y.npy`

Print shapes for each file as verification.

**1b. `scripts/prepare_merfish.py`:**

```python
import numpy as np; from sklearn.decomposition import PCA
X_full = np.load("temp/merfish_100k/merfish_100k_expression.npz")["arr_0"]  # (100000, 1122) float
Y_full = np.load("temp/merfish_100k/merfish_100k_spatial.npz")["arr_0"]     # (100000, 2) float
X10k = X_full[:10000].astype(np.float64)
Y10k = Y_full[:10000].astype(np.float64)
X10k_pca = PCA(50, random_state=42).fit_transform(X10k)                     # (10000, 50)
np.save("research/2026-04-04-tw-perf-scaling/data/merfish/merfish_n10k_x.npy", X10k_pca)
np.save("research/2026-04-04-tw-perf-scaling/data/merfish/merfish_n10k_y.npy", Y10k)
```

Print shapes and PCA explained variance ratio (first 10 components) as verification.

**1c. `scripts/sklearn_reference.py`:**

Compute `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` for each dataset
(Gaussian n=1K–100K, structured n=1K–10K, MERFISH n=10K). Write results to
`results/sklearn_scores.json`:
```json
{
  "gaussian_n1000": {"T_sklearn": 0.9..., "k": 15},
  "gaussian_n10000": {"T_sklearn": 0.9..., "k": 15},
  ...
  "merfish_n10000": {"T_sklearn": 0.9..., "k": 15}
}
```

Note: `sklearn_reference.py` is also the ground truth for the parity check. Save
per-dataset expected T value before running any Rust benchmarks.

**Verification:** All `.npy` files load without error; all sklearn scores output to
JSON. Check `gaussian_n200` via existing fixture: run Rust trustworthiness on
`tests/fixtures/tw_parity/tw_parity.npz` and confirm `|T_rust − T_sklearn| < 1e-6`.

---

### Phase 2: Baseline Characterization

After Phase 0 and Phase 1 complete, run the profiler to establish the per-step
profile and the Criterion baseline.

**2a. `scripts/run_profiling.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"
BIN="cargo run --release --features testing,cli --bin tw_profiler --"

for N in 1000 5000 10000 50000 100000; do
  echo "=== Profiling n=$N ==="
  $BIN \
    --x "$RESEARCH_DIR/data/gaussian/gaussian_n${N}_x.npy" \
    --y "$RESEARCH_DIR/data/gaussian/gaussian_n${N}_y.npy" \
    --k 15 --iters 5 --warmup 2 \
    --output "$RESEARCH_DIR/results/step_timing/gaussian_n${N}.json"
done

# MERFISH profiling (d_x=50 — note: slower per step)
$BIN \
  --x "$RESEARCH_DIR/data/merfish/merfish_n10k_x.npy" \
  --y "$RESEARCH_DIR/data/merfish/merfish_n10k_y.npy" \
  --k 15 --iters 5 --warmup 2 \
  --output "$RESEARCH_DIR/results/step_timing/merfish_n10k.json"
```

Optional n=250K (Gaussian): add with `--warmup 2 --iters 3` and a 300-second
timeout wrapper (`timeout 300`).

**2b. `benches/trustworthiness_bench.rs`:**

Criterion benchmark with the following structure:

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spectral_init::trustworthiness;

fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("tw_baseline");
    for &n in &[1_000usize, 5_000, 10_000, 50_000] {
        let x = make_gaussian_data(n, 10, 42);  // helper: returns Array2<f64>
        let y = make_gaussian_data(n, 2, 43);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| trustworthiness(x.view(), y.view(), 15))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_baseline);
criterion_main!(benches);
```

Include `bench_baseline` plus placeholder groups for each variant
(initially identical to baseline; filled in Phase 3):
- `bench_thread_local`
- `bench_partial_rank`
- `bench_avx2_kernel`
- `bench_combined`

**2c. `scripts/run_criterion.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo bench --features testing --bench trustworthiness_bench 2>&1 | tee \
  research/2026-04-04-tw-perf-scaling/results/criterion/criterion_output.txt
```

Extract summary JSON from Criterion's `target/criterion/` HTML output into
`results/criterion/summary.json` (use a small Python helper or `jq`).

**AVX2 auto-vectorization inspection (H3):**

```bash
cargo asm --features testing spectral_init::metrics::trustworthiness_row_distance_kernel \
  2>&1 | head -100 \
  > research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt
```

If `cargo asm` is unavailable, use:
```bash
cargo rustc --features testing --release -- --emit asm 2>/dev/null
grep -A 30 "vmovupd\|vfmadd\|ymm" target/release/deps/*.s | head -100 \
  >> research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt
```

The file must contain evidence of whether AVX2 instructions (`vmovupd`, `vfmadd213pd`,
`ymm` registers) appear in the distance inner loop. This is the H3 GO/NO-GO gate.

---

### Phase 3: Optimization Implementations

Implementations are gated on Phase 2 profiling results per the following rules:
- **Thread-local buffers (H2):** Always implement (zero correctness risk, pure throughput).
- **Partial-rank X (H1):** Implement only if X-sort step ≥ 40% of total wall-clock at n=50K.
- **Manual AVX2 kernel (H3):** Implement only if `asm_inspection.txt` shows NO AVX2
  instructions in the distance loop. If auto-vectorized, mark NO-GO immediately.
- **AVX-512 kernel (H4):** Implement only if H3 benchmark shows ≥ 20% AVX2 gain over
  scalar; otherwise the ceiling is too low for AVX-512 to matter.
- **Row subsampling (H5):** Always implement (needed for the 250K+ use case regardless).

**3a. Thread-local Vec reuse** (modifies `src/metrics.rs`):

Replace the three per-row `Vec::new()` allocations with `thread_local!` `RefCell<Vec>`
buffers. Pattern:

```rust
thread_local! {
    static DIST_X: RefCell<Vec<(f64, usize)>> = RefCell::new(Vec::new());
    static RANK_X: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}
// In the closure:
DIST_X.with(|buf| {
    let mut dist_x = buf.borrow_mut();
    dist_x.clear(); dist_x.resize(n, (0.0, 0));
    // ... use dist_x ...
});
```

Add a `bench_thread_local` group to `trustworthiness_bench.rs` using a new public
function `trustworthiness_thread_local(x, y, k)` or a feature flag.

Verify: all 5 unit tests + sklearn parity fixture pass. Criterion ratio vs baseline
must be computed at n=5K, 10K, 50K.

**3b. Partial-rank X computation** (modifies `src/metrics.rs`, conditional):

If X-sort ≥ 40%: replace `sort_unstable_by(total_cmp)` + full rank scatter with:
1. `select_nth_unstable_by` at position k (O(n) average, no quadratic worst-case per
   stdlib PR #107522 merged ~2023).
2. For each of the ≤k Y-neighbor indices i, count the number of X-distances strictly
   less than X-distance to i — this is the rank, computable in a single O(n) pass.
3. No `HashSet` needed: iterate Y-heap results, for each j compute
   `rank_j = dist_x.iter().filter(|&&(d, _)| d < dist_x[j]).count()`.

Correctness gate (mandatory before any benchmarking):
- `|T_partial_rank − T_exact| == 0` on the n=200 sklearn fixture.
- All 5 unit tests pass unchanged.
- Run on n=10K Gaussian; compare to baseline T value.

**3c. Row subsampling variant** (new public function `trustworthiness_approx`):

```rust
pub fn trustworthiness_approx(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
    sample: usize,
    seed: u64,
) -> f64
```

Randomly sample `sample` row indices (without replacement, seeded via `rand::SeedableRng`),
run the exact per-row computation only on those rows, and return the mean penalty
normalized by the same formula as `trustworthiness()`. This is NOT the default;
add to `src/metrics.rs` as an additional public function.

Add a `bench_approx` group to the bench (measuring wall-clock at n=10K–50K, various m).

---

### Phase 4: Subsampling Deviation Sweep (addresses STOP-2 and STOP-3)

**Pre-registered production value: m = 5000.** This value is fixed and cannot be
changed by the results of the sweep.

**4a. `scripts/run_subsampling_sweep.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"

# Primary validation: MERFISH-derived structured data (STOP-3 fix)
echo "=== Subsampling sweep: MERFISH structured (n=10K, d=50) ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/merfish/merfish_n10k_x.npy" \
  --y "$RESEARCH_DIR/data/merfish/merfish_n10k_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/merfish_n10k.json"

# Secondary (exploratory only): structured synthetic
echo "=== Subsampling sweep: structured synthetic (n=10K, d=10) ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/structured/structured_n10000_x.npy" \
  --y "$RESEARCH_DIR/data/structured/structured_n10000_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/structured_n10k.json"

# Informational only (NOT used for GO gate): Gaussian
echo "=== Subsampling sweep: Gaussian (n=10K, d=10) — throughput characterization ONLY ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/gaussian/gaussian_n10000_x.npy" \
  --y "$RESEARCH_DIR/data/gaussian/gaussian_n10000_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/gaussian_n10k.json"
```

**`scripts/subsampling_sweep.py`** drives the Rust `tw_profiler` or the
`trustworthiness_approx` function via a Python binding (via subprocess calling
`cargo run --features testing,cli --bin tw_approx_runner`) at each m value
with 5 independent seeds per m. Outputs JSON:
```json
{
  "dataset": "merfish_n10k",
  "n_exact": 10000, "k": 15,
  "results": [
    {"m": 500,  "T_exact": 0.9712, "T_approx_mean": 0.9698, "T_approx_std": 0.0014, "delta_mean": 0.0014, "delta_max": 0.0028},
    {"m": 5000, "T_exact": 0.9712, "T_approx_mean": 0.9710, "T_approx_std": 0.0004, "delta_mean": 0.0002, "delta_max": 0.0005},
    ...
  ]
}
```

This also requires a small additional binary `src/bin/tw_approx_runner.rs` that
calls `trustworthiness_approx` with `--x, --y, --k, --sample, --seed` CLI args.

---

### Phase 5: Dry Run

Before committing to full Criterion sweeps (which can take 30+ minutes):

1. Run `tw_profiler` at n=1K: verify JSON output format, all 6 steps present, fractions sum to ~100%.
2. Run Criterion with `--bench-time 2s` (override via env `CRITERION_BENCH_TIME=2`) at n=1K only: confirm no panics, output is parseable.
3. Run `subsampling_sweep.py` with `--sample-sizes 500` at n=1K: confirm JSON output, delta is finite.
4. Check `asm_inspection.txt` has content and contains meaningful assembly.
5. Verify all unit tests pass: `cargo test --features testing`.

---

## Execution Protocol

The following steps must be run in order after all implementations are complete.

```bash
# 0. Environment
micromamba activate spectral-test
cd /home/talon/projects/spectral-init

# 1. Build all instrumented binaries (verify Phase 0)
cargo build --release --features testing,cli

# 2. Generate all datasets (Phase 1)
python3 research/2026-04-04-tw-perf-scaling/scripts/gen_synthetic.py
python3 research/2026-04-04-tw-perf-scaling/scripts/prepare_merfish.py
python3 research/2026-04-04-tw-perf-scaling/scripts/sklearn_reference.py

# 3. Dry run (Phase 5) — abort if anything fails
bash research/2026-04-04-tw-perf-scaling/scripts/dry_run.sh

# 4. Baseline characterization (Phase 2)
bash research/2026-04-04-tw-perf-scaling/scripts/run_profiling.sh
bash research/2026-04-04-tw-perf-scaling/scripts/run_criterion.sh   # ~30 min
cat research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt  # inspect H3 gate

# 5. Gate check: inspect step_timing JSONs to determine which Phase 3 variants to implement
#    If X-sort < 40%: skip partial_rank implementation
#    If asm_inspection shows AVX2 already: mark H3 NO-GO, skip avx2_kernel

# 6. Run optimization benchmarks (Phase 3, after implementations land)
bash research/2026-04-04-tw-perf-scaling/scripts/run_criterion.sh   # now includes variant groups

# 7. Subsampling sweep (Phase 4) — MERFISH is the GO gate
bash research/2026-04-04-tw-perf-scaling/scripts/run_subsampling_sweep.sh

# 8. Analysis and recommendation table (Phase 5)
python3 research/2026-04-04-tw-perf-scaling/scripts/analyze_results.py \
  --results-dir research/2026-04-04-tw-perf-scaling/results/ \
  --output research/2026-04-04-tw-perf-scaling/results/recommendation_table.md
```

---

## Analysis Plan

**Step 1 — Per-step dominance (H1):** Load all `results/step_timing/gaussian_n*.json`.
For each n, compute `fraction = step_mean_us / total_mean_us`. Plot step fractions
as stacked bar chart vs n. If X-sort fraction < 40% at n≥50K, H1 fails.

**Step 2 — Criterion speedup ratios (H2, H3, H6):** For each variant group, compute
`speedup = baseline_mean / variant_mean` at each n. Include 95% confidence intervals
from Criterion's bootstrap. H2 passes if speedup ∈ [1.05, 1.15] for thread-local.
H6 passes if combined speedup ≥ 3.0 at n=100K.

**Step 3 — AVX2 gate (H3):** Manual inspection of `asm_inspection.txt`. If AVX2
instructions present: mark H3 as "NO-GO — auto-vectorization confirmed, no manual
SIMD needed." If absent: run manual AVX2 implementation and compare Criterion means.

**Step 4 — AVX-512 gate (H4):** Only if H3 shows ≥20% manual AVX2 gain. Otherwise
mark H4 "NO-GO — ceiling too low for AVX-512 to matter."

**Step 5 — Subsampling deviation (H5, MERFISH primary):** Load
`results/subsampling/merfish_n10k.json`. For m=5000 (pre-registered):
- If `delta_max < 0.001`: H5 PASS — GO for subsampling in production.
- If `delta_max ≥ 0.001`: H5 FAIL — report empirical threshold from smallest m
  where `delta_max < 0.001`, or mark NO-GO if no such m ≤ 10000.

Compare MERFISH results to Gaussian (from `gaussian_n10k.json`) to confirm
STOP-3 finding: Gaussian errors should be near-zero by concentration, MERFISH
errors will be structurally different.

**Step 6 — Recommendation table:** Produce `results/recommendation_table.md`:

| Approach | Speedup (n=100K) | CI 95% | Scaling change | Implementation LOC | Parity impact | GO/NO-GO | Rationale |
|---|---|---|---|---|---|---|---|
| Thread-local buffers | {measured}× | ± | None | ~30 | Zero (exact) | {decision} | ... |
| Partial-rank X | {measured}× | ± | O(n²) → O(n²) | ~60 | Zero if correct | {decision} | ... |
| Manual AVX2 kernel | {measured}× | ± | None | ~40 | Zero (bit-identical) | {decision} | ... |
| Row subsampling m=5000 | ~50× | N/A | O(n²)→O(mn) | ~40 | New: ≤0.001 | {decision} | ... |

No multiple-comparison correction is required for the primary GO/NO-GO decisions
because each hypothesis is evaluated against a single pre-specified criterion. The
exploratory m-sweep analysis (H5 secondary) is reported descriptively only.

---

## Success Criteria

- **Conclusive positive:** Per-step profiling shows X-sort ≥ 40% wall-clock at
  n≥50K (H1 supported), AND at least one exact optimization achieves ≥ 1.5×
  Criterion speedup with non-overlapping CI (H2 or H6 supported), AND MERFISH
  subsampling at pre-registered m=5000 produces `delta_max < 0.001` (H5 supported).

- **Conclusive negative:** X-sort < 40% at all measured n (H1 fails; partial-rank
  is NO-GO), AND Criterion speedup for all exact variants ≤ 1.2× (H2/H6 fail),
  AND MERFISH `delta_max ≥ 0.001` at m=5000 (H5 fails). Report the true dominant
  bottleneck revealed by profiling and recommend further investigation.

- **Inconclusive:** Per-step fractions are within 10 percentage points of each
  other at n≥50K (no dominant step), or Criterion runs do not converge (std > 50%
  of mean), or MERFISH dataset preparation fails. Document the specific gap.

---

## Threats to Validity

### Internal

- **Allocation confound (mitigated — STOP-1):** All large-n timing uses ≥2 warm
  iterations discarded before measurement. The cold-start page-fault cost is
  explicitly excluded from the reported speedup. Criterion handles this automatically
  via warm-up iterations; `tw_profiler` implements it manually.

- **Post-hoc sample fraction selection (mitigated — STOP-2):** m=5000 is pre-registered
  in this document. The m-sweep is exploratory secondary analysis. The GO decision for
  H5 is evaluated at m=5000 only, on held-out MERFISH data not used in timing benchmarks.

- **Vacuous Gaussian parity gate (mitigated — STOP-3):** The H5 GO gate uses MERFISH-
  derived PCA-50 data (biological cluster structure). Gaussian results are collected but
  labeled throughput-characterization-only and are not used for any GO/NO-GO decision.

- **MERFISH scope limitation:** T_exact is computed at n=10K (not n=100K) due to the
  d_x=1122 compute cost. This means H5 measures subsampling error at n=10K. Whether
  the subsampling error at n=10K generalizes to n=100K on MERFISH data is not directly
  tested by this experiment. This limitation must be documented in the report.

- **Partial-rank correctness tie-handling:** The rank-by-counting approach requires
  careful handling of ties in X-distances. Equal distances must be broken consistently
  with the original `sort_unstable_by(total_cmp)` behavior. Unit test coverage must
  include a synthetic case with intentional ties.

### External

- **Hardware specificity:** AVX2 and AVX-512 results are specific to the CI target
  (`x86-64-v3` via `target-cpu=native`). Machines without AVX2 will fall back to
  scalar; the speedup table should document CPU features detected at benchmark time.

- **d_x=10 synthetic benchmarks:** The profiling and Criterion speedups are measured
  at d_x=10. Production use with MERFISH (d_x=1122, or PCA-reduced d_x=50) will have
  different step fractions — Step 1 (distance computation) will be proportionally larger.
  The step profiler should also be run on the MERFISH d=50 dataset to characterize this.

- **Rayon thread count:** Results depend on system thread count. Report actual Rayon
  thread count (via `rayon::current_num_threads()`) in every JSON output.

---

## Estimated Resource Requirements

- **Disk:** ~500 MB (synthetic data for n=100K, Criterion HTML, results JSON)
- **Criterion benchmarks (n=1K–50K, 5 variants × 10s each):** ~10 min
- **Large-scale profiling (n=100K, 5 warm iters):** ~40–60 min (estimate; exact
  wall-clock unknown — that's what this experiment measures)
- **Optional n=250K run:** potentially killed; wrap with `timeout 600s`
- **Subsampling sweep (n=10K × 3 datasets × 5 m values × 5 trials):** ~30 min
- **Python data generation + sklearn reference:** ~5 min
- **Total:** ~2–3 hours compute
