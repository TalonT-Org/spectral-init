# Experiment Plan: Trustworthiness Performance — Scaling Analysis and Optimization Evaluation

## Motivation

The `trustworthiness` function in `src/metrics.rs:389-449` is the dominant runtime bottleneck in
the MERFISH benchmark pipeline. At n=100K cells, Python sklearn consumed 651s of the ~683s total
pipeline time. At n=250K, the Rust process was killed after >7 minutes at 805% CPU utilization.
No Criterion benchmark exists for the Rust implementation — meaning no optimization claim is
currently measurable or verifiable. This experiment establishes a reproducible performance
baseline for the Rust implementation, measures per-step bottleneck distribution, and quantifies
speedup for five candidate optimization approaches (AVX2 SIMD, partial select, thread-local
buffers, row subsampling, and a combined exact variant). The results will produce a ranked
GO/NO-GO table to inform implementation priority for production optimization work.

## Hypothesis

**Null hypothesis (H0):** No single optimization approach delivers a ≥2x end-to-end speedup at
n≥10K; the O(n²) scaling cannot be reduced by constant-factor improvements alone, and row
subsampling at practical rates (≤20%) will exceed the 0.01 benchmark gate.

**Alternative hypothesis (H1):** At least one of SIMD distance kernels, partial select, or
thread-local buffer reuse delivers a ≥2x end-to-end speedup at n=10K. Row subsampling at
10% achieves ≥8x speedup at n=50K while staying within the |Δ| < 0.01 MERFISH benchmark gate.
The dominant bottleneck step (>40% of per-row wall time) is the X distance computation, making
the SIMD path the highest-priority exact optimization.

## Independent Variables

| Variable | Values | Rationale |
|---|---|---|
| Optimization strategy | `baseline`, `partial_select`, `thread_local`, `simd_avx2`, `sampled_5pct`, `sampled_10pct`, `sampled_20pct`, `combined` | Covers all scope directions; `combined` = partial_select + thread_local + simd_avx2 |
| Input size n | 1K, 5K, 10K (Criterion); 25K, 50K, 100K (wall-clock timing binary) | Spans Criterion-feasible and large-scale ranges |
| Step (step-level breakdowns only, at n=10K) | `x_distance`, `x_sort`, `x_rank_scatter`, `x_knn_set`, `y_dist_heap`, `penalty` | Isolates bottleneck contribution |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Wall-clock time per call | µs / ms / s | Criterion `b.iter()` or `std::time::Instant` | NEW — no entry in `src/metrics.rs` |
| Speedup vs baseline | dimensionless ratio | Derived: `t_baseline / t_optimized` | NEW |
| Per-step fraction | % of total row time | `Instant` guards in step-profiling binary | NEW |
| trustworthiness score | f64 scalar in [0, 1] | Return value of function call | `trustworthiness` (canonical, lines 389-449) |
| Score deviation from exact | f64 absolute delta | `|trustworthiness_approx − trustworthiness_exact|` | NEW — gate defined as 0.01 in `docs/metrics/structure-preservation.md` |
| Scaling exponent | dimensionless | Fit to log-log(time vs n) | NEW |

All "NEW" metrics are research-internal to this experiment and do not require addition to
`src/metrics.rs`. They measure performance properties, not embedding quality.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|---|---|---|
| Random seed | 42 | Reproducible; matches existing fixtures |
| k (neighborhood size) | 15 | Matches MERFISH benchmark configuration |
| d_x (X feature dimension) | 10 | Matches MERFISH (gene expression) |
| d_y (Y embedding dimension) | 2 | Matches UMAP 2D output |
| Data distribution | `StandardNormal` via `rand_distr` | No structure bias; reproducible |
| `RAYON_NUM_THREADS` | 8 (logged in results JSON) | Eliminates thread-count confound |
| Compiler flags | `.cargo/config.toml`: `-C target-cpu=native` | Already project-wide; AVX2+FMA available |
| Build profile | release (via `cargo bench` / `--release`) | Matches production |

## Inputs and Data

All input data is generated synthetically in Rust using `rand::SeedableRng` + `rand_distr::StandardNormal`. No external data files are required for the core benchmarks. The existing `tests/fixtures/tw_parity/tw_parity.npz` (n=200, k=15) is used only for post-benchmark correctness spot-checks — it is too small for performance measurement.

For subsampling parity verification, a Python script generates (n, 10) × (n, 2) f64 arrays via `np.random.default_rng(42).standard_normal` and runs sklearn `trustworthiness(X, Y, n_neighbors=15)` to obtain reference scores at n=5K, 10K, and 50K. These are compared against Rust `trustworthiness_sampled` CLI output.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| Synthetic (n, 10) × (n, 2) f64 | Generated inline in Rust benchmarks (SmallRng, seed 42) | Gaussian, no structure | Criterion benchmarks n=1K, 5K, 10K |
| Synthetic large-scale | Generated inline in timing binary (same RNG) | Same distribution, larger n | Wall-clock timing at n=25K, 50K, 100K |
| `tw_parity.npz` | `tests/fixtures/tw_parity/` | n=200, k=15, sklearn score | Correctness spot-check after optimization |
| sklearn reference scores | `scripts/sklearn_parity.py` output | n=5K, 10K, 50K; exact sklearn T | Subsampling accuracy gate verification |

Data properties sufficient for valid testing: the Gaussian distribution produces non-trivial trustworthiness scores (neither perfect 1.0 nor degenerate 0.0), and all algorithm steps are exercised on meaningful distance distributions.

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-04-trustworthiness-perf-scaling/
├── environment.yml                     # Lightweight Python env for parity & plotting
├── scripts/
│   ├── run_experiment.sh               # Master driver: all phases in sequence
│   ├── run_criterion_bench.sh          # cargo bench --features testing --bench trustworthiness_bench
│   ├── run_step_timing.sh              # cargo run --release --features testing --bin tw_step_profiler
│   ├── run_large_scale_timing.sh       # cargo run --release --features testing --bin tw_large_scale
│   ├── sklearn_parity.py               # sklearn reference scores for parity gate check
│   └── analyze_results.py             # Parse JSON results → scaling tables + log-log plots
├── data/                               # .gitkeep (generated data excluded from VCS)
├── results/
│   ├── step_timing_n10k.json           # Per-step breakdown at n=10K (5 iterations)
│   ├── large_scale_timing.json         # Wall-clock per variant at n=25K, 50K, 100K
│   ├── parity_check.json               # |approx − exact| per sample_fraction × n
│   ├── scaling_table.md                # Derived: speedup, scaling exponent per variant
│   ├── speedup_by_variant.png          # Bar chart of speedup at n=10K
│   ├── loglog_scaling.png              # log-log wall-clock vs n for each variant
│   └── .gitkeep
└── report.md                           # Written by write-report after experiment
```

**Source-tree artifacts** (required by the experiment but created in their canonical locations):

| File | Location | Purpose |
|------|----------|---------|
| `trustworthiness_bench.rs` | `benches/` | Criterion end-to-end + per-optimization benchmarks |
| `tw_step_profiler.rs` | `src/bin/` | Per-step Instant timing at n=10K |
| `tw_large_scale.rs` | `src/bin/` | Large-scale wall-clock timing for all variants |
| New `[[bench]]` + `[[bin]]` entries | `Cargo.toml` | Wire up all new targets |

## Environment

**Custom environment required (lightweight):**

The Criterion benchmarks and timing binaries run entirely with `cargo bench --features testing`
or `cargo run --release --features testing` — no Python runtime is needed for these.

A minimal Python environment is required for:
1. `sklearn_parity.py` — computing exact sklearn trustworthiness at n=5K, 10K, 50K to validate
   the 0.01 benchmark gate for approximate variants
2. `analyze_results.py` — parsing JSON result files and generating scaling plots and tables

Check whether the existing environment satisfies requirements before creating a new one:
```bash
conda run -p envs/spectral-test python -c "import sklearn, matplotlib, pandas; print('OK')"
```

If the existing `envs/spectral-test` environment satisfies all requirements, use it directly and
skip creating `environment.yml`. If not, create it with:

```yaml
name: tw-perf-scaling
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.26
  - scikit-learn>=1.4
  - matplotlib>=3.8
  - pandas>=2.0
```

Rationale for each dependency: `scikit-learn` for the sklearn reference trustworthiness scores;
`numpy` for array generation and JSON result loading; `matplotlib` for scaling plots; `pandas`
for tabular result aggregation.

## Implementation Phases

### Phase 1: Directory Structure and Cargo Wiring

**Goal:** Create the research directory, verify the public API is accessible from benches, and
add all new Cargo targets.

**Steps:**

1. Create directory structure:
   ```bash
   mkdir -p research/2026-04-04-trustworthiness-perf-scaling/{scripts,data,results}
   touch research/2026-04-04-trustworthiness-perf-scaling/data/.gitkeep
   touch research/2026-04-04-trustworthiness-perf-scaling/results/.gitkeep
   ```

2. Verify public export: Check `src/lib.rs` — confirm `trustworthiness` is re-exported as
   `pub`. Currently in `src/metrics.rs` as `pub fn trustworthiness`. Check if `lib.rs` exposes
   it under the `testing` feature. If not, add:
   ```rust
   #[cfg(feature = "testing")]
   pub use metrics::trustworthiness;
   ```

3. Add to `Cargo.toml` (after existing `[[bench]]` entries):
   ```toml
   [[bench]]
   name = "trustworthiness_bench"
   harness = false
   required-features = ["testing"]

   [[bin]]
   name = "tw_step_profiler"
   path = "src/bin/tw_step_profiler.rs"
   required-features = ["testing"]

   [[bin]]
   name = "tw_large_scale"
   path = "src/bin/tw_large_scale.rs"
   required-features = ["testing"]
   ```

4. Verify compilation: `cargo build --release --features testing 2>&1 | tail -5`

### Phase 2: Criterion End-to-End Baseline Benchmark

**Goal:** A working `cargo bench --features testing --bench trustworthiness_bench` that
measures full-function trustworthiness at n=1K, 5K, 10K.

**Create `benches/trustworthiness_bench.rs`:**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use spectral_init::trustworthiness;
use std::hint::black_box;
use std::time::Duration;

fn generate_data(n: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 10), |_| StandardNormal.sample(&mut rng));
    let y = Array2::from_shape_fn((n, 2), |_| StandardNormal.sample(&mut rng));
    (x, y)
}

fn bench_n1k(c: &mut Criterion) {
    let (x, y) = generate_data(1_000, 42);
    let mut group = c.benchmark_group("trustworthiness_n1k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));
    group.bench_function("baseline", |b| {
        b.iter(|| black_box(trustworthiness(black_box(x.view()), black_box(y.view()), black_box(15))))
    });
    group.finish();
}

fn bench_n5k(c: &mut Criterion) {
    let (x, y) = generate_data(5_000, 42);
    let mut group = c.benchmark_group("trustworthiness_n5k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(60));
    group.bench_function("baseline", |b| {
        b.iter(|| black_box(trustworthiness(black_box(x.view()), black_box(y.view()), black_box(15))))
    });
    group.finish();
}

fn bench_n10k(c: &mut Criterion) {
    let (x, y) = generate_data(10_000, 42);
    let mut group = c.benchmark_group("trustworthiness_n10k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("baseline", |b| {
        b.iter(|| black_box(trustworthiness(black_box(x.view()), black_box(y.view()), black_box(15))))
    });
    group.finish();
}

criterion_group!(benches, bench_n1k, bench_n5k, bench_n10k);
criterion_main!(benches);
```

**Verification:** `cargo bench --features testing --bench trustworthiness_bench -- trustworthiness_n1k 2>&1 | head -20`

Expected: Criterion warm-up and measurement output, no compilation errors.

### Phase 3: Per-Step Profiling Binary

**Goal:** Measure the wall-clock contribution of each of the 6 steps at n=10K to establish
the bottleneck distribution.

**Create `src/bin/tw_step_profiler.rs`:**

The binary:
- Parses `--n <usize>` argument (default 10000) via `pico_args` (already in Cargo.toml)
- Generates (n, 10) X and (n, 2) Y from `SmallRng::seed_from_u64(42)` + `StandardNormal`
- Runs 3 warmup full-function calls, then 5 timed single-row profiling iterations
- For the per-step timing: measures each step for a representative sample of rows (rows 0,
  n/4, n/2, 3n/4, n-1 — 5 rows covering cache-cold and cache-warm states)
- Each step is implemented as `#[inline(never)]` helper functions to prevent inlining from
  collapsing step boundaries

**Per-step helper function signatures:**
```rust
#[inline(never)]
fn step1_x_dist(x: ArrayView2<f64>, row_i: usize) -> Vec<(f64, usize)>
  // Returns sorted unsorted distances from x[row_i] to all rows

#[inline(never)]
fn step2_x_sort(dist_x: &mut Vec<(f64, usize)>)
  // sort_unstable_by in-place

#[inline(never)]
fn step3_x_rank_scatter(sorted: &[(f64, usize)], n: usize) -> Vec<usize>
  // scatter ranks into rank_x[j] = rank

#[inline(never)]
fn step4_x_knn_set(sorted: &[(f64, usize)], k: usize) -> HashSet<usize>
  // collect 1..=k indices

#[inline(never)]
fn step5_y_dist_heap(y: ArrayView2<f64>, row_i: usize, k: usize) -> BinaryHeap<(u64, usize)>
  // streaming top-k Y distances via max-heap

#[inline(never)]
fn step6_penalty(heap: &BinaryHeap<(u64, usize)>, rank_x: &[usize],
                 knn_set: &HashSet<usize>, k: usize) -> u64
  // penalty accumulation
```

**Output format:** JSON written to
`research/2026-04-04-trustworthiness-perf-scaling/results/step_timing_n10k.json`:
```json
{
  "n": 10000, "k": 15, "rayon_threads": 8, "iterations": 5,
  "rows_sampled": [0, 2500, 5000, 7500, 9999],
  "step_times_µs": {
    "x_dist": [[...5 values per row...]],
    "x_sort": [[...]],
    "x_rank_scatter": [[...]],
    "x_knn_set": [[...]],
    "y_dist_heap": [[...]],
    "penalty": [[...]]
  },
  "step_mean_µs": { "x_dist": ..., ... },
  "step_fraction_pct": { "x_dist": ..., ... }
}
```

**Verification:** `cargo run --release --features testing --bin tw_step_profiler -- --n 1000`
Expected: JSON output with 6 step entries, fractions summing to ~100%.

### Phase 4: Optimization Variant Implementations

**Goal:** Implement all optimization variants for benchmarking. These are research-temporary
implementations in `src/metrics.rs` or a dedicated `src/metrics_research.rs` module gated
behind `#[cfg(feature = "testing")]`.

**Variant 4a — Partial select (`trustworthiness_partial_select`):**

Replace Step 2 (full sort) with `select_nth_unstable_by` + lazy rank computation:
1. Partition: `dist_x.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0))` — puts k+1
   smallest in positions 0..=k (unordered)
2. X k-NN set: collect indices from `dist_x[0..=k]` (skip self at index 0)
3. Lazy rank for Y-kNN penalties: for each j in Y-kNN heap, count
   `dist_x.iter().filter(|(d, _)| *d < dist_x_j).count()` — O(n·k) but k=15 ≈ O(n)
4. Eliminates full O(n log n) sort; replaces with O(n) partial select + O(n·k) lazy rank

Note: this changes the inner loop structure but not the O(n²) scaling of the outer rayon loop.

**Variant 4b — Thread-local buffers (`trustworthiness_thread_local`):**

```rust
thread_local! {
    static DIST_X_BUF: RefCell<Vec<(f64, usize)>> = RefCell::new(Vec::new());
    static RANK_X_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}
```

Inside the rayon `par_iter` map closure:
- `DIST_X_BUF.with(|buf| { let mut v = buf.borrow_mut(); v.resize(n, (0.0, 0)); ... })`
- Eliminates per-row `Vec::new()` + allocation; reuses already-allocated capacity

Verify rayon thread-local safety: rayon threads are OS threads with `thread_local!` storage
— this is correct and safe. Each row within a thread reuses the same buffer.

**Variant 4c — AVX2 SIMD distance kernel (`trustworthiness_simd`):**

1. First, check auto-vectorization: `cargo rustc --release --features testing -- --emit=asm 2>/dev/null | grep -c ymm` — if >0, LLVM is already vectorizing; explicit intrinsics may yield only 1.2-1.5x gain.

2. Implement `compute_sq_dist_avx2(a: *const f64, b: *const f64, len: usize) -> f64` in
   `src/metrics.rs` (or `src/simd_distance.rs`) following `src/operator.rs:96-145` pattern:
   - Process d=10 in two full 4-wide iterations (indices 0-3, 4-7) + 2 scalar tail (8-9)
   - Per 4-wide iteration: `_mm256_loadu_pd(a+i)`, `_mm256_loadu_pd(b+i)`, `_mm256_sub_pd`,
     `_mm256_fmadd_pd(diff, diff, acc)` — no gather needed (contiguous access)
   - Horizontal reduction: `_mm256_hadd_pd(acc, acc)` + cross-lane add via `_mm256_extractf128_pd`
   - Runtime dispatch: `#[target_feature(enable = "avx2,fma")]` with fallback scalar path

3. Wrap with: `if is_x86_feature_detected!("avx2") { compute_sq_dist_avx2(...) } else { scalar_sq_dist(...) }`

4. Plug into `trustworthiness_simd`: replace Step 1 and Step 5 distance loops with the
   AVX2 kernel.

**Variant 4d — Row subsampling (`trustworthiness_sampled`):**

```rust
pub fn trustworthiness_sampled(
    x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize,
    sample_fraction: f64, seed: u64
) -> f64
```

Implementation:
- `m = (n as f64 * sample_fraction).ceil() as usize; m = m.max(k + 1)`
- Sample m indices without replacement: `SmallRng::seed_from_u64(seed)` + Knuth shuffle or
  reservoir sampling from `0..n`
- Run the trustworthiness inner loop over only the m sampled anchor rows
- Normalize: `1.0 - (2.0 * penalty_sum) / (m as f64 * k as f64 * (2*n - 3*k - 1) as f64)`
- Accept sample_fraction values: 0.05, 0.10, 0.20, 0.50

**Variant 4e — Combined exact (`trustworthiness_combined`):**
Applies partial_select + thread_local + simd_avx2 together. Implemented as a single function
combining all three modifications.

### Phase 5: Optimization Benchmark Extensions

**Goal:** Extend `benches/trustworthiness_bench.rs` to benchmark all variants at n=10K.

Add a new benchmark group `trustworthiness_opts_n10k`:
- `baseline`: existing `trustworthiness`
- `partial_select`: `trustworthiness_partial_select`
- `thread_local`: `trustworthiness_thread_local`
- `simd_avx2`: `trustworthiness_simd`
- `sampled_10pct`: `trustworthiness_sampled(..., 0.10, 42)`
- `combined`: `trustworthiness_combined`

All variants use identical (n=10K, seed=42) X and Y arrays generated once outside `b.iter()`.
Parameters: `sample_size(10)`, `measurement_time(Duration::from_secs(120))`.

### Phase 6: Large-Scale Timing Binary

**Goal:** Measure wall-clock time for all variants at n=25K, 50K, 100K where Criterion is
infeasible.

**Create `src/bin/tw_large_scale.rs`:**

- `--ns 25000,50000,100000` (comma-separated list of n values)
- `--variants baseline,partial_select,thread_local,simd_avx2,sampled_10pct,sampled_20pct,combined`
- `--iterations 2` (1-3; use 1 for n=100K to limit runtime)
- `--output <path>` (JSON output path)
- For each (variant, n): generate data with SmallRng seed 42; call warmup (1 iter); time
  `iterations` calls with `Instant::now()` / `elapsed()`; record mean_ms and std_ms
- Log `RAYON_NUM_THREADS` from environment (or rayon thread count) into the JSON

Output JSON (append-style, one record per variant×n):
```json
[
  { "variant": "baseline", "n": 25000, "mean_ms": ..., "std_ms": ..., "iterations": 2, "rayon_threads": 8 },
  ...
]
```

Written to `research/2026-04-04-trustworthiness-perf-scaling/results/large_scale_timing.json`.

### Phase 7: Parity Verification

**Goal:** Confirm exact variants preserve sklearn parity and approximate variants stay within 0.01 gate.

**Create `scripts/sklearn_parity.py`:**
```python
import numpy as np
from sklearn.manifold import trustworthiness
import json, sys

results = []
for n in [5000, 10000, 50000]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 10))
    Y = rng.standard_normal((n, 2))
    score = trustworthiness(X, Y, n_neighbors=15)
    results.append({"n": n, "sklearn_score": score})
    print(f"n={n}: sklearn={score:.8f}")

with open("research/2026-04-04-trustworthiness-perf-scaling/results/sklearn_reference.json", "w") as f:
    json.dump(results, f, indent=2)
```

Then compare against Rust CLI output (`src/bin/trustworthiness.rs`) for each exact variant and
against `trustworthiness_sampled` CLI for subsampling variants. Collect deltas into
`results/parity_check.json`.

Note: the Rust and Python random generators produce different arrays for the same seed, so
exact match is not expected. Parity check is done by comparing Rust exact vs Rust approximate,
not Rust vs sklearn. For the approximation gate: run `trustworthiness_sampled` at
sample_fraction=0.05/0.10/0.20/0.50 on the same (n, 42) arrays used in the timing benchmarks,
compare against `trustworthiness_baseline` on those same arrays.

For sklearn parity (exact implementations only): run `src/bin/trustworthiness` on `.npy` files
saved from the Python arrays (save X.npy, Y.npy from sklearn_parity.py), compare to sklearn
output — this verifies the baseline is sklearn-accurate before comparing optimizations to it.

### Phase 8: Analysis and Report Preparation

**Create `scripts/analyze_results.py`:**
- Loads `step_timing_n10k.json`, `large_scale_timing.json`, `parity_check.json`, Criterion
  HTML/JSON outputs from `target/criterion/`
- Computes:
  - Per-step fraction table (sorted descending)
  - Speedup table: for each variant at n=1K, 5K, 10K, 25K, 50K, 100K
  - Scaling exponent fit: `np.polyfit(np.log(ns), np.log(times), 1)` → slope = exponent
  - Parity gate pass/fail table for subsampling variants
- Writes `results/scaling_table.md` (markdown)
- Saves `results/speedup_by_variant.png` (bar chart, n=10K and n=50K side-by-side)
- Saves `results/loglog_scaling.png` (line plot: log-log wall-clock vs n, one line per variant)

## Execution Protocol

```bash
# 0. Verify environment
conda run -p envs/spectral-test python -c "import sklearn, matplotlib, pandas; print('OK')" || \
  micromamba env create -f research/2026-04-04-trustworthiness-perf-scaling/environment.yml

# 1. Verify compilation
export RAYON_NUM_THREADS=8
cargo build --release --features testing 2>&1 | tail -5

# 2. Criterion end-to-end baseline (n=1K, 5K, 10K)
cargo bench --features testing --bench trustworthiness_bench -- trustworthiness_n 2>&1 | \
  tee research/2026-04-04-trustworthiness-perf-scaling/results/criterion_output.txt

# 3. Per-step profiling at n=10K
cargo run --release --features testing --bin tw_step_profiler -- --n 10000
# Writes: research/2026-04-04-trustworthiness-perf-scaling/results/step_timing_n10k.json

# 4. Criterion optimization variants (n=10K)
cargo bench --features testing --bench trustworthiness_bench -- trustworthiness_opts_n10k 2>&1 | \
  tee -a research/2026-04-04-trustworthiness-perf-scaling/results/criterion_output.txt

# 5. Large-scale timing (n=25K, 50K, 100K) — may take 30-120 min
cargo run --release --features testing --bin tw_large_scale -- \
  --ns 25000,50000,100000 \
  --variants baseline,partial_select,thread_local,simd_avx2,sampled_10pct,sampled_20pct,combined \
  --iterations 2 \
  --output research/2026-04-04-trustworthiness-perf-scaling/results/large_scale_timing.json

# 6. Parity verification (exact vs sklearn, approximate vs exact)
conda run -p envs/spectral-test python \
  research/2026-04-04-trustworthiness-perf-scaling/scripts/sklearn_parity.py

# 7. Analysis and plots
conda run -p envs/spectral-test python \
  research/2026-04-04-trustworthiness-perf-scaling/scripts/analyze_results.py
# Writes: results/scaling_table.md, results/speedup_by_variant.png, results/loglog_scaling.png
```

## Analysis Plan

**Bottleneck identification:** From `step_timing_n10k.json`, compute the mean fraction per
step. Key decision rule: if `x_distance > 50%` → SIMD is the primary target; if `x_sort > 25%`
→ partial select is primary; if combined alloc steps (per-row Vec creation) > 10% →
thread-local buffers are material.

**Speedup ranking table (target format):**

| Variant | n=10K speedup | n=50K speedup | n=100K speedup | Parity gate | Recommendation |
|---------|--------------|--------------|---------------|-------------|----------------|
| partial_select | ? | ? | ? | PASS | ? |
| thread_local | ? | ? | ? | PASS | ? |
| simd_avx2 | ? | ? | ? | PASS | ? |
| sampled_10pct | ? | ? | ? | ? | ? |
| sampled_20pct | ? | ? | ? | ? | ? |
| combined | ? | ? | ? | PASS | ? |

**Scaling exponent:** Fit `log(t) = α·log(n) + β` for each variant over n=5K, 10K, 25K, 50K,
100K. Expected: α≈2.0 for exact algorithms; α≈1.0 for 10% subsampling. A measured α
significantly above 2.0 would indicate cache-bound degradation at scale.

**Parity gate for subsampling:** For each sample_fraction ∈ {0.05, 0.10, 0.20, 0.50}, compute
`|trustworthiness_sampled(seed=42) − trustworthiness_baseline|` at n=10K and n=50K over 5
different seeds (42, 43, 44, 45, 46). Report mean and max delta. Minimum safe fraction =
smallest fraction where max(delta) < 0.01.

**AVX2 auto-vectorization check:** Before the SIMD run, check
`cargo rustc --release --features testing -- --emit=asm 2>/dev/null | grep -c ymm`. Report
whether LLVM already vectorizes. If yes, predict that explicit AVX2 gains will be ≤1.5x.

## Success Criteria

- **Conclusive positive (supports H1):** Per-step breakdown shows a single step >40% at n=10K;
  at least one exact optimization achieves ≥2x speedup at n=10K; row subsampling at fraction=0.10
  achieves ≥8x speedup at n=50K with max(|Δ|) < 0.01 across 5 seeds; a ranked GO/NO-GO table
  with ≥5 entries and rationale is produced.

- **Conclusive negative (supports H0):** All exact optimization speedups are <1.5x at n=10K
  (LLVM already auto-vectorizes; sort is not a bottleneck; allocations are negligible); row
  subsampling at fraction=0.10 has max(|Δ|) ≥ 0.01; conclusion: the O(n²) algorithm requires
  a structural change (KD-tree + approximate ranking) beyond the scope of constant-factor
  tuning.

- **Inconclusive:** Step fractions are roughly uniform (10-20% each) with no clear target;
  Criterion variance >50% coefficient of variation for any variant; parity check borderline
  (0.008-0.012 Δ).

## Threats to Validity

### Internal

- **Synthetic data lacks structure:** Real MERFISH cell neighborhoods have biological structure;
  trustworthiness hot-path cache behavior and score distribution may differ. Mitigate by also
  timing against the MERFISH NPZ fixture if `temp/lobpcg_bench/merfish_10k_laplacian.npz`
  can be extended to include (X, Y) arrays.

- **Single-row step profiling:** The per-step timing binary measures individual rows, not the
  full rayon parallel execution. Rayon scheduling, thread synchronization, and work-stealing
  overhead are not captured in step-level timings. Mitigate by also measuring end-to-end
  wall-clock time for the full function and comparing to sum-of-steps.

- **RAYON_NUM_THREADS assumption:** Results are specific to RAYON_NUM_THREADS=8. On machines
  with fewer cores, speedup ratios may differ due to reduced parallelism benefit. Mitigate
  by logging the actual rayon thread count in all result JSON files.

- **Auto-vectorization uncertainty:** LLVM with `target-cpu=native` may already vectorize
  the distance loop, making the explicit AVX2 variant's measured benefit the marginal gain
  over auto-vectorization (not over scalar). This is the expected measurement — report it
  as "gain over auto-vectorized baseline."

### External

- **d_x=10 specificity:** SIMD and partial-select gains are strongly sensitive to d_x. For
  d_x=50 or d_x=2, the speedup profile would be qualitatively different. Results are specific
  to the MERFISH (d_x=10, d_y=2) configuration.

- **n ≤ 100K coverage:** Large-scale timing is capped at n=100K to limit runtime. Extrapolation
  to n=250K or n=4.2M relies on the measured scaling exponent and may underestimate cache-cold
  degradation at larger scales.

- **Hardware specificity:** AVX2+FMA results assume the development machine (verified via
  `is_x86_feature_detected!`). On ARM (Apple Silicon) or machines without AVX2, the SIMD
  variant would fall back to the scalar path and show no gain.

## Estimated Resource Requirements

| Task | Estimated Wall-Clock | Notes |
|------|---------------------|-------|
| Criterion baseline (n=1K, 5K, 10K) | ~15-30 min | 10 samples × 30-120s per group |
| Criterion optimization variants (n=10K) | ~20-60 min | 6 variants × 10 samples × 120s each |
| Per-step profiling (n=10K) | ~10 min | 5 timed iterations × 5 sampled rows |
| Large-scale timing (n=25K-100K, all variants) | ~60-120 min | O(n²) at n=100K ≈ minutes per call |
| Python parity + analysis | ~5-10 min | sklearn at n=50K is fast |
| **Total** | **~2-4 hours** | Dominated by large-scale timing phase |

Disk space: <200 MB (Criterion HTML reports + result JSON files; no large fixture files).

Additional Cargo dependencies: none beyond existing `rand`, `rand_distr`, `ndarray`, `pico_args`.
The `unsafe` AVX2 intrinsics in the SIMD variant follow the existing `src/operator.rs` pattern
and require no new dependencies.
