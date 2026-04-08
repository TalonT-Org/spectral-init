# Experiment Plan: Trustworthiness Performance — Scaling Analysis and Optimization Evaluation

## Motivation

The `trustworthiness()` function in `src/metrics.rs:397-457` is the dominant runtime
bottleneck for the MERFISH evaluation pipeline at practical scales. At n=250K the
process was killed after 7+ minutes at 805% CPU; at n=100K it consumed 651 of 683
total pipeline seconds. No Criterion benchmark, no per-step wall-clock measurement,
and no validated fast path exist anywhere in the codebase.

This experiment will produce a ranked GO/NO-GO recommendation for each candidate
optimization, grounded in measured wall-clock speedups (not theoretical ones) at
production scale (n=100K–250K). The results will directly inform whether to ship
partial-rank computation, thread-local buffers, manual AVX2, and/or row subsampling
as part of the next MERFISH evaluation release.

This plan is the third revision of a prior plan that received two sequential STOP
verdicts. It incorporates all four required fixes from the resolve-design-review
guidance dated 2026-04-04 17:25.

---

## Hypothesis

**Null hypothesis (H0):** No single algorithmic step dominates the wall-clock; the
current implementation's scaling behavior cannot be improved by targeted optimization
without approximation.

**Alternative hypothesis (H1_combined):** At least one exact optimization
(thread-local buffer reuse, partial-rank X computation, or verified auto-vectorization)
achieves ≥1.5× Criterion speedup at n=50K AND ≥1.5× wall-clock speedup at n=100K,
without any change to the output T(k) value.

### Sub-hypotheses

**H1 — X-sort dominates wall-clock; partial-rank computation yields the largest
single-step speedup.**

*Claim:* X-sort (`sort_unstable_by` + rank scatter) accounts for ≥40% of per-row
wall-clock at n≥10K, making it the primary optimization target.

*Falsifiable:* Per-step profiling at n=10K/50K/100K; if X-sort < 30% of total at
n=100K, H1 is refuted and thread-local buffers become the primary target.

---

**H2 — Thread-local buffer reuse provides ≥1.5× throughput improvement at n=100K.**

*Claim:* The 3 fresh Vec allocations per row (2 × n-length Vecs ≈ 3.2 MB per row at
n=100K) generate ~100K × 3 = 300K allocator round-trips per call. Wrapping `dist_x`
and `rank_x` in `thread_local! { static ... } RefCell<Vec>` eliminates per-row
alloc/dealloc cost.

*Falsifiable:* Criterion benchmark with and without thread-local reuse at n=5K–50K;
additionally `tw_profiler` wall-clock at n=100K with `--warmup 2 --iters 5`.
**GO requires ≥1.5× speedup at n=100K (tw_profiler) in addition to Criterion CI
gate at n≤50K.** *(Fix 1)*

---

**H3 — The trustworthiness() inner distance loop is auto-vectorized to AVX2 by the
Rust compiler at `target-cpu=native`; manual AVX2 intrinsics provide negligible
additional benefit.**

*Claim:* With `rustflags = ["-C", "target-cpu=native"]` and the `sum((a-b)^2)` loop
pattern, LLVM auto-vectorizes the d=10 f64 distance loop to use YMM registers and
FMA instructions, making manual intrinsics redundant.

*Falsifiable:* Inspect `cargo asm` output for the distance inner loop. Decision rule:
if ≥1 AVX2/FMA instruction (`vmovupd`, `vfmadd231pd`, `ymm`) appears in the inner
loop body → H3 confirmed (auto-vectorized) → NO-GO for manual SIMD intrinsics. If no
AVX2 instructions present → H3 refuted → implement manual AVX2 kernel.

*(Fix 2 — Option A applied: H3 is defined as a binary detection hypothesis, not a
quantitative ratio test. The prior claim of "< 2× over auto-vectorized baseline" has
been removed because the cargo asm check resolves it without needing a measurement.
If a quantitative bound is desired for supplementary evidence, the `bench_tw_avx2`
Criterion group provides it, but the GO/NO-GO is determined by the asm inspection.)*

---

**H4 — AVX-512 provides marginal or negative benefit for the d=10 f64 distance
kernel due to poor register utilization (62.5%) and potential frequency throttling.**

*Claim:* 8-wide AVX-512 at d=10 uses only 10/16 = 62.5% of the first register and
may trigger core downclocking on affected Intel microarchitectures, offsetting the
wider-lane benefit.

*Falsifiable:* Direct `tw_profiler` benchmark: scalar baseline vs avx2_kernel vs
avx512_kernel at n=100K with `--warmup 2 --iters 5`. **GO requires ≥1.5× speedup
over AVX2 at n=100K (tw_profiler).** NO-GO if AVX-512 speedup < 1.2× over AVX2
at n=100K. *(Fix 1 extended to H4)*

---

**H5 — Row subsampling at m=5000 provides ≥5× speedup with < 0.001 absolute
deviation from exact on MERFISH-type structured data.**

*Claim:* T(k) is a row-mean; sampling m=5000 rows reduces work from O(n² log n) to
O(m·n log n), yielding ~50× at n=250K. The error stays below 0.001 on MERFISH
structured data (MERFISH is more concentrated than Gaussian, so the error may be
tighter than the O(1/sqrt(m)) Gaussian bound).

*Pre-registered value:* m=5000 (sealed in `h5_confirmatory_result.json` before any
sweep runs). The exploratory m-sweep {500, 1000, 2000, 5000, 10000} is secondary
analysis only and does NOT determine the GO threshold.

*Parity gate:* Evaluated on MERFISH-derived PCA-50 data (`tests/fixtures/merfish/`),
NOT on Gaussian data (concentration of measure makes Gaussian results vacuous). A
|T_approx - T_exact| < 0.001 threshold is required on MERFISH data.

*Note:* The existing sklearn parity threshold of `< 1e-6` does NOT apply to any
approximate implementation. H5 requires a separate explicit threshold.

---

**H6 — Combined optimization (thread-local + partial-rank + verified auto-vectorized
distance) achieves 3–6× end-to-end speedup at n=100K with no correctness change.**

*Claim:* These three optimizations are composable, all exact (no approximation), and
each targets a different bottleneck (allocations, sort, compute). Their combined
effect is subadditive but expected in the 3–6× range.

*Combined variant composition pre-locked:* `thread_local + partial_rank +
avx2_kernel` (where avx2_kernel means the auto-vectorized baseline if H3 is
confirmed, otherwise the explicit manual kernel). This composition is fixed in this
plan document and must not be adjusted based on individual speedup results.

*Falsifiable:* Criterion benchmark at n=5K–50K AND `tw_profiler` at n=100K with
`--warmup 2 --iters 5`. GO requires ≥3× wall-clock speedup at n=100K.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm variant | `baseline`, `thread_local`, `partial_rank`, `avx2_kernel`, `avx512_kernel`, `combined` | Isolates each optimization; `combined` is pre-locked composition |
| Dataset size n | 1K, 5K, 10K, 25K, 50K, 100K (and 250K for approximate path only) | Covers practical range; 100K is production target |
| Data structure | Gaussian (throughput), make_blobs 8-center (parity gate), MERFISH-PCA50 (H5 gate) | Separates throughput measurement from correctness verification |
| Subsampling rows m (H5 only) | 500, 1000, 2000, 5000, 10000 | Pre-registered: m=5000 is the confirmatory value |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Per-step wall-clock fraction | % of total | `#[cfg(feature="testing")]` instrumentation in `src/metrics.rs`, emitted via `[timing:tw_*]` | NEW — no entry in `src/metrics.rs` catalog; throughput diagnostic only |
| Wall-clock per iteration (warm) | seconds | `tw_profiler` binary: `--warmup 2 --iters 5`, discard warmup, mean ± std of 5 | NEW |
| Criterion speedup ratio | ratio (95% CI) | `benches/trustworthiness_bench.rs`, Criterion 0.5 `bench.iter()` | NEW |
| AVX2 instruction presence | boolean | `cargo asm` inspection for `ymm`/`vfmadd231pd`/`vmovupd` in inner loop | NEW — H3 only |
| Trustworthiness parity |T_rust - T_sklearn| absolute | `sklearn_reference.py` computes sklearn T; `tw_profiler` (baseline) computes Rust T | Implicit in `tests/integration/test_trustworthiness.rs`; threshold < 1e-6 for exact variants |
| Subsampling absolute deviation | |T_approx - T_exact| absolute | `tw_approx_runner` binary, seeded with --seed 42 for confirmatory run | NEW — approximate path only; threshold < 0.001 on MERFISH data |

All performance metrics are NEW (no existing `MetricResult` catalog entries for
trustworthiness performance). The existing `AssessmentReport` / `MetricResult`
framework in `src/metrics.rs` covers eigensolver quality only; trustworthiness parity
is enforced only via an `assert!` in the integration test, not through the catalog.
No catalog additions are required before the experiment — these metrics are
experiment-internal.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| Neighborhood size k | 15 | Matches production use and existing parity test fixture |
| Input dimensionality d_x | 10 | MERFISH PCA-50 reduced to 10 for synthetic; 50 for real MERFISH |
| Embedding dimensionality d_y | 2 | Standard UMAP output dimension |
| Warmup iterations | 2 (discarded) | Eliminates OS page-fault cost and JIT effects from measurements (STOP-1 fix) |
| Measurement iterations | 5 | Sufficient for mean ± std; matches `tw_profiler` default |
| H5 confirmatory seed | 42 | Reproducibility of sealed GO gate (Fix 4) |
| Gaussian data RNG seed | 0 | Reproducible synthetic inputs |
| make_blobs data seed | 1 | Reproducible structured synthetic inputs |
| Combined variant composition | thread_local + partial_rank + avx2_kernel | Pre-locked before Phase 3 measurements; no post-hoc adjustment |
| Per-variant n=100K validation | Required for H2, H3, H4 | Prevents Goodhart exploitation at sub-production scale (Fix 1) |
| H5 pre-registered m | 5000 | Pre-registered before any sweep (STOP-2 fix); sweep is exploratory only |
| Parity gate data | MERFISH-PCA50, NOT Gaussian | Gaussian concentration of measure renders Gaussian parity gate vacuous (STOP-3 fix) |

---

## Inputs and Data

The experiment requires three classes of data, each serving a distinct validation purpose.

**Synthetic Gaussian data** is used for throughput benchmarking and scaling-law
characterization only. Gaussian vectors at d=10 exhibit concentration of measure
that makes parity checks meaningless; these datasets are labelled "throughput
characterization only" and are NOT used for any GO/NO-GO correctness decision.

**Structured synthetic data** (`make_blobs`, 8 centers, d=10, spread=2.0) is used
for secondary parity verification of exact optimizations. Blobs data has separable
cluster structure where rank orderings differ meaningfully across rows, making it a
more discriminating test of correctness than Gaussian.

**MERFISH-derived PCA-50 data** is the definitive correctness gate for H5. Extracted
from the real biological dataset in `temp/merfish_100k/`, reduced to 10K rows ×
50 PCA components for feasibility. Spatial coordinates are used as the Y embedding.
This data has concentrated cluster structure representative of the production use
case. The 10K subset is committed to `tests/fixtures/merfish/` as a reproducible
artifact (Fix 3).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| gaussian_n{1,5,10,25,50,100}k_x/y.npy | `gen_synthetic.py`, seed=0 | d_x=10, d_y=2, StandardNormal | Throughput benchmarks (H1-H4, H6); NOT for correctness |
| blobs_n{1,5,10}k_x/y.npy | `gen_synthetic.py`, seed=1 | d_x=10, d_y=2, k=8 centers | Secondary parity verification for exact variants |
| merfish_n10k_x.npy | `prepare_merfish.py`, committed to tests/fixtures/merfish/ | 10000×50 f64 PCA-50 from MERFISH 100K expression | H5 parity gate (primary); sklearn parity reference |
| merfish_n10k_y.npy | `prepare_merfish.py`, committed to tests/fixtures/merfish/ | 10000×2 f64 spatial coordinates | H5 parity gate (primary) |

The MERFISH fixture originates from Allen Brain Cell Atlas Mouse Brain MERFISH data
(2023), derived from `temp/merfish_100k/merfish_100k_expression.npz` via PCA-50 reduction.

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-04-tw-perf-scaling/
├── environment.yml                    # Extends spectral-test: numpy, scipy, sklearn, scikit-learn
├── scripts/
│   ├── gen_synthetic.py               # Generate Gaussian + blobs data at all n values
│   ├── prepare_merfish.py             # Extract PCA-50 from temp/ → tests/fixtures/merfish/ + data/merfish/
│   ├── sklearn_reference.py           # Compute sklearn T(k=15) for parity reference
│   ├── subsampling_sweep.py           # H5 m-sweep with --seed parameter (Fix 4)
│   ├── analyze_results.py             # Produce ranked recommendation table
│   ├── dry_run.sh                     # Quick smoke test (n=1K only)
│   ├── run_profiling.sh               # tw_profiler: baseline + all variants at all n
│   ├── run_criterion.sh               # Criterion benchmarks + cargo asm inspection (H3)
│   ├── run_h5_confirmatory.sh         # Sealed H5 gate: m=5000, --seed 42, MERFISH only
│   └── run_subsampling_sweep.sh       # Exploratory m-sweep (checks for sealed file first)
├── data/
│   ├── gaussian/                      # gaussian_n{1k,5k,10k,25k,50k,100k}_{x,y}.npy
│   ├── blobs/                         # blobs_n{1k,5k,10k}_{x,y}.npy
│   └── merfish/                       # Symlinks or copies from tests/fixtures/merfish/
├── results/
│   ├── step_timing/                   # Per-step JSON from tw_profiler (baseline)
│   ├── criterion/                     # Criterion HTML report artifacts
│   ├── asm/                           # cargo asm output for H3 inspection
│   ├── subsampling/                   # h5_confirmatory_result.json + sweep JSON
│   └── analysis/                      # ranked_recommendations.md + final figures
└── report.md                          # Written by write-report skill
```

### File Descriptions

**`environment.yml`** — Conda environment inheriting from the project's
`tests/environment.yml` pattern. Pins numpy 2.2, scipy 1.15, scikit-learn 1.8.
No additional Python packages beyond existing `spectral-test` env needed.

**`scripts/gen_synthetic.py`** — Generates Gaussian (StandardNormal, seed=0) and
make_blobs (8 centers, spread=2.0, seed=1) X arrays plus corresponding random
Y embeddings in 2D. Saves to `data/gaussian/` and `data/blobs/` as `.npy` f64
dense arrays at sizes n=1K, 5K, 10K, 25K, 50K, 100K.

**`scripts/prepare_merfish.py`** — Loads
`temp/merfish_100k/merfish_100k_expression.npz`, extracts expression matrix,
applies PCA (50 components, sklearn, seed=42), takes first 10K rows. Saves to
`tests/fixtures/merfish/merfish_n10k_x.npy` (10000×50 f64) and
`tests/fixtures/merfish/merfish_n10k_y.npy` (10000×2 f64, spatial coordinates).
Checks for existing fixture before deriving: if fixture exists, copies to
`data/merfish/` without re-deriving. *(Fix 3 — Option A)*

**`scripts/sklearn_reference.py`** — For each dataset in `data/`, computes
`sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` and saves to
`results/parity/sklearn_{dataset}.json`.

**`scripts/subsampling_sweep.py`** — Accepts `--x`, `--y`, `--m`, `--n-trials`,
`--seed` CLI flags. `--seed` is wired to `np.random.RandomState(args.seed)` for
row index selection. *(Fix 4)* Saves per-m JSON with mean and max delta.

**`scripts/run_h5_confirmatory.sh`** — Invokes `tw_approx_runner` with `--m 5000
--seed 42` on MERFISH data, saves to `results/subsampling/h5_confirmatory_result.json`.
This file must be committed to git as a sealed artifact before `run_subsampling_sweep.sh`
is allowed to execute. *(Fix 4)*

**`scripts/run_subsampling_sweep.sh`** — Checks for presence of
`results/subsampling/h5_confirmatory_result.json` and exits with error code 1 if
missing. Then invokes `subsampling_sweep.py` over m={500, 1000, 2000, 5000, 10000}
with `--seed 99` (different seed from confirmatory to prevent data leakage).

**`scripts/run_profiling.sh`** — Invokes `tw_profiler` for all variants at all
n values. For `baseline` at all n: emits per-step timing JSON via the
`#[cfg(feature="testing")]` instrumentation. For `thread_local`, `partial_rank`,
and `avx2_kernel` variants: includes an explicit n=100K run with
`--warmup 2 --iters 5`. *(Fix 1)* For `avx512_kernel`: if CPU supports AVX-512,
include n=100K run. Full variant × n matrix:

```bash
# Per-variant n=100K runs (Fix 1 requirement)
for VARIANT in thread_local partial_rank avx2_kernel; do
  $TW_PROFILER \
    --x data/gaussian/gaussian_n100000_x.npy \
    --y data/gaussian/gaussian_n100000_y.npy \
    --k 15 --iters 5 --warmup 2 \
    --variant "$VARIANT" \
    --output results/step_timing/gaussian_n100000_${VARIANT}.json
done
```

**`scripts/run_criterion.sh`** — Runs Criterion benchmarks. Before recording any
speedup, compares AVX2 instruction count between a clean build and the `--features
testing` build by running `cargo asm --features testing spectral_init::metrics::trustworthiness`
and checking that the YMM instruction count is identical. If counts differ, aborts
and emits a warning. After confirmation, runs `cargo criterion --bench trustworthiness_bench`.

**`scripts/analyze_results.py`** — Reads all JSON in `results/`, produces
`results/analysis/ranked_recommendations.md` with columns: approach, measured
speedup range, scaling law change (yes/no), implementation complexity (LOC
estimate), GO/NO-GO, rationale.

---

## Environment

**Custom environment required** for the Python scripts.

The experiment requires numpy, scipy, and scikit-learn, which are already declared
in `tests/environment.yml`. Rather than duplicating them, the experiment's
`environment.yml` uses the same package set. An `environment.yml` is created in the
research directory for self-containment, but no packages beyond the existing
`spectral-test` environment are needed.

```yaml
name: tw-perf-scaling
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2
  - scipy=1.15
  - scikit-learn=1.8
  - pip
  - pip:
    - ndarray-npy  # not needed; .npy files read via numpy directly
```

Rationale: `sklearn.manifold.trustworthiness` (sklearn 1.8) for reference
computation; `numpy` for `.npy` I/O and PCA via `np.linalg.svd`; `scipy` for
optional statistical analysis. All Rust builds use the existing `target-cpu=native`
from `.cargo/config.toml`.

**No new Rust dependencies are required.** All optimization variants are implemented
using existing crate features: `rayon` for parallelism (already in `[dependencies]`),
`std::arch::x86_64` for manual SIMD (stdlib), `std::cell::RefCell` for thread-local
buffers.

---

## Implementation Phases

### Phase 0: Instrumentation (mandatory precondition for all profiling)

Add `#[cfg(feature = "testing")]` per-step timing to `trustworthiness()` in
`src/metrics.rs:397-457`, matching the exact pattern in `src/solvers/mod.rs`:

```rust
// At the top of trustworthiness():
#[cfg(feature = "testing")]
let _t_tw = std::time::Instant::now();

// Before each step:
#[cfg(feature = "testing")]
let _t_x_dist = std::time::Instant::now();
// ... step 1: X-distance computation ...
#[cfg(feature = "testing")]
eprintln!("[timing:tw_x_dist] {}µs", _t_x_dist.elapsed().as_micros());

// Repeat for: tw_x_sort, tw_rank_scatter, tw_x_knn_set, tw_y_heap, tw_penalty
// ...

// Before the single return point:
#[cfg(feature = "testing")]
eprintln!("[timing:tw_total] {}µs", _t_tw.elapsed().as_micros());
```

Six per-step timers: `tw_x_dist`, `tw_x_sort`, `tw_rank_scatter`, `tw_x_knn_set`,
`tw_y_heap`, `tw_penalty`. One total timer `tw_total`. Uses `std::time::Instant::now()`
directly (no imports) and `.as_micros()` for the emit format.

**Verification:** Run `cargo test --features testing 2>&1 | grep "\[timing:tw_"` on
the unit tests to confirm all 7 timing lines emit.

**No functional change:** This phase touches only `src/metrics.rs` and only adds
`#[cfg(feature = "testing")]`-gated `let` bindings and `eprintln!` calls. All
non-testing builds are unaffected.

### Phase 1: Directory Structure and Data Generation

1. Create directory tree:
   ```
   research/2026-04-04-tw-perf-scaling/{scripts,data/{gaussian,blobs,merfish},results/{step_timing,criterion,asm,subsampling,analysis,parity}}
   ```

2. Create `environment.yml` in the research directory (see Environment section).

3. Write and run `scripts/gen_synthetic.py`:
   - Gaussian: `np.random.RandomState(0).randn(n, 10)` for X; `np.random.RandomState(0).randn(n, 2)` for Y
   - Blobs: `sklearn.datasets.make_blobs(n, centers=8, cluster_std=2.0, random_state=1)` for X; same RandomState(1).randn(n, 2) for Y
   - Save as `.npy` f64 with `np.save()`
   - Sizes: n ∈ {1000, 5000, 10000, 25000, 50000, 100000}

4. Write and run `scripts/prepare_merfish.py`:
   - Load `temp/merfish_100k/merfish_100k_expression.npz`
   - Load `temp/merfish_100k/merfish_100k_spatial.npz`
   - Apply `sklearn.decomposition.PCA(n_components=50, random_state=42).fit_transform(expression[:10000])` → shape (10000, 50)
   - Save `tests/fixtures/merfish/merfish_n10k_x.npy` (10000×50, f64)
   - Save `tests/fixtures/merfish/merfish_n10k_y.npy` (10000×2 spatial, f64)
   - Copy both to `data/merfish/`
   - Create `tests/fixtures/merfish/` directory if missing

5. Run `scripts/sklearn_reference.py` on n=10K Gaussian, blobs, and MERFISH data.
   Verify |T_rust_baseline - T_sklearn| < 1e-6 before proceeding.

### Phase 2: Cargo Wiring and Binary Implementations

**`Cargo.toml` additions:**
```toml
[[bench]]
name = "trustworthiness_bench"
harness = false
# NOTE: No required-features — Criterion measures clean production code

[[bin]]
name = "tw_profiler"
path = "src/bin/tw_profiler.rs"
required-features = ["testing", "cli"]

[[bin]]
name = "tw_approx_runner"
path = "src/bin/tw_approx_runner.rs"
required-features = ["cli"]
```

**`src/bin/tw_profiler.rs`** CLI specification:
- `--x <path>` — high-dim input `.npy` (f64)
- `--y <path>` — embedding `.npy` (f64)
- `--k <usize>` — neighborhood size (default 15)
- `--iters <usize>` — measured iterations (default 5)
- `--warmup <usize>` — discarded warmup iterations (default 2)
- `--variant <str>` — one of: `baseline`, `thread_local`, `partial_rank`, `avx2_kernel`, `avx512_kernel`, `combined`
- `--output <path>` — JSON output path
- Output JSON: `{"variant": str, "n": int, "iters": [f64], "mean_s": f64, "std_s": f64, "warmup": int}`
- Implementation: run `--warmup` iterations (call trustworthiness or variant fn, discard result);
  then run `--iters` iterations timing each with `std::time::Instant::now()`. Uses
  `#[cfg(feature = "testing")]` per-step output for the `baseline` variant.

**`src/bin/tw_approx_runner.rs`** CLI specification:
- `--x <path>`, `--y <path>`, `--k <usize>` (default 15)
- `--sample <usize>` — number of rows to sample (m)
- `--seed <u64>` — RNG seed for row sampling
- `--output <path>` — JSON output path
- Output JSON: `{"n": int, "m": int, "seed": int, "t_exact": f64, "t_approx": f64, "delta": f64, "wall_exact_s": f64, "wall_approx_s": f64}`
- Calls `spectral_init::trustworthiness` for exact; calls `spectral_init::trustworthiness_approx`
  for approximate (new function to be added to `src/metrics.rs`).

**`benches/trustworthiness_bench.rs`** Criterion groups (no `required-features`):
- Group `bench_tw_baseline`: `bench.iter(|| trustworthiness(x.view(), y.view(), 15))` at n ∈ {1K, 5K, 10K, 25K, 50K}
- Group `bench_tw_thread_local`: same sizes, calls `trustworthiness_thread_local` variant
- Group `bench_tw_partial_rank`: same sizes, calls `trustworthiness_partial_rank` variant
- Group `bench_tw_avx2`: same sizes, supplementary data for H3 (not decision-determining)
- Group `bench_tw_combined`: same sizes, uses pre-locked combined composition

### Phase 3: Optimization Implementations

All new functions are added to `src/metrics.rs`. All must pass the 5 existing unit
tests and the `tests/integration/test_trustworthiness.rs` sklearn parity test before
being benchmarked.

**3a. Thread-local buffers (H2):**

```rust
pub fn trustworthiness_thread_local(
    x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize
) -> f64
```

Wraps `dist_x: Vec<(f64, usize)>` and `rank_x: Vec<usize>` in
`thread_local! { static DIST_X: RefCell<Vec<(f64, usize)>> = RefCell::new(Vec::new()); }`.
Each iteration: `dist_x_buf.borrow_mut().clear(); dist_x_buf.borrow_mut().resize(n, (0.0, 0));`.
Requires `use std::cell::RefCell;`. All other steps identical to baseline.

**3b. Partial-rank X computation (H1, if X-sort ≥ 40% of wall-clock):**

```rust
pub fn trustworthiness_partial_rank(
    x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize
) -> f64
```

Replaces `sort_unstable_by` + full rank scatter with:
1. Collect Y-heap results first to get the set of ≤k row indices needed
2. Compute full X-distance vector (step 1 unchanged)
3. Use `select_nth_unstable_by` at position k to partition (O(n) average, guaranteed
   O(n log n) worst-case via Median of Medians fallback since Rust 1.75)
4. For each Y-neighbor j: scan the partition to count how many X-distances are
   ≤ dist_x[j] — this gives rank_x[j] exactly
5. Accumulate penalty as before

Correctness requirement: |T_partial - T_exact| = 0 on all 5 unit tests and the n=200
sklearn parity fixture. Any non-zero delta is a bug, not an acceptable approximation.

**3c. AVX2 distance kernel (H3, conditional on asm inspection):**

Only implement if Phase 4 `cargo asm` inspection shows NO AVX2 instructions in the
distance inner loop. If auto-vectorization is confirmed → NO-GO → skip this
implementation. If needed:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2(xi: &[f64], xj: &[f64]) -> f64
```

Pattern from `src/operator.rs:98-145`: 2× `_mm256_loadu_pd` (4-wide loads for d=10),
`_mm256_sub_pd`, `_mm256_mul_pd` (or `_mm256_fmadd_pd` with zero accumulator),
horizontal reduction, scalar tail for element 8+9 (d=10 → indices 8,9). Runtime
dispatch via `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.

**3d. AVX-512 distance kernel (H4, conditional on CPU support):**

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn dist_sq_avx512(xi: &[f64], xj: &[f64]) -> f64
```

8-wide `_mm512_loadu_pd` at d=10: first pass lanes 0-7 (indices 0-7), second pass
only indices 8-9 (6 zero-padded lanes). Horizontal reduction via `_mm512_reduce_add_pd`.
Runtime dispatch via `is_x86_feature_detected!("avx512f")`. If CPU does not support
AVX-512, skip and label H4 as "N/A (hardware not available)".

**3e. Combined variant (H6, composition pre-locked):**

`trustworthiness_combined` = thread-local buffers + partial-rank X + AVX2 kernel
(or baseline auto-vectorized if H3 confirmed). Composition is fixed; implementer
must not adjust it based on individual results.

**3f. Row-subsampling approximate function (H5):**

```rust
pub fn trustworthiness_approx(
    x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize,
    sample: usize, seed: u64
) -> f64
```

Samples `sample` row indices uniformly without replacement using a seeded RNG
(`rand::rngs::SmallRng::seed_from_u64(seed)`). Computes exact trustworthiness on
sampled rows only. Returns scaled result (sum of penalties for sampled rows,
normalized by the same global denominator as the exact formula — this preserves
the T(k) ∈ [0,1] range and the same formula).

### Phase 4: Cargo ASM Inspection for H3

Run before any Criterion benchmarks to determine H3's GO/NO-GO:

```bash
cargo asm --release spectral_init::metrics::trustworthiness \
  | grep -E "(ymm|vfmadd|vmovupd|vsubpd|vmulpd)" \
  > results/asm/trustworthiness_asm_avx2.txt

if [ -s results/asm/trustworthiness_asm_avx2.txt ]; then
  echo "H3: AUTO-VECTORIZED — NO-GO for manual AVX2" | tee results/asm/h3_verdict.txt
else
  echo "H3: NOT AUTO-VECTORIZED — IMPLEMENT manual AVX2 kernel" | tee results/asm/h3_verdict.txt
fi
```

Save `results/asm/h3_verdict.txt`. If "AUTO-VECTORIZED", the `avx2_kernel` variant
becomes a thin wrapper that calls the baseline (bit-identical output, trivially
passes parity); the H3 entry in the ranked table will read "NO-GO — auto-vectorized,
manual intrinsics provide no benefit."

### Phase 5: Dry Run

Before committing to full-scale runs:

```bash
# Minimal sanity check (n=1K only, 1 iteration, no warmup)
cargo build --features "testing cli" --release 2>&1

cargo run --features "testing cli" --release --bin tw_profiler -- \
  --x data/gaussian/gaussian_n1000_x.npy \
  --y data/gaussian/gaussian_n1000_y.npy \
  --k 15 --iters 1 --warmup 0 --variant baseline \
  --output results/step_timing/dry_run.json

python scripts/sklearn_reference.py --n 1000 --output results/parity/dry_run.json

cargo criterion --bench trustworthiness_bench -- baseline/1k
```

Verify all 3 succeed and produce parseable JSON. Then run `scripts/dry_run.sh` which
wraps these three checks.

---

## Execution Protocol

Execute in this order after implementation is complete:

**Step 1 — Environment setup:**
```bash
cd research/2026-04-04-tw-perf-scaling
micromamba env create -f environment.yml
micromamba activate tw-perf-scaling
```

**Step 2 — Data generation:**
```bash
python scripts/gen_synthetic.py
python scripts/prepare_merfish.py
python scripts/sklearn_reference.py
```
Verify `tests/fixtures/merfish/merfish_n10k_{x,y}.npy` exist before proceeding.

**Step 3 — Dry run:**
```bash
bash scripts/dry_run.sh
```
Must exit 0 before continuing.

**Step 4 — Seal H5 confirmatory gate (BEFORE any sweep):**
```bash
bash scripts/run_h5_confirmatory.sh
```
This produces `results/subsampling/h5_confirmatory_result.json`. Commit this file
to git immediately after generation so the sweep cannot overwrite it.

**Step 5 — Cargo asm inspection (H3 verdict):**
```bash
bash scripts/run_criterion.sh --asm-only
```
Read `results/asm/h3_verdict.txt`. If "NOT AUTO-VECTORIZED", implement the manual
AVX2 kernel (Phase 3c) before continuing. This step must complete before any
Criterion benchmarks that include `avx2_kernel`.

**Step 6 — Profiling (per-step timing and per-variant wall-clock):**
```bash
bash scripts/run_profiling.sh
```
Runs `tw_profiler` for all variants and all n values, including n=100K for H2, H3,
H4 individually. Saves per-run JSON to `results/step_timing/`.

**Step 7 — Criterion benchmarks:**
```bash
bash scripts/run_criterion.sh
```
Runs full Criterion suite. HTML report saved to `results/criterion/`. Criterion runs
WITHOUT `--features testing` to measure production code.

**Step 8 — Exploratory subsampling sweep:**
```bash
bash scripts/run_subsampling_sweep.sh
```
Script verifies presence of `results/subsampling/h5_confirmatory_result.json` and
exits 1 if missing. Runs sweep over m={500, 1000, 2000, 5000, 10000} with --seed 99.

**Step 9 — Analysis and ranking:**
```bash
python scripts/analyze_results.py
```
Produces `results/analysis/ranked_recommendations.md`.

---

## Analysis Plan

### H0 (Baseline profile)
Read `results/step_timing/gaussian_n{10,50,100}k_baseline.json` (from
`#[cfg(feature="testing")]` `[timing:tw_*]` output). Compute fraction of `tw_total`
consumed by each step. Report table: step × n with % wall-clock. If any single step
≥50% at n=100K → H0 rejected (single step dominates).

### H1 (X-sort dominance)
From the baseline profiling table: if `tw_x_sort + tw_rank_scatter` ≥ 40% of
`tw_total` at n=100K → H1 supported. Report fraction at each n to show scaling
trend. If X-sort < 30% → H1 refuted, reallocate priority to thread-local buffers.

### H2 (Thread-local buffers)
From `results/step_timing/gaussian_n100000_thread_local.json`: compute speedup ratio
`mean_baseline / mean_thread_local`. GO if ratio ≥ 1.5×. Also read Criterion
comparison at n=50K for confidence interval. Report both.

### H3 (AVX2 auto-vectorization)
Read `results/asm/h3_verdict.txt`. If "AUTO-VECTORIZED" → H3 confirmed → NO-GO.
Include the raw instruction grep output as evidence. If supplementary Criterion data
from `bench_tw_avx2` is available, report the speedup ratio as context, but note it
does not change the binary decision.

### H4 (AVX-512)
If CPU supports AVX-512: compare `results/step_timing/gaussian_n100000_avx512_kernel.json`
vs `gaussian_n100000_avx2_kernel.json`. If ratio < 1.2× → NO-GO. Report frequency
throttling evidence if available (via `perf stat` or similar). If CPU does not
support AVX-512 → H4 "N/A".

### H5 (Row subsampling)
Primary: read `results/subsampling/h5_confirmatory_result.json`. Key field: `delta`
(|T_approx - T_exact| at m=5000, seed=42, MERFISH data). GO if delta < 0.001.
Secondary (descriptive only): read sweep JSON and plot delta vs m. Note that the
m=5000 pre-registered value was sealed before the sweep; the sweep is exploration only.

### H6 (Combined variant)
From `results/step_timing/gaussian_n100000_combined.json` vs baseline: speedup ratio.
GO if ratio ≥ 3× at n=100K. From Criterion `bench_tw_combined` vs baseline at n=50K:
CI. Report both. Cross-check: the combined speedup should be ≤ H2_speedup × H1_speedup_on_xsort
× 1 (if H3 confirmed auto-vectorized) due to Amdahl's law.

### Final Ranked Recommendation Table

`analyze_results.py` produces `results/analysis/ranked_recommendations.md` with:

| Approach | Speedup at n=50K (Criterion CI) | Speedup at n=100K (tw_profiler) | Scaling law change | LOC estimate | GO/NO-GO | Rationale |
|---|---|---|---|---|---|---|
| Thread-local buffers | | | No | ~20 | | |
| Partial-rank X | | | No | ~40 | | |
| Auto-vectorized distance | N/A (auto) | N/A | No | 0 | | |
| Manual AVX2 kernel | | | No | ~60 | | |
| AVX-512 kernel | | | No | ~80 | | |
| Row subsampling | N/A (Criterion inapplicable) | ~50× (projected) | YES: O(m·n log n) | ~30 | | |
| Combined exact | | | No | ~60 (sum) | | |

---

## Success Criteria

- **Conclusive positive (H1_combined):** At least one exact optimization achieves
  ≥1.5× Criterion speedup (95% CI lower bound) at n=50K AND ≥1.5× tw_profiler mean
  speedup at n=100K. Ranked recommendation table delivered with GO verdict.

- **Conclusive positive (H5):** `h5_confirmatory_result.json` delta < 0.001 on
  MERFISH data at pre-registered m=5000. `trustworthiness_approx` recommended for
  n≥100K use with explicit approximate threshold documented.

- **Conclusive negative (H1_combined):** All exact optimizations < 1.5× speedup at
  n=100K. Ranked table delivered with NO-GO verdicts and rationale (bottleneck is
  not addressable without approximation).

- **Conclusive negative (H5):** `h5_confirmatory_result.json` delta ≥ 0.001 on
  MERFISH data. Subsampling declared unreliable for this data class.

- **Inconclusive:** Criterion benchmarks show speedup ≥ 1.5× at n=50K but
  tw_profiler at n=100K shows < 1.5× (scaling reversal). Requires additional
  profiling at n=75K and n=150K to determine the crossover point.

---

## Threats to Validity

### Internal

**Cold-start allocation confound (STOP-1, mitigated):** The `--warmup 2` flag in
`tw_profiler` discards the first two iterations, which incur OS page-fault costs for
the 3.2 MB per-row buffers. All reported wall-clock means are from iterations 3–7
(warm). The per-warmup discard is implemented in the binary, not in the analysis
script.

**Post-hoc sampling calibration (STOP-2, mitigated):** m=5000 is pre-registered in
this plan document. `run_h5_confirmatory.sh` seals the result before the exploratory
sweep runs. The sweep uses a different seed (99 vs 42) to prevent the confirmatory
run from being a cherry-picked draw.

**Vacuous parity gate (STOP-3, mitigated):** All correctness decisions use MERFISH-
derived PCA-50 data. Gaussian results are labelled "throughput characterization only"
and appear only in the speedup tables, not in any GO/NO-GO correctness gate.

**Criterion/rayon interaction (RT-5):** Criterion's measurement loop runs multiple
iterations in a tight loop, which may interact with rayon's thread-pool initialization
on the first call. Mitigated by ensuring the first Criterion warm-up iteration (which
Criterion discards by convention) initializes the thread pool before measurement.
The `bench.iter()` pattern calls rayon on every iteration; by the second iteration
the pool is warm.

**Goodhart exploitation in combined variant (RT-1, mitigated):** The combined variant
composition (thread_local + partial_rank + avx2_kernel) is locked in this plan
document before any Phase 3 measurements. Implementers must not add or remove
components based on individual results.

**Partial-rank correctness risk:** The `select_nth_unstable_by` + linear rank scan
approach requires careful tie-handling (multiple X-distances equal to the k-th
value). The implementation must verify |T_partial - T_exact| = 0 exactly on all unit
tests before being benchmarked. Any non-zero delta is a correctness bug.

### External

**Hardware specificity:** SIMD results (H3, H4) are specific to the benchmark host's
CPU microarchitecture. AVX-512 conclusions on an Intel Sapphire Rapids may differ
from Ice Lake or Zen 4. The plan targets the development hardware; CI uses x86-64-v3
baseline (no AVX-512 guarantee).

**d=10 specificity:** SIMD benefit conclusions apply only to d=10 f64. Higher-
dimensional embeddings (d≥32) would yield substantially different AVX2 utilization.

**Subsampling error bounds:** The empirical < 0.001 threshold on MERFISH 10K data
may not generalize to other structured manifold datasets (e.g., mass cytometry,
scRNA-seq). The error is characterized only for the specific MERFISH expression +
spatial coordinate combination.

**Scale extrapolation:** n=100K is directly measured; n=250K behavior for exact
variants is extrapolated from scaling-law fits. Only `trustworthiness_approx` is
directly validated at sub-100K scale with the MERFISH data; its n=250K behavior is
a projection.

---

## Estimated Resource Requirements

- **Compute (baseline profiling):** tw_profiler at n=100K × 5 variants × 5 iters = ~25
  single-call trustworthiness evaluations. At the current rate (~651s for 100K in
  Python UMAP, Rust expected ~10–60s), budget 60s × 25 = ~25 minutes for profiling.
  Criterion benchmarks at n≤50K: ~30 minutes total.
- **Compute (H5 confirmatory):** `trustworthiness_approx` at m=5000, n=10K (MERFISH):
  O(5000 × 10K × 10) ≈ 500M ops, < 30 seconds.
- **Storage:** Synthetic data for all n values: ~100 MB total (dense f64 arrays).
  MERFISH fixture: ~4 MB. Criterion HTML output: ~20 MB. Total: < 200 MB.
- **Python dependencies:** All available in existing `spectral-test` environment.
  No new Rust dependencies.
- **Rust compilation:** `--features "testing cli"` increases build time by < 10s
  (only adds timing macros and CLI parsing).
