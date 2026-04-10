# Scope Report: Sub-sampled Trustworthiness Error/Speed Trade-off (Rust Implementation)

## Research Question

What is the empirical error/speed trade-off when computing trustworthiness on a random subset of rows using our Rust `trustworthiness()` function? At what subsample sizes does |T_sampled - T_exact| remain acceptable, and what wall-clock speedup does each size yield against our SIMD-optimized implementation (AVX2 looped kernels, Rayon parallelism, introselect, thread-local buffers)?

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| Current behavior | `trustworthiness()` in `src/metrics.rs:518-765` iterates all n rows via `(0..n).into_par_iter()`, computing full pairwise X/Y distances per row with AVX2+FMA (`dist_sq_avx2_looped` for d>=10) and 2D AVX2 batch kernel, introselect for k-NN, rank-counting penalty. Thread-local buffers reuse allocations. | No sub-sampled variant exists in Rust. Actual Rust wall-clock at each (n, m) is unmeasured. |
| Performance | Rust exact: n=10K takes ~3.6s, n=50K takes ~91.4s on MERFISH d=50 (from step-timing research). x_dist dominates at 58-59%, y_dist 25-26%, x_sort 9%, penalty 6%. Python exact: n=10K takes ~1.95s, n=20K ~10.3s (sklearn). | Rust sub-sampled wall-clock times at any m. Whether Rust speedup ratios match Python ratios. How Rayon thread utilization changes at small m (fewer rows to parallelize). |
| Accuracy | Python Approach A: mean|ΔT|=0.00165 at m=2000 n=10K, 0.00195 at m=2000 n=20K. Normalization sanity check passes: T_A(m=n) = T_exact within 1e-10. CLT predicts variance decay O(1/sqrt(m)). Empirical slope -0.657 (faster than CLT). | Whether Rust implementation reproduces same |ΔT| as Python (should, since algorithm is identical, but SIMD floating-point ordering could cause minor divergence). |ΔT| at n=50K for any m (Python research dropped n=50K due to memory). |
| Edge cases | k < n/2 guard matches sklearn. Self-exclusion handled via dist_y[i]=INFINITY. Introselect with tie-breaking by index. | Behavior when m is very small (m < 100) — variance may be too high. Whether thread-local buffer resizing interacts poorly with repeated calls at different n. |
| Prior work | Full Python research completed (PR #260, 560 trials, 2 approaches). Approach A recommended with m=2000 default. Normalization formula verified: `m * k * (2n - 3k - 1)`. Prior Rust sub-sampling attempt (2026-04-05 H5) failed catastrophically (|ΔT|=0.474) — suspected denominator bug using m instead of n. | Why exactly the 2026-04-05 H5 attempt failed (suspected but unconfirmed denominator bug). Whether Rust SIMD kernel behavior under sub-sampling introduces any systematic bias vs scalar path. |

## Prior Art in Codebase

### Existing Trustworthiness Implementation

**`src/metrics.rs:518-765`** — Full exact implementation:
- Signature: `pub fn trustworthiness(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64`
- Rayon parallel outer loop over all n rows
- Per-row pipeline: x_dist (AVX2+FMA looped for d>=10) -> x_sort (introselect) -> y_dist (2D AVX2 batch for d_y==2) -> penalty (rank-counting)
- Thread-local `RefCell<Vec<f64>>` and `RefCell<Vec<usize>>` for scratch buffers
- Normalization: `n * k * (2n - 3k - 1)` matching sklearn
- `profiling` feature gate with four AtomicU64 step timers

### SIMD Kernels

- **`dist_sq_avx2_looped`** (`src/metrics.rs:401-449`): 4-wide YMM loop with FMA for X-space squared Euclidean distance, d>=10. Scalar tail for 0-3 remaining elements.
- **`dist_sq_2d_avx2_batch`** (`src/metrics.rs:463-495`): Processes two 2D target points per SIMD iteration for Y-space. Broadcasts query, loads consecutive row-major pairs.

### No Sub-Sampled Variant

There is no `trustworthiness_subsampled`, no `query_idx` parameter, no partial row selection anywhere in `src/`. The Rayon loop is hardcoded to `(0..n)`.

### Benchmark Infrastructure

- **`benches/trustworthiness_bench.rs`**: Criterion benchmarks at n ∈ {1K, 5K, 10K, 50K}, d_x ∈ {10, 50}, d_y=2, k=15
- **`benches/dist_sq_bench.rs`**: Microbenchmark for AVX2 kernel vs scalar
- **`src/bin/tw_profiler.rs`**: CLI profiling harness with warmup/iterations, JSON output, captures per-step timing from stderr
- **`profiling` feature**: Gates step-level nanosecond timing in trustworthiness()

### Test Infrastructure

- **`tests/integration/test_trustworthiness.rs`**: sklearn parity tests using `.npz` fixtures
- **`tests/common/mod.rs`**: Generic NPY/NPZ loaders (`load_dense<T,D>`, `load_sparse_csr`)
- **Unit tests in `src/metrics.rs:1108-1416`**: 10+ tests covering formula, AVX2 kernel match, introselect parity, self-exclusion

### MERFISH Fixtures (all confirmed present)

- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy`
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy`
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy`
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy`

### NPY Loading Infrastructure

- `ndarray_npy::read_npy` for `.npy` files (used in `src/bin/trustworthiness.rs` and `src/bin/tw_profiler.rs`)
- `ndarray_npy::NpzReader` for `.npz` files (used in test infrastructure)
- Available as dev-dependency (always in tests) and behind `cli` feature for binaries

### Prior Research Chain

| Research | Key Finding |
|----------|-------------|
| `2026-04-04-trustworthiness-performance-sc` | Scope/plan only. Identified 6 optimization candidates including sub-sampling. |
| `2026-04-05-tw-perf-rerun-clean` | thread_local 1.54x, avx2 1.49x at n=10K. Sub-sampling (H5) catastrophic failure: mean\|ΔT\|=0.474 (suspected denominator bug). y_heap dominated at 70.3%. |
| `2026-04-06-y-heap-bottleneck-optimization` | Replaced BinaryHeap Y-kNN with flat SIMD + introselect. This is now in production. |
| `2026-04-07-kdtree-y-knn-trustworthiness` | KD-tree definitively slower than flat_simd (36-44% slower). Y-space only ~28% of cost at d=50. Recommendation: sub-sampling is the correct path for further speedup. |
| `2026-04-08-tw-merfish-step-timing` | Real MERFISH step-level profiling. x_dist=58.9%, y_dist=25.4%, x_sort=9.3%, penalty=6.3% at n=10K. |
| `2026-04-09-subsampled-tw-tradeoff` (PR #260) | **Python-only.** Both Approach A and B pass. Approach A recommended: m=2000, mean\|ΔT\|=0.00165, 4.1x speedup at n=10K. Denominator verified. **Explicitly defers Rust validation.** |

## External Research

### Sub-Sampled Trustworthiness — No Published Canonical Method

There is no formally published method for sub-sampled trustworthiness. The technique is used informally in practice but lacks peer-reviewed treatment. The correctness relies on standard statistical theory:

- **Unbiased estimator:** Each point's penalty contribution is i.i.d. under uniform random sampling. By the Law of Large Numbers, `T_m(k)` converges to `T(k)` as m -> n.
- **Variance:** By CLT, standard error scales as O(1/sqrt(m)). Hoeffding's inequality gives concentration bounds: `P(|T_m - T| >= eps) <= 2*exp(-2m*eps^2/M_max^2)` where M_max is the maximum per-point penalty.
- **Empirical evidence:** The Python research confirms variance decays faster than -0.5 CLT expectation (slope -0.657), likely because MERFISH has correlated local structure.

### Trustworthiness Metric Definition

**Venna & Kaski (2001, 2006, 2010):**
```
T(k) = 1 - [2 / (n*k*(2n-3k-1))] * sum_i sum_{j in U_i(k)} (r(i,j) - k)
```
The normalization `n*k*(2n-3k-1)` ensures T in [0,1] and equals the maximum possible penalty sum. Constraint: k < n/2.

### Scalable DR Evaluation

- **sklearn bottleneck:** O(n^2) pairwise distance matrix + O(n^2 log n) full argsort. The n x n inverted index matrix requires 80 GB at n=100K.
- **Our approach avoids both:** Row-by-row distance + introselect (no full sort) + rank-counting (no inverted index). Memory is O(n) per thread, not O(n^2).
- **ZADU (2023):** Shares preprocessing across multiple metrics. Not relevant here (single metric).
- **TopOMetry:** Accelerates kNN with approximate libraries. Not applicable — our introselect is already O(n) per row.
- **Out-of-core DR (2024):** GPU-batched trustworthiness up to n=3.2M. Beyond that, qualitative-only evaluation. No standard sub-sampled alternative.

### Introselect for k-NN

Rust's `select_nth_unstable_by` implements introselect (O(n) average). This is the key to our row-by-row approach: we find k-NN in O(n) without sorting, then count rank in O(n) per violating neighbor. Sub-sampling doesn't change the per-row algorithm — only the number of rows processed changes from n to m.

## Technical Context

### Architecture for Sub-Sampling

The existing `trustworthiness()` architecture is naturally compatible with row sub-sampling:

1. **Outer loop change only:** Replace `(0..n).into_par_iter()` with `query_idx.into_par_iter()` where `query_idx` is a Vec<usize> of m randomly selected row indices.

2. **Inner loop unchanged:** For each selected row i:
   - X-distances from row i to ALL n rows (O(n*d)) — unchanged, still needs full population
   - Introselect for X-kNN (O(n)) — unchanged
   - Y-distances from row i to ALL n rows (O(n*d_y)) — unchanged
   - Introselect for Y-kNN (O(n)) — unchanged
   - Penalty rank-counting (O(n) per violating neighbor) — unchanged

3. **Thread-local buffers:** Still length n (full population). No change needed.

4. **Normalization:** Change `n * k * (2n-3k-1)` to `m * k * (2n-3k-1)` — critical: use n from `x.nrows()` for the population size, m for the sample size.

5. **Complexity:** O(m * n * d) for distances + O(m * n) for introselect + O(m * k * n) for rank-counting = O(m * n * (d + k + 1)). Linear in m, so m/n fraction of exact cost (approximately).

### Potential Implementation Approaches

**Option 1 — New function `trustworthiness_subsampled(x, y, k, query_idx)`:**
- Takes explicit query indices
- Caller handles random sampling
- Most flexible, testable

**Option 2 — Wrapper `trustworthiness_approx(x, y, k, m, seed)`:**
- Generates query_idx internally via `StdRng::seed_from_u64(seed)`
- Convenience API for common use case
- Calls Option 1 internally

**Option 3 — Modify existing function with optional parameter:**
- `trustworthiness(x, y, k, sample: Option<usize>)` or builder pattern
- Minimizes API surface but changes existing signature

### Rayon Parallelism at Small m

At m << n, Rayon has fewer tasks to distribute across threads. With default thread pool (num_cpus), m=500 on a 16-core machine means ~31 rows per thread — still sufficient for good load balancing since each row's work is O(n). At m=100 on 16 cores, only ~6 rows/thread — may see overhead. This is a minor concern for m >= 500.

### Step Timing Profile (MERFISH d=50)

From the 2026-04-08 research on the current production code:

| Step | n=10K (ms, %) | n=50K (ms, %) |
|------|---------------|---------------|
| x_dist | 2127.6 (58.9%) | 53346.3 (58.4%) |
| x_sort | 336.3 (9.3%) | 8029.0 (8.8%) |
| y_dist | 918.1 (25.4%) | 24251.9 (26.5%) |
| penalty | 228.3 (6.3%) | 5740.5 (6.3%) |
| **Total** | **3610.4 ms** | **91367.6 ms** |

Sub-sampling attacks ALL steps proportionally (each step is per-row). Expected speedup factor ≈ n/m.

## Hypotheses

### H1 — Accuracy at m=2000

At m=2000, the Rust sub-sampled trustworthiness on MERFISH n=10K will have mean|ΔT| < 0.01 across 5+ seeds, consistent with the Python finding of 0.00165.

**Falsifiable:** Run 5+ seeds, compute mean|ΔT|. Reject if mean|ΔT| >= 0.01.

### H2 — Linear Speedup in m

Wall-clock speedup of exact/sub-sampled will be approximately n/m (linear in the reduction factor) for the Rust implementation, because all four pipeline steps scale linearly with the number of query rows.

**Falsifiable:** Measure wall-clock at 4+ m values. Fit speedup vs n/m. Reject if R^2 < 0.95.

### H3 — Variance Decay

Standard deviation of T_sampled across seeds will decay as O(1/sqrt(m)) or faster, matching the Python observation (slope -0.657).

**Falsifiable:** Compute std(T) at 4+ m values with 5+ seeds each. Fit log(std) vs log(m). Reject if slope > -0.3 (too slow).

### H4 — Rust Speedup Exceeds Python Speedup

Because the Rust exact baseline is faster than Python's (SIMD + Rayon + introselect vs numpy + sklearn), the *absolute* time savings from sub-sampling may differ, but the *speedup ratio* (exact/sub-sampled) should still be approximately n/m regardless of implementation language.

**Falsifiable:** Compare Rust speedup ratio to Python speedup ratio at same (n, m). Reject if they differ by more than 2x.

### H5 — n=50K Validation

At n=50K, m=2000, Rust sub-sampled trustworthiness will have mean|ΔT| < 0.01 (the Python research could not test this due to memory constraints, but our Rust implementation has O(n) memory, not O(n^2)).

**Falsifiable:** Run on MERFISH n=50K fixtures with m=2000, 5+ seeds. Reject if mean|ΔT| >= 0.01.

### H6 — Normalization Sanity

At m=n, the sub-sampled function will produce exactly the same T as the exact function (within floating-point tolerance 1e-10). This validates that the denominator correction is correctly implemented.

**Falsifiable:** Compare T_subsampled(m=n) to T_exact. Reject if |delta| >= 1e-10.

## Proposed Investigation Directions

### Direction A — Minimal Experiment Binary (Recommended)

Create a standalone Rust experiment binary (e.g., `research/.../scripts/run_experiment.rs` compiled via `cargo run --example`) that:

1. Loads MERFISH .npy fixtures using `ndarray_npy::read_npy`
2. Computes exact T via existing `spectral_init::trustworthiness()`
3. Implements sub-sampled T inline (copy the trustworthiness inner loop but iterate over `query_idx` instead of `0..n`)
4. Runs all (n, m, seed) combinations, measures wall-clock via `std::time::Instant`
5. Outputs structured JSON for analysis

**Trade-offs:** Most faithful to the existing Rust pipeline. Uses the actual SIMD kernels, thread-local buffers, and Rayon parallelism. Avoids modifying `src/metrics.rs` during research phase. Requires implementing the sub-sampled loop from scratch (but it's structurally identical to the existing code with a different outer iterator).

### Direction B — Parameterize Existing Function

Add a `query_indices: Option<&[usize]>` parameter to the existing `trustworthiness()` (or create `trustworthiness_subsampled()`), then benchmark via Criterion or the tw_profiler binary.

**Trade-offs:** Directly tests the production code path. But modifies `src/metrics.rs` during research, which mixes experiment artifacts with library code. If the experiment shows sub-sampling isn't worth shipping, the changes must be reverted.

### Direction C — Hybrid: Feature-Gated Experiment in Library

Add `trustworthiness_subsampled()` to `src/metrics.rs` behind a `#[cfg(feature = "testing")]` gate. Write the experiment as an integration test or binary that uses this function.

**Trade-offs:** Tests the exact production SIMD/Rayon code path without polluting the public API. The function is only available in test/bench builds. If results are positive, promote to public API. If negative, remove cleanly.

## Success Criteria

1. **All benchmarks run against Rust `trustworthiness()`** — no Python-only results
2. **Error/speed table** across >= 4 subsample sizes on MERFISH data with Rust wall-clock times:
   - n=10K: m in {500, 1000, 2000, 5000}
   - n=50K: m in {1000, 2000, 5000, 10000}
3. **Variance estimates** with >= 5 random seeds per (n, m) cell
4. **Normalization sanity check:** T_subsampled(m=n) = T_exact within 1e-10
5. **Recommended default subsample size** with Rust-specific speedup justification
6. **Cross-validation against Python numbers** from PR #260 (at overlapping (n, m) points)
7. **Hypotheses H1-H6 evaluated** with clear PASS/FAIL verdicts

## Metric Context

### Relevant Canonical Metrics from `src/metrics.rs`

The research question touches the **Performance** quality dimension. The relevant metrics and thresholds:

| Metric | Dimension | Current Threshold | Relevance |
|--------|-----------|-------------------|-----------|
| `trustworthiness(x, y, k)` | Performance | No explicit threshold (returns raw T in [0,1]) | The exact function being sub-sampled |
| `RSVD_QUALITY_THRESHOLD` | Accuracy | 1e-2 | Not directly relevant |
| `DENSE_EVD_QUALITY_THRESHOLD` | Accuracy | 1e-6 | Not directly relevant |
| `LOBPCG_QUALITY_THRESHOLD` | Accuracy | 2e-5 | Not directly relevant |
| `SUBSPACE_GRAM_DET_THRESHOLD` | Parity | 0.95 | Not directly relevant |

### Gaps

- **No explicit threshold for sub-sampled T accuracy.** The issue specifies |ΔT| < 0.01 as the acceptance threshold, but this is not codified in `src/metrics.rs`.
- **No `assess_performance` test function** in `test_metrics_assess.rs`. Accuracy and Parity dimensions have structured assessment; Performance does not.
- **No `MetricResult` dimension tagging.** The `dimension` field on `MetricResult` is always set to `0`; no enum distinguishes Accuracy/Parity/Performance programmatically.
- **`trustworthiness_subsampled` does not exist.** The 2026-04-09 research recommends shipping it but it has not been implemented.
