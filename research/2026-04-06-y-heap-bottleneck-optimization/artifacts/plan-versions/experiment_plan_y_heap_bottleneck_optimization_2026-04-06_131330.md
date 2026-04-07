# Experiment Plan: y_heap Bottleneck Optimization in Trustworthiness Computation

## Motivation

The `trustworthiness()` function spends **70.3% of CPU time** in the `y_heap`
step — a brute-force O(n log k) `BinaryHeap<(u64, usize)>` scan per point to
find k-nearest neighbors in Y-space. This is the dominant cost identified in
PR #226/#229, with x_dist (already SIMD+thread_local optimized) consuming only
13%. Two dependency-free approaches exist to accelerate this step:

- **H1**: Replace the per-row `BinaryHeap` allocation with a reusable
  thread-local flat buffer + `select_nth_unstable_by` (mirrors the established
  `COMB_DIST_X`/`COMB_INDICES` pattern used for x_dist).
- **H2**: Add a 2D-specialized AVX2 batch distance kernel (processes 4 Y-rows
  per SIMD cycle, vs current scalar 1 row per cycle).

Results will directly inform whether to adopt H1+H2 as the production path or
escalate to a KD-tree dependency (H3) in a follow-up experiment. The experiment
is bounded by the 3-hour Criterion budget constraint from the scope report.

---

## Hypothesis

**Null hypothesis (H0):** Replacing `BinaryHeap` with a thread-local flat
Y-distance buffer and `select_nth_unstable_by` (H1), optionally combined with
a 2D AVX2 batch distance kernel (H2), produces a total `trustworthiness`
speedup of < 1.5× at n=10K, k=15 in Criterion wall-time measurements (95% CI
overlaps 1.0×).

**Alternative hypothesis (H1_alt):** The H1+H2 combination yields ≥ 1.5×
speedup on total `trustworthiness` wall-time at n=10K (95% CI lower bound
> 1.0×), and reduces the y_heap step fraction from 70.3% to ≤ 40%.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| y_heap implementation variant | `baseline` (current BinaryHeap), `h1_introselect` (thread-local flat buf + select_nth_unstable_by), `h2_simd` (h1 + 2D AVX2 batch kernel) | Tests allocation elimination (H1) and arithmetic acceleration (H2) in isolation and combined; enables attribution of speedup to each mechanism |
| n (number of points) | 1_000, 5_000, 10_000 | 10K is the profiled reference scale where 70.3% was measured; smaller values catch scaling anomalies and guard against SIMD overhead at small n |

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighbors) | 15 | Matches profiling configuration from PR #229; all 70.3% measurements used k=15 |
| d_x (X dimension) | 10 | Matches existing bench; ensures x_dist step contribution is stable across runs |
| d_y (embedding dimension) | 2 | Canonical UMAP output; AVX2 kernel specialized for this value |
| Random seed | 42 | Matches existing `trustworthiness_bench.rs`; ensures identical inputs across variants |
| RUSTFLAGS | `-C target-cpu=native` | Already set in `.cargo/config.toml`; AVX2+FMA required for H2 |
| Rayon thread count | System default | Matches production deployment |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| trustworthiness wall-time | µs | Criterion `y_heap_variants_bench.rs` — mean ± 95% CI per (variant × n) | NEW — `trustworthiness_wall_time_us` |
| speedup ratio | dimensionless | `baseline_mean / variant_mean` from Criterion JSON estimates | NEW — `y_heap_speedup_ratio` |
| y_heap step fraction | fraction [0,1] | `AtomicU64` accumulator inside par_iter → `[timing:y_heap]` stderr line → `tw_profiler --stderr-capture` JSON | NEW — `y_heap_step_fraction` |
| trustworthiness score delta | f64 (absolute) | Direct comparison: `(trustworthiness_h1(x,y,k) - trustworthiness(x,y,k)).abs()` | Derived from existing `trustworthiness()` output; correctness gate |

**NEW metric definitions** (must be registered in `src/metrics.rs` before the
experiment is finalized — none currently have entries):

- **`trustworthiness_wall_time_us`**: Quality dimension = Performance. Formula:
  Criterion `BenchmarkResult.mean.point_estimate` in nanoseconds, divided by
  1000. Unit: µs. No SLO threshold (research context); success criterion is
  the speedup ratio derived from it.

- **`y_heap_speedup_ratio`**: Quality dimension = Performance. Formula:
  `baseline_wall_time_us / variant_wall_time_us`. Unit: dimensionless. Success
  threshold: ≥ 1.5 (lower bound of 95% CI strictly > 1.0).

- **`y_heap_step_fraction`**: Quality dimension = Performance. Formula:
  `y_heap_total_ns / trustworthiness_wall_time_ns` where `y_heap_total_ns` is
  the sum of per-row y_heap timings across all Rayon threads (accumulated via
  `AtomicU64`). Unit: fraction [0, 1]. Success threshold: drop from 0.703
  to ≤ 0.40 for H2 variant.

---

## Inputs and Data

All data is generated synthetically in-process using the established `make_data`
pattern from `benches/trustworthiness_bench.rs`. No fixtures, no file I/O
during the bench itself.

```rust
fn make_data(n: usize, d_x: usize, d_y: usize, seed: u64)
    -> (Array2<f64>, Array2<f64>)
// Uses SmallRng::seed_from_u64(seed); both X and Y are uniform [0,1).
```

For the `tw_profiler` step, `scripts/gen_data.py` generates `.npy` files into
`data/` using numpy, matching the same shape and distribution.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| synthetic_n1000 | generated in-process | n=1K, d_x=10, d_y=2, seed=42 | Regression guard; detects overhead at small n |
| synthetic_n5000 | generated in-process | n=5K, d_x=10, d_y=2, seed=42 | Mid-scale validation |
| synthetic_n10000 | generated in-process (bench) + npy (profiler) | n=10K, d_x=10, d_y=2, seed=42 | Primary measurement scale matching 70.3% profile |

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-06-y-heap-optimization/
├── scripts/
│   ├── run_bench.sh          # Runs Criterion variant comparison; copies JSON to results/
│   ├── run_profiler.sh       # Generates npy data; runs tw_profiler per variant; saves JSON
│   └── analyze_results.py    # Parses Criterion + profiler JSON; prints speedup table
├── data/                     # x_10k.npy, y_10k.npy (generated by run_profiler.sh)
├── results/
│   ├── criterion/            # Criterion JSON copied from target/criterion/
│   └── profiler/             # tw_profiler JSON outputs per variant
└── report.md                 # Final report (written by write-report skill)
```

**`run_bench.sh`** (follows `set -euo pipefail`, derives `PROJECT_ROOT` from
`SCRIPT_DIR`):
- Runs `cargo bench --bench y_heap_variants_bench` from project root
- Copies `$PROJECT_ROOT/target/criterion/y_heap_variants/` → `results/criterion/`

**`run_profiler.sh`**:
- Calls `scripts/gen_data.py` to write `data/x_10k.npy`, `data/y_10k.npy`
- Runs `cargo run --release --bin tw_profiler -- --x data/x_10k.npy --y data/y_10k.npy --output results/profiler/baseline.json --iters 5 --warmup 2 --stderr-capture`
- Repeats with `--variant h1` and `--variant h2` flags (requires `tw_profiler`
  to accept a `--variant` flag dispatching to the appropriate function — see
  Phase 3)

**`analyze_results.py`** (stdlib only — no external packages):
- Reads Criterion `estimates.json` for each (variant × n) combination
- Computes speedup ratio = `baseline_mean / variant_mean`, 95% CI bounds
- Reads profiler JSONs for `step_timing.y_heap` fraction
- Prints a Markdown comparison table

**Source code changes** (implemented in `src/` as part of the experiment — not
created in `research/`, but planned here):
1. `src/metrics.rs`: add `COMB_DIST_Y` + `COMB_INDICES_Y` thread-locals
2. `src/metrics.rs`: add `pub(crate) fn trustworthiness_h1(...)` and
   `pub(crate) fn trustworthiness_h2(...)`
3. `src/metrics.rs`: add `AtomicU64` step timing in all three variants
4. `src/bin/tw_profiler.rs`: add `--variant {baseline,h1,h2}` CLI flag
5. `benches/y_heap_variants_bench.rs`: new Criterion bench (new file)
6. `Cargo.toml`: register new bench under `[[bench]]`

---

## Environment

**No custom environment needed.** The project's existing Rust toolchain is
sufficient for H1 and H2 (primary experiment):

- `.cargo/config.toml` already sets `rustflags = ["-C", "target-cpu=native"]`,
  enabling AVX2+FMA on any supporting x86_64 host without additional flags.
- The `unsafe fn + #[target_feature(enable = "avx2,fma")]` pattern is
  established in `src/metrics.rs:384–409` and can be replicated exactly.
- `criterion = "0.5"` (with `html_reports`) is already in dev-dependencies.
- `rand = "0.9"`, `ndarray = "0.17"`, `rayon` are in existing production deps.
- `analyze_results.py` uses only Python stdlib (`json`, `pathlib`, `statistics`).

No `environment.yml` will be created.

**Stretch goal (H3 — KD-tree, not part of this experiment):** Would require
adding `kiddo = "4"` (verify exact current version at crates.io before pinning)
to `[dependencies]`. Leave for a follow-up experiment if H1+H2 results are
inconclusive or if n=100K performance matters.

---

## Implementation Phases

### Phase 1: Directory Structure and Benchmark Scaffolding

1. Create `research/2026-04-06-y-heap-optimization/` with `scripts/`, `data/`,
   `results/`, `results/criterion/`, `results/profiler/` subdirectories.
2. Create `benches/y_heap_variants_bench.rs`:
   - Copy `make_data` verbatim from `benches/trustworthiness_bench.rs`.
   - Create group `"y_heap_variants"` with `SamplingMode::Flat`,
     `sample_size(10)`, `warm_up_time(Duration::from_secs(10))`.
   - Add benchmark IDs for `baseline`, `h1_introselect`, `h2_simd` at
     n = [1_000, 5_000, 10_000] using `BenchmarkId::new("n", n)`.
   - Import `spectral_init::{trustworthiness, trustworthiness_h1, trustworthiness_h2}`.
3. Register in `Cargo.toml`:
   ```toml
   [[bench]]
   name = "y_heap_variants_bench"
   harness = true
   ```
4. Verify: `cargo bench --bench y_heap_variants_bench --no-run` compiles
   (stubs for `trustworthiness_h1`/`h2` will be added in Phase 2).

### Phase 2: H1 Implementation — Thread-local Flat Buffer + Introselect

All changes in `src/metrics.rs`:

1. Add two new thread-locals immediately after the existing `COMB_DIST_X` /
   `COMB_INDICES` block (currently near line 455):
   ```rust
   thread_local! {
       static COMB_DIST_Y:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
       static COMB_INDICES_Y: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
   }
   ```

2. Add `pub(crate) fn trustworthiness_h1(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64`
   by duplicating `trustworthiness()` and replacing the `y_heap` section
   (lines 499–509) with:
   ```rust
   // Thread-local flat Y-dist buffer
   let y_knn_set: HashSet<usize> = COMB_DIST_Y.with(|dist_buf| {
       COMB_INDICES_Y.with(|idx_buf| {
           let dist_y = &mut *dist_buf.borrow_mut();
           let indices_y = &mut *idx_buf.borrow_mut();
           dist_y.clear();
           dist_y.resize(n, 0.0);
           indices_y.clear();
           indices_y.extend(0..n);
           for j in 0..n {
               let d: f64 = yi.iter().zip(y.row(j).iter())
                   .map(|(&a, &b)| (a - b) * (a - b)).sum();
               dist_y[j] = d;
           }
           dist_y[i] = f64::MAX; // exclude self
           indices_y.select_nth_unstable_by(k, |&a, &b| {
               dist_y[a].partial_cmp(&dist_y[b]).unwrap()
           });
           HashSet::from_iter(indices_y[..=k].iter().copied().filter(|&j| j != i))
       })
   });
   ```

3. Add `AtomicU64` step timing for the profiler (defined before `into_par_iter`):
   ```rust
   let y_heap_ns_total = std::sync::atomic::AtomicU64::new(0);
   // inside per-row closure, wrapping the y_heap block:
   let t_y = std::time::Instant::now();
   // ... y_heap code ...
   y_heap_ns_total.fetch_add(t_y.elapsed().as_nanos() as u64,
       std::sync::atomic::Ordering::Relaxed);
   // after into_par_iter().sum():
   eprintln!("[timing:y_heap]{}", y_heap_ns_total.load(
       std::sync::atomic::Ordering::Relaxed));
   ```
   Add the same pattern to `trustworthiness()` (baseline) so the profiler has
   a before/after comparison.

### Phase 3: H2 Implementation — 2D AVX2 Batch Kernel

All changes in `src/metrics.rs`:

1. Add the batch distance kernel following the established AVX2 pattern
   (see `dist_sq_avx2` at lines 384–409):
   ```rust
   #[cfg(all(target_arch = "x86_64", target_feature = "avx2",
             target_feature = "fma"))]
   #[target_feature(enable = "avx2,fma")]
   unsafe fn fill_dist_2d_avx2(
       yi: &[f64],      // 2 elements: [yi0, yi1]
       y_flat: &[f64],  // row-major flat: [y00, y01, y10, y11, ...]
       out: &mut [f64], // output distances, length n
       skip: usize,     // index to set to f64::MAX (self-exclusion)
   ) { ... }
   ```
   The kernel processes 4 Y-rows per SIMD pass:
   - Broadcast `yi0`, `yi1` into 256-bit registers.
   - Load `[y_j0, y_{j+1}_0, y_{j+2}_0, y_{j+3}_0]` and
     `[y_j1, y_{j+1}_1, y_{j+2}_1, y_{j+3}_1]` via `_mm256_loadu_pd`.
   - Compute `(yi0 - y_col0)^2 + (yi1 - y_col1)^2` via `_mm256_sub_pd` +
     `_mm256_mul_pd` + `_mm256_add_pd`.
   - Store 4 results to `out[j..j+4]` via `_mm256_storeu_pd`.
   - Handle tail (n mod 4) with scalar fallback.
   - After loop: `out[skip] = f64::MAX`.

2. Add `pub(crate) fn trustworthiness_h2(...)` by copying `trustworthiness_h1`
   and replacing the scalar distance fill loop with:
   ```rust
   if use_avx2 && d_y == 2 && y.is_standard_layout() {
       let y_flat = y.as_slice().unwrap();
       unsafe { fill_dist_2d_avx2(yi.as_slice().unwrap(), y_flat, dist_y, i) }
   } else {
       // scalar fallback (identical to h1)
       for j in 0..n { ... }
       dist_y[i] = f64::MAX;
   }
   ```

3. Add `--variant` flag to `src/bin/tw_profiler.rs`:
   - Parse `--variant {baseline,h1,h2}` (default: `baseline`).
   - Dispatch to `trustworthiness`, `trustworthiness_h1`, or `trustworthiness_h2`.

### Phase 4: Correctness Verification

1. Run full test suite: `cargo test` — all existing tests must pass, including:
   - `test_trustworthiness::sklearn_parity_synthetic` (`|Δ| < 1e-6` vs sklearn)
   - `t_tw_01` through `t_tw_07`
2. Add correctness assertions in `benches/y_heap_variants_bench.rs` (or a
   dedicated test in `tests/`) for fixed inputs at n=1000, seed=42:
   ```rust
   let (x, y) = make_data(1000, 10, 2, 42);
   let t_base = spectral_init::trustworthiness(x.view(), y.view(), 15);
   let t_h1   = spectral_init::trustworthiness_h1(x.view(), y.view(), 15);
   let t_h2   = spectral_init::trustworthiness_h2(x.view(), y.view(), 15);
   assert!((t_h1 - t_base).abs() < 1e-12, "H1 score delta: {}", (t_h1-t_base).abs());
   assert!((t_h2 - t_base).abs() < 1e-12, "H2 score delta: {}", (t_h2-t_base).abs());
   ```
3. Run `cargo test` again after adding assertions to confirm no regressions.

### Phase 5: Benchmark Collection

1. Run: `cargo bench --bench y_heap_variants_bench`
   - Criterion produces HTML + JSON under `target/criterion/y_heap_variants/`.
   - Each (variant × n) combination: 10 samples, 10s warmup.
2. Copy results: `cp -r target/criterion/y_heap_variants/ research/2026-04-06-y-heap-optimization/results/criterion/`
3. Run `cargo bench --bench trustworthiness_bench` to verify no regression on
   the existing main benchmark (sanity check that source changes did not break
   the baseline `trustworthiness()` function).

### Phase 6: Step-Fraction Profiling

1. Create `scripts/gen_data.py`:
   ```python
   import numpy as np, pathlib
   rng = np.random.default_rng(42)
   out = pathlib.Path("data")
   out.mkdir(exist_ok=True)
   n, d_x, d_y = 10_000, 10, 2
   np.save(out / "x_10k.npy", rng.random((n, d_x)))
   np.save(out / "y_10k.npy", rng.random((n, d_y)))
   ```
2. Run `scripts/run_profiler.sh` from the experiment directory:
   - Calls `gen_data.py` to populate `data/`
   - Runs `cargo run --release --bin tw_profiler -- --x ../../data/x_10k.npy --y ../../data/y_10k.npy --output ../../results/profiler/baseline.json --iters 5 --warmup 2 --stderr-capture`
   - Repeats with `--variant h1` → `results/profiler/h1.json`
   - Repeats with `--variant h2` → `results/profiler/h2.json`
3. The `step_timing.y_heap` field in each JSON contains the `AtomicU64`
   accumulator total (in nanoseconds) from the final warm iteration. Divide by
   `mean_s × 1e9` to get the step fraction.

---

## Execution Protocol

After implementation is complete and Phase 4 correctness verification passes:

1. **From project root**, run the full experiment in order:
   ```bash
   cd research/2026-04-06-y-heap-optimization
   python3 scripts/gen_data.py
   bash scripts/run_bench.sh
   bash scripts/run_profiler.sh
   python3 scripts/analyze_results.py
   ```
2. Inspect the speedup table printed by `analyze_results.py`.
3. Copy `target/criterion/trustworthiness/` results for the existing bench as
   a regression baseline (compare before/after the source changes).
4. If Criterion shows ≥ 1.5× speedup with CI lower bound > 1.0:
   - H1_alt supported → proceed to `write-report` skill.
5. If Criterion shows < 1.5× but > 1.2×:
   - Inconclusive — report as partial improvement; recommend H3 (KD-tree) follow-up.
6. If Criterion shows ≤ 1.05×:
   - H0 supported — allocation was not the bottleneck; investigate arithmetic
     or other dominant costs; report as inconclusive.

---

## Analysis Plan

**Primary analysis:** Compute speedup ratio = `baseline_mean / variant_mean`
from Criterion `estimates.json` for each variant at n=10K. Report 95% CI.
Criterion's comparison output (`--baseline`) also provides this directly.

**Attribution analysis:** Compare H1 vs H2 speedup to isolate:
- H1 alone: allocation elimination contribution
- H2 − H1: SIMD arithmetic contribution

Expected decomposition (from scope):
- If H1 alone gives ~20–50% reduction, allocation was the dominant sub-cost.
- If H2 adds another 1.5–2×, arithmetic throughput is a secondary bottleneck.

**Step fraction analysis:** Compare `y_heap_step_fraction` from profiler JSON
across variants. If the fraction drops proportionally to the wall-time speedup,
the improvement is genuine (not a measurement artifact).

**Scaling analysis:** Plot speedup ratio vs n (1K, 5K, 10K). If speedup
increases with n, the O(n) allocation cost dominates (favoring H1 at all
scales). If speedup is flat, instruction throughput dominates.

**Statistical criterion for significance:** The 95% CI lower bound for the
speedup ratio must be strictly > 1.0 (Criterion's reported CI must not include
1.0 for the primary n=10K benchmark).

---

## Success Criteria

- **Conclusive positive:** Criterion at n=10K shows ≥ 1.5× speedup for H2
  (or H1 alone) with 95% CI lower bound > 1.0; all existing tests pass
  (`cargo test`); sklearn parity delta < 1e-6; h2 trustworthiness score delta
  < 1e-12 vs baseline on identical inputs.

- **Conclusive negative:** Criterion at n=10K shows < 1.1× speedup for H2
  with CI overlapping 1.0; or trustworthiness score delta ≥ 1e-6 (kNN set
  differs — indicates a correctness bug in the variant).

- **Inconclusive:** Speedup between 1.1× and 1.5× with CI overlapping 1.0
  (marginal improvement but not statistically significant at the 10-sample
  budget). Recommend increasing `sample_size` or investigating W4 cache
  warm-state interactions (the anomaly documented for the x_dist combined
  result in PR #229).

---

## Threats to Validity

### Internal

- **W4 cache warm-state anomaly**: Prior research (PR #229) showed the combined
  x_dist improvements were inconclusive at 1.03× due to cache interactions.
  The same risk exists for H1+H2 — the y_heap improvement may be masked or
  exaggerated depending on L2/L3 warm state during Criterion runs. Mitigation:
  run baseline and variants in separate Criterion invocations with cold starts.

- **Rayon thread count non-determinism**: Parallel execution means timing
  variance includes scheduling noise. Mitigation: `SamplingMode::Flat` with
  `sample_size(10)` is already the established convention; do not reduce below
  10 samples.

- **`AtomicU64` ordering overhead**: `Relaxed` ordering adds ~1ns per fetch_add
  (vs no instrumentation). At n=10K rows, this is ~10µs added to the timed
  region — negligible but should be noted in the report.

- **AVX2 assumption**: H2 correctness relies on `is_x86_feature_detected!("avx2")`
  returning true. If the CI runner lacks AVX2, H2 silently falls back to scalar
  (H1). The bench must assert `use_avx2 == true` or report which path was
  actually exercised.

- **select_nth_unstable tie-breaking**: The current BinaryHeap uses `(d.to_bits(), j)`
  — ties are broken by point index. `select_nth_unstable_by` is not
  deterministic for equal distances. Mitigation: test correctness on synthetic
  data where ties are possible; if the parity test passes with delta < 1e-12,
  tie-breaking is functionally equivalent for the test inputs.

### External

- **Generalizability to other k values**: The 70.3% measurement and the heap
  optimization are calibrated for k=15. For k=100, the log(k) factor grows to
  ~6.6, and introselect may lose its allocation advantage. Results apply to
  k=15 only unless tested at other k values.

- **Generalizability to other n scales**: The experiment validates n ≤ 10K.
  Extrapolation to n=100K is untested — KD-tree may dominate at that scale.

- **Non-uniform Y distributions**: Synthetic uniform Y is the worst case for
  KD-trees (not tested here) but not necessarily worst for brute force. Real
  UMAP embeddings are clustered — the actual production distribution may
  yield different speedup ratios.

---

## Estimated Resource Requirements

- **Compile time**: ~30s incremental build after source changes.
- **Criterion bench runtime**: 3 variants × 3 n values × 10 samples × 10s warmup
  ≈ 15–20 minutes wall time total.
- **Profiler runtime**: 3 variants × (2 warmup + 5 iters) × ~500ms per iter at
  n=10K ≈ 10 minutes.
- **Total**: ~30 minutes wall time. Well within the 3-hour budget.
- **Disk**: Criterion HTML + JSON < 10MB; profiler JSON < 100KB; npy files ~1.6MB.
- **Dependencies added**: None for H1+H2 (primary). H3 stretch would add
  `kiddo` (~3MB compiled).
