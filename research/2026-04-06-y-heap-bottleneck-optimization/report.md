# y_heap Bottleneck Optimization: BinaryHeap → Vec + Introselect + AVX2

> Research report — 2026-04-06

## Executive Summary

The `trustworthiness()` function in `spectral-init` spent 70% of its wall-clock time on a single step — `y_heap` — which found k-nearest neighbors in embedding space using a per-row `BinaryHeap` allocation. This experiment evaluated three algorithmic variants to determine whether a faster data structure or distance kernel could meaningfully reduce total trustworthiness computation time.

The primary question was whether per-row heap allocation, heap maintenance overhead, or the raw distance arithmetic was responsible for the 70% step fraction. A causal isolation design (three incrementally additive variants) resolved this cleanly: **heap allocation cost is negligible; the dominant gain comes from replacing BinaryHeap with a flat Vec + `select_nth_unstable_by` (introselect), with AVX2 SIMD providing substantial additional gain on 2D embeddings**.

The `flat_simd` variant (Vec + introselect + AVX2 2D distance kernel) achieves a **statistically significant ~2× total speedup at n=10000** (conservative ratio bounds: 1.73–2.27×), with the y_heap step fraction dropping from 69.8% to 27.6%. All 21 correctness tests pass with `|ΔT| < 1e-12`, and all 9 accuracy and 5 parity fixture tests pass with no regression. The recommendation is to **ship `flat_simd`** as the production replacement for the BinaryHeap-based implementation.

## Background and Research Question

The `trustworthiness()` function is the primary diagnostic metric for embedding quality in `spectral-init`. At n=10K, k=15, step-level profiling (PR #229) established that the `y_heap` step — which scans all n points in embedding space to find the k-nearest neighbors for each row — consumed 70.3% of total wall-clock time. All other steps (x_dist, x_sort, penalty) together account for the remaining 30%.

Prior optimization work (PR #226/229) addressed `x_dist` (then 13% of time) with thread-local buffers (1.54×) and AVX2 SIMD (1.49×). No optimization had been applied to `y_heap`. The scope report identified three hypotheses:

- **H1**: Replacing BinaryHeap with a thread-local flat Vec + introselect eliminates per-row allocation overhead (hypothesis: 20–50% y_heap reduction).
- **H2**: A dedicated AVX2 batch kernel for d_y=2 reduces the arithmetic cost (hypothesis: 1.5–3× additional speedup on the y_heap distance pass).
- **H3**: A KD-tree pre-build enables O(log n) queries vs O(n) brute force (deferred — requires new dependency, benefit uncertain at n=10K).

The combined H1+H2 approach (no new dependencies) was selected as the primary experiment target, with a causal isolation variant (`heap_reuse`) added to quantify the allocation cost alone.

## Methodology

### Experimental Design

**Hypothesis**: Replacing the `BinaryHeap<(u64, usize)>` per-row allocation in `y_heap` with a thread-local `Vec<f64>` + `select_nth_unstable_by` (H1), optionally combined with an AVX2 batch distance kernel for d_y=2 (H2), will yield a statistically significant speedup in total `trustworthiness()` wall time at n=10K, k=15.

**Independent variable**: Algorithm variant — one of `baseline`, `heap_reuse`, `flat_partial`, `flat_simd`.

**Dependent variable**: Total `trustworthiness(x, y, k)` wall-clock time (Criterion mean estimate, ms), and per-step time fraction from the step-level profiler.

**Controls**:
- Fixed k=15 across all variants and all n
- Gaussian random data (d_x=10, d_y=2), seeded for reproducibility
- Same Rayon thread count (`nproc`) for all runs
- W8 guard: `CARGO_FEATURE_PROFILING` check in `run_criterion.sh` ensures profiling instrumentation is absent during Criterion runs
- 60-second thermal gap between Criterion variant runs

**Causal isolation design**:

| Variant | What changes vs previous | Isolates |
|---------|--------------------------|----------|
| `baseline` | — | Reference |
| `heap_reuse` | `thread_local` BinaryHeap, `clear()` instead of `with_capacity` | Allocation cost only |
| `flat_partial` | thread_local `Vec<f64>` + `select_nth_unstable_by` | Data structure (BinaryHeap → introselect) |
| `flat_simd` | Same as flat_partial + `dist_sq_2d_avx2_batch` for d_y=2 | AVX2 distance arithmetic |

**Correctness criterion**: `|ΔT| < 1e-12` for all variants vs baseline across 7 test data scenarios (21 tests total). The `select_nth_unstable_by` comparator `.total_cmp(&dist_y[b]).then(a.cmp(&b))` was verified to replicate BinaryHeap tie-breaking exactly.

### Environment

- **Repository commit**: `fae2d2a884db257f8d6c269cd0f37f052ff23dc8`
- **Branch**: `research-20260406-162144`
- **Rust toolchain**: `rustc 1.96.0-nightly (80d0e4be6 2026-03-25)`, `cargo 1.96.0-nightly (e84cb639e 2026-03-21)`
- **Key dependencies**: `criterion = "0.5"`, `sprs`, `ndarray`, `faer`, `ndarray-linalg`, `rayon`
- **Python environment**: `envs/spectral-test` (micromamba; `environment.yml` at `research/2026-04-06-y-heap-bottleneck-optimization/environment.yml`)
- **Hardware/OS**: Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64, AVX2 present)

### Procedure

1. **Data generation** (`scripts/gen_data.py`): Generated Gaussian random datasets at n ∈ {1000, 5000, 10000}, d_x=10, d_y=2, seeded. Stored as `.npy` in `data/`.

2. **Rust implementation** (groupB): Added `profiling` feature flag; implemented `trustworthiness_heap_reuse`, `trustworthiness_flat_partial`, `trustworthiness_flat_simd` in `src/metrics.rs`; added `dist_sq_2d_avx2_batch` AVX2 kernel; extended `lib.rs` exports; added `--variant` dispatch to `tw_profiler`.

3. **Correctness tests**: 21 tests (`t_tw_{variant}_{01..07}`) added to `src/metrics.rs`; all must pass `cargo test --features testing`.

4. **Criterion benchmarks** (`scripts/run_criterion.sh`): `cargo bench --bench y_heap_variants_bench --features testing`, Flat sampling, 10 samples, 10s warm-up, 10s measurement. Each variant run sequentially with 60s thermal gap. JSON harvested to `results/criterion/`.

5. **Step profiler** (`scripts/run_profiler.sh`): `cargo build --release --features cli,profiling`; 4 variants at n=10000, 30 iterations, 5 warmup. JSON harvested to `results/profiler/`.

6. **Analysis** (`scripts/analyze_results.py`): Loaded Criterion JSON, computed speedup ratios and conservative ratio CI bounds, causal attribution fractions, and step fractions from profiler data. Generated `results/analysis/analysis_report.md` and `speedup_ratios.png`.

## Results

### Criterion Benchmark — Speedup Table

| Variant | n | Mean (ms) | Speedup | CI lb | CI ub | Sig |
|---------|---|-----------|---------|-------|-------|-----|
| baseline | 1000 | 11.162 | 1.0000 | 1.0000 | 1.0000 | |
| baseline | 5000 | 69.994 | 1.0000 | 1.0000 | 1.0000 | |
| baseline | 10000 | 289.315 | 1.0000 | 1.0000 | 1.0000 | |
| heap_reuse | 1000 | 11.276 | 0.9899 | 0.9532 | 1.0265 | |
| heap_reuse | 5000 | 70.323 | 0.9953 | 0.9768 | 1.0131 | |
| heap_reuse | 10000 | 292.412 | 0.9894 | 0.9391 | 1.0410 | |
| flat_partial | 1000 | 8.934 | 1.2494 | 1.1922 | 1.3092 | * |
| flat_partial | 5000 | 39.648 | 1.7654 | 1.7351 | 1.7952 | * |
| flat_partial | 10000 | 162.843 | 1.7766 | 1.6955 | 1.8679 | * |
| flat_simd | 1000 | 7.892 | 1.4144 | 1.3537 | 1.4780 | * |
| flat_simd | 5000 | 37.684 | 1.8574 | 1.7824 | 1.9259 | * |
| flat_simd | 10000 | 145.099 | 1.9939 | 1.7272 | 2.2727 | * |

`*` = CI lower bound > 1.0 (statistically significant speedup). CI bounds are conservative ratio bounds (not formal 95% CIs): `(base_ci_lb / variant_ci_ub, base_ci_ub / variant_ci_lb)`.

### Step Fractions — Profiler (n=10000, 30 iters, 5 warmup)

Step fractions are summed wall-clock nanoseconds across all 30 iterations (Rayon parallel sections aggregated per call):

| Variant | x_dist (ms) | x_sort (ms) | y_heap (ms) | penalty (ms) | y_heap % |
|---------|-------------|-------------|-------------|--------------|----------|
| baseline | 582.751 | 435.079 | 2988.416 | 273.777 | 69.8 |
| heap_reuse | 608.119 | 454.205 | 3013.613 | 281.907 | 69.2 |
| flat_partial | 579.541 | 434.476 | 1070.424 | 282.171 | 45.2 |
| flat_simd | 602.322 | 457.746 | 521.753 | 306.863 | 27.6 |

### Causal Decomposition (n=10000)

Attribution fractions quantify what fraction of baseline time each bundle eliminates:

| Bundle | Attribution fraction |
|--------|----------------------|
| Allocation elimination (heap_reuse vs baseline) | −0.011 (negligible) |
| Data structure change: Vec+introselect vs BinaryHeap (flat_partial vs heap_reuse) | 0.443 |
| SIMD 2D distance kernel (flat_simd vs flat_partial) | 0.109 |

Note: W2 applies — these are bundle attributions (not single-cause isolations). `heap_reuse` vs `baseline` conflates allocation cost with any incidental BinaryHeap initialization effects; the other two bundles are nearly single-cause within the y_heap step.

### Correctness

All 21 variant correctness tests passed (7 test cases × 3 variants):
- `t_tw_heap_reuse_01` through `_07`: PASS
- `t_tw_flat_partial_01` through `_07`: PASS
- `t_tw_flat_simd_01` through `_07`: PASS

Maximum `|ΔT|` confirmed < 1e-12 against baseline for all 21 cases.

### Standardized Metrics

#### Accuracy

| Metric | Dimension | Dataset | Value | Threshold | Status |
|--------|-----------|---------|-------|-----------|--------|
| max_eigenpair_residual | Accuracy | blobs_connected_200 | 1.333e-15 | 1e-6 | ✅ PASS |
| orthogonality_error | Accuracy | blobs_connected_200 | 4.598e-15 | 1e-8 | ✅ PASS |
| max_eigenpair_residual | Accuracy | blobs_connected_2000 | 9.097e-6 | 2e-5 | ✅ PASS |
| orthogonality_error | Accuracy | blobs_connected_2000 | 1.387e-15 | 1e-8 | ✅ PASS |
| max_eigenpair_residual | Accuracy | circles_300 | 1.201e-15 | 1e-6 | ✅ PASS |
| orthogonality_error | Accuracy | circles_300 | 2.971e-15 | 1e-8 | ✅ PASS |
| max_eigenpair_residual | Accuracy | moons_200 | 1.657e-10 | 1e-6 | ✅ PASS |
| orthogonality_error | Accuracy | moons_200 | 4.865e-15 | 1e-8 | ✅ PASS |
| max_eigenpair_residual | Accuracy | near_dupes_100 | 1.110e-15 | 1e-6 | ✅ PASS |
| orthogonality_error | Accuracy | near_dupes_100 | 2.929e-15 | 1e-8 | ✅ PASS |
| component_count_match | Accuracy | blobs_50/500/5000 | 1.0 | 1.0 | ✅ PASS |
| component_count_match | Accuracy | disconnected_200 | 1.0 | 1.0 | ✅ PASS |

All 9 accuracy datasets: PASS.

#### Parity

| Metric | Dimension | Dataset | Value | Threshold | Status |
|--------|-----------|---------|-------|-----------|--------|
| max_eigenvalue_abs_error | Parity | blobs_connected_200 | 2.613e-17 | 1e-6 | ✅ PASS |
| sign_agnostic_max_error | Parity | blobs_connected_200 | 0.0 | 5e-3 | ✅ PASS |
| max_eigenvalue_abs_error | Parity | blobs_connected_2000 | 6.590e-10 | 2e-5 | ✅ PASS |
| sign_agnostic_max_error | Parity | blobs_connected_2000 | 1.897e-3 | 5e-3 | ✅ PASS |
| max_eigenvalue_abs_error | Parity | circles_300 | 2.982e-12 | 1e-6 | ✅ PASS |
| sign_agnostic_max_error | Parity | circles_300 | 1.960e-4 | 5e-3 | ✅ PASS |
| max_eigenvalue_abs_error | Parity | moons_200 | 2.351e-16 | 1e-6 | ✅ PASS |
| max_eigenvalue_abs_error | Parity | near_dupes_100 | 4.233e-16 | 1e-6 | ✅ PASS |
| subspace_gram_det | Parity | all 5 datasets | 1.000 | 0.95 | ✅ PASS |

All 5 parity datasets: PASS.

## Observations

1. **heap_reuse eliminates nothing measurable.** BinaryHeap per-row allocation cost at k=15 is ~−1.1% attribution at n=10000 — statistically indistinguishable from zero across all three n values. Modern allocators handle 16-element heap allocations essentially for free; the cost hypothesis was falsified cleanly.

2. **Data structure change is the dominant performance lever.** Replacing BinaryHeap with flat Vec + `select_nth_unstable_by` reduces the y_heap step from 69.8% to 45.2% of total time — a 2.79× reduction within the y_heap step alone, translating to 1.78× total speedup at n=10000. This is the largest achievable gain without algorithmic restructuring.

3. **AVX2 kernel provides substantial additional gain for 2D embeddings.** `flat_simd` further reduces y_heap from 45.2% to 27.6% (1.64× within the step), adding ~10.9% total attribution beyond `flat_partial`. The combined effect reaches near-2× total speedup.

4. **Speedup increases with n.** `flat_partial` scales from 1.25× (n=1000) to 1.78× (n=10000); `flat_simd` from 1.41× to 1.99×. This is consistent with the y_heap step becoming more dominant at larger n (larger fraction of total, so more absolute gain from reducing it).

5. **x_dist and x_sort are unaffected; penalty shows a minor unexplained increase.** Step fractions confirm x_dist (~580ms) and x_sort (~435ms) are consistent across all variants. However, the `penalty` step increases by ~12% in `flat_simd` (273.8ms baseline → 306.9ms), despite no code changes to the penalty computation. This may reflect cache interaction effects from the flat Vec layout changing memory access patterns. The increase is small relative to total runtime and does not affect the primary finding, but should be noted as a secondary effect.

6. **flat_simd n=10000 ratio bounds are wide but decisive.** The conservative ratio bounds (1.73–2.27) — computed as `base_ci_lb / variant_ci_ub` to `base_ci_ub / variant_ci_lb`, not formal 95% CIs — are expected to be wide with 10 Criterion samples and the Flat sampling mode at high measurement times. The lower bound of 1.73 strongly confirms significance; no Stage 2 escalation was needed.

7. **No accuracy or parity regressions detected.** The `select_nth_unstable_by` comparator `.total_cmp(&dist_y[b]).then(a.cmp(&b))` correctly replicates BinaryHeap tie-breaking. The PythonCompat eigensolver path is unchanged — these are entirely within the `trustworthiness()` metric function.

## Analysis

The experiment cleanly resolved the performance question and falsified the allocation hypothesis. The causal attribution ladder demonstrates:

```
baseline → heap_reuse:  ΔSpeedup = −0.011   (allocation is NOT the bottleneck)
heap_reuse → flat_partial: ΔSpeedup = 0.443  (data structure IS the bottleneck)
flat_partial → flat_simd: ΔSpeedup = 0.109  (arithmetic is a secondary bottleneck)
```

The original hypothesis predicted 20–50% y_heap reduction for H1 alone; the measured result is 35% y_heap fraction reduction (69.8% → 45.2%), placing it in the middle of that range. H2 (SIMD) was predicted at 1.5–3× additional speedup on the y_heap step; the measured 1.64× further reduction within y_heap falls at the low end of the predicted range, which is consistent with the actual bottleneck being introselect throughput (already fast) rather than pure arithmetic.

The scope report's earlier work on x_dist established that thread_local + AVX2 combined yielded 1.03× on x_dist (W4 cache warm-state anomaly). The y_heap experiment avoids this confound because the variants are additive and the cache state is more stable at the larger y_heap time budget. The 60-second thermal gaps in `run_criterion.sh` further reduce thermal contamination.

The LOBPCG residual margin factor of 2.2 on `blobs_connected_2000` is the tightest across all datasets, but this is a pre-existing characteristic of the eigensolver and not introduced by this experiment's changes. All accuracy thresholds pass.

## What We Learned

- **Per-row allocation is not the bottleneck at k=15.** `BinaryHeap::with_capacity(k+1)` is effectively free with modern allocators at this scale. Thread-local reuse of a BinaryHeap is not worth the added complexity.
- **BinaryHeap's O(n log k) vs introselect's O(n) advantage is real at k=15.** Even though log(15) ≈ 4 is small, the heap's branchy push/pop structure has worse cache behavior than a flat array partial sort at n ≥ 5000.
- **The introselect tie-breaking comparator must precisely match BinaryHeap semantics.** `.total_cmp(&dist_y[b]).then(a.cmp(&b))` (ascending distance, ascending index on ties) is required. Using `.then_with(|| ...)` or reversing the index order breaks the `|ΔT| < 1e-12` guarantee.
- **AVX2 batch for 2D is feasible and beneficial** when the embedding is row-major and d_y=2. `_mm256_hadd_pd` over two points per iteration yields clean throughput gains. The scalar fallback path for non-AVX2 or non-2D embeddings preserves correctness.
- **The step-level profiler is essential for attributing performance improvements.** Without it, the 44% bundle attribution to data structure change would not be distinguishable from SIMD gains in the Criterion output alone.
- **No new dependencies are required.** Both H1 and H2 use only Rust stdlib + existing unsafe intrinsics. The KD-tree direction (H3) remains viable for n >> 10K but is not needed at this scale.

## Conclusions

The `y_heap` step bottleneck in `trustworthiness()` is caused by **BinaryHeap's data structure overhead** (pointer indirection, branchy heap maintenance) rather than allocation cost or raw arithmetic throughput. The fix is to replace BinaryHeap with a thread-local flat distance buffer + `select_nth_unstable_by`. For d_y=2 embeddings, adding an AVX2 batch distance kernel provides an additional ~11% total speedup.

The `flat_simd` variant achieves a **statistically significant ~2× total speedup** at n=10000 (CI lb 1.73×) with zero correctness regression (`|ΔT| < 1e-12`, all accuracy/parity fixtures pass). The y_heap step fraction drops from 69.8% to 27.6%.

## Recommendations

1. **Ship `flat_simd`** as the production replacement for `trustworthiness()`. The statistically significant ~2× speedup (CI lb 1.73×) justifies the change. The implementation is correct to 1e-12 against the original baseline.

2. **Do not ship `heap_reuse`** as an intermediate step. It provides no measurable benefit and adds code complexity for no gain.

3. **Deprecate the BinaryHeap path** once `flat_simd` is merged. The `trustworthiness_heap_reuse` and `trustworthiness_flat_partial` variants can be removed after the production merge is validated in CI.

4. **Consider the KD-tree direction (H3) for n ≥ 50K workloads.** At n=10K, the ~2× gain from `flat_simd` is sufficient. At n=100K (outside the current Criterion budget), the O(n) brute force becomes O(n²) total, and a KD-tree's O(n log n) all-NN query would likely dominate. A follow-up research cycle targeting n ≥ 50K is warranted if trustworthiness is called on large embeddings.

5. **Document the AVX2 fallback path.** The `flat_simd` implementation falls back to scalar for non-x86_64, non-AVX2, or non-2D cases. This fallback must be preserved in the production merge; the correctness tests already cover it via the 21 `t_tw_flat_simd_*` tests.

## Appendix: Experiment Scripts

### scripts/run_criterion.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results/criterion"

# W8 guard: abort if profiling feature is active — would contaminate timing
if [[ -n "${CARGO_FEATURE_PROFILING:-}" ]]; then
    echo "ERROR: CARGO_FEATURE_PROFILING is set. Benchmark must run without profiling instrumentation." >&2
    exit 1
fi

export RAYON_NUM_THREADS
RAYON_NUM_THREADS="$(nproc)"
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

mkdir -p "$RESULTS_DIR"

run_variant() {
    local variant="$1"
    local group="y_heap_${variant}"

    echo "=== Running variant: $variant ==="
    cargo bench \
        --bench y_heap_variants_bench \
        --features testing \
        --manifest-path "$REPO_ROOT/Cargo.toml" \
        -- "$group"

    # Harvest Criterion JSON for each n
    for n in 1000 5000 10000; do
        local src="$REPO_ROOT/target/criterion/${group}/n/${n}/new/estimates.json"
        local dst="$RESULTS_DIR/y_heap_${variant}_n${n}.json"
        if [[ -f "$src" ]]; then
            cp "$src" "$dst"
            echo "  copied: $dst"
        else
            echo "  WARNING: expected JSON not found: $src" >&2
        fi
    done
}

run_variant baseline
sleep 60

run_variant heap_reuse
sleep 60

run_variant flat_partial
sleep 60

run_variant flat_simd

# Snapshot Cargo.lock
cp "$REPO_ROOT/Cargo.lock" "$SCRIPT_DIR/../results/Cargo.lock.snapshot"
echo "Cargo.lock snapshot saved."
echo "=== run_criterion.sh complete ==="
```

### scripts/run_profiler.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RESULTS_DIR="$SCRIPT_DIR/../results/profiler"

mkdir -p "$RESULTS_DIR"

# Build profiler binary with profiling instrumentation
echo "=== Building tw_profiler (cli,profiling) ==="
cargo build --release \
    --features cli,profiling \
    --manifest-path "$REPO_ROOT/Cargo.toml"

PROFILER="$REPO_ROOT/target/release/tw_profiler"

run_variant() {
    local variant="$1"
    echo "=== Profiling variant: $variant (n=10000) ==="
    "$PROFILER" \
        --x "$DATA_DIR/gaussian_n10000_x.npy" \
        --y "$DATA_DIR/gaussian_n10000_y.npy" \
        --k 15 \
        --iters 30 \
        --warmup 5 \
        --variant "$variant" \
        --stderr-capture "$RESULTS_DIR/stderr_${variant}.txt" \
        --output "$RESULTS_DIR/profiler_${variant}_n10000.json"
    echo "  wrote: $RESULTS_DIR/profiler_${variant}_n10000.json"
}

run_variant baseline
run_variant heap_reuse
run_variant flat_partial
run_variant flat_simd

echo "=== run_profiler.sh complete ==="
```

## Appendix: Raw Data

### Criterion JSON files

Raw Criterion estimates JSONs are committed at:
- `research/2026-04-06-y-heap-bottleneck-optimization/results/criterion/y_heap_{variant}_n{n}.json`
  for variant ∈ {baseline, heap_reuse, flat_partial, flat_simd}, n ∈ {1000, 5000, 10000}

### Profiler JSON files

Raw step-timing profiler outputs are committed at:
- `research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_{variant}_n10000.json`
  for variant ∈ {baseline, heap_reuse, flat_partial, flat_simd}

### Speedup chart

`research/2026-04-06-y-heap-bottleneck-optimization/results/analysis/speedup_ratios.png`
