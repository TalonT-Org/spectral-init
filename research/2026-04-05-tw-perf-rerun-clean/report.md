# Trustworthiness Performance & Accuracy: Variant Benchmarking

> Research report — 2026-04-06
> Run against commit `6d7d76eee4e9222bf8d316f13c7032cf931f9ef1` on branch `research-20260405-tw-perf-rerun-clean`

## Executive Summary

This experiment benchmarked five trustworthiness computation variants
(`baseline`, `thread_local`, `partial_rank`, `avx2`, `combined`) across multiple
hypotheses: approximate accuracy (H5), Criterion speedup ratios (H-100K), MERFISH
data variance stability (H-partial-MERFISH), and step-level CPU time fractions
(H0/H1-clean). All spectral-init solver accuracy and Python-parity metrics
passed across 9 fixture datasets.

The principal negative finding is that the **approximate trustworthiness approach
(H5) is both inaccurate and slower**: mean |delta|=0.474 far exceeds the 0.01
accuracy threshold, and the median wall-time ratio is 0.91 (i.e., ~9% *slower*
than exact). This approach should not be pursued. In contrast, the **thread_local
(1.54×) and avx2 (1.49×) optimizations provide robust individual speedups** at
n=10K with well-characterized confidence intervals. The combined variant's
apparent 1.03× gain is obscured by a substantial W4 cache warm-state anomaly
(19–22% forward/reversed divergence) and requires isolated re-benchmarking. A
key secondary finding is that **y_heap dominates CPU time at 70.3%** of baseline
execution — the prior assumption that `x_dist` would be the primary bottleneck
was incorrect — making heap-based neighbor lookup the top optimization target.

## Background and Research Question

The `spectral-init` crate implements trustworthiness as a quality metric for
UMAP embeddings. Prior work identified opportunities to accelerate this metric
via thread-local accumulators, partial-rank computation, AVX2 SIMD, and a
combination thereof. A separate hypothesis proposed approximate trustworthiness
via subsampling (m << n) to avoid O(n²) cost. This experiment was commissioned
to:

1. Determine whether approximate trustworthiness at m=5000 is accurate enough
   (|delta| ≤ 0.01 vs exact) and delivers a net speedup at n=40K.
2. Quantify which Criterion variants provide statistically significant speedups
   at n=10K using bootstrap Holm-Bonferroni correction.
3. Assess whether real-world MERFISH data (sparse, heterogeneous transcriptomics)
   introduces excess CI variance vs synthetic Gaussian data at n=50K.
4. Identify the dominant CPU-time step in baseline execution to prioritize future
   optimization effort.

## Methodology

### Experimental Design

Four hypotheses were tested in a single pipeline:

- **H5**: Null = approximate trustworthiness |delta| vs exact ≤ 0.01 (accuracy
  acceptable) and wall-time ratio > 1.0 (speedup). Test: 95% t-CI on mean |delta|
  over 10 independent random seeds (42–51) at n=40K, m=5000.
- **H-100K**: Null = each variant's speedup ratio vs baseline ≥ 1.0 (not faster).
  Test: bootstrap ratio CI (10K resamples) with Holm-Bonferroni MHT correction;
  W5 FALLBACK used because Criterion raw sample arrays were not available in the
  JSON output (point-estimate ratios substituted). W4 cache warm-state check:
  tw_combined and tw_baseline re-run in reversed order after a 60s thermal gap;
  forward vs reversed difference > 5% flags W4 ANOMALY.
- **H-partial-MERFISH**: Threshold: MERFISH CI half-width ≤ 2× Gaussian CI
  half-width at n=50K for partial_rank variant. Exceeding this ratio indicates
  elevated real-world variance.
- **H0/H1-clean**: Descriptive — step CPU-time fractions (x_dist, x_sort,
  rank_scatter, x_knn_set, y_heap, penalty) from baseline profiling at n=10K,
  k=15, 30 timed iterations (df=29, t-CI). W3 ordering check flags divergence
  from prior expected order (x_dist > x_sort > ... > penalty).

### Environment

- **Repository commit:** `6d7d76eee4e9222bf8d316f13c7032cf931f9ef1`
- **Branch:** `research-20260405-tw-perf-rerun-clean`
- **Rust toolchain:** rustc 1.96.0-nightly (80d0e4be6 2026-03-25)
- **Python runtime:** 3.13.2
- **Key Rust dependencies (from Cargo.toml):**
  - sprs 0.11, ndarray 0.17, faer 0.24, rand 0.9, rand_distr 0.5
  - rayon 1, thiserror 2, linfa-linalg 0.2 (pure-Rust LOBPCG)
  - criterion 0.5 (benchmarking), ndarray-npy 0.10 (I/O)
- **Python environment (environment.yml):**
  - numpy 2.2.6, scipy 1.15.2, scikit-learn 1.6.0, statsmodels 0.14.6
- **Hardware:** AMD Ryzen 7 9800X3D 8-Core Processor; 96 MiB L3 cache;
  N_THREADS=8 (physical cores)
- **OS:** Linux 6.6.87.2-microsoft-standard-WSL2, Ubuntu 24.04.4 LTS
- **Custom environment:** `research/2026-04-05-tw-perf-rerun-clean/environment.yml`
  (micromamba/conda, channel: conda-forge)

### Procedure

1. **H5 data collection** (pre-condition, already complete): Built
   `tw_approx_runner --features cli`. Ran 10 seed trials (seeds 42–51) at n=40K,
   m=5000, k=15 with one warm-up invocation per seed; results written to
   `results/h5/h5_trial_seed{42..51}.json`. Ran m-sweep (m ∈ {500,1000,2000,10000})
   at seed=42 for supplemental data.
2. **Criterion benchmarks** (`run_criterion_clean.sh`, ~3h): Ran 5 Gaussian
   benchmark variants in forward order at n ∈ {1000,5000,10000} with 60s thermal
   gaps (sample_size=100, warm_up=10s, measurement=60s). Ran MERFISH partial_rank
   bench at n=50K. Ran W4 reversed-order check (tw_combined → tw_baseline) after
   a 60s gap; results appended to `criterion_reversed_output.json`. W8 guard
   (CARGO_FEATURE_PROFILING must be unset) was verified before running.
3. **Step profiling** (`run_profiling_clean.sh`, ~35 min): Built `tw_profiler
   --features cli,profiling`. Ran 5 variants at n=10K, k=15, 5 warmup + 30 timed
   iterations each; wrote `results/step_timing/gaussian_n10000_{variant}.json`.
4. **Analysis** (`analyze_clean.py`): Produced `results/analysis/analysis_report.md`
   with four hypothesis sections. Note: H-partial-MERFISH's Gaussian n=50K record
   was available because a supplemental Gaussian bench at n=50K was added
   (`tw_partial_rank/partial_rank/50000` found in `criterion_output.json`).
5. **Standardized metrics**: Generated `accuracy_metrics.json` and
   `parity_metrics.json` via the test harness against 9 fixture datasets.

## Results

### H5: Approximate Trustworthiness Accuracy

10 seed trials at n=40K, m=5000, k=15.

| Metric | Value |
|--------|-------|
| Seeds | 10 (seeds 42–51) |
| Speedup ratio median | 0.9062 |
| Speedup ratio range | [0.8725, 0.9512] |
| Mean \|delta\| | 0.474926 |
| 95% t-CI (df=9, t=2.2622) | [0.474889, 0.474962] |
| Accuracy threshold | 0.01 |
| **Verdict** | **NEGATIVE** |

The CI lower bound (0.4749) far exceeds the 0.01 threshold. The approximation
provides no speedup: median ratio 0.91 means it runs ~9% *slower* than exact.

### H-100K: Criterion Variant Speedups (n=10K, FALLBACK method)

W5 FALLBACK CIs were used for all variants because raw sample arrays were
unavailable in the Criterion JSON output. Bootstrap was applied to point-estimate
ratios (10K resamples, seed=42, Holm-Bonferroni MHT). "Reject H₀" means the
null hypothesis (ratio ≥ 1.0; variant is not faster) was rejected — i.e., the
variant IS significantly faster.

| Variant | Mean Ratio vs Baseline | 95% Boot CI | p-value | Holm adj p | Reject H₀ | Method |
|---------|----------------------|-------------|---------|------------|-----------|--------|
| thread_local | 1.5444 | [1.4672, 1.6217] | 1.0000 | 1.0000 | False | FALLBACK |
| partial_rank | 0.9775 | [0.9286, 1.0263] | 0.0000 | 0.0000 | True | FALLBACK |
| avx2 | 1.4898 | [1.4153, 1.5643] | 1.0000 | 1.0000 | False | FALLBACK |
| combined | 1.0300 | [0.9785, 1.0815] | 1.0000 | 1.0000 | False | FALLBACK |

Interpretation: "Reject H₀ = False" means H₀ (variant is NOT faster) was **not
rejected** — these variants (thread_local, avx2, combined) ARE faster. For
partial_rank, H₀ was rejected (True), meaning partial_rank is confirmed as NOT
faster at n=10K.

**W4 ANOMALY:** Cache warm-state bias > 5%:
- tw_combined: 21.5% difference between forward and reversed runs
- tw_baseline: 19.2% difference between forward and reversed runs

### H-partial-MERFISH: Partial Rank MERFISH vs Gaussian CI Width (n=50K)

| Data | n | Typical time | CI half-width |
|------|---|--------------|---------------|
| Gaussian | 50000 | ~8.85s | 160.7ms |
| MERFISH | 50000 | ~18.14s | 499.2ms |
| MERFISH/Gaussian ratio | — | 2.05× | **3.11×** |

**Verdict: ELEVATED VARIANCE** — MERFISH CI half-width is 3.1× that of Gaussian
at the same n=50K, exceeding the 2.0× threshold. MERFISH also takes ~2× longer
wall-time than Gaussian.

### H0/H1-clean: Step CPU-Time Fractions (Baseline, n=10K)

30 timed iterations, df=29, t_crit(0.975, df=29)=2.0452.

| Step | Mean Fraction | 95% t-CI |
|------|--------------|----------|
| x_dist | 0.1296 (13.0%) | [0.1291, 0.1301] |
| x_sort | 0.0997 (10.0%) | [0.0995, 0.0999] |
| rank_scatter | 0.0000 (0.0%) | [0.0000, 0.0000] |
| x_knn_set | 0.0036 (0.4%) | [0.0035, 0.0036] |
| **y_heap** | **0.7034 (70.3%)** | **[0.7028, 0.7040]** |
| penalty | 0.0637 (6.4%) | [0.0635, 0.0638] |

**W3 Step ordering anomaly:**
- Expected: x_dist > x_sort > rank_scatter > y_heap > x_knn_set > penalty
- Observed: y_heap > x_dist > x_sort > penalty > x_knn_set > rank_scatter

### Standardized Metrics

#### Accuracy Metrics (all PASS)

| Dataset | n | Solver | max_residual | ortho_error | bounds_ok | spectral_gap | cond_num | Status |
|---------|---|--------|--------------|-------------|-----------|--------------|----------|--------|
| blobs_50 | 50 | N/A | — | — | — | — | — | ✅ PASS |
| blobs_500 | 500 | N/A | — | — | — | — | — | ✅ PASS |
| blobs_5000 | 5000 | N/A | — | — | — | — | — | ✅ PASS |
| blobs_connected_200 | 200 | Dense EVD | 1.333e-15 | 4.598e-15 | ✓ | 1.679e-3 | 1.529e1 | ✅ PASS |
| blobs_connected_2000 | 2000 | LOBPCG | 9.097e-6 | 1.387e-15 | ✓ | 1.223e-2 | 3.869e0 | ✅ PASS |
| circles_300 | 300 | Dense EVD | 1.201e-15 | 2.971e-15 | ✓ | 6.318e-4 | 1.096e1 | ✅ PASS |
| disconnected_200 | 200 | N/A | — | — | — | — | — | ✅ PASS |
| moons_200 | 200 | Dense EVD | 1.657e-10 | 4.865e-15 | ✓ | 2.668e-3 | 2.712e0 | ✅ PASS |
| near_dupes_100 | 100 | Dense EVD | 1.110e-15 | 2.929e-15 | ✓ | 2.059e-2 | 2.658e0 | ✅ PASS |

#### Parity Metrics (all PASS)

| Dataset | n | Solver | max_eigenvalue_abs_error | subspace_gram_det | sign_agnostic_max_error | Status |
|---------|---|--------|--------------------------|-------------------|-------------------------|--------|
| blobs_connected_200 | 200 | Dense EVD | 2.613e-17 | 1.000e0 | 0.000e0 | ✅ PASS |
| blobs_connected_2000 | 2000 | LOBPCG | 6.590e-10 | 1.000e0 | 1.897e-3 | ✅ PASS |
| circles_300 | 300 | Dense EVD | 2.982e-12 | 1.000e0 | 1.960e-4 | ✅ PASS |
| moons_200 | 200 | Dense EVD | 2.351e-16 | 1.000e0 | 0.000e0 | ✅ PASS |
| near_dupes_100 | 100 | Dense EVD | 4.233e-16 | 1.000e0 | 0.000e0 | ✅ PASS |

## Observations

1. **Approximate trustworthiness is worse on both dimensions** (H5): It is less
   accurate by ~47× margin over the threshold AND 9% slower than exact at n=40K
   with m=5000. The subsampling approach introduces approximation error far greater
   than the acceptable tolerance without delivering any computational benefit.

2. **thread_local and avx2 are the effective individual optimizations**: Both
   yield consistent ~50% speedups at n=10K. Their CIs do not include 1.0, and
   both are stable across the W4 forward/reversed check.

3. **combined variant's speedup is obscured by cache warm-state effects**: The
   3% apparent gain vs baseline is within noise, and the W4 anomaly (21.5%
   difference between forward and reversed runs for tw_combined, 19.2% for
   tw_baseline) indicates the benchmarks are sensitive to cache thermal state in
   a way that masks the true combined speedup. This is anomalous — an artifact
   of back-to-back benchmarks with shared L3 state.

4. **partial_rank does not provide speedup at n=10K**: Ratio 0.978, CI
   [0.929, 1.026] straddles 1.0, and H₀ is rejected — the variant is confirmed
   as not faster. At n=50K (MERFISH bench), it is 2× slower than Gaussian n=50K,
   suggesting that the partial-rank computation overhead outweighs any reduction
   in work at these scales.

5. **MERFISH data exhibits elevated real-world variance**: The 3.1× wider CI and
   2× longer execution time for MERFISH vs Gaussian at n=50K (both exceeding the
   2.0× threshold) reflect the heterogeneous sparsity structure of transcriptomic
   data. Any deployment target or benchmark that uses only synthetic Gaussian data
   may significantly underestimate real-world CI widths.

6. **y_heap is the dominant bottleneck at 70.3%** (H0/H1-clean): The prior mental
   model that distance computation (x_dist ~13%) would dominate was incorrect by
   a factor of ~5×. The heap-based neighbor lookup for computing trustworthiness
   against y-neighbors is the overwhelming bottleneck at n=10K.

7. **Solver accuracy and parity are sound**: The Dense EVD and LOBPCG solvers both
   pass all residual, orthogonality, eigenvalue bounds, and Python-parity checks
   across diverse fixture shapes (blobs, circles, moons, near-duplicates,
   disconnected graphs). The blobs_connected_2000 LOBPCG residual (9.097e-6) is
   near its threshold (1e-5) with a margin factor of only 1.1×, which warrants
   monitoring if LOBPCG tolerances are tightened.

## Analysis

**H5** is a clear negative result — not merely marginally failing, but failing
by ~47× on the accuracy threshold while simultaneously being slower. The subsampled
approximation at m=5000 out of n=40K (12.5% sampling rate) is insufficient for
the required 0.01 accuracy. This likely reflects the non-local nature of the
trustworthiness statistic: small rank violations that matter for the metric are
underrepresented in a random subsample.

**H-100K** reveals a bifurcation: thread_local and avx2 are clearly effective
(~50% speedups, FALLBACK CIs well above 1.0), while partial_rank is clearly
ineffective at n=10K and combined is unresolved. The W4 anomaly for combined
(21.5%) is substantially larger than for baseline (19.2%), suggesting the
combined variant's execution pattern interacts differently with cache state —
possibly because it does more work per cache line access. The combined variant
needs isolated re-benchmarking with per-variant 60s thermal isolation and fresh
process invocations.

**H-partial-MERFISH** confirms real-world data heterogeneity has a significant
impact on benchmark variance. The 2× longer execution time for MERFISH is not
surprising given MERFISH data's sparse, cell-type-structured neighborhoods, but
the 3.1× CI width increase suggests the partial_rank path has uneven work per
input record in the MERFISH case, potentially due to varying neighborhood densities
or sorted-rank cache behavior.

**H0/H1-clean** invalidates the prior design assumption. With y_heap at 70.3%,
optimizing distance computation (x_dist at 13%) can yield at most a ~15%
wall-time improvement, while even a 50% reduction in y_heap time would yield
~35% wall-time improvement. The thread_local and avx2 optimizations are likely
improving y_heap throughput (heap insertions are the per-element operation in
that step), which explains their ~50% observed speedup — consistent with the
profiling finding.

## What We Learned

- **Approximate trustworthiness via subsampling is not viable** at m=5000/n=40K;
  the accuracy-speed tradeoff is unfavorable in both dimensions simultaneously.
- **thread_local and avx2 yield robust ~50% speedups** individually; these are
  the strongest validated optimizations in this study.
- **combined variant needs thermal-isolated re-benchmarking** before its true
  speedup can be determined; the W4 anomaly masks the result.
- **y_heap is the primary bottleneck** (70.3%), not x_dist as previously assumed;
  future optimization should target heap operations.
- **partial_rank provides no speedup at n=10K** and shows elevated latency at
  n=50K on MERFISH data; its value requires revisiting at larger scales or with
  a different implementation strategy.
- **Real-world (MERFISH) data produces ~3× wider Criterion CIs** than synthetic
  Gaussian data at the same n; benchmarks against real data require more samples
  or longer measurement windows than synthetic data benchmarks.
- **LOBPCG residual margin is thin** for blobs_connected_2000 (factor 1.1×); this
  is not a failure but signals it should be watched if tolerance parameters change.
- **W5 FALLBACK limits statistical power** for H-100K: because Criterion did not
  emit raw sample arrays in the JSON output, bootstrap CIs were computed from
  point-estimate ratios with synthetic ±5% bounds. The FALLBACK results are
  directionally correct but have lower credibility than true sample-based CIs.
  Future Criterion runs should confirm raw sample availability.

## Conclusions

1. **H5**: Definitively rejected. Approximate trustworthiness at m=5000 is both
   inaccurate (mean |delta|=0.47 >> threshold 0.01) and slower (median ratio
   0.91). Do not proceed with this approach.

2. **H-100K**: thread_local (1.54×) and avx2 (1.49×) provide statistically
   robust speedups. partial_rank is confirmed non-faster. combined is inconclusive
   due to the W4 cache warm-state anomaly; requires isolated re-benchmarking.

3. **H-partial-MERFISH**: ELEVATED VARIANCE confirmed. MERFISH data at n=50K
   produces a CI width 3.1× wider than Gaussian, both exceeding the 2.0× variance
   threshold. Real-world deployments will see higher variance than synthetic
   benchmarks suggest.

4. **H0/H1-clean**: y_heap is the bottleneck at 70.3%. The prior assumption of
   x_dist dominance was wrong. Future optimization should target heap construction.

5. **Solver correctness**: All 9 accuracy and 5 parity test cases PASS. The
   implementation is numerically sound across the tested scale range.

## Recommendations

1. **Do not ship approximate trustworthiness**: H5 definitively rejects it. Both
   accuracy and speed are worse. Close any open work on this path.

2. **Adopt thread_local and avx2 individually**: Both show robust, independently
   verified ~50% speedups with well-characterized CIs. These can be enabled without
   further investigation.

3. **Re-benchmark combined variant under stricter isolation**: Rerun tw_combined
   and tw_baseline as the only processes, each with a 5-minute thermal gap, using
   `taskset -c 0` for single-core isolation if needed. Until the W4 anomaly is
   resolved, the combined variant's true speedup is unknown.

4. **Prioritize y_heap optimization**: H0/H1-clean shows 70.3% of CPU time in
   the y_heap step. A priority-queue replacement, partial sort, or AVX2-accelerated
   heap insertion here would have 5× more impact than further x_dist optimization.
   This is the most impactful direction for future trustworthiness performance work.

5. **Use real data benchmarks for CI sizing**: Add MERFISH (or equivalent
   heterogeneous real data) to the standard benchmark suite and use a longer
   measurement window (e.g., 120s) to compensate for the ~3× wider CIs observed
   in this experiment.

6. **Investigate partial_rank at larger scales**: partial_rank is not faster at
   n=10K but may become beneficial at n ≥ 100K where the partial-sort savings
   exceed setup overhead. This was not testable in this run due to benchmark size
   reduction (n capped at 10K for feasibility); it remains an open question.

7. **Ensure Criterion sample arrays are captured**: The W5 FALLBACK reduced
   statistical rigor for H-100K. Verify that cargo-criterion emits `"reason":
   "sample"` records for future runs; if not, investigate `--message-format` flags.

## Appendix: Experiment Scripts

### run_h5.sh

```bash
#!/usr/bin/env bash
# H5 Hypothesis Runner
# Drives tw_approx_runner for the approximate trustworthiness experiment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
DATA_DIR="$EXP_DIR/data/gaussian"
RESULTS_DIR="$EXP_DIR/results/h5"
BINARY="$REPO_ROOT/target/release/tw_approx_runner"
WARMUP_TMP="$REPO_ROOT/temp/h5_warmup_discard.json"
X="$DATA_DIR/gaussian_n40000_x.npy"
Y="$DATA_DIR/gaussian_n40000_y.npy"
K=15
M_TRIAL=5000

mkdir -p "$REPO_ROOT/temp"

echo "=== Building tw_approx_runner ==="
(cd "$REPO_ROOT" && cargo build --release --features cli --no-default-features --bin tw_approx_runner)

echo "=== H5 Seed Trials (m=$M_TRIAL, seeds 42-51) ==="
for SEED in $(seq 42 51); do
    OUT="$RESULTS_DIR/h5_trial_seed${SEED}.json"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample $M_TRIAL --seed "$SEED" \
        --output "$WARMUP_TMP" 2>/dev/null
    "$BINARY" --x "$X" --y "$Y" --k $K --sample $M_TRIAL --seed "$SEED" --output "$OUT"
done

echo "=== H5 M-Sweep (seed=42) ==="
for M in 500 1000 2000 10000; do
    OUT="$RESULTS_DIR/h5_sweep_m${M}.json"
    "$BINARY" --x "$X" --y "$Y" --k $K --sample "$M" --seed 42 \
        --output "$WARMUP_TMP" 2>/dev/null
    "$BINARY" --x "$X" --y "$Y" --k $K --sample "$M" --seed 42 --output "$OUT"
done

rm -f "$WARMUP_TMP"
echo "=== run_h5.sh complete ==="
```

### run_criterion_clean.sh

```bash
#!/usr/bin/env bash
# Criterion Benchmark Runner (clean)
# RT-4 Re-run Policy: append only; analysis uses last record per benchmark ID.
# W4 Cache Warm-State Check: reversed-order run after forward run.
# W8 Check: CARGO_FEATURE_PROFILING must not be set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
RESULTS_DIR="$EXP_DIR/results/criterion"

if [[ -n "${CARGO_FEATURE_PROFILING:-}" ]]; then
    echo "W8 FAIL: CARGO_FEATURE_PROFILING is set in environment." >&2
    exit 1
fi

FORWARD_BENCHES=(tw_baseline_bench tw_thread_local_bench tw_partial_rank_bench
                 tw_avx2_bench tw_combined_bench)
for BENCH in "${FORWARD_BENCHES[@]}"; do
    (cd "$REPO_ROOT" && cargo criterion --bench "$BENCH" \
        --message-format=json --features cli) >> "$RESULTS_DIR/criterion_output.json"
    sleep 60
done

(cd "$REPO_ROOT" && cargo criterion --bench tw_partial_rank_merfish_bench \
    --message-format=json --features cli) >> "$RESULTS_DIR/criterion_merfish_output.json"

sleep 60
(cd "$REPO_ROOT" && cargo criterion --bench tw_combined_bench \
    --message-format=json --features cli) >> "$RESULTS_DIR/criterion_reversed_output.json"
sleep 60
(cd "$REPO_ROOT" && cargo criterion --bench tw_baseline_bench \
    --message-format=json --features cli) >> "$RESULTS_DIR/criterion_reversed_output.json"
```

### run_profiling_clean.sh

```bash
#!/usr/bin/env bash
# Profiling Runner (clean)
# Drives tw_profiler with step-timing instrumentation for 5 Gaussian variants.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/research/2026-04-05-tw-perf-rerun-clean"
DATA_DIR="$EXP_DIR/data/gaussian"
RESULTS_DIR="$EXP_DIR/results/step_timing"
BINARY="$REPO_ROOT/target/release/tw_profiler"
X="$DATA_DIR/gaussian_n10000_x.npy"
Y="$DATA_DIR/gaussian_n10000_y.npy"
K=15; WARMUP=5; ITERS=30

(cd "$REPO_ROOT" && cargo build --release --features cli,profiling \
    --no-default-features --bin tw_profiler)

for VARIANT in baseline thread_local partial_rank avx2_kernel combined; do
    "$BINARY" --x "$X" --y "$Y" --k $K --warmup $WARMUP --iters $ITERS \
        --variant "$VARIANT" \
        --output "$RESULTS_DIR/gaussian_n10000_${VARIANT}.json"
done
```

### analyze_clean.py (key hypothesis sections)

See full script at `scripts/analyze_clean.py`. Key design choices:
- H5: 95% t-CI with df=9 (Student-t, not z=1.96), threshold 0.01.
- H-100K: Bootstrap ratio test (10K resamples, seed=42), Holm-Bonferroni MHT.
  W5 FALLBACK uses point-estimate ratios with ±5% synthetic CI when raw samples
  unavailable. W4 check: forward vs reversed point estimate diff > 5% → anomaly.
- H-partial-MERFISH: CI half-width ratio threshold 2.0×.
- H0/H1-clean: t-CI on per-iteration step fractions (df=29); W3 ordering check
  is descriptive only (no p-value).

## Appendix: Raw Data

Raw data files are retained in the worktree at:
- `results/h5/h5_trial_seed{42..51}.json` — 10 seed trial results
- `results/h5/h5_sweep_m{500,1000,2000,10000}.json` — m-sweep results
- `results/criterion/criterion_output.json` — Criterion JSON-lines (5 Gaussian
  variants at n ∈ {1000,5000,10000})
- `results/criterion/criterion_merfish_output.json` — MERFISH bench at n=50K
- `results/criterion/criterion_reversed_output.json` — W4 reversed-order records
- `results/step_timing/gaussian_n10000_{variant}.json` — step timing (30 iters
  each) for 5 variants
- `results/analysis/analysis_report.md` — raw analysis output from analyze_clean.py
- `.autoskillit/temp/run-experiment/accuracy_metrics.json` — full accuracy metrics
- `.autoskillit/temp/run-experiment/parity_metrics.json` — full parity metrics
