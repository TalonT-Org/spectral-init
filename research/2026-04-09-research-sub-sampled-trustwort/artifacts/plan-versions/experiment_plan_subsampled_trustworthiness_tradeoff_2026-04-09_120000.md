# Experiment Plan: Sub-sampled Trustworthiness Error/Speed Trade-off

## Motivation

Exact trustworthiness computation is O(n^2) and takes ~3.6s on MERFISH n=10K. For
large datasets (n>50K), this cost becomes prohibitive for iterative workflows
(hyperparameter sweeps, visual evaluation pipelines). Sub-sampling is the only
known approach that changes the fundamental O(n^2) scaling, but its accuracy on
real data with non-uniform cluster structure has never been measured correctly.
The prior H5 experiment (2026-04-05) produced systematically biased results due
to a normalization bug, and the "10K subsample gives ~1% error" claim from PR
#238 has no published source.

This experiment will produce the first correct, multi-seed characterization of
sub-sampled trustworthiness on MERFISH data, covering both the practitioner
approach (subset embedding, Approach B) and the unbiased estimator (row
sub-sampling, Approach A). The results will determine whether a
`trustworthiness_subsampled` function should be added to `src/metrics.rs`, and
if so, what default subsample size to recommend.

## Hypothesis

**Null hypothesis (H0):** Sub-sampling trustworthiness does not achieve |ΔT| < 0.01
at any subsample size m < n/2 on MERFISH data, or the variance across seeds is
too high (std > 0.005) to be useful as a drop-in replacement for exact computation.

**Alternative hypothesis (H1):** There exists a subsample size m << n such that
|T_subsampled - T_exact| < 0.01 with std < 0.005 across 10 seeds on MERFISH
data, and the wall-clock speedup is at least 3x over exact computation.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Approach | A (row-subsampled, full distances), B (subset embedding) | A is an unbiased estimator of full T; B is what practitioners actually do (slice then compute). Both must be characterized. |
| Subsample size m (n=10K) | 250, 500, 1000, 2000, 5000 | Spans 2.5%–50% of n. Below 250 is statistically unreliable; above 5000 is >50% of n. |
| Subsample size m (n=50K) | 250, 500, 1000, 2000, 5000, 10000, 25000 | Extends range to larger n. 10K and 25K test the "10K subsample" claim directly. |
| Dataset | MERFISH (real manifold), Gaussian (synthetic) | MERFISH has non-uniform cluster structure; Gaussian is uniform. Comparison tests H5. |
| Dataset size n | 10000, 50000 | Available fixtures. n=50K enables speedup extrapolation. |
| Random seed | 0–9 | 10 seeds per configuration for robust variance estimation. |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| T_exact | dimensionless [0,1] | sklearn.manifold.trustworthiness on full dataset | `trustworthiness` (existing in src/metrics.rs) |
| |ΔT| = |T_sub - T_exact| | dimensionless | Absolute difference of sub-sampled vs exact T | NEW — requires formula/threshold definition |
| mean(|ΔT|) | dimensionless | Mean of |ΔT| across 10 seeds at fixed (approach, dataset, n, m) | NEW |
| std(T_sub) | dimensionless | Sample standard deviation of T_sub across 10 seeds | NEW |
| Wall-clock time | seconds | time.perf_counter around each trustworthiness call | NEW |
| Speedup ratio | dimensionless | T_exact_time / T_sub_time | NEW |

For metrics marked "NEW":
- **|ΔT|**: Formula: `abs(T_sub - T_exact)`. Threshold: < 0.01 (matching visual eval pipeline tolerance per scope report §Metric Context). Needs codifying as a constant if experiment succeeds.
- **std(T_sub)**: Formula: `np.std(T_sub_values, ddof=1)`. Threshold: < 0.005 (half the |ΔT| tolerance to ensure reproducibility).
- **Wall-clock time**: Raw seconds, measured per individual trustworthiness call.
- **Speedup ratio**: Derived metric. No threshold — purely descriptive.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (n_neighbors) | 15 | Standard UMAP default, matches all prior trustworthiness experiments in this project |
| Distance metric | Euclidean (L2) | sklearn default, matches Rust implementation |
| Data dtype | float64 | Matches fixture format; avoids precision artifacts |
| Python implementation | sklearn 1.8.x | Validated ground truth; avoids normalization bugs |
| Warm-up | 1 timed call discarded before measurement | Stabilize JIT/cache effects |

## Inputs and Data

The experiment requires high-dimensional input data (X) and 2D UMAP embeddings
(Y) for both real and synthetic datasets. All primary fixtures already exist on
disk.

- **MERFISH data:** Real gene expression data processed with PCA(50), with
  pre-computed 2D UMAP embeddings. Non-uniform cluster structure makes this the
  critical test case — sub-sampling accuracy depends on whether cluster boundary
  points contribute disproportionately to trustworthiness.
- **Gaussian data:** Synthetic isotropic Gaussian in 50 dimensions with random
  2D projection as "embedding." Uniform local structure means per-point
  contributions are nearly identical, giving a best-case scenario for
  sub-sampling. Must be generated at d_x=50 to match MERFISH dimensionality
  (existing Gaussian fixtures use d_x=10).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| merfish_n10k | Existing fixture | n=10000, d_x=50, d_y=2, f64, real manifold | Primary accuracy target |
| merfish_n50k | Existing fixture | n=50000, d_x=50, d_y=2, f64, real manifold | Speedup scaling, large-n behavior |
| gaussian_n10k_50d | Generated | n=10000, d_x=50, d_y=2, f64, isotropic Gaussian | H5 comparison: synthetic vs real |

**Fixture locations (verified on disk):**
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` (3.9 MB)
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` (157 KB)
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` (20 MB)
- `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` (782 KB)

**Why d_x=50 Gaussian must be generated:** Existing Gaussian fixtures use d_x=10.
Dimensionality affects the distance concentration phenomenon and could confound
the MERFISH vs. synthetic comparison. Generating d_x=50 Gaussian data eliminates
this confound. The generation script (`gen_synthetic.py` in the prior research
directory) already supports `--d 50`.

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-09-subsampled-tw-tradeoff/
├── environment.yml                  # Micromamba/conda env
├── scripts/
│   ├── gen_gaussian_50d.py          # Generate d=50 Gaussian data for H5
│   ├── compute_exact.py             # Exact T via sklearn for all datasets
│   ├── sweep_approach_b.py          # Approach B: sklearn on X[idx],Y[idx]
│   ├── sweep_approach_a.py          # Approach A: custom row-subsampled T
│   └── analyze_results.py           # Aggregate, fit models, produce tables/plots
├── data/
│   ├── merfish/                     # Symlinks to existing MERFISH fixtures
│   └── gaussian/                    # Generated d=50 Gaussian data
├── results/                         # JSON outputs from sweeps
│   ├── exact_trustworthiness.json   # Ground-truth T values per dataset
│   ├── approach_b_results.json      # Per-trial results for Approach B
│   ├── approach_a_results.json      # Per-trial results for Approach A
│   └── analysis/                    # Tables and plots
│       ├── error_speed_table.csv    # Summary table
│       ├── delta_t_vs_m.png         # |ΔT| vs subsample size
│       ├── std_vs_m_loglog.png      # Variance scaling (CLT check)
│       ├── speedup_vs_m.png         # Speedup ratios
│       └── merfish_vs_gaussian.png  # H5 comparison
└── report.md                        # Final report (written by write-report)
```

### File Descriptions

**`environment.yml`** — Micromamba environment with Python 3.11, numpy, scipy,
scikit-learn, matplotlib. Matches project conventions.

**`scripts/gen_gaussian_50d.py`** — Generates isotropic Gaussian data at n=10K,
d_x=50 with a random 2D linear projection as the "embedding." Uses
`np.random.default_rng(seed=42)` for reproducibility. Adapted from
`research/2026-04-05-tw-perf-rerun-clean/scripts/gen_synthetic.py`.

**`scripts/compute_exact.py`** — Loads each dataset (MERFISH 10K, MERFISH 50K,
Gaussian 10K), computes exact `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)`,
records T_exact and wall-clock time. Outputs JSON to `results/exact_trustworthiness.json`.
Adapted from `research/2026-04-04-tw-perf-scaling/scripts/sklearn_reference.py`.

**`scripts/sweep_approach_b.py`** — For each (dataset, m, seed) configuration:
draws random indices via `np.random.default_rng(seed)`, slices X[idx] and Y[idx],
calls `sklearn.manifold.trustworthiness(X[idx], Y[idx], n_neighbors=k)`, records
T_sub and wall-clock time. This is Approach B (subset embedding). Outputs per-trial
JSON rows to `results/approach_b_results.json`.

**`scripts/sweep_approach_a.py`** — Custom Python implementation of Approach A
(row sub-sampling with full-population distances). For each (dataset, m, seed):
draws random indices, computes `pairwise_distances(X[idx], X)` for the (m, n)
distance slab, ranks via argsort, uses `NearestNeighbors.fit(Y).kneighbors(Y[idx])`
for output-space k-NN, applies the trustworthiness penalty formula with
denominator `m*k*(2*n - 3*k - 1)`. Records T_sub and wall-clock time. Memory
constraint: for n=50K, Approach A is capped at m=5000 (the (m,n) distance matrix
at m=5K, n=50K is ~2GB; beyond this, memory requirements become prohibitive
without chunking).

**`scripts/analyze_results.py`** — Reads all results JSON files, computes:
- mean(|ΔT|) and std(T_sub) per (approach, dataset, n, m)
- Fits std vs. m on log-log scale (CLT check: slope should be ~-0.5)
- Computes speedup ratios
- Produces CSV summary table and 4 PNG plots
- Tests each hypothesis (H1–H6) against measured data
- Outputs analysis to `results/analysis/`.

## Environment

**Custom environment required.**

The experiment is Python-only and requires scikit-learn for ground-truth
trustworthiness computation. The project's existing Rust/cargo toolchain is not
involved. A lightweight micromamba environment follows the established project
convention (every prior research experiment has its own environment.yml).

```yaml
name: subsampled-tw-tradeoff
channels:
  - conda-forge
dependencies:
  - python=3.11.*
  - numpy=2.2.*
  - scipy=1.15.*
  - scikit-learn=1.8.*
  - matplotlib=3.10.*
```

**Rationale for each dependency:**
- `numpy` — Load .npy fixtures, array operations, random number generation
- `scipy` — Used internally by sklearn for distance computations; also provides
  `scipy.spatial.distance` if needed for Approach A
- `scikit-learn` — `sklearn.manifold.trustworthiness` as ground truth (Approach B);
  `sklearn.neighbors.NearestNeighbors` for Approach A k-NN queries
- `matplotlib` — Generate error/speed plots for analysis

Version pins match the project's established conventions across all prior
research environments.

## Implementation Phases

### Phase 1: Directory Structure and Environment

1. Create `research/2026-04-09-subsampled-tw-tradeoff/` and all subdirectories
   (`scripts/`, `data/merfish/`, `data/gaussian/`, `results/`, `results/analysis/`)
2. Write `environment.yml` with the specification above
3. Create symlinks in `data/merfish/` pointing to the existing MERFISH fixtures at
   `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`
4. Build the environment: `micromamba create -f environment.yml -y`
5. Verify: `micromamba run -n subsampled-tw-tradeoff python -c "import sklearn; print(sklearn.__version__)"`

### Phase 2: Data Generation

1. Create `scripts/gen_gaussian_50d.py`:
   - Generate X: `rng.standard_normal((10000, 50))` with seed=42
   - Generate Y: `X @ rng.standard_normal((50, 2))` (random linear projection)
   - Save to `data/gaussian/gaussian_n10k_50d_x.npy` and `gaussian_n10k_50d_y.npy`
2. Run: `micromamba run -n subsampled-tw-tradeoff python scripts/gen_gaussian_50d.py`
3. Verify shapes and dtypes: assert X.shape == (10000, 50), Y.shape == (10000, 2), both float64

### Phase 3: Exact Trustworthiness Baselines

1. Create `scripts/compute_exact.py`:
   - Load each dataset (MERFISH 10K, MERFISH 50K, Gaussian 10K)
   - For each: call `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` with
     wall-clock timing (3 runs, take median)
   - Output JSON: `{"dataset": str, "n": int, "k": 15, "T_exact": float, "time_s": float}`
2. Run: `micromamba run -n subsampled-tw-tradeoff python scripts/compute_exact.py`
3. Verify: T_exact for MERFISH 10K should be a reasonable value (likely 0.95–0.99
   for a good UMAP embedding). MERFISH 50K similar. Gaussian should be ~0.5 (no
   meaningful structure to preserve).
4. Record T_exact values — these are the ground truth for all subsequent |ΔT| computations.

### Phase 4: Approach B Sweep (Subset Embedding)

1. Create `scripts/sweep_approach_b.py`:
   - Configuration matrix:
     - MERFISH 10K: m ∈ {250, 500, 1000, 2000, 5000}, seeds 0–9
     - MERFISH 50K: m ∈ {250, 500, 1000, 2000, 5000, 10000, 25000}, seeds 0–9
     - Gaussian 10K: m ∈ {250, 500, 1000, 2000, 5000}, seeds 0–9
   - For each (dataset, m, seed):
     - `idx = np.random.default_rng(seed).choice(n, size=m, replace=False)`
     - `T_sub = sklearn.manifold.trustworthiness(X[idx], Y[idx], n_neighbors=min(k, m-1))`
     - Record wall-clock time via `time.perf_counter`
   - Output: JSONL file with one row per trial
   - Note: clamp `n_neighbors` to `min(k, m-1)` when m <= k (applies to m=250 with k=15: fine,
     but defensive check is warranted)
2. Run: `micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_b.py`
3. Verify: spot-check that T_sub at m=5000 on MERFISH 10K is close to T_exact.
   Total trials: (5 + 7 + 5) × 10 = 170 trials. Expected runtime: ~10–30 min
   (dominated by n=50K large-m trials).

### Phase 5: Approach A Sweep (Row Sub-sampling)

1. Create `scripts/sweep_approach_a.py`:
   - Custom trustworthiness_approach_a(X_full, Y_full, idx, k) function:
     ```
     dist_X = pairwise_distances(X_full[idx], X_full)  # (m, n)
     rank_X = argsort(argsort(dist_X, axis=1), axis=1) # rank of each point
     nn = NearestNeighbors(n_neighbors=k).fit(Y_full)
     ind_Y = nn.kneighbors(Y_full[idx], return_distance=False)  # (m, k)
     penalty = sum of max(0, rank_X[i,j] - k) for j in ind_Y[i] for all i
     T = 1 - 2/(m*k*(2*n-3*k-1)) * penalty
     ```
   - Configuration matrix:
     - MERFISH 10K: m ∈ {250, 500, 1000, 2000, 5000}, seeds 0–9
     - MERFISH 50K: m ∈ {250, 500, 1000, 2000, 5000}, seeds 0–9
       (capped at m=5000 due to memory: (5000, 50000) × 8 bytes = 2 GB)
     - Gaussian 10K: m ∈ {250, 500, 1000, 2000, 5000}, seeds 0–9
   - Output: JSONL file with one row per trial
2. Run: `micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_a.py`
3. Verify: T_sub at m=5000 on MERFISH 10K should be very close to T_exact
   (Approach A is an unbiased estimator). Total trials: (5 + 5 + 5) × 10 = 150.
   Expected runtime: ~15–60 min (Approach A is slower per trial due to (m,n) distances).

### Phase 6: Dry Run

Before committing to full sweeps, run each script with a minimal configuration:
- `compute_exact.py` on MERFISH 10K only
- `sweep_approach_b.py` with m=1000, seeds=[0], MERFISH 10K only
- `sweep_approach_a.py` with m=1000, seeds=[0], MERFISH 10K only
- `analyze_results.py` on the minimal outputs

Verify: all scripts complete without error, JSON outputs are well-formed,
|ΔT| at m=1000 is a plausible value (not 0.47 — that would indicate a
normalization bug repeat).

### Phase 7: Analysis

1. Create `scripts/analyze_results.py`:
   - Load exact baselines and both sweep result files
   - Compute per-configuration statistics: mean(|ΔT|), std(T_sub), mean(time), mean(speedup)
   - Produce `results/analysis/error_speed_table.csv` with columns:
     [approach, dataset, n, m, mean_delta_t, std_t_sub, mean_time_s, mean_speedup]
   - Generate 4 plots:
     - `delta_t_vs_m.png`: |ΔT| vs m, one line per (approach, dataset), with std error bars
     - `std_vs_m_loglog.png`: log(std) vs log(m), fit line, report slope (expect ~-0.5)
     - `speedup_vs_m.png`: speedup ratio vs m for both approaches
     - `merfish_vs_gaussian.png`: |ΔT| comparison at matched m values for H5
   - Test each hypothesis:
     - H1: Check if Approach A on MERFISH 10K achieves |ΔT| < 0.01, std < 0.005 at m >= 2000
     - H2: Check if Approach B on MERFISH 10K achieves |ΔT| < 0.01, std < 0.005 at m >= 5000
     - H3: Check speedup scaling (linear for A, quadratic for B)
     - H4: Check log-log slope of std vs m (expect ~-0.5)
     - H5: Compare std on MERFISH vs Gaussian at matched m
     - H6: Extrapolate error curve to assess "10K from 100K" claim
   - Print hypothesis verdicts to stdout
2. Run: `micromamba run -n subsampled-tw-tradeoff python scripts/analyze_results.py`

## Execution Protocol

All commands run from `research/2026-04-09-subsampled-tw-tradeoff/`.

```bash
# 1. Set up environment
micromamba create -f environment.yml -y
micromamba run -n subsampled-tw-tradeoff python -c "import sklearn; print(sklearn.__version__)"

# 2. Generate synthetic data
micromamba run -n subsampled-tw-tradeoff python scripts/gen_gaussian_50d.py

# 3. Compute exact baselines (run once, ~5 min for n=50K)
micromamba run -n subsampled-tw-tradeoff python scripts/compute_exact.py

# 4. Dry run (minimal config, verify correctness)
micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_b.py --dry-run
micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_a.py --dry-run

# 5. Full sweeps (after dry run passes)
micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_b.py
micromamba run -n subsampled-tw-tradeoff python scripts/sweep_approach_a.py

# 6. Analysis
micromamba run -n subsampled-tw-tradeoff python scripts/analyze_results.py
```

Each script should print progress to stderr (dataset/m/seed being processed) and
write structured results to the `results/` directory. Scripts should be idempotent
— re-running overwrites previous results cleanly.

## Analysis Plan

### Primary Analysis: Error/Speed Table

For each (approach, dataset, n, m), compute:
- mean(|ΔT|) and 95% CI: `mean ± 1.96 * std / sqrt(10)`
- std(T_sub) across 10 seeds
- mean wall-clock time (seconds)
- speedup = T_exact_time / mean(T_sub_time)

### Variance Scaling (CLT Check)

Fit `log(std) = a * log(m) + b` across m values for each (approach, dataset).
If the CLT applies, the slope `a` should be approximately -0.5. Deviation from
-0.5 indicates non-independent or non-identically-distributed per-point
contributions (expected for real data with cluster structure).

### Approach A vs. B Comparison

At matched (dataset, n, m) values, compare:
- |ΔT|: Approach A should have lower bias (it's an unbiased estimator)
- std: Approach A may have higher variance per trial (fewer points)
- Speedup: Approach B should have quadratic speedup vs linear for A
- Pareto frontier: plot |ΔT| vs speedup to identify the optimal approach for
  each accuracy/speed trade-off point

### Hypothesis Testing

Each hypothesis is tested by checking the measured values against the stated
thresholds. No formal statistical tests (t-tests, etc.) are needed — the
hypotheses are stated as threshold comparisons, and the 10-seed variance
provides confidence intervals directly.

### Extrapolation

For the "10K from 100K" claim (H6): if the error-vs-m/n relationship is
monotonic, extrapolate from the n=50K data. Specifically, check whether
|ΔT| at m/n = 0.2 (10K/50K) is consistent with what m/n = 0.1 (10K/100K)
would predict under the measured variance scaling model.

## Success Criteria

- **Conclusive positive:** At least one (approach, m) combination achieves
  mean(|ΔT|) < 0.01 AND std(T_sub) < 0.005 on MERFISH data with speedup >= 3x.
  This identifies a viable default subsample size for `trustworthiness_subsampled`.

- **Conclusive negative:** No (approach, m < n/2) combination achieves
  mean(|ΔT|) < 0.01 on MERFISH data. Sub-sampling is not viable for this use case
  at the 1% accuracy tolerance. The "10K subsample" claim is empirically falsified.

- **Inconclusive:** Error is below threshold only at m >= n/2 (marginal benefit),
  or variance across seeds is too high to distinguish signal from noise (std > 0.01).
  Would require either more seeds or larger n to resolve.

## Threats to Validity

### Internal

1. **Normalization bug (repeat of H5 failure):** The prior experiment produced
   |ΔT| = 0.47 due to mixed Approach A/B semantics in the denominator. Mitigation:
   Approach B uses sklearn directly (no custom normalization). Approach A uses a
   clearly documented custom implementation with explicit denominator
   `m*k*(2*n - 3*k - 1)`. The dry run (Phase 6) specifically checks for this
   failure mode by verifying |ΔT| at m=1000 is not ~0.47.

2. **sklearn version differences:** The exact trustworthiness implementation may
   differ slightly between sklearn versions. Mitigation: pin sklearn=1.8.x in
   environment.yml; both exact and subsampled computations use the same sklearn.

3. **Wall-clock measurement noise:** Python's GC, OS scheduling, and other
   processes introduce timing noise. Mitigation: use `time.perf_counter` (high
   resolution), discard 1 warm-up run, take median of 3 timed runs for exact
   computation. Sub-sampled runs are naturally replicated (10 seeds).

4. **Approach A self-distance handling:** When computing `pairwise_distances(X[idx], X)`,
   the query points appear in both the query set and the target set. The self-distance
   must be excluded from ranking. Mitigation: explicitly set `dist_X[i, idx[i]] = inf`
   for each query point i.

5. **k clamping for small m:** When m < 2k, `n_neighbors=k` exceeds the number of
   available neighbors. Mitigation: use `min(k, m-1)` and flag these configurations
   in results. For m=250, k=15 is fine (m >> 2k). The smallest m=250 with k=15 is
   safe.

### External

1. **MERFISH specificity:** MERFISH (n=10K–50K, d_x=50, gene expression data)
   may not be representative of all UMAP use cases. Image data, text embeddings,
   or graph data may have different local structure. Results should be stated as
   specific to MERFISH-like data.

2. **Embedding quality dependence:** The accuracy of sub-sampled trustworthiness
   may depend on how well the embedding preserves structure. A poor embedding
   (low T_exact) may behave differently under sub-sampling than a good one.
   Mitigation: the Gaussian comparison (T_exact ~ 0.5) provides a contrast point.

3. **Python vs. Rust performance:** Wall-clock speedups measured in Python
   (sklearn) may not transfer directly to the Rust implementation, which uses
   SIMD, Rayon, and scratch-buffer optimizations. The speedup ratios characterize
   the algorithmic scaling (m/n and (m/n)^2), but absolute times will differ in
   Rust. A follow-up Rust benchmark would be needed for production performance
   claims.

4. **Subsample size scaling:** The n=10K and n=50K datasets are small enough that
   exact computation is already feasible. The real value of sub-sampling is at
   n >= 100K, which would require either extrapolation or larger fixtures.

## Estimated Resource Requirements

- **Compute time:** ~1–2 hours total
  - Exact baselines: ~5 min (dominated by n=50K)
  - Approach B sweep: ~10–30 min (170 trials, most fast; n=50K m=25K is slowest)
  - Approach A sweep: ~15–60 min (150 trials; (m,n) distance computation is heavy)
  - Analysis: < 1 min
- **Peak memory:** ~2 GB (Approach A at m=5000, n=50K: the (5000, 50000) distance
  matrix is 2 GB)
- **Disk space:** < 100 MB (fixtures already on disk; results are small JSON/CSV/PNG)
- **Dependencies:** Python 3.11, numpy, scipy, scikit-learn, matplotlib (all via
  micromamba, ~200 MB installed)
