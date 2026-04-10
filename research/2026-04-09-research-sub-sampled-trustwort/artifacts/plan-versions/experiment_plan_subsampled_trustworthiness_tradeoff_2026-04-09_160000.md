# Experiment Plan: Sub-sampled Trustworthiness Error/Speed Trade-off

## Motivation

Exact trustworthiness computation is O(n²) and becomes the dominant cost when evaluating embedding quality on large datasets. Sub-sampling is the only approach that changes this fundamental scaling (prior research eliminated KD-tree and ANN alternatives). However, no correct sub-sampled trustworthiness implementation exists in this project — the sole prior attempt (H5 in `2026-04-05-tw-perf-rerun-clean`) produced systematically biased results due to a normalization bug. This experiment will establish the empirical error/speed trade-off for two distinct sub-sampling approaches on real-world (MERFISH) data, producing the evidence needed to decide whether to ship a `trustworthiness_subsampled` function in `src/metrics.rs` and what default subsample size to recommend.

## Hypothesis

Two independent hypothesis pairs, one per sub-sampling approach. Each is evaluated independently with its own pass/fail verdict.

### H1_A — Approach A (Row Sub-sampling, Unbiased Estimator of Full-n T)

**H0_A:** mean(|ΔT_A|) ≥ 0.01 OR max(|ΔT_A|) ≥ 0.02 OR std(T_sub_A) ≥ 0.005 at m=2000 on MERFISH n=10K, k=15

**H1_A:** mean(|ΔT_A|) < 0.01 AND max(|ΔT_A|) < 0.02 AND std(T_sub_A) < 0.005 at m=2000 on MERFISH n=10K, k=15

### H1_B — Approach B (Subset Embedding, Estimator of Subset-T)

**H0_B:** mean(|ΔT_B|) ≥ 0.01 OR max(|ΔT_B|) ≥ 0.02 OR std(T_sub_B) ≥ 0.005 at m=5000 on MERFISH n=10K, k=15

**H1_B:** mean(|ΔT_B|) < 0.01 AND max(|ΔT_B|) < 0.02 AND std(T_sub_B) < 0.005 at m=5000 on MERFISH n=10K, k=15

### Pre-specified Outcome Table

| A verdict | B verdict | Outcome | Operational consequence |
|-----------|-----------|---------|------------------------|
| H1_A supported | H1_B supported | Both pass | Ship `trustworthiness_subsampled` with Approach A (preferred, unbiased estimator) as default; Approach B supported as an alternative for subset-T use cases |
| H1_A supported | H0_B not rejected | A passes only | Ship `trustworthiness_subsampled` with Approach A only; Approach B not recommended at m=5000 for accuracy parity |
| H0_A not rejected | H1_B supported | B passes only | Approach B estimates a different quantity (subset-T, not full-n T); evaluate whether subset-T is an acceptable product metric before shipping |
| H0_A not rejected | H0_B not rejected | Both fail | Sub-sampling does not provide acceptable accuracy on MERFISH at the tested m values; larger m or alternative approaches required |

### Secondary Hypotheses (Exploratory)

**H2 (Variance scaling — exploratory):** Standard deviation of the sub-sampled estimator scales as C/√m for both approaches, consistent with CLT. Evaluated by fitting std vs. m on a log-log plot and checking slope ≈ −0.5. No formal H0; reported descriptively.

**H3 (MERFISH vs. synthetic — exploratory/descriptive):** The error profile on MERFISH data differs from synthetic Gaussian data — specifically, variance is higher on MERFISH due to heterogeneous cluster structure. Evaluated by comparing std at matched m values on both data types. Reported descriptively.

**H4 (Speed scaling):** Wall-clock speedup is approximately linear in n/m for Approach A and quadratic (n/m)² for Approach B. On MERFISH n=50K → m=5K, Approach A gives ~10× speedup, Approach B gives ~100× speedup. Evaluated by measuring wall times and fitting the scaling relationship.

**H5 (n=50K qualitative consistency — exploratory):** The error/speed trade-off curve at n=50K is qualitatively consistent with n=10K: the crossover m/n ratio (where |ΔT| first drops below 0.01) is within 2× of the n=10K value. Reported descriptively.

**H6 (Extrapolation — out-of-distribution projection):** The "10K subsample from n=100K gives ~1% error and ~100× speedup" claim from PR #238 is evaluated by extrapolating the measured error curve. Clearly marked as out-of-distribution projection, not empirical falsification.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Subsample size m | {250, 500, 1000, 2000, 5000, 7500} for n=10K; add {10000, 25000} for n=50K | Spans from 2.5% to 75% of n=10K; primary cells at m=2000 (A) and m=5000 (B); finer grid around decision boundary |
| Sub-sampling approach | {A (row sub-sampling), B (subset embedding)} | Two fundamentally different estimands; scope report recommends testing both |
| Dataset size n | {10000, 50000} | MERFISH at two scales; n=50K tests scaling behavior |
| Random seed | {0, 1, 2, ..., 9} (10 seeds per cell) | CLT-motivated: 10 trials gives ±1.96·σ/√10 ≈ 0.62·σ for 95% CI width |
| Data type | {MERFISH (real), Gaussian (synthetic)} | For H3/H5 comparison only; primary hypotheses tested on MERFISH |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| T_exact | dimensionless [0,1] | `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` on full dataset | `trustworthiness` (existing in `src/metrics.rs`) |
| T_sub_A | dimensionless [0,1] | Custom Approach A function (m query rows, full population distances) | NEW — `trustworthiness_row_subsampled` |
| T_sub_B | dimensionless [0,1] | `sklearn.manifold.trustworthiness(X[idx], Y[idx], n_neighbors=15)` | NEW — `trustworthiness_subset_embedding` |
| |ΔT_A| | dimensionless | `abs(T_exact - T_sub_A)` | NEW — `tw_subsampling_abs_error_A` |
| |ΔT_B| | dimensionless | `abs(T_exact - T_sub_B)` | NEW — `tw_subsampling_abs_error_B` |
| std(T_sub) | dimensionless | Sample std-dev across 10 seeds | NEW — `tw_subsampling_std` |
| wall_exact_s | seconds | `time.perf_counter()` around exact computation | NEW — `tw_wall_exact_s` |
| wall_sub_s | seconds | `time.perf_counter()` around sub-sampled computation | NEW — `tw_wall_sub_s` |
| speedup | ratio | `wall_exact_s / wall_sub_s` | NEW — `tw_subsampling_speedup` |

All metrics marked "NEW" are experiment-local — they exist only in the experiment scripts, not in `src/metrics.rs`. If the experiment leads to shipping a Rust function, canonical names and thresholds would be added to `src/metrics.rs` in a follow-up.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (n_neighbors) | 15 | Standard UMAP default; matches all prior TW benchmarks in this project |
| Distance metric | Squared Euclidean | Matches `src/metrics.rs` production implementation and sklearn default |
| MERFISH embedding | Pre-computed UMAP Y arrays from existing fixtures | Same embedding across all trials; isolates sub-sampling variance |
| Python runtime | CPython 3.11 via micromamba | Consistent across all measurements |
| sklearn version | 1.6.0 | Matches prior validated research environments |
| Hardware | Same machine for all runs | Wall-clock comparisons only valid within a single hardware config |
| Warmup | 1 warmup run discarded before timed runs | Eliminates cold-cache effects for timing measurements |

## Inputs and Data

### Existing Datasets (verified on disk)

All MERFISH fixtures confirmed at `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`:

| File | Shape | Dtype | Size |
|------|-------|-------|------|
| `merfish_n10k_x.npy` | (10000, 50) | f64 | 3.9 MB |
| `merfish_n10k_y.npy` | (10000, 2) | f64 | 160 KB |
| `merfish_n50k_x.npy` | (50000, 50) | f64 | 20 MB |
| `merfish_n50k_y.npy` | (50000, 2) | f64 | 800 KB |

### Datasets to Generate

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| `gaussian_n10k_x.npy` | `gen_synthetic.py --d 50 --sizes 10000` | (10000, 50) f64, i.i.d. N(0,1) | H3: synthetic comparison at matched d=50 |
| `gaussian_n10k_y.npy` | Same script | (10000, 2) f64, i.i.d. N(0,1) | H3: synthetic Y (no manifold structure) |
| `gaussian_n50k_x.npy` | `gen_synthetic.py --d 50 --sizes 50000` | (50000, 50) f64, i.i.d. N(0,1) | H3/H5: synthetic comparison at n=50K |
| `gaussian_n50k_y.npy` | Same script | (50000, 2) f64, i.i.d. N(0,1) | H3/H5: synthetic Y at n=50K |

Generator script: Reuse `research/2026-04-05-tw-perf-rerun-clean/scripts/gen_synthetic.py` with `--d 50`. The existing Gaussian fixtures are d=10, which does not match MERFISH d=50; new d=50 files are needed for a fair H3 comparison.

### Data Properties

- MERFISH X has real manifold structure with non-uniform cluster density — this is precisely why sub-sampling accuracy may differ from Gaussian (H3).
- MERFISH Y are actual UMAP 2D embeddings; Gaussian Y are random 2D projections with no neighborhood preservation. Gaussian data should have T ≈ 0.5 (no structure), making |ΔT| interpretation different — the comparison is about variance, not absolute error.

## Experiment Directory Layout

```
research/2026-04-09-subsampled-tw-tradeoff/
├── environment.yml                     # Micromamba/conda environment
├── scripts/
│   ├── gen_data.py                     # Generate Gaussian d=50 fixtures
│   ├── compute_exact.py                # Compute and cache exact T values
│   ├── run_subsampling.py              # Main experiment: both approaches, all m values, all seeds
│   ├── analyze_results.py              # Aggregate JSONs → tables, plots, verdicts
│   └── utils.py                        # Shared: Approach A implementation, I/O helpers
├── data/
│   ├── merfish/                        # Symlinks to existing MERFISH fixtures
│   └── gaussian/                       # Generated Gaussian d=50 fixtures
├── results/
│   ├── raw/                            # One JSON per (approach, m, seed, dataset) trial
│   └── analysis/                       # Aggregated tables, plots, verdict summary
└── report.md                           # Final report (written by write-report)
```

### File Descriptions

- **`environment.yml`** — Micromamba environment with numpy, scipy, scikit-learn, matplotlib.
- **`scripts/gen_data.py`** — Generates Gaussian N(0,1) arrays at d=50 for n=10K and n=50K. Saves to `data/gaussian/`. Deterministic seed for reproducibility.
- **`scripts/compute_exact.py`** — Computes exact T(k=15) via sklearn on MERFISH and Gaussian data at all n values. Caches results to `results/raw/exact_*.json`. Run once before the main experiment.
- **`scripts/run_subsampling.py`** — Main experiment driver. For each (approach, m, seed, dataset) combination: draws random indices, computes T_sub, records |ΔT|, wall time. Outputs one JSON per trial to `results/raw/`. Supports `--dry-run` for smoke testing with a single seed and single m value.
- **`scripts/analyze_results.py`** — Reads all raw JSONs, computes per-cell statistics (mean, std, max of |ΔT|), evaluates H1_A and H1_B verdicts against thresholds, generates the error/speed table, fits variance scaling, and writes the analysis report to `results/analysis/`.
- **`scripts/utils.py`** — Contains the Approach A `trustworthiness_row_subsampled(X, Y, k, query_idx)` implementation, shared I/O helpers for loading data and writing result JSONs, and common constants (k=15, seed list, m values).

## Environment

**Custom environment required.**

The experiment requires Python with numpy, scikit-learn, and matplotlib, which are not part of the Rust project's standard toolchain. A micromamba environment will be created.

```yaml
name: subsampled-tw-tradeoff
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2.6
  - scipy=1.15.2
  - scikit-learn=1.6.0
  - matplotlib=3.10.1
```

**Rationale:**
- `python=3.11` — Matches all prior research environments in this project.
- `numpy=2.2.6` — Consistent with recent experiments; required for .npy I/O and array operations.
- `scipy=1.15.2` — Transitive dependency of scikit-learn; pinned for reproducibility.
- `scikit-learn=1.6.0` — Provides `trustworthiness()` and `NearestNeighbors`; matches the version validated in `tw-perf-rerun-clean`.
- `matplotlib=3.10.1` — For error/speed plots; follows `kdtree-y-knn` precedent.

## Implementation Phases

### Phase 1: Directory Structure and Environment

1. Create `research/2026-04-09-subsampled-tw-tradeoff/` and all subdirectories (`scripts/`, `data/merfish/`, `data/gaussian/`, `results/raw/`, `results/analysis/`).
2. Create `environment.yml` with the specification above.
3. Create symlinks in `data/merfish/` pointing to the existing MERFISH fixtures at `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`.
4. Build the environment: `micromamba create -f environment.yml -y && micromamba activate subsampled-tw-tradeoff`.
5. Verify: `python -c "from sklearn.manifold import trustworthiness; print('OK')"`.

### Phase 2: Data Generation and Shared Utilities

1. Create `scripts/utils.py` containing:
   - Constants: `K = 15`, `SEEDS = list(range(10))`, `M_VALUES_10K = [250, 500, 1000, 2000, 5000, 7500]`, `M_VALUES_50K = [250, 500, 1000, 2000, 5000, 7500, 10000, 25000]`.
   - `trustworthiness_row_subsampled(X, Y, k, query_idx)` — Approach A implementation:
     ```python
     def trustworthiness_row_subsampled(X, Y, k, query_idx):
         """Approach A: m query rows, distances to ALL n points."""
         n = X.shape[0]
         m = len(query_idx)
         # (m, n) pairwise distances in X-space, query rows to all points
         dist_X = pairwise_distances(X[query_idx], X)
         for i, gi in enumerate(query_idx):
             dist_X[i, gi] = np.inf  # exclude self
         # Ranks: argsort of argsort gives rank (0-indexed), +1 for 1-indexed
         ranks_X = np.argsort(np.argsort(dist_X, axis=1), axis=1) + 1
         # X-space k-NN: columns with rank <= k
         x_knn_mask = ranks_X <= k  # (m, n) boolean
         # Y-space k-NN: fit on ALL n points, query m rows
         nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Y)
         y_knn_idx = nn.kneighbors(Y[query_idx], return_distance=False)  # (m, k)
         # Penalty: for each Y-neighbor not in X-kNN, add (rank_X - k)
         penalty = 0.0
         for i in range(m):
             for j_col in y_knn_idx[i]:
                 if not x_knn_mask[i, j_col]:
                     penalty += ranks_X[i, j_col] - k
         denom = m * k * (2 * n - 3 * k - 1)
         return 1.0 - 2.0 * penalty / denom
     ```
   - I/O helpers: `load_npy_pair(data_dir, prefix, n)`, `save_result_json(path, result_dict)`.
2. Create `scripts/gen_data.py`:
   - Generate Gaussian arrays at d_x=50, d_y=2, n=10K and n=50K with `np.random.RandomState(42)`.
   - Save to `data/gaussian/gaussian_n{n}_x.npy` and `gaussian_n{n}_y.npy`.
3. Run `python scripts/gen_data.py` and verify output shapes.

### Phase 3: Experiment Scripts

1. Create `scripts/compute_exact.py`:
   - Load each (dataset, n) combination.
   - Compute exact T(k=15) via `sklearn.manifold.trustworthiness`.
   - Time with `time.perf_counter()` (1 warmup + 3 timed runs, take median).
   - Save to `results/raw/exact_{dataset}_{n}.json` with fields: `{dataset, n, k, T_exact, wall_median_s, wall_runs}`.
2. Create `scripts/run_subsampling.py`:
   - Accept `--dry-run` flag (runs single seed, single m, both approaches — for CI smoke testing).
   - For each (dataset, n, approach, m, seed):
     - Draw `m` random indices via `np.random.RandomState(seed).choice(n, size=m, replace=False)`.
     - Approach A: call `trustworthiness_row_subsampled(X, Y, k=15, query_idx)`.
     - Approach B: call `sklearn.manifold.trustworthiness(X[idx], Y[idx], n_neighbors=15)`.
     - Load cached T_exact, compute |ΔT|.
     - Time with `time.perf_counter()`.
     - Save to `results/raw/sub_{approach}_{dataset}_{n}_m{m}_s{seed}.json` with fields: `{approach, dataset, n, m, seed, k, T_sub, T_exact, delta_T, abs_delta_T, wall_s}`.
   - Print progress to stderr.
3. Create `scripts/analyze_results.py`:
   - Glob all `results/raw/sub_*.json` files.
   - Group by (approach, dataset, n, m).
   - Compute per-cell: mean(|ΔT|), max(|ΔT|), std(T_sub), mean(wall_s), mean(speedup).
   - Evaluate H1_A: check thresholds at m=2000, MERFISH n=10K.
   - Evaluate H1_B: check thresholds at m=5000, MERFISH n=10K.
   - Map to four-cell outcome table.
   - Fit variance scaling: log-log regression of std vs. m, report slope (expect ≈ −0.5).
   - H3: compare std at matched m between MERFISH and Gaussian (use m=2000 and m=5000 as comparison points; specify this pre-analysis per HF-8 guidance).
   - H5: compute crossover m/n ratio at n=10K and n=50K, report whether within 2×.
   - H6: extrapolate error curve to n=100K (clearly marked as out-of-distribution projection).
   - Generate markdown tables and save to `results/analysis/summary.md`.
   - Generate plots (error vs. m, speedup vs. m, std vs. m log-log) and save to `results/analysis/`.
4. Verify each script individually with small test inputs before proceeding.

### Phase 4: Dry Run

1. Run `python scripts/compute_exact.py` on MERFISH n=10K only (skip n=50K for dry run).
2. Run `python scripts/run_subsampling.py --dry-run` — single seed (0), single m (2000), both approaches, MERFISH n=10K.
3. Run `python scripts/analyze_results.py` on the dry-run output.
4. Verify: JSON schema is correct, analysis script reads and aggregates without errors, plots generate, verdicts render.
5. If dry run passes, proceed to full execution.

## Execution Protocol

### Step 1 — Environment Setup
```bash
cd research/2026-04-09-subsampled-tw-tradeoff
micromamba create -f environment.yml -y
micromamba activate subsampled-tw-tradeoff
```

### Step 2 — Generate Data
```bash
python scripts/gen_data.py
# Verify: data/gaussian/gaussian_n10k_x.npy (10000,50), gaussian_n50k_x.npy (50000,50)
```

### Step 3 — Compute Exact Baselines
```bash
python scripts/compute_exact.py
# Expected output: results/raw/exact_merfish_10000.json, exact_merfish_50000.json,
#                  exact_gaussian_10000.json, exact_gaussian_50000.json
# Expected wall time: ~3.6s for MERFISH n=10K (from prior profiling), ~90s for n=50K
```

### Step 4 — Dry Run
```bash
python scripts/run_subsampling.py --dry-run
python scripts/analyze_results.py
# Inspect results/analysis/summary.md for correctness
```

### Step 5 — Full Experiment
```bash
python scripts/run_subsampling.py
# Total trials: 2 approaches × (6 m-values for n=10K + 8 m-values for n=50K) × 10 seeds × 2 datasets = 560 trials
# Estimated wall time: ~2-4 hours (dominated by Approach A at large n with full pairwise distances)
```

### Step 6 — Analysis
```bash
python scripts/analyze_results.py
# Outputs: results/analysis/summary.md, results/analysis/*.png
```

### Step 7 — Review
Inspect `results/analysis/summary.md` for:
- H1_A verdict (primary cell: Approach A, m=2000, MERFISH n=10K)
- H1_B verdict (primary cell: Approach B, m=5000, MERFISH n=10K)
- Four-cell outcome mapping
- Error/speed table
- Variance scaling fit
- MERFISH vs. Gaussian comparison

## Analysis Plan

### Primary Analysis: Hypothesis Testing

For each primary cell (H1_A at m=2000, H1_B at m=5000), compute across 10 seeds:
- `mean_abs_delta_T = mean(|ΔT|)` — must be < 0.01 for H1
- `max_abs_delta_T = max(|ΔT|)` — must be < 0.02 for H1
- `std_T_sub = std(T_sub)` — must be < 0.005 for H1

**All three conditions must hold** for H1 to be supported. If any fails, H0 is not rejected.

Map the two verdicts to the four-cell outcome table to determine the operational consequence.

### Inconclusive Zone (per HF-7)

If mean(|ΔT|) falls in [0.008, 0.012], report the verdict but flag it as "near threshold" for each hypothesis independently. H1_A and H1_B can each independently be conclusive, near-threshold, or rejected.

### Secondary Analyses (Exploratory)

**Variance scaling (H2):** Fit `log(std) = a·log(m) + b` across all m values. Report slope `a` and R². If `a ≈ −0.5` (within ±0.1), CLT scaling holds.

**MERFISH vs. Gaussian (H3):** Compare `std(T_sub)` at m=2000 and m=5000 (pre-specified comparison points per HF-8) between MERFISH and Gaussian datasets. Report the ratio `std_MERFISH / std_Gaussian`. Values > 1 suggest heterogeneous manifold structure increases variance.

**Speed scaling (H4):** Fit `log(speedup) = c·log(n/m) + d`. For Approach A, expect `c ≈ 1` (linear). For Approach B, expect `c ≈ 2` (quadratic).

**n=50K consistency (H5):** Compute crossover m/n ratio (smallest m/n where mean(|ΔT|) < 0.01) at n=10K and n=50K. Report whether within 2×.

**Extrapolation (H6):** Extrapolate the fitted error curve to n=100K. Report the predicted |ΔT| at m=10K. Clearly mark as out-of-distribution projection, not empirical evidence.

## Success Criteria

- **H1_A conclusive positive:** mean(|ΔT_A|) < 0.01 AND max(|ΔT_A|) < 0.02 AND std(T_sub_A) < 0.005 at m=2000, MERFISH n=10K, k=15
- **H1_A conclusive negative:** Any of the three conditions is violated at m=2000
- **H1_B conclusive positive:** mean(|ΔT_B|) < 0.01 AND max(|ΔT_B|) < 0.02 AND std(T_sub_B) < 0.005 at m=5000, MERFISH n=10K, k=15
- **H1_B conclusive negative:** Any of the three conditions is violated at m=5000
- **Inconclusive (per-hypothesis):** mean(|ΔT|) in [0.008, 0.012] — near threshold, requires additional seeds or larger m to resolve

The experiment is complete when both H1_A and H1_B have been evaluated, the four-cell outcome table populated, and the error/speed trade-off table produced across all tested m values.

## Threats to Validity

### Internal

1. **Normalization bug risk:** The prior H5 experiment failed due to mixed semantics between Approach A and B denominators. Mitigation: the Approach A implementation is specified explicitly in `utils.py` with the correct denominator `m * k * (2n - 3k - 1)`, and will be validated against sklearn's exact result at m=n (where Approach A should equal exact T). A sanity check asserting `|T_A(m=n) - T_exact| < 1e-10` will be the first test in the dry run.
2. **sklearn version sensitivity:** Different sklearn versions may produce slightly different trustworthiness values due to tie-breaking in KNN. Mitigation: pin `scikit-learn=1.6.0` in `environment.yml`.
3. **Warm-cache effects on timing:** Re-using the same data across seeds means later runs benefit from cached memory. Mitigation: 1 warmup run discarded; timing is secondary to accuracy measurements.
4. **Approach A memory pressure:** `pairwise_distances(X[query_idx], X)` allocates an `(m, n)` matrix. At m=25K, n=50K, f64: ~10 GB. This may force swapping. Mitigation: monitor memory usage; if OOM occurs at large m, fall back to batch computation and document the limitation.

### External

1. **MERFISH-specific results:** Error profiles measured on MERFISH d=50 may not generalize to other data distributions or dimensionalities. The Gaussian comparison (H3) partially addresses this but is not a substitute for diverse real-world datasets.
2. **Python vs. Rust performance:** Wall-clock speedups measured in Python do not directly predict Rust speedups (SIMD, Rayon parallelism, different memory layout). The experiment characterizes accuracy, which transfers directly; speed results are indicative only.
3. **k=15 specificity:** Results are for k=15 only. Different k values may shift the error/speed curve. This experiment does not vary k.
4. **Single hardware configuration:** Wall-clock measurements are machine-specific and not portable.

## Estimated Resource Requirements

- **Compute time:** ~2–4 hours for the full experiment (560 trials). Dominated by Approach A at n=50K (full pairwise distance computation per trial). Approach B trials are fast (sub-second at small m).
- **Peak memory:** ~10 GB for Approach A at m=25K, n=50K (pairwise distance matrix). All other configurations fit in < 2 GB.
- **Disk space:** ~50 MB for generated data; ~5 MB for result JSONs; ~2 MB for plots. Total < 60 MB.
- **Dependencies:** micromamba, Python 3.11, numpy, scipy, scikit-learn, matplotlib (all via conda-forge).
