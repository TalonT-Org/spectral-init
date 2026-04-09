# Experiment Plan: Sub-sampled Trustworthiness Error/Speed Trade-off

## Revision Guidance Disposition

This plan addresses all findings from the design review (`revision_guidance_subsampled_trustworthiness_tradeoff_2026-04-09_121500.md`). Each critical finding (R1–R7), warning (W1–W9), and red-team decision point (RT1–RT9) is resolved below.

| ID | Resolution |
|----|------------|
| R1 | Speedup is **descriptive only** — removed from acceptance logic in success criteria. Timing methodology specified but speedup ratios are reported, not thresholded. |
| R2 | **Pre-selected primary cells**: m=2000 for Approach A, m=5000 for Approach B, both on MERFISH n=10K. All other cells are exploratory. Only the primary cells determine hypothesis verdicts. |
| R3 | Denominator `m·k·(2n−3k−1)` **validated by derivation** (see Appendix A). Dry-run includes m=n convergence check: T_A(m=n) must equal T_exact to within floating-point tolerance (|ΔT| < 1e-12). |
| R4 | Data manifest includes **byte sizes, shapes, dtypes, and SHA-256 checksums**. MERFISH provenance documented in Data Provenance section. |
| R5 | Symlink creation step **added to Execution Protocol** as Step 2, between environment setup and exact baseline computation. |
| R6 | Recommendation **bounded to n ≤ 50K**. Extrapolation to n > 50K explicitly marked as out-of-distribution with stated uncertainty. H6 downgraded from confirmatory to exploratory. |
| R7 | Fixed vs. varied components **documented explicitly** in Variance Protocol section. Seeds govern only subsample index selection. |
| W1 | **Symmetric warm-up**: 1 warm-up call + median of 3 timed runs for all measurements (exact, Approach A, Approach B). |
| W2 | Results at m < k+1 where k-clamping activates are **segregated** and reported separately, excluded from primary analysis. |
| W3 | Approach B explicitly defined as measuring **trustworthiness of the subset embedding**, not an estimator of full-n T. Research question updated accordingly. |
| W4 | Threshold provenance: |ΔT| < 0.01 from visual eval pipeline tolerance (scope report §Metric Context §2.2). std < 0.005 derived as half the |ΔT| threshold (ensuring 95% CI ≈ mean ± 2·std stays within 0.01). |
| W5 | All recommendations **scoped to "MERFISH-like data"** (gene expression, d=50, clustered manifold). Generalization requires additional datasets. |
| W6 | Timing methodology **symmetric** across all measurements — see Timing Protocol section. |
| W7 | Environment pins **exact versions** (not wildcards) for numpy, scikit-learn, matplotlib. Noted that BLAS backend is not pinned; acknowledged as limitation. |
| W8 | CLT slope deviation impact **defined**: if slope ∉ [-0.7, -0.3], the variance scaling model is rejected and extrapolation to untested m values is flagged as unreliable. |
| W9 | **max(|ΔT|) across seeds** added as supplementary metric. Primary cells must satisfy both mean(|ΔT|) < 0.01 AND max(|ΔT|) < 0.02 across 10 seeds. |
| RT1 | Approach B evaluates **trustworthiness of the subset** (different estimand). Research question and analysis explicitly distinguish the two estimands. |
| RT2 | Threshold provenance documented (W4). No pilot needed — thresholds anchored to existing visual eval pipeline tolerance. |
| RT3 | H6 **downgraded to exploratory** with bounded extrapolation and stated uncertainty. |
| RT5 | Scope: **MERFISH-specific validation only**. API decision requires additional datasets. |
| RT6 | **m=n convergence check** added to dry-run protocol (R3). |
| RT8 | **Primary cells pre-selected** (R2). Remaining cells are exploratory. |
| RT9 | Approach A has a custom implementation; Approach B uses sklearn directly. **Asymmetry documented**; Approach A includes additional correctness safeguards (m=n convergence, small-m sanity checks). |

---

## Motivation

Sub-sampled trustworthiness is the only known approach to break the O(n^2) scaling barrier for embedding quality assessment. The exact `trustworthiness` function in `src/metrics.rs` is fully optimized (AVX2+FMA SIMD, Rayon parallelism, introselect) but remains O(n^2) — prohibitive for n > 50K. Before adding a `trustworthiness_subsampled` function to the production crate, we need empirical evidence on:

1. **Accuracy**: How much error does sub-sampling introduce on real manifold data?
2. **Variance**: How reliable is a single sub-sampled estimate?
3. **Scaling**: How does error and variance scale with subsample size m?

This experiment characterizes two sub-sampling approaches on MERFISH spatial transcriptomics data (the project's primary real-world benchmark) and synthetic Gaussian data, producing a recommended default subsample size with quantitative justification. The results will inform whether to ship `trustworthiness_subsampled` in `src/metrics.rs` and what default parameters to use.

A prior attempt (H5 in `2026-04-05-tw-perf-rerun-clean`) produced systematically biased results (|ΔT| ≈ 0.47) due to a normalization bug. This experiment uses sklearn as ground truth for Approach B and a validated custom implementation for Approach A, eliminating the normalization risk.

## Hypothesis

**Null hypothesis (H0):** Sub-sampling does not provide acceptable accuracy — mean(|ΔT|) >= 0.01 or max(|ΔT|) >= 0.02 at the pre-selected primary subsample sizes (m=2000 for Approach A, m=5000 for Approach B) on MERFISH n=10K with k=15.

**Alternative hypothesis (H1):** At the pre-selected primary subsample sizes, mean(|ΔT|) < 0.01, max(|ΔT|) < 0.02, and std(T_sub) < 0.005 across 10 seeds, indicating sub-sampling provides sufficient accuracy for embedding quality monitoring on MERFISH-like data at n ≤ 50K.

### Secondary Hypotheses (Exploratory)

**H2 (Variance scaling):** Standard deviation of sub-sampled T scales as C/sqrt(m), consistent with CLT. Empirical log-log slope of std vs. m is in [-0.7, -0.3].

**H3 (MERFISH vs. synthetic):** The error profile on MERFISH differs from synthetic Gaussian — specifically, variance is higher on MERFISH due to heterogeneous cluster structure.

**H4 (Approach comparison):** Approach A (row sub-sampling, unbiased estimator of full-n T) achieves lower |ΔT| than Approach B (subset embedding, different estimand) at matched m, because A preserves the full population context.

**H5 (n=50K scaling — exploratory):** The accuracy relationship observed at n=10K holds qualitatively at n=50K for matched m/n ratios.

**H6 (Extrapolation — exploratory, bounded):** Error/speed curves can be extrapolated to n=100K with stated uncertainty bounds. This hypothesis is explicitly out-of-distribution and results are reported as bounded projections, not validated claims.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Approach | A (row sub-sampling), B (subset embedding) | Two fundamentally different sub-sampling strategies with different estimands |
| Subsample size m | 250, 500, 1000, **2000** (A primary), 3000, **5000** (B primary), 7500, 10000 (n=50K only), 25000 (n=50K only) | Spans from aggressive sub-sampling to near-full. Bold = pre-selected primary cells for hypothesis testing |
| Dataset | MERFISH n=10K, MERFISH n=50K, Gaussian n=10K | Real manifold vs. uniform structure comparison |
| Seed | 0–9 (10 seeds) | Sufficient for std estimation; seeds govern subsample index selection only |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name | Status |
|--------|------|-------------------|----------------|--------|
| T_exact | dimensionless [0,1] | `sklearn.manifold.trustworthiness(X, Y, n_neighbors=k)` | `trustworthiness` | EXISTS in src/metrics.rs |
| T_sub (per seed) | dimensionless [0,1] | Approach-specific computation (see Method) | NEW | Experiment-level |
| abs_delta_T | dimensionless | `abs(T_sub - T_exact)` per seed | NEW | Derived |
| mean_abs_delta_T | dimensionless | `mean(abs_delta_T)` across 10 seeds | NEW | Aggregated |
| max_abs_delta_T | dimensionless | `max(abs_delta_T)` across 10 seeds | NEW | Aggregated (W9) |
| std_T_sub | dimensionless | `std(T_sub, ddof=1)` across 10 seeds | NEW | Aggregated |
| wall_time_s | seconds | `time.perf_counter` with symmetric warm-up protocol | NEW | Descriptive only |
| speedup_ratio | dimensionless | `time_exact / time_sub` | NEW | **Descriptive only — not thresholded** (R1) |
| clt_slope | dimensionless | log-log regression slope of std_T_sub vs. m | NEW | Derived, exploratory |

For metrics marked "NEW": these are computed in the experiment scripts, not added to `src/metrics.rs`. They exist only in the experiment's analysis pipeline.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (n_neighbors) | 15 | Matches UMAP default and prior research |
| X, Y data | Fixed per dataset | MERFISH: loaded from .npy fixtures. Gaussian: generated once with seed=42, saved to data/, reused across all trials |
| sklearn version | 1.8.0 | Pinned for reproducibility |
| numpy version | 2.2.6 | Pinned for reproducibility |
| BLAS backend | System default (openblas via conda-forge) | Not pinned — acknowledged limitation (W7) |
| Warm-up protocol | 1 warm-up + median of 3 timed runs | Symmetric across all measurements (W1, W6) |
| Thread count | 1 (OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1) | Eliminate threading variance in timing |

### Variance Protocol (R7)

The following components are **fixed** across all seeds within a (dataset, n, m, approach) cell:
- X (high-dimensional input): loaded once, immutable
- Y (embedding): loaded once (MERFISH) or generated once with seed=42 (Gaussian), immutable
- k: constant at 15
- m: constant per cell

The **only** component varied across the 10 seeds is the **subsample index selection** (`np.random.default_rng(seed).choice(n, size=m, replace=False)`). Therefore, std(T_sub) across seeds measures exclusively sub-sampling variance, not run-to-run or data-generation noise.

## Inputs and Data

### Data Provenance (R4)

**MERFISH fixtures**: Spatial transcriptomics gene expression data from the MERFISH protocol. The n=10K subset is a random sample of the full MERFISH dataset; the n=50K subset is a larger sample. Y embeddings were produced by Python UMAP with default parameters (n_neighbors=15, min_dist=0.1, metric=euclidean). These fixtures have been used in prior experiments (`2026-04-05-tw-perf-rerun-clean`, `2026-04-08-tw-merfish-step-timing`) and are verified on disk.

**Gaussian fixtures**: Available pre-generated at `research/2026-04-05-tw-perf-rerun-clean/data/gaussian/`. For this experiment, we generate a fresh n=10K Gaussian dataset with seed=42 for controlled conditions (X ~ N(0,1) in R^50, Y from UMAP with default params via sklearn's trustworthiness as T_exact baseline).

### Data Manifest

| Dataset | Source Path | Shape | Dtype | Approx Size | Source Type | Verification |
|---------|-------------|-------|-------|-------------|-------------|--------------|
| merfish_n10k_x | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | (10000, 50) | float64 | 3.9 MB | gitignored fixture | Shape/dtype assertion + SHA-256 logged at runtime |
| merfish_n10k_y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | (10000, 2) | float64 | 157 KB | gitignored fixture | Shape/dtype assertion + SHA-256 logged at runtime |
| merfish_n50k_x | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | (50000, 50) | float64 | 20 MB | gitignored fixture | Shape/dtype assertion + SHA-256 logged at runtime |
| merfish_n50k_y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | (50000, 2) | float64 | 782 KB | gitignored fixture | Shape/dtype assertion + SHA-256 logged at runtime |
| gaussian_n10k_x | Generated by `gen_gaussian.py` (seed=42) | (10000, 50) | float64 | ~3.9 MB | generated | Regenerated deterministically; shape/dtype assertion |
| gaussian_n10k_y | Generated by `gen_gaussian.py` (seed=42) | (10000, 2) | float64 | ~157 KB | generated | Regenerated deterministically; shape/dtype assertion |

All MERFISH fixtures are verified present on disk. Symlinks will be created in the experiment's `data/merfish/` directory pointing to the originals (same filesystem, symlinks confirmed functional).

### Data Verification Script

`scripts/verify_inputs.py` will:
1. Assert shape and dtype of every input file
2. Compute and log SHA-256 checksums for MERFISH fixtures (logged to `results/data_checksums.json`)
3. Verify Gaussian data regeneration is deterministic (generate twice, compare)

## Experiment Directory Layout

```
research/2026-04-09-tw-subsample-tradeoff/
├── environment.yml                    # Micromamba env specification
├── experiment-plan.md                 # This plan (copied from .autoskillit/temp/)
├── scripts/
│   ├── verify_inputs.py               # Data integrity checks + SHA-256 logging
│   ├── gen_gaussian.py                # Generate Gaussian n=10K X,Y with seed=42
│   ├── compute_exact.py               # Compute T_exact for all datasets, save baselines
│   ├── run_subsample_sweep.py         # Main experiment: sweep (approach, m, seed)
│   ├── analyze_results.py             # Aggregate, compute stats, generate tables
│   └── plot_results.py                # Generate error/speed plots (optional)
├── data/
│   ├── merfish/                        # Symlinks to MERFISH fixtures
│   │   ├── merfish_n10k_x.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
│   │   ├── merfish_n10k_y.npy -> ...
│   │   ├── merfish_n50k_x.npy -> ...
│   │   └── merfish_n50k_y.npy -> ...
│   └── gaussian/                       # Generated by gen_gaussian.py
│       ├── gaussian_n10k_x.npy
│       └── gaussian_n10k_y.npy
├── results/
│   ├── data_checksums.json             # SHA-256 of all input files
│   ├── exact_baselines.json            # T_exact for each dataset
│   ├── sweep_results.json              # Raw per-(approach, dataset, m, seed) results
│   ├── analysis/
│   │   ├── summary_table.md            # Primary results table
│   │   ├── variance_scaling.json       # CLT slope fits
│   │   └── analysis_report.md          # Full analysis with hypothesis verdicts
│   └── plots/                          # PNG figures (optional)
│       ├── error_vs_m.png
│       ├── speed_vs_m.png
│       └── std_vs_m_loglog.png
└── report.md                           # Final report (written by write-report skill)
```

### Script Descriptions

**`verify_inputs.py`**: Loads each data file, asserts shape/dtype, computes SHA-256 hash, writes `results/data_checksums.json`. Exits non-zero if any assertion fails. Addresses R4.

**`gen_gaussian.py`**: Generates X ~ N(0,1) shape (10000, 50) and Y via sklearn UMAP-like random projection (since we only need T_exact as a reference, Y is a random 2D projection with seed=42). Saves to `data/gaussian/`. Deterministic regeneration verified by `verify_inputs.py`.

**`compute_exact.py`**: Loads each dataset, computes `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` with the symmetric timing protocol (1 warm-up + median of 3). Saves `results/exact_baselines.json` with structure: `{dataset: {T_exact: float, time_s: float, n: int, k: int}}`.

**`run_subsample_sweep.py`**: The main experiment script. For each (approach, dataset, m, seed) cell:

- **Approach B** (subset embedding): `sklearn.manifold.trustworthiness(X[idx], Y[idx], n_neighbors=min(k, m-1))`. Direct sklearn call — no custom code, no normalization risk.
- **Approach A** (row sub-sampling): Custom implementation that iterates over m query rows, computes distances to all n points in both X and Y spaces, finds k-NN in each space using the full population, and applies penalty with denominator `m·k·(2n−3k−1)`. Includes safeguards:
  - At m=n: assert |T_A - T_exact| < 1e-12 (convergence check, R3/RT6)
  - At small m where min(k, m-1) < k: flag as k-clamped (W2)

Timing: 1 warm-up call + median of 3 timed calls per cell (W1, W6). Thread control: `OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1`.

Output: `results/sweep_results.json` — array of records: `{approach, dataset, n, m, seed, T_sub, abs_delta_T, time_s, k_clamped}`.

**`analyze_results.py`**: Reads sweep results and exact baselines. Computes per-cell aggregates (mean, max, std of |ΔT|, mean time, speedup ratio). Fits CLT slope (log-log regression of std vs. m). Generates:
- `results/analysis/summary_table.md`: Primary results table
- `results/analysis/variance_scaling.json`: CLT slope per (approach, dataset)
- `results/analysis/analysis_report.md`: Full analysis with hypothesis verdicts

Primary cell evaluation uses **only** the pre-selected cells (R2/RT8):
- Approach A, MERFISH n=10K, m=2000
- Approach B, MERFISH n=10K, m=5000

**`plot_results.py`**: Optional visualization. Generates error-vs-m, speed-vs-m, and std-vs-m (log-log) plots. Uses `matplotlib.use("Agg")` (non-interactive). Skipped gracefully if matplotlib unavailable.

## Environment

**Custom environment required** (W7):

The experiment requires Python with numpy, scikit-learn, and matplotlib. While these are available in the system Python, an isolated environment ensures reproducibility. Following the project convention of per-experiment environments.

```yaml
name: tw-subsample-tradeoff
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2.6
  - scikit-learn=1.8.0
  - matplotlib=3.10.3
```

**Rationale for each dependency:**
- `python=3.11`: Matches project convention across all research environments.
- `numpy=2.2.6`: Required for .npy file I/O and array operations. Pinned to exact version for reproducibility.
- `scikit-learn=1.8.0`: Provides `sklearn.manifold.trustworthiness` (ground truth) and `sklearn.neighbors.NearestNeighbors` (for Approach A's full-population KNN). Pinned to exact version.
- `matplotlib=3.10.3`: Optional plotting. Pinned for reproducibility of figures.

**Acknowledged limitation (W7):** Transitive dependencies (joblib, threadpoolctl, BLAS/LAPACK backend) are not pinned. Timing results may not reproduce exactly across machines. This is acceptable because speedup is descriptive only (R1).

## Implementation Phases

### Phase 1: Directory Structure and Environment
- Create `research/2026-04-09-tw-subsample-tradeoff/` and all subdirectories (`scripts/`, `data/merfish/`, `data/gaussian/`, `results/`, `results/analysis/`, `results/plots/`)
- Create `environment.yml` with the specification above
- Create symlinks in `data/merfish/` pointing to MERFISH fixtures at `research/2026-04-05-tw-perf-rerun-clean/data/merfish/` (R5)
- Build the environment: `micromamba create -f environment.yml -y && micromamba activate tw-subsample-tradeoff`
- Verify: `python -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"`

### Phase 2: Data Generation and Verification
- Create `scripts/gen_gaussian.py`:
  - Generate X = `np.random.default_rng(42).standard_normal((10000, 50))`
  - Generate Y = random 2D linear projection of X (deterministic with seed=42) — sufficient for trustworthiness baseline since we only need a fixed Y
  - Save to `data/gaussian/gaussian_n10k_x.npy` and `gaussian_n10k_y.npy`
- Create `scripts/verify_inputs.py`:
  - Load all 6 data files, assert shapes and dtypes
  - Compute SHA-256 checksums for MERFISH files
  - Verify Gaussian regeneration determinism
  - Write `results/data_checksums.json`
- Run: `python scripts/gen_gaussian.py && python scripts/verify_inputs.py`

### Phase 3: Exact Baselines
- Create `scripts/compute_exact.py`:
  - Load each dataset (MERFISH n=10K, MERFISH n=50K, Gaussian n=10K)
  - Compute `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)`
  - Timing: 1 warm-up + median of 3 timed calls (symmetric protocol)
  - Save `results/exact_baselines.json`
- Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/compute_exact.py`

### Phase 4: Main Experiment Script
- Create `scripts/run_subsample_sweep.py`:
  - Implement Approach B using sklearn directly
  - Implement Approach A with custom row-subsampled trustworthiness:
    - For each query point i in subsample S: compute pairwise distances to all n points in X and Y
    - Find k-NN in each space against full population
    - Compute penalty with denominator `m·k·(2n−3k−1)`
  - Add Approach A safeguards:
    - m=n convergence test (|T_A(m=n) - T_exact| < 1e-12) — run once at startup for MERFISH n=10K
    - k-clamping flag for m < k+1
  - Timing: 1 warm-up + median of 3 per cell
  - Output: `results/sweep_results.json`
  - CLI args: `--datasets`, `--approaches`, `--dry-run` (minimal m values for pipeline validation)
- Run dry-run first: `python scripts/run_subsample_sweep.py --dry-run`

### Phase 5: Analysis and Plotting
- Create `scripts/analyze_results.py`:
  - Read sweep results, compute per-cell aggregates
  - Evaluate primary cells against thresholds
  - Fit CLT slopes
  - Generate summary table, variance scaling JSON, analysis report
  - Segregate k-clamped results (W2)
  - Scope all recommendations to "MERFISH-like data, n ≤ 50K" (W5, R6)
- Create `scripts/plot_results.py`:
  - Error vs. m curves (both approaches, all datasets)
  - Speed vs. m curves
  - std vs. m log-log with fitted slope
- Run: `python scripts/analyze_results.py && python scripts/plot_results.py`

### Phase 6: Dry Run (End-to-End Pipeline Validation)
- Execute the full pipeline with `--dry-run` flag (m={1000, 2000, 5000} only, 3 seeds)
- Verify:
  - Approach A m=n convergence check passes
  - All output files are created with correct structure
  - |ΔT| values are in a plausible range (not ≈ 0.47 — the prior bug signature)
  - Timing data is collected
  - Analysis script produces summary table
- Fix any issues before proceeding to full run

## Execution Protocol

### Step 1 — Environment Setup
```bash
cd research/2026-04-09-tw-subsample-tradeoff
micromamba create -f environment.yml -y
micromamba activate tw-subsample-tradeoff
```

### Step 2 — Data Acquisition (R5)
```bash
# Symlinks already created in Phase 1; verify they resolve
python scripts/verify_inputs.py
# Generate Gaussian data
python scripts/gen_gaussian.py
# Re-verify all inputs including generated data
python scripts/verify_inputs.py
```

### Step 3 — Exact Baselines
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/compute_exact.py
# Inspect: cat results/exact_baselines.json
```

### Step 4 — Dry Run
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_subsample_sweep.py --dry-run
python scripts/analyze_results.py --dry-run
```

Verify:
- [ ] Approach A m=n convergence: |T_A - T_exact| < 1e-12
- [ ] No |ΔT| ≈ 0.47 (prior bug signature)
- [ ] Output JSON has correct schema
- [ ] Analysis produces summary table

### Step 5 — Full Experiment
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_subsample_sweep.py
```

Expected runtime: ~30–60 minutes for MERFISH n=50K cells (dominated by Approach A at large m, which requires m × n distance computations in Python).

### Step 6 — Analysis
```bash
python scripts/analyze_results.py
python scripts/plot_results.py
```

### Step 7 — Review Results
Inspect `results/analysis/analysis_report.md` for hypothesis verdicts. Verify primary cell results match expectations. Check CLT slope fit quality.

## Analysis Plan

### Primary Analysis (Confirmatory)

Evaluate the two pre-selected primary cells:

1. **Approach A, MERFISH n=10K, m=2000**: Does mean(|ΔT|) < 0.01 AND max(|ΔT|) < 0.02 AND std(T_sub) < 0.005?
2. **Approach B, MERFISH n=10K, m=5000**: Same criteria.

If both pass: H1 supported — sub-sampling is viable for MERFISH-like data at these m values.
If one passes and one fails: partial support — the passing approach is recommended.
If both fail: H0 not rejected — sub-sampling at these m values is insufficient.

### Secondary Analyses (Exploratory)

**Error scaling curves**: Plot mean(|ΔT|) vs. m for each (approach, dataset). Identify the minimum m where all three criteria are met ("crossover point").

**Variance scaling (H2)**: For each (approach, dataset), fit log(std) = a + b·log(m). If b ∈ [-0.7, -0.3], CLT model is supported and can be used for extrapolation. If b ∉ [-0.7, -0.3], the variance model is rejected and extrapolation is flagged as unreliable (W8).

**MERFISH vs. Gaussian (H3)**: Compare std(T_sub) at matched m between MERFISH n=10K and Gaussian n=10K. Higher std on MERFISH supports H3.

**Approach comparison (H4)**: At matched m, compare mean(|ΔT|) between A and B. Note: A estimates full-n T (unbiased); B estimates subset T (different quantity, RT1). The comparison is informative but the two approaches answer different questions.

**n=50K scaling (H5)**: Compare error profiles at MERFISH n=10K and n=50K at matched m/n ratios (e.g., m=1000 at n=10K vs. m=5000 at n=50K, both 10% subsample).

**Extrapolation (H6 — bounded)**: If CLT model holds, extrapolate std to n=100K. Report with explicit uncertainty: "projected std at n=100K, m=5000 is X ± Y, assuming CLT scaling holds beyond the measured range." Do not present as validated (R6).

### k-Clamped Results (W2)

Results where m < k+1 (m < 16 for k=15) trigger k-clamping in Approach B: `n_neighbors = min(k, m-1)`. These cells are:
- Not applicable for any m in our sweep (minimum m=250 >> k=15)

If k-clamping were triggered, those results would be reported in a separate table and excluded from the primary analysis.

### Approach B Structural Bias (W3)

Approach B computes trustworthiness within the m-point subset. The effective problem is "easier" because the k-NN structure in a subset can differ from the full population. This means:
- |ΔT_B| may understate the actual divergence from full-n trustworthiness
- A low |ΔT_B| at small m may reflect reduced problem difficulty, not sub-sampling accuracy

The analysis report will note this structural difference and recommend Approach A when the goal is estimating full-n trustworthiness.

## Success Criteria

- **Conclusive positive (H1 supported):** At least one pre-selected primary cell satisfies ALL of: mean(|ΔT|) < 0.01, max(|ΔT|) < 0.02, std(T_sub) < 0.005. This supports adding `trustworthiness_subsampled` to `src/metrics.rs` with the passing approach and m as defaults, scoped to MERFISH-like data at n ≤ 50K.

- **Conclusive negative (H0 not rejected):** Both pre-selected primary cells fail at least one criterion. Sub-sampling at these m values does not provide acceptable accuracy on MERFISH data. Larger m values or alternative approaches needed.

- **Inconclusive:** Results are near-threshold (e.g., mean(|ΔT|) ∈ [0.008, 0.012]) or CLT slope is outside [-0.7, -0.3], making extrapolation unreliable. Recommend follow-up with more seeds or larger n.

**Speedup** is reported descriptively (R1): expected ~5× for Approach A at m=2000 (n=10K) and ~4× for Approach B at m=5000 (n=10K), based on O(m·n) vs O(n^2) and O(m^2) vs O(n^2) scaling respectively. These are informative but do not gate the accept/reject decision.

## Threats to Validity

### Internal

1. **Approach A custom implementation correctness**: The custom Python implementation of row-subsampled trustworthiness could contain bugs. **Mitigation**: m=n convergence check must produce |ΔT| < 1e-12. Additional sanity check at m=n/2 with 50 seeds to verify mean converges to T_exact.

2. **sklearn version sensitivity**: Different sklearn versions may produce slightly different trustworthiness values due to floating-point ordering. **Mitigation**: Pinned to 1.8.0. Exact baselines computed in the same environment.

3. **BLAS backend variance**: Unpinned BLAS backend could affect floating-point results across machines. **Mitigation**: Acceptable because |ΔT| threshold (0.01) is orders of magnitude above BLAS-induced variance (~1e-15).

4. **Seed selection bias**: Seeds 0–9 are arbitrary. **Mitigation**: 10 seeds is sufficient for std estimation; results would be similar with any 10 seeds given CLT.

### External

1. **Single real dataset (W5)**: MERFISH (gene expression, d=50, clustered manifold) may not represent other data modalities (images, text embeddings, graphs). **Mitigation**: All recommendations explicitly scoped to "MERFISH-like data."

2. **Python vs. Rust performance (R6)**: Python timing does not predict Rust performance. Speedup ratios in Python reflect algorithmic scaling only, not production wall-clock. **Mitigation**: Speedup is descriptive; Rust implementation benchmarks are a separate follow-up.

3. **n ≤ 50K only (R6)**: The primary use case (n ≥ 100K) is not directly tested. **Mitigation**: Recommendation bounded to n ≤ 50K. Extrapolation reported with explicit uncertainty.

4. **k=15 only**: Results may not generalize to other k values. **Mitigation**: k=15 is the UMAP default and the most common use case.

## Estimated Resource Requirements

- **Compute time**: ~30–60 minutes for the full sweep (dominated by Approach A on MERFISH n=50K at large m values, which requires m×50K distance computations in Python). Dry run: ~2–5 minutes.
- **Disk space**: ~50 MB for data (mostly symlinks) + ~5 MB for results and plots.
- **Dependencies**: Python 3.11, numpy, scikit-learn, matplotlib (all available via micromamba/conda-forge).
- **No GPU required**: All computation is CPU-only.
- **No Rust compilation required**: This is a Python-only experiment.

## Appendix A: Approach A Denominator Derivation (R3)

The standard trustworthiness normalization constant derives from the maximum possible penalty sum.

**Standard formula** (Venna & Kaski 2006):
```
T(k) = 1 − (2 / (n·k·(2n−3k−1))) · Σ_{i=1}^{n} Σ_{j ∈ U_i(k)} (r(i,j) − k)
```

**Maximum penalty per query point i**: In the worst case, all k Y-neighbors of point i are the k points ranked lowest in X-space (ranks n−k through n−1, 1-indexed). The penalty for point i is:
```
Σ_{l=0}^{k-1} (n − 2k + l) = k·(n − 2k) + k·(k−1)/2 = k·(2n − 3k − 1) / 2
```

**For n query points**: max total penalty = `n · k·(2n−3k−1)/2`. The normalization is `2 · max_penalty = n·k·(2n−3k−1)`.

**For Approach A (m query points, full-n ranks)**: The per-query-point worst case is identical because ranks are still computed against all n points. Only the number of query points changes from n to m:
```
max_penalty_A = m · k·(2n−3k−1)/2
normalization_A = 2 · max_penalty_A = m·k·(2n−3k−1)
```

**Convergence check**: At m = n, `m·k·(2n−3k−1) = n·k·(2n−3k−1)`, which is exactly the standard normalization. T_A(m=n) = T_exact by construction.

This derivation confirms the denominator is mathematically correct. The experiment includes an empirical convergence check (m=n, |ΔT| < 1e-12) as a runtime safeguard.
