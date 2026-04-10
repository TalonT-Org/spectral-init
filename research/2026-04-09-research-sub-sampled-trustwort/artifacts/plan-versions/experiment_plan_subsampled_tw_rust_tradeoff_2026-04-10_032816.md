# Experiment Plan: Sub-Sampled Trustworthiness Error/Speed Trade-off (Rust Validation)

## Motivation

The Python sub-sampling research (PR #260) demonstrated that computing trustworthiness on a random subset of m=2000 rows yields mean|ΔT| = 0.00165 with a 4.1x speedup at n=10K. This experiment validates that finding in the Rust implementation, which uses fundamentally different low-level machinery (AVX2+FMA SIMD kernels, Rayon work-stealing parallelism, introselect k-NN) that could in principle produce different numerical behavior. The results will inform whether the m=2000 default recommendation from Python carries over to Rust, and quantify the Rust-specific speedup curve. This is explicitly a **confirmatory replication study** — it confirms Python-established thresholds on the same dataset class, not a discovery of new thresholds.

## Hypothesis

**Null hypothesis (H0):** The Rust sub-sampled trustworthiness at m=2000 has mean|ΔT| >= 0.01 compared to the Rust exact trustworthiness, OR the speedup does not scale linearly with the reduction factor n/m.

**Alternative hypothesis (H1):** The Rust sub-sampled trustworthiness at m=2000 has mean|ΔT| < 0.01, and the speedup scales approximately linearly with n/m, confirming the Python findings carry over to the Rust implementation.

### Individual Hypotheses

**H1 — Accuracy at m=2000 (n=10K):**
At m=2000, mean|ΔT| < 0.01 across 10 seeds on MERFISH n=10K.
- *Test:* One-sample one-sided t-test. H0: μ(|ΔT|) >= 0.01, H1: μ(|ΔT|) < 0.01.
- *Alpha:* 0.01 (pre-specified).
- *Decision rule:* Reject H0 if the upper bound of the 99% one-sided confidence interval on mean|ΔT| is < 0.01.
- *Additionally report:* max|ΔT| across all seeds (worst-case tail risk).

**H2 — Linear Speedup:**
Wall-clock speedup ratio (exact_time / sub-sampled_time) is approximately linear in n/m.
- *Test:* OLS regression of speedup vs n/m over 7 m-values. Bootstrap R² (1000 resamples).
- *Alpha:* 0.05.
- *Decision rule:* PASS if the lower bound of the 95% bootstrap CI on R² exceeds 0.90.

**H3 — Variance Decay:**
Standard deviation of T_sampled across seeds decays as O(1/sqrt(m)) or faster.
- *Test:* OLS on log(std(T)) ~ β·log(m) over 7 m-values, each with 10 seeds.
- *Alpha:* 0.05.
- *Decision rule:* PASS if OLS slope estimate β <= -0.3 AND one-sided p-value for H0: β >= -0.3 is < 0.05.

**H4 — Cross-Validation Against Python (Conditional):**
Rust speedup ratio at overlapping (n, m) points matches Python speedup ratio within 2x.
- *Precondition:* Python reference JSON files from PR #260 are present in the data directory.
- *If absent:* H4 verdict is SKIPPED (not FAIL). This is pre-specified.
- *Comparability criteria:* Results are comparable if both use the same dataset (MERFISH), same k=15, and same m values. If the Python runs used a fundamentally different machine class (no AVX2), the speed comparison is NOT_COMPARABLE and should be noted, but accuracy overlap remains valid.
- *Decision rule:* For each overlapping (n, m), |rust_speedup - python_speedup| / python_speedup < 1.0 (within 2x).

**H5 — Large-Scale Accuracy (n=50K, Exploratory):**
At n=50K, m=2000, mean|ΔT| < 0.01 across 10 seeds.
- *Test:* Same one-sided t-test as H1. Alpha = 0.01.
- *Framing:* This is exploratory — the 0.01 threshold was established at n=10K and is extrapolated to n=50K. Python could not test this due to O(n²) memory. The Rust implementation has O(n) memory, making this the first validation at this scale.

**H6 — Normalization Sanity (Deterministic):**
T_sub(m=n) = T_exact within 1e-10.
- *No statistical test needed.* This is a deterministic check of the normalization formula.
- *What is being validated:* That the denominator correction `m * k * (2n - 3k - 1)` reduces to `n * k * (2n - 3k - 1)` when m=n, producing identical output. This is a mathematical identity AND an implementation correctness check.
- *Independence enhancement:* Additionally compare T_exact against the Python-computed T_exact = 0.5362038060873342 (from PR #260 results) if the reference file is present. This provides an independent normalization validation. Report the delta and whether it is < 1e-6.

### Multiple Testing Acknowledgment

Six hypotheses are tested with per-test alpha levels (0.01 for H1/H5; 0.05 for H2/H3/H4; deterministic for H6). Family-wise error rate (FWER) correction is **intentionally omitted** because:
1. This is a confirmatory replication of Python findings, not an exploratory study.
2. The hypotheses address qualitatively distinct properties (accuracy, scaling, variance, cross-language parity, large-n, normalization) — they are not multiple comparisons within a single dimension.
3. H6 is deterministic (no Type I error possible). H4 is conditional on data availability.
4. The nominal FWER at uncorrected per-test rates is bounded at ~15% across the 4 stochastic tests, which is acceptable for a confirmatory design where prior evidence strongly favors all hypotheses passing.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Subsample size m | 250, 500, 1000, 2000, 4000, 7500, n (7 levels) | Geometric spread from very small to full population. 7 points provides adequate resolution for regression (H2, H3). m=2000 is the recommended default. m=n is the normalization sanity anchor (H6). |
| Random seed | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 seeds) | 10 seeds provides ~90% power to detect true mean|ΔT| of 0.005 at α=0.01 given expected σ ≈ 0.002. |
| Dataset size n | 10000, 50000 (2 levels) | n=10K enables cross-validation against Python (PR #260). n=50K extends to a scale Python could not test. |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| T_exact | dimensionless [0,1] | `spectral_init::trustworthiness(x, y, k)` | `trustworthiness` (EXISTS in src/metrics.rs) |
| T_sub | dimensionless [0,1] | Sub-sampled variant with query_idx | NEW — computed in experiment binary |
| abs_delta_T | dimensionless | `(T_exact - T_sub).abs()` | NEW — derived metric, no threshold in src/metrics.rs |
| wall_time_ms | milliseconds | `std::time::Instant` elapsed, median of 5 reps | NEW — collected by experiment binary |
| speedup_ratio | dimensionless | exact_median_ms / sub_median_ms | NEW — derived from timing |
| std_T_sub | dimensionless | std dev of T_sub across 10 seeds at fixed m | NEW — computed in analysis |

For metrics marked "NEW": these are experiment-specific derived metrics that do not require addition to `src/metrics.rs` (they are not eigensolver quality thresholds). The acceptance threshold |ΔT| < 0.01 is justified as: at T ≈ 0.54 (MERFISH), a 0.01 absolute error represents < 2% relative error, which is negligible for the purpose of comparing embeddings during UMAP optimization. This threshold was also used in the Python study.

**Secondary tightened threshold:** In addition to the primary 0.01 threshold, report whether mean|ΔT| < 0.003 (approximately 2x the Python observed value of 0.00165). This addresses the concern that the 6x margin between observed and threshold could mask implementation flaws.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighborhood size) | 15 | Matches all prior research. Note: k=15 is the ONLY validated configuration. The m=2000 recommendation should be explicitly scoped to k=15. |
| d_x (input dimension) | 50 | MERFISH dataset dimensionality |
| d_y (embedding dimension) | 2 | Standard UMAP 2D embedding |
| Dataset | MERFISH (Yao et al. 2023) | Matches prior research chain. Note: conclusions are scoped to "scRNA-seq-class high-dimensional data on AVX2 x86 hardware" |
| Rust toolchain | stable (current) | Production-representative |
| Rayon thread pool | default (num_cpus) | Production configuration |
| Build profile | --release | Production-representative |
| SIMD path | AVX2+FMA (automatic) | Hardware-determined; no scalar fallback tested |

## Inputs and Data

### Data Acquisition Strategy

All MERFISH fixture files reside in a prior research directory and are NOT committed to git. They will be absent in a fresh worktree. The experiment uses **symlinks** to reference shared data (established project pattern).

| Dataset | Source Path | Properties | Purpose |
|---------|-------------|------------|---------|
| MERFISH n=10K X | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | f64, shape (10000, 50) | Input high-dimensional data |
| MERFISH n=10K Y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | f64, shape (10000, 2) | Input 2D embedding |
| MERFISH n=50K X | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | f64, shape (50000, 50) | Large-scale input |
| MERFISH n=50K Y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | f64, shape (50000, 2) | Large-scale embedding |
| Python reference JSON | `research/2026-04-09-subsampled-tw-tradeoff/results/raw/*.json` | JSON, per-trial results | H4 cross-validation |

**Pre-flight verification (Phase 1 gate):** Before any experiment execution, a verification script must confirm:
1. All 4 `.npy` symlink targets exist and are readable
2. Each `.npy` file has the expected shape (load and check `x.shape`)
3. Python reference files are present (for H4) or explicitly flagged as absent

**Reconstitution if absent:** If MERFISH `.npy` files are missing, they can be regenerated via:
```bash
cd research/2026-04-05-tw-perf-rerun-clean
python3 scripts/prepare_merfish.py --npz-dir ../../temp/merfish_100k --n 10000
python3 scripts/prepare_merfish.py --npz-dir ../../temp/merfish_100k --n 50000
```
This requires manually downloading the Allen Brain Cell Atlas MERFISH dataset (Yao et al. 2023) into `temp/merfish_100k/`. The download is manual; no automated script exists.

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-10-subsampled-tw-rust-tradeoff/
├── experiment-plan.md              # This plan (copied from .autoskillit/temp/)
├── scripts/
│   ├── run_experiment.sh           # Master orchestrator: build, preflight, execute, analyze
│   ├── preflight_check.py          # Validates data presence and shapes
│   ├── analyze_results.py          # Processes JSON, computes stats, writes verdicts
│   └── plot_supplementary.py       # Optional PNG plots (not gated on)
├── data/
│   ├── .gitkeep
│   ├── merfish_n10k_x.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
│   ├── merfish_n10k_y.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy
│   ├── merfish_n50k_x.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy
│   ├── merfish_n50k_y.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy
│   └── python_ref/ -> ../../2026-04-09-subsampled-tw-tradeoff/results/raw/
├── results/
│   ├── raw/                        # Per-trial JSON files from Rust binary
│   ├── analysis/
│   │   ├── verdicts.json           # Machine-readable hypothesis verdicts
│   │   ├── summary.md              # Human-readable summary with tables
│   │   └── *.png                   # Supplementary plots
│   └── rayon_nondeterminism/       # Duplicate-run delta measurements
└── report.md                       # Final report (written by write-report skill)
```

### File Descriptions

**`src/bin/tw_subsample_experiment.rs`** — Rust experiment binary added to the main project (behind `cli` feature). This is the core measurement tool. It:
- Loads `.npy` fixtures via `ndarray_npy::read_npy`
- Computes exact T via `spectral_init::trustworthiness(x, y, k)`
- Implements sub-sampled T by generating `query_idx` via `StdRng::seed_from_u64(seed)`, iterating over the selected rows using the same inner pipeline (AVX2 x_dist, introselect x_sort, 2D AVX2 y_dist, rank-counting penalty) with normalization `m * k * (2n - 3k - 1)`
- Runs 5 timed repetitions per trial, reports median wall-clock
- Outputs one JSON object per trial to stdout

**`scripts/run_experiment.sh`** — Master orchestrator:
1. Runs preflight check
2. Builds the Rust binary (`cargo build --release --features cli`)
3. Generates randomized trial order
4. Executes all trials, writing JSON to `results/raw/`
5. Runs Rayon non-determinism measurement
6. Runs analysis script
7. Exits with code reflecting verdict status

**`scripts/preflight_check.py`** — Validates:
- All data symlinks resolve to readable files
- Each `.npy` has expected shape
- Python reference directory presence (H4 conditional flag)
- Reports READY or lists missing items

**`scripts/analyze_results.py`** — The primary analysis tool:
- Loads all JSON from `results/raw/`
- Computes per-hypothesis statistics and verdicts
- Writes `results/analysis/verdicts.json` (machine-readable)
- Writes `results/analysis/summary.md` (human-readable)
- Optionally generates supplementary plots

**`results/analysis/verdicts.json`** — Machine-readable verdict file:
```json
{
  "experiment": "subsampled-tw-rust-tradeoff",
  "timestamp": "2026-04-10T...",
  "hypotheses": {
    "H1": {
      "verdict": "PASS",
      "mean_abs_delta_T": 0.00170,
      "max_abs_delta_T": 0.00350,
      "ci_upper_99": 0.00250,
      "t_statistic": -12.5,
      "p_value": 1.2e-7,
      "n_seeds": 10,
      "secondary_threshold_0.003": true
    },
    "H2": { "verdict": "PASS", "R2": 0.987, "R2_ci_lower_95": 0.942 },
    "H3": { "verdict": "PASS", "slope": -0.62, "slope_se": 0.08, "p_value": 0.001 },
    "H4": { "verdict": "PASS|SKIPPED|NOT_COMPARABLE", "..." : "..." },
    "H5": { "verdict": "PASS", "..." : "..." },
    "H6": { "verdict": "PASS", "delta_T_exact": 1.2e-15, "delta_vs_python": 3.4e-7 }
  },
  "rayon_nondeterminism": {
    "max_delta_T_same_seed": 1.5e-14,
    "flagged": false
  },
  "overall": "PASS"
}
```

## Environment

**No custom environment needed.**

The project's existing toolchain is sufficient:
- **Rust:** The experiment binary uses existing Cargo dependencies (`ndarray-npy`, `serde_json`, `rayon`, `rand`) all available under the `cli` feature flag. Build with `cargo build --release --features cli`.
- **Python:** System Python 3.13.2 (micromamba-managed) already has numpy 2.2.6, scipy 1.15.2, and matplotlib 3.10.3. These cover all analysis requirements: JSON parsing (stdlib), statistics (scipy.stats), linear regression (scipy.stats.linregress), bootstrap CIs (scipy.stats.bootstrap), and supplementary plots (matplotlib).

No `environment.yml` will be created for this experiment. The prior experiment's `environment.yml` at `research/2026-04-09-subsampled-tw-tradeoff/environment.yml` documents the Python package versions for reproducibility reference.

## Implementation Phases

### Phase 1: Directory Structure, Data Symlinks, and Pre-flight

**Files to create:**
- `research/2026-04-10-subsampled-tw-rust-tradeoff/` and all subdirectories
- Symlinks in `data/` pointing to MERFISH fixtures and Python reference results
- `data/.gitkeep`
- `scripts/preflight_check.py`

**Acceptance criteria (machine-verifiable):**
- All 4 MERFISH symlinks resolve: `test -f data/merfish_n10k_x.npy && test -f data/merfish_n10k_y.npy && test -f data/merfish_n50k_x.npy && test -f data/merfish_n50k_y.npy`
- Python reference symlink resolves OR is flagged as absent
- `python3 scripts/preflight_check.py` exits 0 and prints "READY"

### Phase 2: Rust Experiment Binary

**Files to create:**
- `src/bin/tw_subsample_experiment.rs` — the core measurement binary

**Implementation details:**
The binary implements the sub-sampled trustworthiness loop. It does NOT modify `src/metrics.rs`. The sub-sampling logic is self-contained in the binary:

```
fn trustworthiness_subsampled(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize, query_idx: &[usize]) -> f64
```

Key implementation points:
- Generate `query_idx` by sampling m indices without replacement from 0..n using `rand::seq::index::sample(rng, n, m)`
- Iterate `query_idx.into_par_iter()` instead of `(0..n).into_par_iter()`
- Use the SAME inner pipeline as `trustworthiness()` in `src/metrics.rs`:
  - X-distances: `dist_sq_avx2_looped` for d >= 10, scalar fallback otherwise
  - X-kNN: `select_nth_unstable_by` (introselect)
  - Y-distances: `dist_sq_2d_avx2_batch` for d_y == 2
  - Penalty: rank-counting with self-exclusion via `dist_y[i] = INFINITY`
  - Thread-local buffers for scratch allocations
- Normalization: `m * k * (2 * n - 3 * k - 1)` where n = x.nrows(), m = query_idx.len()
- The binary copies the inner loop logic from `src/metrics.rs` rather than modifying the library. This keeps research artifacts separate from production code.

**CLI interface:**
```bash
tw_subsample_experiment --data-dir <path> --n <10000|50000> --k 15 \
    --m <subsample_size> --seed <seed> --reps 5 --mode <subsample|exact|sanity>
```

Outputs one JSON line to stdout per invocation:
```json
{
  "n": 10000, "m": 2000, "k": 15, "seed": 0, "reps": 5,
  "T_sub": 0.5345, "T_exact": 0.5362,
  "abs_delta_T": 0.0017,
  "wall_sub_median_ms": 720.3,
  "wall_sub_times_ms": [725.1, 720.3, 718.9, 722.0, 719.5],
  "wall_exact_median_ms": 3610.0,
  "wall_exact_times_ms": [3615.2, 3610.0, 3608.5, 3612.1, 3609.8]
}
```

**Acceptance criteria:**
- `cargo build --release --features cli --bin tw_subsample_experiment` succeeds
- Dry run with `--n 10000 --m 500 --seed 0 --reps 1 --mode subsample` produces valid JSON
- `--mode sanity` with m=n produces |T_sub - T_exact| < 1e-10

**Cargo.toml addition:**
```toml
[[bin]]
name = "tw_subsample_experiment"
path = "src/bin/tw_subsample_experiment.rs"
required-features = ["cli"]
```

### Phase 3: Experiment Scripts

**Files to create:**
- `scripts/run_experiment.sh` — master orchestrator
- `scripts/analyze_results.py` — statistical analysis and verdict generation
- `scripts/plot_supplementary.py` — optional visualization

**`run_experiment.sh` execution protocol:**

1. Run preflight check: `python3 scripts/preflight_check.py`
2. Build: `cargo build --release --features cli --bin tw_subsample_experiment`
3. **Exact baseline (both n):**
   - Run exact trustworthiness for n=10K and n=50K with 5 timed repetitions
4. **Rayon non-determinism measurement:**
   - For 3 representative (m, seed) configs: run each TWICE identically
   - Record max|T_run1 - T_run2| across all configs
   - If max delta > 1e-6, flag in results
5. **Sub-sampled trials (randomized order):**
   - Generate randomized trial order across all (n, m, seed) cells
   - For n=10K: m ∈ {250, 500, 1000, 2000, 4000, 7500, 10000}, seeds 0-9
   - For n=50K: m ∈ {250, 500, 1000, 2000, 4000, 7500, 50000}, seeds 0-9
   - Randomize the order of trials within each n-block (randomized block design)
   - Each trial: 5 timed reps, median reported
6. **Analysis:** Run `python3 scripts/analyze_results.py`

**`analyze_results.py` verdict logic:**

For each hypothesis:
- **H1:** Load all n=10K, m=2000 trials. Compute mean|ΔT|, max|ΔT|, one-sided t-test against μ=0.01, 99% CI upper bound. PASS if CI upper < 0.01. Report secondary threshold (mean|ΔT| < 0.003).
- **H2:** For each n, fit OLS: speedup ~ n/m. Bootstrap R² (1000 resamples). PASS if R²_CI_lower_95 > 0.90.
- **H3:** For each n, compute std(T_sub) per m across 10 seeds. Fit log(std) ~ β·log(m). PASS if β <= -0.3 AND p < 0.05.
- **H4:** If Python reference present: compare speedup ratios at overlapping (n, m). PASS if within 2x. If absent: SKIPPED.
- **H5:** Same as H1 but on n=50K data. Labeled as exploratory.
- **H6:** Check |T_sub(m=n) - T_exact| < 1e-10. Additionally check |T_exact - 0.5362038060873342| < 1e-6 if Python ref present.

Write `results/analysis/verdicts.json` and `results/analysis/summary.md`.

**Acceptance criteria (machine-verifiable):**
- `scripts/run_experiment.sh` is executable
- `python3 scripts/analyze_results.py --help` runs without error
- Experiment directory structure matches the layout specification: `test -d results/raw && test -d results/analysis && test -d results/rayon_nondeterminism`

### Phase 4: Dry Run

Execute the full pipeline with minimal inputs to verify end-to-end correctness:

1. Run preflight check
2. Execute 3 trials only: (n=10K, m=2000, seed=0), (n=10K, m=10000, seed=0), (n=10K, m=500, seed=0)
3. Run analysis script on the 3-trial subset
4. Verify `verdicts.json` is produced with correct structure (all hypothesis keys present, values are the expected types)
5. Verify H6 sanity check passes (m=n trial)

**Acceptance criteria:**
- All 3 trials produce valid JSON
- Analysis script runs without error
- `verdicts.json` has all 6 hypothesis keys
- H6 verdict is PASS with delta < 1e-10

## Execution Protocol

After implementation is verified via dry run, execute the full experiment:

**Step 1 — Pre-flight:**
```bash
cd research/2026-04-10-subsampled-tw-rust-tradeoff
python3 scripts/preflight_check.py
```

**Step 2 — Build:**
```bash
cargo build --release --features cli --bin tw_subsample_experiment
```

**Step 3 — Exact baselines (run first, used by all subsequent analysis):**
```bash
# n=10K exact (5 reps)
./target/release/tw_subsample_experiment --data-dir data --n 10000 --k 15 --m 10000 --seed 0 --reps 5 --mode exact > results/raw/exact_n10000.json

# n=50K exact (5 reps)
./target/release/tw_subsample_experiment --data-dir data --n 50000 --k 15 --m 50000 --seed 0 --reps 5 --mode exact > results/raw/exact_n50000.json
```

**Step 4 — Rayon non-determinism measurement:**
```bash
for config in "10000 2000 0" "10000 500 5" "50000 2000 3"; do
  read n m s <<< "$config"
  for rep in 1 2; do
    ./target/release/tw_subsample_experiment --data-dir data --n $n --k 15 --m $m --seed $s --reps 1 --mode subsample \
      > results/rayon_nondeterminism/rayon_n${n}_m${m}_s${s}_rep${rep}.json
  done
done
```

**Step 5 — Sub-sampled trials (randomized order):**
The `run_experiment.sh` script generates a shuffled list of all (n, m, seed) trials and executes them sequentially:
```bash
# For n=10K: 7 m-values × 10 seeds = 70 trials
# For n=50K: 7 m-values × 10 seeds = 70 trials
# Total: 140 sub-sampled trials + 2 exact baselines + 6 Rayon-nondeterminism runs = 148 invocations
```

Each trial:
```bash
./target/release/tw_subsample_experiment --data-dir data --n $N --k 15 --m $M --seed $S --reps 5 --mode subsample \
  > results/raw/sub_n${N}_m${M}_s${S}.json
```

**Step 6 — Analysis:**
```bash
python3 scripts/analyze_results.py --results-dir results/raw --rayon-dir results/rayon_nondeterminism --python-ref data/python_ref --output-dir results/analysis
```

**Step 7 — Verdict:**
```bash
cat results/analysis/verdicts.json | python3 -c "import json,sys; v=json.load(sys.stdin); print(v['overall']); sys.exit(0 if v['overall']=='PASS' else 1)"
```

## Analysis Plan

### Primary Analysis

1. **Accuracy table:** For each (n, m), compute mean|ΔT|, max|ΔT|, std(|ΔT|) across 10 seeds. Report alongside Python values from PR #260 at overlapping points.

2. **Speed table:** For each (n, m), compute median wall-clock (ms) and speedup ratio. Estimate expected speedup as n/m; compute residuals.

3. **Statistical tests:**
   - H1/H5: One-sample t-test, report t, p, 99% CI upper bound
   - H2: OLS + bootstrap R² CI
   - H3: OLS on log-log, report slope ± SE, p-value
   - H4: Direct ratio comparison (if data present)
   - H6: Point comparison

4. **Rayon non-determinism bound:** max|T_run1 - T_run2| across duplicate runs. Report as a noise floor that is subtracted from the |ΔT| budget discussion.

### Interpretation Guide

- **All H1-H6 PASS:** The Python m=2000 recommendation is confirmed for Rust. Ship `trustworthiness_subsampled()` with m=2000 default, scoped to MERFISH-class scRNA-seq data (d~50) on AVX2 x86 hardware with k=15.
- **H1 PASS, H5 FAIL:** m=2000 works at n=10K but may need adjustment at larger scales. Investigate whether a larger m is needed for n >= 50K.
- **H2 FAIL:** Speedup is not linear — investigate Rayon overhead at small m or cache effects. The sub-sampling optimization may still be worthwhile at specific m values.
- **H1 FAIL:** The Rust implementation produces systematically different results from Python at m=2000. Investigate normalization, SIMD rounding, or introselect behavior before shipping.

### No Visual Inspection Gates

All hypothesis verdicts are determined by the machine-readable `verdicts.json` file. PNG plots are produced as supplementary documentation only and are never consulted for pass/fail decisions.

## Success Criteria

- **Conclusive positive:** All of H1, H2, H3, H5, H6 PASS. H4 PASS or SKIPPED. The `verdicts.json` file contains `"overall": "PASS"`. This confirms the Rust sub-sampled trustworthiness is ready to ship with m=2000 default (for MERFISH-class data, k=15, AVX2 x86).

- **Conclusive negative:** H1 or H6 FAIL. If H1 fails, the sub-sampling accuracy is insufficient at m=2000. If H6 fails, the normalization is broken. Either blocks shipping.

- **Inconclusive:** H5 fails but H1 passes (large-n behavior differs; need more investigation). H2 or H3 fail but H1 passes (accuracy is fine but the speedup model doesn't fit cleanly; sub-sampling may still be worth shipping without the neat theoretical story).

## Threats to Validity

### Internal

1. **Rayon floating-point non-determinism:** Work-stealing reorders parallel summation, introducing O(ε_machine) variation between identical runs. Mitigated by measuring this variation explicitly (Rayon non-determinism step) and bounding its contribution to |ΔT|.

2. **Thermal/cache confounds on timing:** Monotonic trial ordering could confound timing with m magnitude. Mitigated by randomized block design for trial order and 5-rep median aggregation with warmup discard.

3. **Code duplication divergence:** The experiment binary copies the inner loop from `src/metrics.rs` rather than calling it with modified indices. If the copy diverges from the source (e.g., missing a SIMD optimization), the experiment measures a different function than what would be shipped. Mitigated by: (a) the H6 sanity check verifying identical output at m=n, and (b) the implementer must base the code on a direct copy of the production inner loop.

4. **Single machine:** All timing measurements come from one hardware configuration. Speedup ratios are hardware-contingent.

### External

1. **Single dataset:** All results are from MERFISH (scRNA-seq, d=50). The m=2000 recommendation may not generalize to other data types, dimensionalities, or local structure patterns.

2. **Single k value:** Only k=15 is tested. Smaller k (e.g., k=5) may be more sensitive to sub-sampling omission; larger k (e.g., k=50) may be more robust. The recommendation must be scoped to k=15.

3. **Hardware class:** Results are scoped to AVX2 x86 with multi-core Rayon parallelism. Non-SIMD or low-core-count hardware may see different speedup ratios (though the accuracy results are hardware-independent since they depend only on the set of sampled indices and the metric computation, not on timing).

4. **Threshold leakage:** The 0.01 threshold was calibrated on MERFISH in the Python study and is confirmed on the same dataset here. This is acknowledged as a confirmatory study, not out-of-sample validation.

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| **n=10K trials** | 70 sub-sampled + 1 exact + 3 Rayon-dup. Each sub-sampled trial: ~(m/10000) × 3.6s × 5 reps ≈ varies. Exact: ~3.6s × 5 reps ≈ 18s. Total n=10K: ~15-20 minutes. |
| **n=50K trials** | 70 sub-sampled + 1 exact + 3 Rayon-dup. Exact: ~91s × 5 reps ≈ 7.5 min. Total n=50K: ~60-90 minutes. |
| **Total compute** | ~90-120 minutes wall-clock (sequential execution) |
| **Disk space** | < 100 MB (JSON results are small; .npy fixtures are shared via symlinks) |
| **Dependencies** | None beyond existing Cargo.toml + system Python |
| **Build time** | < 2 minutes (incremental after first build) |
