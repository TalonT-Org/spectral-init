# Experiment Plan: Sub-Sampled Trustworthiness Error/Speed Trade-off (Rust Validation)

## Motivation

The Python sub-sampling research (PR #260) established that computing trustworthiness on a random subset of m rows yields mean|ΔT| < 0.002 at m=2000 on MERFISH n=10K, with ~4x speedup. This experiment validates that finding in the Rust implementation, which uses a fundamentally different computational pipeline (AVX2+FMA SIMD kernels, Rayon parallelism, introselect k-NN) and has O(n) memory per thread (not O(n^2)). The results will determine whether to ship `trustworthiness_subsampled()` in the library with m=2000 as a recommended default for MERFISH-class data.

This is a **same-dataset confirmatory study** — m=2000 and the 0.01 threshold were selected on MERFISH data in PR #260. This experiment confirms that the Rust implementation reproduces those results under identical conditions. It does **not** constitute out-of-sample validation or claim generalization beyond MERFISH-class data (k=15, PCA-50 features, AVX2 x86_64).

## Hypothesis

**Top-level framing:** The individual hypotheses H1–H6 are evaluated independently. There is no composite null hypothesis — each hypothesis stands alone with its own pass/fail criterion. The study outcome is a structured verdict across all six.

### H1 — Accuracy at m=2000 (Confirmatory)
**H0:** Mean|ΔT| ≥ 0.01 at m=2000, n=10K across 10 seeds.
**H1:** Mean|ΔT| < 0.01 at m=2000, n=10K across 10 seeds.
**Test:** One-sample t-test, α=0.025 (one-sided), against threshold 0.01.
**PASS:** Upper 97.5% CI bound of mean|ΔT| < 0.01.
**FAIL:** Upper 97.5% CI bound of mean|ΔT| ≥ 0.01.

### H2 — Linear Speedup in m (Confirmatory)
**H0:** Speedup ratio is not linearly related to n/m (R² ≤ 0.90).
**H1:** Speedup ratio is approximately linear in n/m (R² > 0.90).
**Test:** OLS regression of speedup_ratio ~ n/m, fitted **per-stratum** (separate fits for n=10K and n=50K). Report both per-stratum R² values. The hypothesis passes if **both** per-stratum R² bootstrap 95% CI lower bounds exceed 0.90.
**PASS:** Both per-stratum R² CI lower bounds > 0.90.
**FAIL:** Either per-stratum R² CI lower bound ≤ 0.90.
**Linearity check:** Additionally compute residuals of a linear fit vs a log-linear fit. If the log-linear fit reduces RMSE by >20%, report "monotone non-linear" rather than "linear" even if R² > 0.90.

### H3 — Variance Decay (Confirmatory)
**H0:** Log-log slope of std(T) vs m is > -0.3 (variance decays too slowly).
**H1:** Log-log slope of std(T) vs m is ≤ -0.3 (variance decays at least as fast as CLT baseline).
**Test:** OLS on log(std(T)) ~ log(m) across 7 m-values with 10 seeds each. One-sided t-test on slope, α=0.025.
**PASS:** Slope estimate ≤ -0.3 AND p < 0.025.
**FAIL:** Slope estimate > -0.3 AND p < 0.025.
**INCONCLUSIVE:** p ≥ 0.025 (insufficient evidence to determine slope direction relative to threshold).

### H4 — Rust vs Python Speedup Ratio Parity (Conditional)
**H0:** Rust and Python speedup ratios at same (n, m) differ by more than 2x.
**H1:** Rust and Python speedup ratios at same (n, m) agree within 2x.
**Pre-condition:** Python reference timing data from PR #260 must be present as hardcoded constants. If absent, H4 is reported as `"NOT_EVALUATED"` with rationale.
**PASS:** For all overlapping (n, m) points, |log2(rust_speedup / python_speedup)| < 1.
**FAIL:** Any overlapping point has |log2(rust_speedup / python_speedup)| ≥ 1.

### H5 — n=50K Accuracy (Exploratory)
**H0:** Mean|ΔT| ≥ 0.01 at m=2000, n=50K across 10 seeds.
**H1:** Mean|ΔT| < 0.01 at m=2000, n=50K across 10 seeds.
**Test:** Same as H1 (one-sample t-test, α=0.025, one-sided).
**Status:** Exploratory — this tests the same construct as H1 at a larger population size that the Python study could not evaluate due to O(n²) memory. H5 is **not** included in the confirmatory family and its result does not affect the overall study verdict. It is reported separately as supplementary evidence.
**PASS/FAIL/INCONCLUSIVE:** Same decision rules as H1.

### H6 — Normalization Sanity (Confirmatory)
**H0:** |T_subsampled(m=n) - T_exact| ≥ 1e-10.
**H1:** |T_subsampled(m=n) - T_exact| < 1e-10.
**Test:** Direct comparison, no statistical test needed.
**PASS:** |delta| < 1e-10 for both n=10K and n=50K.
**FAIL:** |delta| ≥ 1e-10 for either.

### Multiple Testing Correction

The confirmatory family comprises H1, H2, H3, and H6 (4 tests). H4 is conditional and H5 is exploratory — neither is in the confirmatory family.

**FWER justification:** This is a pre-registered confirmatory replication of PR #260's findings. The four confirmatory hypotheses test qualitatively distinct constructs: accuracy (H1), scaling relationship (H2), variance behavior (H3), and implementation correctness (H6). H1 and H5 test the same construct (accuracy at m=2000) but H5 is excluded from the confirmatory family as exploratory.

**Pre-declared FWER ceiling:** With 4 confirmatory tests at α=0.025 each (one-sided), the worst-case FWER under independence is 1-(1-0.025)^4 ≈ 0.096. Under the observed correlation structure (H1/H2/H3 share data), the effective FWER is lower. We accept a FWER ceiling of 0.10 for this confirmatory replication.

**Bonferroni-adjusted alternative:** If the reviewer prefers formal correction, applying Bonferroni to the 4-test family yields α_adj = 0.025/4 = 0.00625 per test. Given the expected effect sizes (mean|ΔT| ≈ 0.002 vs threshold 0.01), power remains adequate under Bonferroni.

### Power Analysis

**H1:** With 10 seeds, expected σ ≈ 0.002 (from PR #260 Python study: std of |ΔT| at m=2000, n=10K was 0.00165), threshold μ₀=0.01, expected mean ≈ 0.002. Effect size d = (0.01 - 0.002)/0.002 = 4.0. Power > 0.999 at α=0.025 one-sided. Even at σ=0.005 (conservative 2.5x inflation), d=1.6, power > 0.95.

**H2:** Per-stratum OLS with 7 m-values (7 points per fit). Under the theoretical model (perfect linearity), R² ≈ 1.0. The threshold is R² > 0.90. With n=7 data points, even moderate noise (σ_residual up to 20% of signal range) yields R² > 0.90 for a truly linear relationship. The primary risk is not power but model misspecification (non-linearity). The linearity check (linear vs log-linear RMSE comparison) addresses this.

**H3:** Log-log OLS with 7 m-values. Expected slope ≈ -0.66 (from PR #260), threshold -0.3. With 7 points, residual σ must be characterized empirically. Under the Python study's observed log-log residual variance (R² = 0.998), the estimated slope SE is ~0.02, yielding z = (-0.66 - (-0.3))/0.02 = 18.0 — power effectively 1.0. Even with 10x residual variance inflation (R² ≈ 0.98), power remains > 0.99.

**H5:** Same structure as H1. At n=50K, σ may differ from n=10K (unknown — Python could not test). If σ scales as O(1/sqrt(n)), σ_50K ≈ 0.002 * sqrt(10K/50K) ≈ 0.0009, power > 0.999. If σ is invariant to n, power matches H1. Conservative scenario: σ_50K = 0.005, power > 0.95. H5 is exploratory so formal power is informational only.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| n (population size) | {10000, 50000} | MERFISH fixtures available; n=50K tests scaling beyond Python study's memory limit |
| m (subsample size) | {500, 1000, 2000, 3000, 5000, 7500, 10000} for n=10K; {1000, 2000, 5000, 10000, 20000, 35000, 50000} for n=50K | 7 points per stratum for OLS regression; includes m=n for H6 sanity |
| seed | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} | 10 seeds per (n, m) cell for variance estimation and power |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| T_exact | dimensionless [0,1] | `spectral_init::trustworthiness(x, y, k)` | NEW — bare f64 return from `trustworthiness()` |
| T_sub | dimensionless [0,1] | Inline sub-sampled loop in experiment binary | NEW — not yet in library |
| abs_delta_T | dimensionless | `|T_sub - T_exact|` | NEW |
| wall_exact_ms | milliseconds | `std::time::Instant` around `trustworthiness()` call | NEW |
| wall_sub_ms | milliseconds | `std::time::Instant` around sub-sampled call | NEW |
| speedup_ratio | dimensionless | `median(wall_exact_ms) / median(wall_sub_ms)` | NEW |
| std_T_sub | dimensionless | `std(T_sub)` across 10 seeds at fixed (n, m) | NEW |
| cpu_model | string | `/proc/cpuinfo` or `sysinfo` crate | NEW — controlled variable |
| core_count | integer | `std::thread::available_parallelism()` | NEW — controlled variable |

All metrics are NEW — none have canonical names in `src/metrics.rs`. These are experiment-level measurements, not library metrics. They do not require addition to the metrics catalog. If the experiment concludes positively and `trustworthiness_subsampled()` is shipped, the relevant metrics (accuracy threshold, speed threshold) should be added to `src/metrics.rs` as part of the implementation PR — not in this experiment.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighbor count) | 15 | Matches sklearn default and all prior research |
| Dataset | MERFISH (PCA-50 features, 2D spatial coords) | Same dataset as PR #260 Python study for direct comparison |
| Rust toolchain | nightly-2026-03-26 | Matches all prior experiment toolchain pins |
| SIMD path | AVX2+FMA (compile-time via `-C target-cpu=native`) | Production optimization level |
| Rayon thread pool | Default (num_cpus) | Production configuration; core_count recorded per run |
| Compilation | `--release` with `--features cli` | Production optimization level |
| Repetitions per trial | 5 timed + 1 warmup (discarded) | Matches tw_profiler convention |

## Inputs and Data

The experiment operates on pre-existing MERFISH fixtures from prior research. These are **pre-positioned on the developer workstation** — the experiment does not download or regenerate them.

### Data Acquisition Model

This experiment assumes the **developer-workstation model**: MERFISH fixtures are already present from prior experiments. The preflight check (Phase 1) verifies their presence and shape before any compute is spent.

If fixtures are absent, the experiment **stops with a clear error** directing the user to run `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_data.sh` (requires raw MERFISH NPZ files in `$MERFISH_NPZ_DIR`). No automated download is provided because the raw data source has no stable public URL.

### Fixture Verification

For each fixture, verify:
1. File exists at expected path
2. Shape matches expected dimensions (via `ndarray_npy::read_npy` and `.shape()`)
3. Dtype is f64

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| merfish_n10k_x.npy | Pre-existing fixture | (10000, 50) f64, PCA-50 features | High-dim X-space for n=10K trials |
| merfish_n10k_y.npy | Pre-existing fixture | (10000, 2) f64, spatial coords | 2D Y-space for n=10K trials |
| merfish_n50k_x.npy | Pre-existing fixture | (50000, 50) f64, PCA-50 features | High-dim X-space for n=50K trials |
| merfish_n50k_y.npy | Pre-existing fixture | (50000, 2) f64, spatial coords | 2D Y-space for n=50K trials |

All fixtures located at: `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`

## Experiment Directory Layout

```
research/2026-04-10-subsampled-tw-rust/
├── rust-toolchain.toml               # Pin nightly-2026-03-26
├── environment.yml                    # Python env for analysis only
├── scripts/
│   ├── tw_subsample_experiment.rs     # Rust experiment binary (placed as example)
│   ├── run_experiment.sh              # Orchestrator: builds, runs all trials
│   ├── analyze_results.py             # Aggregation, hypothesis testing, verdicts
│   └── utils.py                       # Shared constants (K, SEEDS, M_VALUES)
├── data/                              # Symlinks to MERFISH fixtures
│   └── merfish/
│       ├── merfish_n10k_x.npy -> ../../../../research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
│       ├── merfish_n10k_y.npy -> ...
│       ├── merfish_n50k_x.npy -> ...
│       └── merfish_n50k_y.npy -> ...
├── results/
│   ├── raw/                           # One JSON per trial
│   │   ├── trial_n10000_m500_s0.json
│   │   ├── trial_n10000_m500_s1.json
│   │   └── ...
│   └── analysis/
│       ├── verdicts.json              # Structured hypothesis verdicts
│       ├── summary.md                 # Human-readable summary
│       ├── error_vs_m.png             # |ΔT| vs m plot
│       ├── speedup_vs_m.png           # Speedup ratio vs m plot
│       └── variance_decay.png         # log(std) vs log(m) plot
└── report.md                          # Final report (written by write-report)
```

### File Descriptions

**`rust-toolchain.toml`**: Pins `channel = "nightly-2026-03-26"` to match all prior experiments.

**`environment.yml`**: Minimal Python env for analysis only — `numpy`, `scipy`, `matplotlib`, `scikit-learn` (for reference constants). Reuses the `subsampled-tw-tradeoff` env spec.

**`scripts/tw_subsample_experiment.rs`**: The core Rust experiment binary, registered as an example in `Cargo.toml` (under `[[example]]` with `required-features = ["cli"]`). Implements:
- `--mode exact`: Calls `spectral_init::trustworthiness(x, y, k)` — the **same library function** used in production, with the same AVX2/Rayon optimizations. This ensures no asymmetric optimization between exact and sub-sampled paths (addresses R7/RT3).
- `--mode subsample`: Implements the sub-sampled trustworthiness loop inline, copying the exact per-row pipeline from `src/metrics.rs` (AVX2 distance kernels, introselect, rank-counting, thread-local buffers) but iterating over `query_idx` instead of `0..n`. The per-row computation is identical; only the outer iterator and normalization denominator change.
- `--mode sanity`: Runs sub-sampled with m=n and compares to exact. Reports both `T_exact` and `T_sub` in the JSON output under fields `"t_exact"` and `"t_sub"`.
- JSON output per trial with fields: `n`, `m`, `k`, `seed`, `mode`, `t_exact`, `t_sub`, `abs_delta_t`, `wall_exact_ms` (array of 5 timed repetitions), `wall_sub_ms` (array of 5 timed repetitions), `warmup_exact_ms`, `warmup_sub_ms`, `cpu_model`, `core_count`, `rust_version`.
- Warmup: 1 untimed iteration before 5 timed iterations. The warmup iteration initializes the Rayon thread pool and warms CPU caches. Warmup wall-clock is recorded in JSON (as `warmup_exact_ms` / `warmup_sub_ms`) but excluded from median calculations.

**`scripts/run_experiment.sh`**: Shell orchestrator that:
1. Runs preflight checks (fixture existence, shape verification via a `--preflight` mode in the binary)
2. Records environment metadata (CPU model, core count, Rust version, git commit)
3. Runs all trial combinations sequentially (each trial writes one JSON to `results/raw/`)
4. Calls the analysis script at the end

**`scripts/analyze_results.py`**: Python analysis script that:
1. Loads all raw JSONs from `results/raw/`
2. Computes aggregate statistics per (n, m) cell
3. Evaluates all 6 hypotheses with the specified statistical tests
4. Writes `verdicts.json` with structure: `{"H1": {"verdict": "PASS"|"FAIL"|"INCONCLUSIVE", "details": {...}}, ...}`
5. When insufficient data for a hypothesis (e.g., during dry-run), the verdict is `"INSUFFICIENT_DATA"` with a `"reason"` field
6. Generates plots and `summary.md`

**`scripts/utils.py`**: Shared constants:
```python
K = 15
SEEDS = list(range(10))
M_VALUES_10K = [500, 1000, 2000, 3000, 5000, 7500, 10000]
M_VALUES_50K = [1000, 2000, 5000, 10000, 20000, 35000, 50000]
# Python reference values from PR #260 for H4
PYTHON_SPEEDUP_10K = {500: 18.2, 1000: 9.1, 2000: 4.1, 5000: 1.7}
PYTHON_MEAN_DELTA_T_10K_M2000 = 0.00165
```

## Environment

**Option A — No custom environment needed (Rust side):**

The project's existing Rust toolchain is sufficient. The experiment binary uses only dependencies already in `Cargo.toml`: `ndarray`, `ndarray-npy` (via `cli` feature), `rayon`, `rand`, and `spectral_init` itself. The `cli` feature already exists in `Cargo.toml` (line 24) and gates `ndarray-npy`, `pico-args`, `serde_json`, and `libc` — all required by the experiment binary.

The experiment binary is registered as a `[[example]]` with `required-features = ["cli"]`, following the pattern of existing examples. No new dependencies are needed.

A `rust-toolchain.toml` is placed in the experiment directory pinning `nightly-2026-03-26` (matching all prior experiments). This ensures consistent codegen (AVX2 auto-vectorization behavior) across runs.

**Option B — Minimal Python environment for analysis:**

The analysis script requires `numpy`, `scipy`, `matplotlib`, and `scikit-learn`. These are already available in the existing `subsampled-tw-tradeoff` conda environment. A copy of the environment spec is placed in the experiment directory for reproducibility:

```yaml
name: subsampled-tw-rust
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2
  - scipy=1.15
  - scikit-learn=1.6
  - matplotlib=3.10
```

## Implementation Phases

### Phase 1: Directory Structure, Environment, and Preflight

**Files to create:**
- `research/2026-04-10-subsampled-tw-rust/` directory tree (scripts/, data/, results/raw/, results/analysis/)
- `research/2026-04-10-subsampled-tw-rust/rust-toolchain.toml`
- `research/2026-04-10-subsampled-tw-rust/environment.yml`
- Symlinks in `data/merfish/` pointing to the MERFISH fixtures

**Register the experiment binary:**
- Add `[[example]]` entry to `Cargo.toml`:
  ```toml
  [[example]]
  name = "tw_subsample_experiment"
  path = "research/2026-04-10-subsampled-tw-rust/scripts/tw_subsample_experiment.rs"
  required-features = ["cli"]
  ```

**Preflight checks (implemented as `--preflight` mode in binary):**
1. Verify each fixture file exists and can be opened
2. Load each `.npy` file and verify shape: n10k_x is (10000, 50), n10k_y is (10000, 2), n50k_x is (50000, 50), n50k_y is (50000, 2)
3. Verify dtype is f64
4. Print `"PREFLIGHT OK"` or `"PREFLIGHT FAILED: {reason}"` and exit

**Acceptance criterion:** `cargo run --release --features cli --example tw_subsample_experiment -- --preflight --data-dir research/2026-04-10-subsampled-tw-rust/data/merfish` prints `PREFLIGHT OK`.

### Phase 2: Experiment Binary

**File to create:** `research/2026-04-10-subsampled-tw-rust/scripts/tw_subsample_experiment.rs`

**Implementation details:**

The binary has three modes: `exact`, `subsample`, and `sanity`.

**CLI interface:**
```
tw_subsample_experiment --mode {exact|subsample|sanity|preflight}
    --x <path> --y <path>
    --k <int>             # default 15
    --m <int>             # subsample size (required for subsample/sanity modes)
    --seed <int>          # RNG seed (required for subsample mode)
    --reps <int>          # timed repetitions, default 5
    --warmup <int>        # warmup repetitions, default 1
    --output <path>       # JSON output path
    --data-dir <path>     # for preflight mode
```

**`--mode exact`:** Calls `spectral_init::trustworthiness(x.view(), y.view(), k)` directly — the identical library function with identical AVX2/Rayon optimizations. Runs `warmup` untimed iterations, then `reps` timed iterations. Records `t_exact`, `wall_exact_ms` (array of per-rep times), `warmup_exact_ms`.

**`--mode subsample`:** Implements the sub-sampled computation inline:
1. Generate `query_idx`: sample m unique indices from 0..n using `rand::seq::index::sample` with `StdRng::seed_from_u64(seed)`.
2. Compute sub-sampled T by iterating `query_idx.into_par_iter()` instead of `(0..n).into_par_iter()`. The per-row pipeline is copied from `src/metrics.rs:trustworthiness()`:
   - X-distances from row i to ALL n rows using the same AVX2+FMA `dist_sq_avx2_looped` kernel (for d>=10) and scalar fallback
   - Introselect via `select_nth_unstable_by` for X-kNN
   - Y-distances from row i to ALL n rows using `dist_sq_2d_avx2_batch` (for d_y==2)
   - Introselect for Y-kNN (to detect false neighbors needing penalty)
   - Rank-counting penalty (identical logic)
   - Thread-local `Vec<f64>` and `Vec<usize>` scratch buffers
3. Normalization: `m * k * (2*n - 3*k - 1)` — note: `n` is the **population** size (x.nrows()), `m` is the **sample** size.
4. Also compute T_exact via the library function (once, with warmup).
5. Output: `t_exact`, `t_sub`, `abs_delta_t`, `wall_exact_ms`, `wall_sub_ms`, `warmup_exact_ms`, `warmup_sub_ms`.

**`--mode sanity`:** Runs subsample with m=n (all indices 0..n) and compares to exact. The JSON output includes `t_exact`, `t_sub`, and `abs_delta_t`. The acceptance criterion is `abs_delta_t < 1e-10`.

**Environment metadata in all JSON output:**
```json
{
  "cpu_model": "<from /proc/cpuinfo>",
  "core_count": "<from std::thread::available_parallelism()>",
  "rust_version": "<from rustc --version>",
  "git_commit": "<from env or git rev-parse HEAD>"
}
```

**Rayon non-determinism check:** Before the main trials, run 2 identical calls to `trustworthiness(x, y, k)` and record both values. If `|T_run1 - T_run2| > 1e-6`, **abort the experiment** with an error message. This is a fatal pre-condition, not a "flag and proceed" situation — Rayon non-determinism invalidating floating-point reproducibility would undermine all accuracy measurements.

**Acceptance criterion:** `cargo run --release --features cli --example tw_subsample_experiment -- --mode sanity --x data/merfish/merfish_n10k_x.npy --y data/merfish/merfish_n10k_y.npy --m 10000 --output results/raw/sanity_n10k.json` produces JSON with `abs_delta_t < 1e-10`.

### Phase 3: Orchestration and Analysis Scripts

**Files to create:**
- `research/2026-04-10-subsampled-tw-rust/scripts/run_experiment.sh`
- `research/2026-04-10-subsampled-tw-rust/scripts/analyze_results.py`
- `research/2026-04-10-subsampled-tw-rust/scripts/utils.py`

**`run_experiment.sh`** orchestrates:
1. `cd` to project root
2. Build: `cargo build --release --features cli --example tw_subsample_experiment`
3. Run preflight: `--preflight --data-dir research/2026-04-10-subsampled-tw-rust/data/merfish`
4. Run Rayon determinism check (2 exact runs, compare T values)
5. Run sanity checks: `--mode sanity` for n=10K and n=50K
6. Run exact baseline: `--mode exact` for n=10K and n=50K (5 timed reps + 1 warmup each)
7. Run all subsample trials: nested loop over n, m, seed — each writes one JSON to `results/raw/`
8. Run analysis: `micromamba run -n subsampled-tw-rust python scripts/analyze_results.py`

**`analyze_results.py`** implements:
1. Load all JSON files from `results/raw/`
2. For each (n, m) cell: compute mean|ΔT|, max|ΔT|, std(T_sub), median wall times
3. H1: One-sample t-test on |ΔT| values at (n=10K, m=2000) against 0.01
4. H2: Per-stratum OLS of speedup_ratio ~ n/m; bootstrap 95% CI on R²; linear vs log-linear RMSE comparison
5. H3: Log-log OLS of std(T_sub) ~ m; one-sided t-test on slope against -0.3
6. H4: Compare Rust speedup ratios to Python reference constants (from utils.py)
7. H5: Same as H1 but at n=50K, m=2000
8. H6: Check sanity JSON |delta| < 1e-10
9. Write `verdicts.json`:
   ```json
   {
     "H1": {"verdict": "PASS", "mean_abs_delta_t": 0.00165, "ci_upper": 0.003, "p_value": 0.001},
     "H2": {"verdict": "PASS", "r2_10k": 0.98, "r2_50k": 0.97, "linearity": "linear"},
     "H3": {"verdict": "PASS", "slope": -0.65, "p_value": 0.001},
     "H4": {"verdict": "PASS|NOT_EVALUATED", "details": {...}},
     "H5": {"verdict": "PASS", "mean_abs_delta_t": 0.001, "note": "exploratory"},
     "H6": {"verdict": "PASS", "delta_10k": 1e-15, "delta_50k": 1e-15}
   }
   ```
10. When fewer than the required trials are present (e.g., dry-run with 3 trials), each hypothesis key still appears in `verdicts.json` with `"verdict": "INSUFFICIENT_DATA"` and `"reason": "N trials < required minimum"`.
11. Generate plots: `error_vs_m.png`, `speedup_vs_m.png`, `variance_decay.png`
12. Write `summary.md`

**Seed protocol:** Seeds 0–9 are exhaustive. No post-hoc exclusion is permitted. If any seed produces anomalous results (e.g., |ΔT| > 3σ from the cell mean), it is reported as an outlier in the summary but **included** in all aggregate statistics and hypothesis tests. The protocol is: report-and-include, never exclude (addresses RT4).

### Phase 4: Dry Run

Execute the full pipeline with a minimal 3-trial subset to verify end-to-end correctness before committing to the full ~150-trial run.

**Dry-run trials:**
1. `(n=10K, m=2000, seed=0)` — tests sub-sampling
2. `(n=10K, m=10000, seed=0)` — this is m=n=10000, which satisfies the H6 sanity check (sub-sampled with all rows = exact). Note: this trial simultaneously serves as a sanity check because m equals n.
3. `(n=50K, m=2000, seed=0)` — tests n=50K path

**Dry-run acceptance criteria:**
1. All 3 trial JSONs written successfully with all expected fields
2. Sanity trial (n=10K, m=10000): `abs_delta_t < 1e-10`
3. Analysis script runs without error and produces `verdicts.json` with all 6 hypothesis keys (most will be `"INSUFFICIENT_DATA"`)
4. Plots are generated (even if sparse)

**What the dry-run verdicts contain:** With only 3 trials, H1 has only 1 data point at (n=10K, m=2000) — `"INSUFFICIENT_DATA"`. H2 has only 2 m-values — `"INSUFFICIENT_DATA"`. H3 has only 2 m-values — `"INSUFFICIENT_DATA"`. H4 depends on H2 — `"INSUFFICIENT_DATA"`. H5 has 1 data point — `"INSUFFICIENT_DATA"`. H6 can be evaluated from the sanity trial — may produce `"PASS"` or `"FAIL"`.

## Execution Protocol

After implementation and dry-run validation, execute the full experiment:

```bash
# From project root
cd /home/talon/projects/spectral-init

# 1. Build the experiment binary
cargo build --release --features cli --example tw_subsample_experiment

# 2. Preflight check
./target/release/examples/tw_subsample_experiment --preflight \
  --data-dir research/2026-04-10-subsampled-tw-rust/data/merfish

# 3. Rayon determinism check (built into run_experiment.sh)
# Runs 2 exact trustworthiness calls, aborts if |T1-T2| > 1e-6

# 4. Sanity checks (m=n)
./target/release/examples/tw_subsample_experiment --mode sanity \
  --x research/2026-04-10-subsampled-tw-rust/data/merfish/merfish_n10k_x.npy \
  --y research/2026-04-10-subsampled-tw-rust/data/merfish/merfish_n10k_y.npy \
  --m 10000 --output research/2026-04-10-subsampled-tw-rust/results/raw/sanity_n10k.json

./target/release/examples/tw_subsample_experiment --mode sanity \
  --x research/2026-04-10-subsampled-tw-rust/data/merfish/merfish_n50k_x.npy \
  --y research/2026-04-10-subsampled-tw-rust/data/merfish/merfish_n50k_y.npy \
  --m 50000 --output research/2026-04-10-subsampled-tw-rust/results/raw/sanity_n50k.json

# 5. Exact baselines (1 warmup + 5 timed reps each)
./target/release/examples/tw_subsample_experiment --mode exact \
  --x .../merfish_n10k_x.npy --y .../merfish_n10k_y.npy \
  --output .../results/raw/exact_n10k.json

./target/release/examples/tw_subsample_experiment --mode exact \
  --x .../merfish_n50k_x.npy --y .../merfish_n50k_y.npy \
  --output .../results/raw/exact_n50k.json

# 6. Subsample trials (140 trials: 7 m-values × 10 seeds × 2 n-values)
# For each n in {10000, 50000}:
#   For each m in M_VALUES:
#     For each seed in {0..9}:
for n in 10000 50000; do
  for m in $(python3 -c "from scripts.utils import *; print(' '.join(map(str, M_VALUES_10K if $n==10000 else M_VALUES_50K)))"); do
    for seed in $(seq 0 9); do
      ./target/release/examples/tw_subsample_experiment --mode subsample \
        --x .../merfish_n${n_label}_x.npy --y .../merfish_n${n_label}_y.npy \
        --m $m --seed $seed \
        --output .../results/raw/trial_n${n}_m${m}_s${seed}.json
    done
  done
done

# 7. Analysis
micromamba run -n subsampled-tw-rust python \
  research/2026-04-10-subsampled-tw-rust/scripts/analyze_results.py
```

**Trial ordering:** n=10K trials first (faster, ~3.6s exact baseline), then n=50K (slower, ~91s exact baseline). Within each n, m values are run in ascending order. Seeds are sequential within each (n, m) cell.

**Total estimated trials:** 2 sanity + 2 exact + 140 subsample = 144 trials.

## Analysis Plan

### Primary Analysis

1. **Error table:** For each (n, m) cell, compute mean|ΔT|, max|ΔT|, std(T_sub), and count across 10 seeds.

2. **Speed table:** For each (n, m) cell, compute median wall_sub_ms, median wall_exact_ms, and speedup_ratio.

3. **H1 evaluation:** One-sample t-test on the 10 |ΔT| values at (n=10K, m=2000). Report t-statistic, p-value, 97.5% CI upper bound. Compare to threshold 0.01.

4. **H2 evaluation:** For each stratum (n=10K, n=50K):
   - OLS: speedup_ratio ~ (n/m)
   - Bootstrap R² (1000 resamples) to get 95% CI
   - Compare linear vs log-linear RMSE
   - Report per-stratum R², CI, linearity classification

5. **H3 evaluation:** Log-log OLS on std(T_sub) vs m (pooling both n strata if patterns are consistent, else per-stratum). Report slope, SE, one-sided p-value against -0.3.

6. **H4 evaluation:** Compare Rust speedup ratios to Python reference constants at overlapping (n=10K, m) points. Report log2(ratio) for each point.

7. **H5 evaluation:** Same as H1 at (n=50K, m=2000). Reported separately as exploratory.

8. **H6 evaluation:** Check sanity JSON delta values.

### Secondary Analysis

- **Cross-validation with Python:** Compare Rust T_exact to Python T_exact (should agree within ~1e-10 since both compute the same formula). Compare Rust mean|ΔT| to Python mean|ΔT| at overlapping points.
- **Step-level timing (if profiling feature enabled):** Break down sub-sampled wall-clock by step to confirm all steps scale proportionally with m.
- **Rayon utilization:** At small m (e.g., m=500 on 16 cores), note whether speedup deviates from linear — this would indicate Rayon overhead dominating at low task counts.

## Success Criteria

- **Conclusive positive (ship recommendation):** H1 PASS, H2 PASS, H3 PASS, H6 PASS. This confirms that the Rust sub-sampled trustworthiness implementation reproduces the Python study's accuracy and scaling results on MERFISH data with m=2000 default. **Scope of recommendation:** "Rust implementation reproduces Python study results on MERFISH-class data (k=15, PCA-50, AVX2 x86_64). Generalization to other scRNA-seq datasets or parameter regimes is unvalidated."
- **Conclusive negative (do not ship):** H1 FAIL or H6 FAIL. Accuracy is unacceptable or the implementation has a correctness bug.
- **Inconclusive:** H1 PASS but H2 FAIL — sub-sampling is accurate but speedup is non-linear (possible Rayon overhead issue). Further investigation needed.
- **H4 and H5 outcomes do not affect the shipping decision.** H4 is informational (cross-language comparison). H5 is exploratory (larger n).

## Threats to Validity

### Internal

1. **Warmup effects:** First call to `trustworthiness()` initializes the Rayon thread pool, triggers page faults, and populates CPU caches. **Mitigation:** 1 warmup iteration discarded before 5 timed iterations. Warmup time recorded separately in JSON.

2. **Rayon floating-point non-determinism:** Parallel reduction with floating-point addition is not associative. Rayon's work-stealing may change summation order between runs. **Mitigation:** Pre-experiment determinism check (2 identical runs, abort if |ΔT| > 1e-6). This is a hard gate — if violated, the experiment cannot proceed until the cause is identified.

3. **System load interference:** Background processes can affect wall-clock timing. **Mitigation:** 5 timed repetitions with median (not mean) to reduce outlier sensitivity. No formal system isolation is enforced — this is a developer workstation experiment.

4. **SIMD floating-point ordering:** AVX2+FMA may produce different rounding than scalar code. Sub-sampled and exact paths both use the same SIMD kernels, so any rounding differences are symmetric. **Mitigation:** H6 sanity check confirms m=n produces identical T.

5. **Asymmetric optimization (R7/RT3):** The exact path uses `spectral_init::trustworthiness()` (library function), while the sub-sampled path uses an inline copy of the per-row loop. If the inline copy is less optimized (e.g., different inlining decisions), speedup ratios would be inflated. **Mitigation:** Both paths use identical SIMD kernels and thread-local buffer patterns. The only difference is the outer iterator. Code review during implementation must verify the per-row pipeline is identical.

### External

1. **Dataset specificity:** All results are on MERFISH PCA-50 + 2D spatial data. The error/speed trade-off may differ for datasets with different dimensionality, cluster structure, or noise profiles. **Scope limitation acknowledged in success criteria.**

2. **Threshold leakage (RT2/RT6):** m=2000 and the 0.01 threshold were calibrated on MERFISH in PR #260. This study validates the same parameter on the same dataset. **Mitigation:** Success criteria explicitly state this is same-dataset confirmation, not out-of-sample validation. The shipping recommendation is scoped as "Rust implementation parity with Python study" not "validated cross-dataset default."

3. **Hardware specificity:** Results depend on CPU model (AVX2 support, core count, cache hierarchy). Speedup ratios are not portable across hardware. **Mitigation:** CPU model and core count recorded in every trial JSON.

4. **k=15 specificity:** All experiments use k=15. The error/speed trade-off at other k values is unknown.

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| **n=10K subsample trials** | 70 trials × ~2s each ≈ 2.5 min |
| **n=50K subsample trials** | 70 trials × ~20s each ≈ 23 min |
| **n=10K exact baselines** | 2 runs × 5 reps × ~3.6s ≈ 36s |
| **n=50K exact baselines** | 2 runs × 5 reps × ~91s ≈ 15 min |
| **Sanity checks** | 2 runs × ~4s + ~91s ≈ 2 min |
| **Total compute time** | ~45 min |
| **Disk space (results)** | ~144 JSON files × ~1KB ≈ 150 KB |
| **Disk space (data)** | Symlinks to existing ~25 MB fixtures |
| **Dependencies** | Existing Rust nightly + Python conda env |
