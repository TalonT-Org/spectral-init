# Experiment Plan: Sub-Sampled Trustworthiness — Rust Error/Speed Trade-off Validation

## Motivation

The Python sub-sampling research (PR #260, `research/2026-04-09-subsampled-tw-tradeoff/`) demonstrated that computing trustworthiness on a random subset of m rows yields mean|ΔT| < 0.002 at m=2000, with ~4.1x speedup at n=10K. However, the Rust `trustworthiness()` implementation has a fundamentally different performance profile: AVX2+FMA SIMD distance kernels, Rayon work-stealing parallelism, introselect (O(n) partial sort), and thread-local buffer reuse. Before shipping a `trustworthiness_subsampled()` function in the Rust crate, we must empirically validate that:

1. **Accuracy** matches Python findings (mean|ΔT| < 0.01 at m=2000)
2. **Speedup** is linear in the reduction factor m/n
3. **SIMD floating-point ordering** does not introduce systematic bias vs the Python scalar path
4. **n=50K works** — Python could not test this due to O(n²) memory; our O(n)-per-thread implementation can
5. **Normalization** is correct (T_sub(m=n) == T_exact within 1e-10)

This experiment's results will determine: (a) what default subsample size to ship, (b) whether the Rust speedup justifies the same m=2000 recommendation as Python, and (c) whether any Rust-specific corrections are needed.

## Hypothesis

**Null hypothesis (H0):** Sub-sampling rows in the Rust trustworthiness computation introduces unacceptable error (mean|ΔT| >= 0.01 at m=2000) or yields sub-linear speedup (speedup ratio significantly less than n/m), making it unsuitable for production use.

**Alternative hypothesis (H1):** Sub-sampling in Rust yields mean|ΔT| < 0.01 at m=2000 with speedup approximately linear in n/m, matching the Python research findings and confirming that the SIMD/Rayon implementation behaves equivalently under row sub-sampling.

### Sub-Hypotheses (from scope report)

- **H1 (Accuracy):** At m=2000, n=10K, mean|ΔT| < 0.01 across 10 seeds. *Reject if* mean|ΔT| >= 0.01.
- **H2 (Linear Speedup):** Speedup ≈ n/m. Fit speedup vs n/m across 6+ m values. *Reject if* R² < 0.95.
- **H3 (Variance Decay):** std(T_sub) decays as O(1/√m) or faster. *Reject if* log-log slope > -0.3.
- **H4 (Cross-Language Parity):** Rust speedup ratio at same (n,m) matches Python within 2x. *Reject if* ratio differs > 2x.
- **H5 (n=50K Validation):** At n=50K, m=2000, mean|ΔT| < 0.01 across 10 seeds. *Reject if* mean|ΔT| >= 0.01.
- **H6 (Normalization Sanity):** T_sub(m=n) == T_exact within 1e-10. *Reject if* |delta| >= 1e-10.

## Independent Variables

| Variable | Values (n=10K) | Values (n=50K) | Rationale |
|----------|----------------|-----------------|-----------|
| n (population) | 10000 | 50000 | Two dataset sizes; n=50K is the novel Rust-only contribution |
| m (subsample) | 250, 500, 1000, 2000, 5000, 7500 | 500, 1000, 2000, 5000, 10000, 25000 | Covers 2.5%-75% of n; matches Python's m values at n=10K for cross-validation |
| seed | 0, 1, 2, ..., 9 | 0, 1, 2, ..., 9 | 10 seeds per cell for variance estimation |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Exact trustworthiness | dimensionless [0,1] | `spectral_init::trustworthiness(x, y, k)` | `trustworthiness` (existing) |
| Sub-sampled trustworthiness | dimensionless [0,1] | `trustworthiness_subsampled(x, y, k, query_idx)` | NEW: `tw_subsampled` |
| Absolute error | dimensionless | `|T_sub - T_exact|` | NEW: `tw_sample_abs_error` — threshold: < 0.01 |
| Wall-clock time (exact) | seconds | `std::time::Instant` | `wall_exact_s` (reuses tw_profiler pattern) |
| Wall-clock time (sub-sampled) | seconds | `std::time::Instant` | `wall_sub_s` |
| Speedup ratio | dimensionless | `wall_exact_s / wall_sub_s` | NEW: `tw_speedup_ratio` |
| Variance across seeds | dimensionless | `std(T_sub)` over 10 seeds at fixed (n, m) | NEW: `tw_sample_std` |
| Normalization error | dimensionless | `|T_sub(m=n) - T_exact|` | NEW: `tw_normalization_error` — threshold: < 1e-10 |

For metrics marked "NEW": these are experiment-specific derived measurements. If the experiment succeeds and sub-sampling is shipped, `tw_sample_abs_error` (threshold 0.01) and `tw_normalization_error` (threshold 1e-10) should be added to `src/metrics.rs` as canonical thresholds.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighbors) | 15 | Matches Python experiment (K=15) and all prior trustworthiness research |
| Dataset | MERFISH | Real-world scRNA-seq data; matches all prior experiments in chain |
| d_x (input dim) | 50 | MERFISH dimensionality |
| d_y (embedding dim) | 2 | Standard UMAP output |
| Hardware | Same machine, same run | Eliminates hardware variance for timing comparisons |
| Rust toolchain | nightly-2026-03-26 | Pinned for SIMD intrinsics reproducibility |
| Build profile | `--release` | Production-representative optimization |
| Rayon threads | Default (num_cpus) | Production-representative parallelism |
| Warmup | 1 iteration (discarded) | Ensures CPU caches and branch predictors are warm |

## Inputs and Data

All data already exists as MERFISH .npy fixtures from prior research. No data generation is required.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| merfish_n10k_x.npy | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/` | (10000, 50) float64 | X-space input, n=10K trials |
| merfish_n10k_y.npy | Same directory | (10000, 2) float64 | Y-space embedding, n=10K trials |
| merfish_n50k_x.npy | Same directory | (50000, 50) float64 | X-space input, n=50K trials |
| merfish_n50k_y.npy | Same directory | (50000, 2) float64 | Y-space embedding, n=50K trials |

The Python experiment's reference results (exact T values, sub-sampled T values) are in `research/2026-04-09-subsampled-tw-tradeoff/results/raw/` for cross-validation:
- `exact_merfish_10000.json`: T_exact = 0.5362038060873342 (k=15, n=10K)
- `sub_A_merfish_10000_m{m}_s{seed}.json`: Per-trial sub-sampled results

**Data properties relevant to validity:**
- MERFISH has correlated local structure, which the Python research found causes variance to decay faster than CLT predicts (slope -0.657 vs theoretical -0.5). This is a feature, not a bug — it means sub-sampling works better on real data than worst-case theory suggests.
- n=50K is the maximum available fixture size. The Python experiment could not test n=50K due to sklearn's O(n²) memory; our implementation's O(n)-per-thread memory makes this feasible.

## Experiment Directory Layout

```
research/2026-04-09-subsampled-tw-rust-validation/
├── experiment-plan.md                      # This plan (copied from .autoskillit/temp/)
├── scripts/
│   ├── run_experiment.sh                   # Shell orchestrator: build, hardware profile, run
│   └── analyze_results.py                  # Python analysis: aggregate, plot, evaluate hypotheses
├── data/                                   # Symlinks to MERFISH fixtures
│   └── merfish/
│       ├── merfish_n10k_x.npy -> ../../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
│       ├── merfish_n10k_y.npy -> ...
│       ├── merfish_n50k_x.npy -> ...
│       └── merfish_n50k_y.npy -> ...
├── results/
│   ├── raw/                                # Per-n JSON output from experiment binary
│   │   ├── merfish_n10000.json
│   │   └── merfish_n50000.json
│   ├── analysis/
│   │   ├── summary.md                      # Hypothesis verdicts table
│   │   ├── error_vs_m.png                  # |ΔT| vs m (both n values)
│   │   ├── speedup_vs_m.png               # Speedup ratio vs n/m
│   │   ├── variance_vs_m.png              # std(T_sub) vs m (log-log)
│   │   └── cross_validation.png           # Rust vs Python |ΔT| comparison
│   └── hardware_profile.txt                # CPU, cores, cache sizes
└── report.md                               # Final report (written by write-report)
```

### File Descriptions

**`src/bin/tw_subsample_experiment.rs`** — Rust experiment binary (added to project). Loads MERFISH .npy fixtures, computes exact T once, then iterates over all (m, seed) combinations computing sub-sampled T. Outputs one JSON file per n value containing exact result, normalization sanity check, and all trial results. Uses `pico-args` for CLI, `ndarray-npy` for data loading, `serde_json` for output. Declared as `[[bin]]` with `required-features = ["cli", "testing"]`.

**`scripts/run_experiment.sh`** — Shell orchestrator following `tw_profiler` pattern. Builds the binary with `cargo build --release --features cli,testing --bin tw_subsample_experiment`, captures hardware profile, runs the binary for n=10K and n=50K, directs output to `results/raw/`.

**`scripts/analyze_results.py`** — Python analysis script. Reads JSON results, computes per-(n,m) statistics (mean/std of |ΔT|, mean/std of wall-clock, speedup ratio), fits linear speedup and variance decay models, compares against Python reference values from `research/2026-04-09-subsampled-tw-tradeoff/results/raw/`, produces plots and `summary.md` with hypothesis verdicts.

**`data/merfish/`** — Symlinks to existing MERFISH fixtures. Avoids data duplication.

## Environment

**Option A — No custom environment needed (for Rust experiment):**

The project's existing Rust toolchain (nightly-2026-03-26, Cargo features `cli` + `testing`) is fully sufficient for building and running the experiment binary. All required dependencies (`ndarray-npy`, `pico-args`, `serde_json`, `rand`) are already declared in `Cargo.toml`. No external libraries, system packages, or non-Rust tools are needed for the core experiment.

**Python analysis (minimal, optional):**

The analysis script requires Python with numpy and matplotlib. These are standard packages available in any Python 3.x environment. A dedicated environment.yml is not necessary — the implementer may use any available Python with these packages. If a reproducible environment is desired, the following suffices:

```yaml
name: tw-rust-validation
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.24
  - matplotlib>=3.7
```

This is optional — the core experiment results (JSON files) are produced by Rust alone. The Python script is only for post-hoc aggregation and visualization.

## Implementation Phases

### Phase 1: Library Function — `trustworthiness_subsampled()`

**Goal:** Add the sub-sampled trustworthiness function to `src/metrics.rs`, reusing the existing SIMD kernels and inner-loop logic.

**Files to modify:**

1. **`src/metrics.rs`** — Add `trustworthiness_subsampled()`:
   ```rust
   #[cfg(feature = "testing")]
   pub fn trustworthiness_subsampled(
       x: ArrayView2<f64>,
       y: ArrayView2<f64>,
       k: usize,
       query_idx: &[usize],
   ) -> f64
   ```
   Implementation is structurally identical to `trustworthiness()` with two changes:
   - Outer parallel loop: `query_idx.par_iter().map(|&i| ...)` instead of `(0..n).into_par_iter()`
   - Normalization denominator: `m * k * (2n - 3k - 1)` where `m = query_idx.len()`, `n = x.nrows()`
   - The inner loop body (x_dist, x_sort, y_dist, penalty) is identical — same SIMD kernels, same thread-local buffers, same introselect

   The function must validate:
   - `query_idx` values are in range `[0, n)`
   - `query_idx` is non-empty
   - Same `k < n/2` guard as existing function

2. **`src/metrics.rs`** — Add unit test for `trustworthiness_subsampled`:
   - Test that `trustworthiness_subsampled(x, y, k, &(0..n).collect::<Vec<_>>())` equals `trustworthiness(x, y, k)` within 1e-10 (normalization sanity)
   - Use the existing small test fixture (e.g., the 5-point test used in `test_trustworthiness_formula`)

**Verification:** `cargo test --features testing -- trustworthiness_subsampled`

### Phase 2: Experiment Binary

**Goal:** Create the Rust experiment binary that drives all benchmark trials.

**Files to create:**

1. **`src/bin/tw_subsample_experiment.rs`** — Main experiment binary.

   Template: `src/bin/tw_profiler.rs` (same CLI pattern, JSON output, Instant timing).

   CLI arguments:
   - `--x <path>` — Path to X .npy file
   - `--y <path>` — Path to Y .npy file
   - `--k <int>` — Number of neighbors (default 15)
   - `--m-values <comma-separated>` — Subsample sizes (e.g., "250,500,1000,2000,5000,7500")
   - `--seeds <int>` — Number of seeds (default 10; uses seeds 0..N-1)
   - `--warmup <int>` — Warmup iterations for exact T (default 1)
   - `--output <path>` — Output JSON file path

   Execution flow:
   ```
   1. Parse args, load x and y via ndarray_npy::read_npy
   2. Warmup: call trustworthiness(x, y, k) once (discard)
   3. Exact T: time trustworthiness(x, y, k), record T_exact and wall_exact_s
   4. Normalization sanity: call trustworthiness_subsampled(x, y, k, &(0..n).collect())
      Record delta = |T_sanity - T_exact|, pass = delta < 1e-10
   5. For each m in m_values:
        For each seed in 0..num_seeds:
          a. Generate query_idx: use StdRng::seed_from_u64(seed), sample m indices
             from 0..n without replacement (use rand::seq::index::sample)
          b. Time trustworthiness_subsampled(x, y, k, &query_idx)
          c. Record: m, seed, t_sub, delta_t, abs_delta_t, wall_sub_s
   6. Write JSON to --output
   ```

   JSON output schema:
   ```json
   {
     "n": 10000,
     "d_x": 50,
     "d_y": 2,
     "k": 15,
     "dataset": "merfish",
     "exact": {
       "t": 0.5362038060873342,
       "wall_s": 3.61
     },
     "sanity": {
       "m": 10000,
       "t_sub": 0.5362038060873342,
       "delta": 1.2e-15,
       "pass": true
     },
     "trials": [
       {
         "m": 250,
         "seed": 0,
         "t_sub": 0.534,
         "delta_t": -0.002,
         "abs_delta_t": 0.002,
         "wall_s": 0.09
       }
     ]
   }
   ```

2. **`Cargo.toml`** — Add `[[bin]]` entry:
   ```toml
   [[bin]]
   name = "tw_subsample_experiment"
   path = "src/bin/tw_subsample_experiment.rs"
   required-features = ["cli", "testing"]
   ```

**Verification:** `cargo build --release --features cli,testing --bin tw_subsample_experiment`

### Phase 3: Experiment Infrastructure

**Goal:** Create the experiment directory, orchestrator script, and analysis script.

**Files to create:**

1. **`research/2026-04-09-subsampled-tw-rust-validation/experiment-plan.md`** — Copy of this plan.

2. **`research/2026-04-09-subsampled-tw-rust-validation/data/merfish/`** — Symlinks to MERFISH fixtures:
   ```bash
   ln -s ../../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
   ln -s ../../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy
   ln -s ../../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy
   ln -s ../../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy
   ```

3. **`research/2026-04-09-subsampled-tw-rust-validation/scripts/run_experiment.sh`** — Shell orchestrator:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
   EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
   PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
   RESULTS_RAW="$EXPERIMENT_DIR/results/raw"
   DATA_DIR="$EXPERIMENT_DIR/data/merfish"

   mkdir -p "$RESULTS_RAW" "$EXPERIMENT_DIR/results/analysis"

   # Hardware profile
   lscpu > "$EXPERIMENT_DIR/results/hardware_profile.txt" 2>/dev/null || true

   # Build
   echo "Building tw_subsample_experiment..."
   cargo build --release --features cli,testing --bin tw_subsample_experiment \
     --manifest-path "$PROJECT_ROOT/Cargo.toml"

   BIN="$PROJECT_ROOT/target/release/tw_subsample_experiment"

   # n=10K
   echo "Running n=10K..."
   "$BIN" \
     --x "$DATA_DIR/merfish_n10k_x.npy" \
     --y "$DATA_DIR/merfish_n10k_y.npy" \
     --k 15 \
     --m-values "250,500,1000,2000,5000,7500" \
     --seeds 10 \
     --warmup 1 \
     --output "$RESULTS_RAW/merfish_n10000.json"

   # n=50K
   echo "Running n=50K..."
   "$BIN" \
     --x "$DATA_DIR/merfish_n50k_x.npy" \
     --y "$DATA_DIR/merfish_n50k_y.npy" \
     --k 15 \
     --m-values "500,1000,2000,5000,10000,25000" \
     --seeds 10 \
     --warmup 1 \
     --output "$RESULTS_RAW/merfish_n50000.json"

   echo "Done. Results in $RESULTS_RAW"
   ```

4. **`research/2026-04-09-subsampled-tw-rust-validation/scripts/analyze_results.py`** — Python analysis:
   - Read `results/raw/merfish_n10000.json` and `merfish_n50000.json`
   - For each (n, m): compute mean|ΔT|, std(|ΔT|), mean(T_sub), std(T_sub), mean wall-clock, speedup
   - Fit linear model: speedup vs n/m (for H2)
   - Fit log-log model: std(T_sub) vs m (for H3)
   - Cross-validate against Python reference: load `research/2026-04-09-subsampled-tw-tradeoff/results/raw/exact_merfish_10000.json` and `sub_A_merfish_10000_m*_s*.json` for H4
   - Generate plots: `error_vs_m.png`, `speedup_vs_m.png`, `variance_vs_m.png`, `cross_validation.png`
   - Write `results/analysis/summary.md` with hypothesis verdict table
   - Evaluate H1-H6 with PASS/FAIL/INCONCLUSIVE verdicts

### Phase 4: Dry Run

**Goal:** Verify the end-to-end pipeline works before committing to the full experiment.

1. Build the binary: `cargo build --release --features cli,testing --bin tw_subsample_experiment`
2. Run a minimal trial (n=10K, m=2000 only, 2 seeds):
   ```bash
   ./target/release/tw_subsample_experiment \
     --x research/2026-04-09-subsampled-tw-rust-validation/data/merfish/merfish_n10k_x.npy \
     --y research/2026-04-09-subsampled-tw-rust-validation/data/merfish/merfish_n10k_y.npy \
     --k 15 --m-values "2000" --seeds 2 --warmup 1 \
     --output /tmp/dry_run.json
   ```
3. Verify JSON output: check T_exact matches known value (~0.5362), sanity check passes, trial results are present
4. Verify sub-sampled T is in a reasonable range (within 0.05 of exact)
5. Run `cargo test --features testing -- trustworthiness_subsampled` to verify unit tests pass

## Execution Protocol

After implementation is complete and dry run passes:

1. **Ensure clean machine state:** Close unnecessary processes. Verify no competing CPU-intensive workloads.
2. **Run the full experiment:**
   ```bash
   cd research/2026-04-09-subsampled-tw-rust-validation
   bash scripts/run_experiment.sh 2>&1 | tee results/experiment_log.txt
   ```
3. **Verify outputs exist:**
   - `results/raw/merfish_n10000.json` (should have 1 exact + 1 sanity + 60 trials)
   - `results/raw/merfish_n50000.json` (should have 1 exact + 1 sanity + 60 trials)
4. **Spot-check sanity results:** Both files should show `sanity.pass: true`.
5. **Run analysis:**
   ```bash
   python scripts/analyze_results.py
   ```
6. **Review outputs:** `results/analysis/summary.md` for hypothesis verdicts, PNG plots for visual inspection.

## Analysis Plan

### Per-(n, m) Aggregation

For each (n, m) cell across 10 seeds:
- `mean_delta_t` = mean of `abs_delta_t` across seeds
- `std_delta_t` = std of `abs_delta_t` across seeds
- `mean_t_sub` = mean of `t_sub` across seeds
- `std_t_sub` = std of `t_sub` across seeds
- `mean_wall_s` = mean of `wall_s` across seeds
- `speedup` = `exact.wall_s / mean_wall_s`

### Hypothesis Evaluation

**H1 (Accuracy at m=2000, n=10K):**
- Compute `mean_delta_t` at (n=10K, m=2000)
- PASS if < 0.01, FAIL if >= 0.01
- Report exact value and compare to Python's 0.00165

**H2 (Linear Speedup):**
- Plot speedup vs n/m for each n
- Fit linear regression: speedup = a * (n/m) + b
- PASS if R² >= 0.95, FAIL if R² < 0.95
- Report slope (expect ~1.0) and intercept (expect ~0.0)

**H3 (Variance Decay):**
- Compute std(T_sub) at each m (across 10 seeds)
- Fit log-log regression: log(std) = slope * log(m) + intercept
- PASS if slope <= -0.3, FAIL if slope > -0.3
- Compare slope to Python's -0.657

**H4 (Cross-Language Parity):**
- At overlapping (n=10K, m) values: compare Rust speedup ratio to Python speedup ratio
- PASS if all ratios are within 2x of each other
- Note: absolute times will differ (Rust is faster); the ratio n/m should be similar

**H5 (n=50K at m=2000):**
- Compute `mean_delta_t` at (n=50K, m=2000)
- PASS if < 0.01, FAIL if >= 0.01
- This is novel data — Python could not test n=50K

**H6 (Normalization Sanity):**
- Read `sanity.delta` from both JSON files
- PASS if both < 1e-10, FAIL if either >= 1e-10

### Summary Table

Produce a markdown table:

| Hypothesis | Criterion | Observed | Verdict |
|------------|-----------|----------|---------|
| H1 | mean\|ΔT\| < 0.01 (m=2000, n=10K) | {value} | PASS/FAIL |
| H2 | R² >= 0.95 (speedup linearity) | {value} | PASS/FAIL |
| H3 | Variance slope <= -0.3 | {value} | PASS/FAIL |
| H4 | Rust/Python speedup ratio within 2x | {values} | PASS/FAIL |
| H5 | mean\|ΔT\| < 0.01 (m=2000, n=50K) | {value} | PASS/FAIL |
| H6 | \|T_sub(m=n) - T_exact\| < 1e-10 | {value} | PASS/FAIL |

### Overall Verdict

- **Ship sub-sampling:** All H1-H6 PASS. Recommend default m=2000 (or whatever m minimizes error within acceptable speedup).
- **Ship with caveat:** H1-H3 and H5-H6 PASS but H4 shows Rust-specific speedup characteristics. Recommend Rust-specific default m.
- **Do not ship:** Any of H1, H5, or H6 FAIL. Investigate root cause before proceeding.

## Success Criteria

- **Conclusive positive:** H1-H6 all PASS. Error/speed table shows monotonic improvement. Recommended default m with Rust-specific speedup justification.
- **Conclusive negative:** H1 or H5 FAIL (error too large at m=2000), or H6 FAIL (normalization bug). Clear diagnosis of root cause.
- **Inconclusive:** H2 or H3 marginal (e.g., R² = 0.93). May need more m values or more seeds. H4 fails due to non-comparable measurement conditions.

## Threats to Validity

### Internal

1. **CPU thermal throttling:** Long runs (especially n=50K) may cause frequency scaling. Mitigation: warmup iteration, discard first run, monitor CPU frequency if possible.
2. **Rayon thread pool startup cost:** First parallel call has pool initialization overhead. Mitigation: warmup iteration uses exact T, which initializes the pool. All subsequent calls reuse it.
3. **Memory allocator fragmentation:** Thread-local buffers may interact differently with allocator at small vs large m. Mitigation: monitor peak RSS; unlikely to matter since buffers are always n-length regardless of m.
4. **SIMD floating-point non-determinism:** AVX2 FMA operations may give slightly different results than scalar path due to rounding. Mitigation: this is the point of the experiment — we're measuring whether this divergence matters in practice.
5. **Seed selection bias:** Seeds 0-9 are arbitrary. Mitigation: 10 seeds provides reasonable variance estimate; uniform random sampling from `StdRng` is well-tested.

### External

1. **Dataset specificity:** Results are for MERFISH d=50 only. Other datasets with different local structure may show different error/speed trade-offs.
2. **Hardware specificity:** AVX2 SIMD speedup is Intel/AMD x86 specific. ARM (NEON) or non-SIMD paths may show different absolute times (but speedup ratios should be similar since sub-sampling is architecture-agnostic).
3. **k sensitivity:** All experiments use k=15. Larger k values increase per-row work and penalty sum, which could change error characteristics. However, k=15 is the standard UMAP default.
4. **Rayon core count:** Results depend on available parallelism. Machines with fewer cores may see different speedup at small m due to reduced parallelism opportunity.

## Estimated Resource Requirements

- **Compute time:** ~20 minutes total
  - n=10K: ~65 seconds (1 exact + 60 sub-sampled trials + 1 sanity)
  - n=50K: ~17 minutes (1 exact + 60 sub-sampled trials + 1 sanity)
- **Disk space:** < 5 MB (JSON results + PNG plots). MERFISH fixtures are symlinked, not copied.
- **Memory:** Peak ~400 MB (n=50K: two 50K×50 f64 arrays + thread-local n-length buffers)
- **Dependencies:** No new dependencies. Existing Cargo.toml covers everything (`ndarray-npy`, `pico-args`, `serde_json`, `rand` all present). Python analysis needs numpy + matplotlib (standard).
- **Build time:** < 30 seconds incremental (new binary + one function addition to metrics.rs)
