# Experiment Plan: Trustworthiness Step-Timing Validation on MERFISH Data

## Motivation

PR #238 established that X-space operations (x_dist + x_sort) consume ~56% of trustworthiness runtime on synthetic Gaussian data (d_x=10, n=10K), making X-space ANN the productive next optimization target. However, that conclusion rests entirely on synthetic data with uniform/Gaussian geometry. Before committing engineering effort to an X-space ANN optimization, we need to verify that the step-timing breakdown holds on a real dataset.

This experiment profiles trustworthiness step timing on a single MERFISH spatial transcriptomics dataset (PCA-50, n=10K and n=50K) and compares the results against a fresh Gaussian baseline run on the same binary and hardware. The result informs whether to proceed with X-space ANN optimization or investigate data-geometry-dependent bottleneck shifts.

**Scope limitation (per R1):** This experiment validates step timing on one specific MERFISH dataset (mouse hypothalamic preoptic region, PCA-50 reduction). Results apply to this dataset's geometry and dimensionality. Generalization to "real biological data" broadly requires additional datasets with different tissue types, assays, and PCA dimensionalities.

## Hypothesis

**Null hypothesis (H0):** On this MERFISH 10K dataset (d_x=50), X-space operations (x_dist + x_sort) account for less than 50% of total trustworthiness step time, meaning PR #238's X-dominance conclusion does not transfer to this data configuration.

**Alternative hypothesis (H1):** X-space operations account for 50% or more of total trustworthiness step time on this MERFISH 10K dataset, confirming that X-space ANN remains the productive optimization target for this data configuration. Given d_x=50 (5x the synthetic baseline's d_x=10), X-space dominance is expected to be stronger than on synthetic data.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Dataset | MERFISH 10K (d_x=50), Gaussian 10K (d_x=10) | Primary comparison: real vs synthetic geometry at matched n |
| Dataset size (stretch) | MERFISH 50K (d_x=50) | Validates fraction stability across n on real data |

**Confound acknowledgment (per W1-W4):** The Gaussian baseline differs from MERFISH in both d_x (10 vs 50) and data geometry. This comparison conflates dimensionality and geometry effects. A matched-d_x=50 Gaussian control would be needed to isolate geometry effects alone; that is out of scope for this exploratory experiment. The primary comparison answers whether X-dominance holds on this MERFISH configuration, not whether the shift (if any) is caused by dimensionality vs geometry.

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| x_space_pct | percentage points (pp) | (x_dist + x_sort) / sum(all steps) × 100 | NEW |
| x_dist_pct | pp | x_dist / sum(all steps) × 100 | NEW |
| x_sort_pct | pp | x_sort / sum(all steps) × 100 | NEW |
| y_dist_pct | pp | y_dist / sum(all steps) × 100 | NEW |
| penalty_pct | pp | penalty / sum(all steps) × 100 | NEW |
| wall_time_s | seconds | tw_profiler `iters[]` array | NEW |
| tw_score | dimensionless (0,1] | tw_profiler `score` field | NEW |
| x_dist_ns | nanoseconds (thread-sum) | step_timing.x_dist raw values | NEW |

All metrics are marked "NEW" because `src/metrics.rs` does not define canonical metric names for step-timing fractions. These are research-only measurements, not CI-gated thresholds. No additions to `src/metrics.rs` are required for this experiment.

**DV priority ordering (per W24):** `x_space_pct` is the **primary** dependent variable. The four individual step fractions (x_dist_pct, x_sort_pct, y_dist_pct, penalty_pct) are **secondary** DVs providing breakdown detail. wall_time_s and tw_score are **tertiary** sanity checks.

**Proxy limitation (per W8-W11):** Step timings are thread-summed nanosecond counters from `AtomicU64::fetch_add` across all Rayon threads. They represent aggregate compute-share, not wall-clock share. The proxy systematically overweights high-SIMD-throughput steps because SIMD instructions retire more FLOPs per nanosecond. At d_x=50, the AVX2 x_dist kernel processes 8 f64/cycle (~6.25 loop iterations for 50 dims), while the batched y_dist kernel processes 2 points per loop iteration at d_y=2. "X-space dominance confirmed" is a statement about compute-time share, with a caveat that SIMD utilization asymmetry means the wall-clock share may differ slightly from the thread-ns share.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighbors) | 15 | Matches PR #238 synthetic baselines and prior experiments |
| Timed iterations | 5 | Sufficient for CI estimation; validated in dry run (per RT-3) |
| Warmup iterations | 2 | Flushes JIT/cache cold-start artifacts |
| RAYON_NUM_THREADS | 16 | Pins to all available cores for reproducibility (per W15) |
| Binary build | `cargo build --release --features cli,profiling --bin tw_profiler` | Single build for all runs ensures identical codepath |
| Execution order | Gaussian 10K → MERFISH 10K → MERFISH 50K | Fixed order with acknowledged bias (per W5-W7) |

**Execution order bias (per W5-W7):** Sequential execution without cache flushing between datasets means: (1) the first dataset (Gaussian) runs with cold page cache; (2) replicate 1 of each dataset has different I/O characteristics than replicates 2-5 due to page cache warming; (3) thermal accumulation may slightly slow later runs. These biases are systematic and directional but small relative to the step-fraction differences under test (~20-30pp expected). They are accepted for this exploratory experiment.

## Inputs and Data

### MERFISH Fixtures (per R2)

The MERFISH .npy files are **git-committed** (not Git LFS) in commit `6cfa1df` ("Add research artifacts: 20260405-tw-perf-rerun-clean (#229)"). They are regular files, not symlinks. Any clone of this repository has access to them.

**Provenance:** Generated by `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish.py` from raw MERFISH data (`merfish_100k_expression.npz` + `merfish_100k_spatial.npz`) using `sklearn.decomposition.PCA(n_components=50, random_state=42)`. The raw source data is not in the repository.

| Dataset | Path | Shape | dtype | Purpose |
|---------|------|-------|-------|---------|
| MERFISH 10K X | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | (10000, 50) | float64 | Primary real-data X-space input |
| MERFISH 10K Y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | (10000, 2) | float64 | Primary real-data Y-space input |
| MERFISH 50K X | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | (50000, 50) | float64 | Stretch: scaling validation |
| MERFISH 50K Y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | (50000, 2) | float64 | Stretch: scaling validation |

### Gaussian Baseline

Generated fresh by `gen_gaussian_baseline.py` to ensure matched conditions (same binary, same hardware, same session). Uses the established pattern from `research/2026-04-05-tw-perf-rerun-clean/scripts/gen_synthetic.py` with `np.random.default_rng(seed=2026)`, d_x=10, d_y=2.

| Dataset | Source | Shape | dtype | Purpose |
|---------|--------|-------|-------|---------|
| Gaussian 10K X | Generated (seed=2026) | (10000, 10) | float64 | Fresh baseline for same-session comparison |
| Gaussian 10K Y | Generated (seed=2026) | (10000, 2) | float64 | Fresh baseline Y embedding |

### Historical Reference

The post-flat_simd Gaussian baseline from `research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_flat_simd_n10000.json` provides a historical reference point (30 iters, k=15, d_x=10, n=10K):

| Step | Fraction |
|------|----------|
| x_dist | 31.9% |
| x_sort | 24.4% |
| y_heap (now y_dist) | 27.6% |
| penalty | 16.2% |
| **X-space total** | **56.2%** |

**Step name note:** Historical results use the key `y_heap`; current production code emits `y_dist`. The analysis script must use `y_dist` for new runs.

## Experiment Directory Layout

```
research/2026-04-08-tw-merfish-step-timing/
├── scripts/
│   ├── verify_inputs.py          # Verify npy shapes, dtypes, d_x confirmation
│   ├── gen_gaussian_baseline.py  # Generate d_x=10 Gaussian data (seed=2026)
│   ├── run_profiler.sh           # Build tw_profiler and run on all configs
│   └── analyze_results.py        # Compute fractions, CIs, comparison table
├── data/
│   └── gaussian/                 # Generated Gaussian baseline data
├── results/
│   ├── profiler/                 # tw_profiler JSON output + stderr captures
│   └── analysis/                 # Comparison tables, summary
└── report.md                     # Final report (written by write-report)
```

### File Descriptions

**`scripts/verify_inputs.py`**: Reads the npy header of each input file using `numpy.load(..., mmap_mode='r')`, prints shape, dtype, and confirms d_x. Records results to stdout. Validates that all four MERFISH fixtures are readable and have expected shapes. This is the first script run to confirm d_x=50.

**`scripts/gen_gaussian_baseline.py`**: Generates `data/gaussian/gaussian_n10k_x.npy` (10000, 10) and `data/gaussian/gaussian_n10k_y.npy` (10000, 2) using `np.random.default_rng(seed=2026)`. X is drawn from N(0,1), Y from U(0,1). Matches the established pattern from prior experiments (reference: `research/2026-04-05-tw-perf-rerun-clean/scripts/gen_synthetic.py`).

**`scripts/run_profiler.sh`**: Builds `tw_profiler` once with `cargo build --release --features cli,profiling --bin tw_profiler`. Then runs it sequentially on three configurations:
1. Gaussian 10K: `--x data/gaussian/gaussian_n10k_x.npy --y data/gaussian/gaussian_n10k_y.npy`
2. MERFISH 10K: `--x <merfish_10k_x_path> --y <merfish_10k_y_path>`
3. MERFISH 50K (stretch): `--x <merfish_50k_x_path> --y <merfish_50k_y_path>`

All runs use `--k 15 --iters 5 --warmup 2 --stderr-capture <path>`. Exports `RAYON_NUM_THREADS=16`. Records hardware info (nproc, cpu model) to `results/hardware_profile.txt`.

**`scripts/analyze_results.py`**: Reads profiler JSON files, extracts `step_timing` arrays (using key `y_dist` for current runs), computes per-step mean/std/CI, produces a side-by-side comparison table (MERFISH vs Gaussian vs historical reference). Outputs a markdown summary to `results/analysis/comparison_table.md` and prints the primary verdict (x_space_pct with CI).

## Environment

**No custom environment needed.** The project's existing toolchain is sufficient:

| Component | Version | Status |
|-----------|---------|--------|
| Rust (nightly) | 1.96.0 | Available — builds tw_profiler with cli,profiling features |
| Python 3 | 3.13.2 | Available — numpy, scipy installed system-wide |
| NumPy | 2.2.6 | Available — npy I/O and analysis |
| SciPy | 1.15.2 | Available — t-distribution CIs |
| jq | 1.7 | Available — ad-hoc JSON inspection |

No `environment.yml` will be created. All dependencies are present in the base system.

## Implementation Phases

### Phase 1: Directory Structure and Input Verification

**Files to create:**
- `research/2026-04-08-tw-merfish-step-timing/` directory tree (scripts/, data/gaussian/, results/profiler/, results/analysis/)
- `scripts/verify_inputs.py`

**Actions:**
1. Create the directory structure
2. Write `verify_inputs.py` — uses numpy to load each MERFISH fixture with `mmap_mode='r'`, prints shape and dtype
3. Run: `python3 scripts/verify_inputs.py`
4. Confirm d_x=50, dtype=float64, n=10000/50000 for the four fixtures

**Acceptance:** All four MERFISH fixtures are readable and report expected shapes. d_x is recorded.

### Phase 2: Data Generation

**Files to create:**
- `scripts/gen_gaussian_baseline.py`
- `data/gaussian/gaussian_n10k_x.npy` (generated)
- `data/gaussian/gaussian_n10k_y.npy` (generated)

**Actions:**
1. Write `gen_gaussian_baseline.py`:
   - `rng = np.random.default_rng(seed=2026)`
   - X = `rng.standard_normal((10000, 10))` → float64
   - Y = `rng.uniform(0.0, 1.0, (10000, 2))` → float64
   - Save to `data/gaussian/`
2. Run: `python3 scripts/gen_gaussian_baseline.py`
3. Verify output shapes

**Acceptance:** Gaussian files exist with correct shapes (10000, 10) and (10000, 2), dtype float64.

### Phase 3: Profiler Script

**Files to create:**
- `scripts/run_profiler.sh`

**Actions:**
1. Write `run_profiler.sh`:
   - Set `RAYON_NUM_THREADS=16`
   - Build: `cargo build --release --features cli,profiling --bin tw_profiler`
   - Record hardware: `nproc`, `lscpu | grep "Model name"` → `results/hardware_profile.txt`
   - Run Gaussian 10K: output to `results/profiler/gaussian_n10k.json`, stderr to `results/profiler/stderr_gaussian_n10k.txt`
   - Run MERFISH 10K: output to `results/profiler/merfish_n10k.json`, stderr to `results/profiler/stderr_merfish_n10k.txt`
   - Run MERFISH 50K (stretch): output to `results/profiler/merfish_n50k.json`, stderr to `results/profiler/stderr_merfish_n50k.txt`
   - All runs: `--k 15 --iters 5 --warmup 2`
2. **Do NOT pass `--variant` flag** — the current `tw_profiler` binary does not accept it; it profiles the production code path directly

**Acceptance:** Script is syntactically valid and references correct file paths.

### Phase 4: Analysis Script

**Files to create:**
- `scripts/analyze_results.py`

**Actions:**
1. Write `analyze_results.py`:
   - Load JSON files from `results/profiler/`
   - Extract `step_timing` dict; use keys `x_dist`, `x_sort`, `y_dist`, `penalty` (NOT `y_heap`)
   - For each configuration, compute per-step: mean_ns, std_ns, fraction (mean_ns / sum_of_means)
   - Compute x_space_pct = (x_dist_mean + x_sort_mean) / total_mean × 100
   - Compute 95% CI using scipy.stats.t.interval with df = len(iters) - 1
   - Produce comparison table: Gaussian 10K | MERFISH 10K | MERFISH 50K | Historical reference
   - Include the historical flat_simd reference from `research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_flat_simd_n10000.json` (mapping `y_heap` → `y_dist`)
   - Write markdown table to `results/analysis/comparison_table.md`
   - Print primary verdict: x_space_pct with CI for MERFISH 10K

**Acceptance:** Script runs on sample JSON and produces correctly formatted output.

### Phase 5: Dry Run

**Actions:**
1. Run `scripts/run_profiler.sh` with modified parameters: `--iters 2 --warmup 1` on Gaussian 10K only
2. Verify JSON output contains `step_timing` with all four keys (x_dist, x_sort, y_dist, penalty)
3. Run `scripts/analyze_results.py` on the dry-run output
4. Verify comparison table is produced with correct formatting
5. **Check within-run CV (per RT-3):** If CV of step fractions > 15% across the 2 dry-run iters, increase iters for the full run (up to 10). If CV is acceptable, proceed with iters=5.
6. Delete dry-run outputs (they are not part of the final results)

**Acceptance:** End-to-end pipeline produces valid JSON with step_timing, analysis script generates comparison table. Iteration count is validated or adjusted.

## Execution Protocol

After implementation is complete and dry run passes:

1. **Clean state:** Ensure no other CPU-intensive processes are running
2. **Set environment:**
   ```bash
   export RAYON_NUM_THREADS=16
   ```
3. **Run the full profiler sweep:**
   ```bash
   cd research/2026-04-08-tw-merfish-step-timing
   bash scripts/run_profiler.sh
   ```
   Expected runtimes:
   - Gaussian 10K: ~1-2 minutes (7 iterations × ~0.13s each)
   - MERFISH 10K: ~2-5 minutes (higher d_x = more x_dist time)
   - MERFISH 50K: ~30-120 minutes (25x scaling from n² complexity)
4. **Run analysis:**
   ```bash
   python3 scripts/analyze_results.py
   ```
5. **Inspect results:**
   ```bash
   cat results/analysis/comparison_table.md
   ```
6. **Record verdict:** Is x_space_pct ≥ 50% on MERFISH 10K?

## Analysis Plan

### Primary Analysis

Compute x_space_pct = (mean(x_dist) + mean(x_sort)) / (mean(x_dist) + mean(x_sort) + mean(y_dist) + mean(penalty)) × 100 for each configuration.

Compare MERFISH 10K x_space_pct against:
1. The 50% threshold (H0 boundary)
2. The fresh Gaussian 10K x_space_pct (same-session baseline)
3. The historical flat_simd Gaussian 10K x_space_pct (56.2%)

### Confidence Intervals

For R=5 timed iterations (df=4): if between-run std = 3pp, 95% CI half-width = 3 × 2.776 / sqrt(5) = 3.7pp. This provides informative bounds for the primary question.

For R=3 (if iters are reduced): df=2, half-width = 3 × 4.30 / sqrt(3) = 7.4pp. Less informative but still useful for a 20-30pp expected effect.

**Post-hoc threshold acknowledgment (per W24):** The 50% threshold is not a pre-registered decision boundary. It is a post-hoc guidepost anchored to the Gaussian baseline result (56.2%). The report must not present this threshold as a pre-specified statistical decision rule.

### Analytical Prediction

Before examining results, compute a rough FLOP-ratio prediction:
- At d_x=50, d_y=2: x_dist FLOPs are 25x y_dist FLOPs per row
- AVX2 throughput: x_dist processes ~8 f64/cycle, y_dist batched kernel ~4 f64/cycle (2 points × 2 dims)
- Predicted x_dist share increase relative to d_x=10 baseline: approximately proportional to d_x ratio (50/10 = 5x more x_dist FLOPs, with similar y_dist/penalty costs)
- This predicts x_dist fraction should increase substantially from its d_x=10 value of 31.9%

Compare measured x_dist_pct against this prediction to validate the FLOP model.

### Joint Interpretation (per W27)

The analysis inspects 5 DVs (x_space_pct + 4 individual fractions) across 2-3 configurations. Individual CI non-overlaps should be interpreted jointly as a pattern, not as 5-15 independent statistical findings.

## Success Criteria

- **Conclusive positive (H1 supported):** x_space_pct on MERFISH 10K is ≥ 50%, with 95% CI lower bound > 40%. X-space ANN optimization is validated as the productive target for this MERFISH configuration.
- **Conclusive negative (H0 supported):** x_space_pct on MERFISH 10K is < 50%, with 95% CI upper bound < 55%. X-space is not dominant on this data; further investigation of the actual bottleneck is needed before committing to X-space ANN work.
- **Inconclusive:** 95% CI for x_space_pct on MERFISH 10K spans both sides of 50% (e.g., CI = [45%, 58%]), or CI half-width > 15pp (per RT-1). The measurement is not informative enough to draw a conclusion.
- **Informative uncertainty (per RT-1):** At least one configuration must produce a 95% CI half-width for x_space_pct < 15pp for the experiment to be considered informative.
- **Structural completeness:**
  - d_x of MERFISH fixtures is recorded (expected: 50)
  - All four step fractions are present in JSON output for every run
  - Comparison table (MERFISH vs Gaussian vs historical) is produced
  - Explicit verdict on X-space ANN optimization target validity

**Replicate policy (per RT-4):** If the initial R=5 result is inconclusive, replicates will NOT be extended in this experiment. A follow-up experiment with R=10+ would be designed separately.

## Threats to Validity

### Internal

1. **Dimensionality confound (W1-W4):** The Gaussian baseline uses d_x=10 while MERFISH uses d_x=50. Any difference in step fractions conflates data geometry and dimensionality effects. This experiment does not attempt to isolate these — it answers whether X-dominance holds on this specific MERFISH configuration, not why.

2. **Thread-ns proxy bias (W8-W11):** Step timings are thread-summed nanoseconds, not wall-clock partitions. SIMD-heavy steps (x_dist) accumulate more compute per nanosecond than scalar steps (penalty). The compute-share metric is a proxy for the actual optimization opportunity, with an unquantified bias. The direction of bias favors x_dist (SIMD-efficient step appears proportionally larger in thread-ns than in wall-clock share).

3. **Execution order effects (W5-W7):** Fixed sequential execution (Gaussian → MERFISH 10K → MERFISH 50K) creates page-cache warming, thermal accumulation, and first-replicate I/O asymmetries. Estimated magnitude: a few percent, small relative to expected effect sizes.

4. **Evaluation collision (RT-5):** The profiling instrumentation is embedded in the measured function. Timing overhead from `Instant::now()` and `fetch_add` is present in both the measurement and the measured code. This is inherent and accepted.

5. **HARKing risk (RT-6):** The directional hypothesis and post-hoc thresholds create risk of post-hoc rationalization. Mitigation: the report must carry forward the acknowledgment that thresholds are exploratory guideposts, not pre-registered decision boundaries.

### External

1. **Single-dataset limitation (R1):** Results apply to this specific MERFISH dataset (mouse hypothalamic preoptic region, PCA-50, n=10K/50K). They do not generalize to other tissue types, assays, gene panels, or PCA dimensionalities without additional experiments.

2. **Hardware specificity:** Step fractions depend on CPU microarchitecture, cache hierarchy, and SIMD instruction set. Results on this 16-core system may differ on ARM, different Intel/AMD generations, or systems with different L1/L2/L3 cache sizes.

3. **Single k value:** Only k=15 is tested. The penalty step scales as O(n²·k); at large k, penalty's fraction would grow relative to distance steps. The X-dominance conclusion is specific to k=15.

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| Disk space | ~50 MB (data) + ~5 MB (results) |
| Gaussian 10K runtime | ~1-2 minutes |
| MERFISH 10K runtime | ~2-5 minutes (higher d_x) |
| MERFISH 50K runtime | 30-120 minutes (n² scaling) |
| Total compute time | ~35-130 minutes including stretch |
| Build time | ~1-3 minutes (release build with features) |
| Analysis time | < 1 minute |
| Python dependencies | numpy, scipy (both installed) |
| No additional hardware or cloud resources required |
