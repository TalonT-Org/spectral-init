# Experiment Plan: Trustworthiness Step-Timing Validation on MERFISH Real Data

## Motivation

PR #238 established that X-space computation (x_dist + x_sort) dominates trustworthiness runtime on synthetic Gaussian data (~56% at n=10K), identifying X-space ANN as the productive next optimization target. However, this conclusion was drawn exclusively from synthetic data with known d_x=10 and uniform/Gaussian geometry. Before committing engineering effort to X-space ANN optimization, we need to verify that the step-timing breakdown holds on real biological data with different dimensionality, distribution shape, and clustering structure. This experiment measures step fractions on MERFISH spatial transcriptomics data and compares them against a fresh synthetic Gaussian baseline collected under identical conditions.

**Design revision context:** This plan incorporates revisions addressing 8 critical and 4 warning findings from design review (revision guidance dated 2026-04-07 17:00:12). Key changes from the original scope:

| Finding | Resolution |
|---------|------------|
| #1 Thread-ns vs wall-clock proxy | Metric explicitly framed as compute-share proxy; limitation documented |
| #2 Baseline asymmetry | Fresh Gaussian baseline re-run under identical conditions |
| #3 Warmup contamination | Analysis strips first `warmup` entries from step_timing vectors |
| #4 No statistical justification | Exploratory framing; distributional reporting with CI, no binary threshold |
| #5 Single process invocation | R=3 independent process invocations per configuration |
| #6 n=10K understates at scale | n=50K is required (not stretch) |
| #7 d_x ambiguity | d_x resolved from .npy header as prerequisite gate before profiling |
| #8 Reproducibility | Commit hash, toolchain, CPU, file checksums all recorded |
| #9 Hypothesis structure | Single exploratory framing replaces compound H0/H1 |
| #10 Causal mechanism | Conclusions scoped to observed difference, not causal attribution |
| #11 n=50K triggering | Required, not conditional |
| #12 Generalization boundary | Explicit scope: one dataset, one hardware, one tissue type |

## Hypothesis

This experiment is **exploratory/descriptive**, not confirmatory. Per revision guidance finding #4, the synthetic baseline was observed before threshold selection, making pre-registered hypothesis testing inappropriate.

**Research question:** What are the trustworthiness step-timing fractions on MERFISH real data, and how do they compare to synthetic Gaussian data profiled under identical conditions?

**Expected outcome (not a formal H1):** X-space compute share (x_dist + x_sort) on MERFISH data is qualitatively similar to or greater than the synthetic Gaussian baseline (~56% at n=10K), because x_dist cost scales with d_x and MERFISH d_x is likely >= 10.

**Null expectation:** Step fractions on MERFISH data fall within the between-run variability of the synthetic Gaussian baseline — i.e., data geometry does not measurably shift the compute-share distribution.

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Dataset | MERFISH n=10K, MERFISH n=50K, Gaussian n=10K | Real vs synthetic comparison; n=50K tests scale stability |
| n (sample size) | 10,000 and 50,000 | Addresses finding #6: n=10K may understate X-dominance at production scale |

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| x_dist compute share | % of total thread-aggregate ns | Derived: x_dist_ns / sum(all_step_ns) × 100 | NEW — derived post-hoc from profiling counters |
| x_sort compute share | % of total thread-aggregate ns | Same derivation | NEW — derived |
| y_dist compute share | % of total thread-aggregate ns | Same derivation | NEW — derived |
| penalty compute share | % of total thread-aggregate ns | Same derivation | NEW — derived |
| x_space_pct | % | (x_dist + x_sort) / sum(all) × 100 | NEW — primary comparison metric |
| wall_time | seconds | `mean_s` field in tw_profiler JSON output | EXISTS as `mean_s` in tw_profiler.rs:49-73 |
| step_fraction_cv | dimensionless | std(x_space_pct across timed iters) / mean(x_space_pct) | NEW — per-run variance indicator |
| between_run_std | pp (percentage points) | std of per-run x_space_pct means across R=3 replicate invocations | NEW — between-run variance indicator |

**Metric interpretation caveat (addresses finding #1, RT-1):** All step-share metrics are computed from **thread-aggregate nanoseconds** — the sum of per-row `Instant` timings across all Rayon threads. This measures compute density, not wall-clock contribution. Thread-aggregate share equals wall-clock share only when all steps have equal parallelization efficiency. Since x_dist and y_dist use different SIMD kernels (8-wide AVX2+FMA for x_dist at d_x>=10; batched 2-point kernel for y_dist at d_y=2), their per-thread throughput differs. The thread-ns fraction is the best available proxy without modifying the profiler to add per-step wall-clock instrumentation. Results should be interpreted as "compute-share" not "wall-clock share."

All metrics marked NEW are derived in the analysis script — no changes to `src/metrics.rs` are required. No new canonical metrics are added to the codebase.

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (nearest neighbors) | 15 | Default for trustworthiness; matches all prior baselines |
| warmup iterations | 2 | Sufficient for JIT/cache warmup; kept small to minimize total runtime |
| timed iterations | 5 | Per-invocation sample size; variance characterized via R=3 replicate invocations |
| replicate invocations (R) | 3 | Independent process launches for between-run variance (addresses finding #5) |
| RAYON_NUM_THREADS | unset (system default = 16) | All configurations use same thread pool; recorded in environment log |
| Binary build | Single `cargo build --release --features cli,profiling` at pinned commit | Same binary for all runs (addresses finding #2) |
| Machine state | Sequential runs in single session, no concurrent heavy workloads | Best-effort control; between-run variance captures residual jitter |

**Baseline comparability (addresses finding #2, RT-3):** The synthetic Gaussian baseline is collected **fresh** in this experiment using the same binary, same machine, same session, same warmup/iter counts. Historical baselines are referenced for context only, not used in the primary comparison.

## Inputs and Data

### MERFISH Fixtures (existing)

| Dataset | Path | Expected Shape | Purpose |
|---------|------|----------------|---------|
| merfish_n10k_x | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | (10000, d_x) — d_x TBD | X-space input, primary comparison |
| merfish_n10k_y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | (10000, 2) | Y-space input (2D UMAP embedding) |
| merfish_n50k_x | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | (50000, d_x) | X-space input, scale test |
| merfish_n50k_y | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | (50000, 2) | Y-space input, scale test |

### Synthetic Gaussian Fixtures (existing)

| Dataset | Path | Shape | Purpose |
|---------|------|-------|---------|
| gaussian_n10k_x | `research/2026-04-05-tw-perf-rerun-clean/data/gaussian/gaussian_n10000_x.npy` | (10000, 10) f64 | Fresh baseline X-space input |
| gaussian_n10k_y | `research/2026-04-05-tw-perf-rerun-clean/data/gaussian/gaussian_n10000_y.npy` | (10000, 2) f64 | Fresh baseline Y-space input |

These fixtures are true standard-normal data generated with `np.random.default_rng(seed=2026).standard_normal(...)`, d_x=10, d_y=2.

### d_x Resolution (prerequisite gate — addresses finding #7)

The actual d_x of the MERFISH fixtures is unknown. File-size analysis suggests ~48 dimensions (at f64) or ~97 (at f32), which is inconsistent with the typical MERFISH PCA range of 8-20. Phase 1 resolves this by reading the .npy header. The experiment proceeds regardless of d_x value, but the d_x value is critical context for interpreting x_dist fractions.

**Decision point (RT-4):** The MERFISH dataset was selected because it was the only real-data fixture already available in the repository, not because preliminary results suggested any particular outcome. Dataset selection is convenience-based.

### Generalization boundary (addresses finding #12)

This experiment covers exactly one biological dataset (MERFISH hypothalamic spatial transcriptomics), one tissue type, one PCA reduction, one hardware configuration (AMD Ryzen 7 9800X3D), and k=15. Results do not generalize to other biological datasets, tissue types, dimensionality ranges, or hardware without additional coverage.

## Experiment Directory Layout

```
research/2026-04-07-tw-merfish-step-timing/
├── scripts/
│   ├── record_environment.sh     # Record commit, toolchain, CPU, file checksums
│   ├── check_dimensions.py       # Read .npy headers; report shape, dtype, d_x
│   ├── run_profiler.sh           # Build tw_profiler; run R=3 invocations per dataset
│   └── analyze_fractions.py      # Parse JSON, strip warmup, compute fractions, compare
├── data/                         # Symlinks to fixture files (no data duplication)
│   ├── merfish_n10k_x.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy
│   ├── merfish_n10k_y.npy -> ...
│   ├── merfish_n50k_x.npy -> ...
│   ├── merfish_n50k_y.npy -> ...
│   ├── gaussian_n10k_x.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/gaussian/gaussian_n10000_x.npy
│   └── gaussian_n10k_y.npy -> ../../2026-04-05-tw-perf-rerun-clean/data/gaussian/gaussian_n10000_y.npy
├── results/
│   ├── environment.json          # Reproducibility anchors
│   ├── dimensions.json           # .npy shape/dtype for all fixtures
│   ├── profiler/                 # Per-dataset, per-replicate JSON + stderr
│   │   ├── merfish_10k_rep1.json
│   │   ├── merfish_10k_rep1_stderr.txt
│   │   ├── merfish_10k_rep2.json
│   │   ├── merfish_10k_rep2_stderr.txt
│   │   ├── merfish_10k_rep3.json
│   │   ├── merfish_10k_rep3_stderr.txt
│   │   ├── merfish_50k_rep1.json
│   │   ├── merfish_50k_rep1_stderr.txt
│   │   ├── merfish_50k_rep2.json
│   │   ├── merfish_50k_rep2_stderr.txt
│   │   ├── merfish_50k_rep3.json
│   │   ├── merfish_50k_rep3_stderr.txt
│   │   ├── gaussian_10k_rep1.json
│   │   ├── gaussian_10k_rep1_stderr.txt
│   │   ├── gaussian_10k_rep2.json
│   │   ├── gaussian_10k_rep2_stderr.txt
│   │   ├── gaussian_10k_rep3.json
│   │   └── gaussian_10k_rep3_stderr.txt
│   ├── analysis.json             # Structured analysis output
│   └── comparison_table.md       # Human-readable comparison table
└── report.md                     # Final report (written by write-report skill)
```

### File Descriptions

- **`scripts/record_environment.sh`**: Captures `git rev-parse HEAD`, `rustc --version`, `cargo --version`, CPU model from `/proc/cpuinfo`, core count, `RAYON_NUM_THREADS` value (or "unset"), and SHA256 checksums of all input .npy files and the built `tw_profiler` binary. Writes JSON to `results/environment.json`.

- **`scripts/check_dimensions.py`**: Reads .npy file headers (not full arrays) using `numpy.lib.format.read_magic` and `read_array_header_*` for each fixture file. Reports shape, dtype, and computed d_x. Writes JSON to `results/dimensions.json`. This is a **prerequisite gate** — if d_x cannot be determined, the experiment cannot be interpreted.

- **`scripts/run_profiler.sh`**: Builds `tw_profiler` once (`cargo build --release --features cli,profiling --bin tw_profiler`). Then for each dataset configuration (merfish_10k, merfish_50k, gaussian_10k) and each replicate (1..R), invokes `tw_profiler` as a fresh process with `--k 15 --warmup 2 --iters 5 --stderr-capture`. Each invocation produces one JSON file and one stderr capture file. Total: 9 independent process invocations (3 datasets × 3 replicates).

- **`scripts/analyze_fractions.py`**: For each JSON file: reads `step_timing` vectors, strips the first `warmup=2` entries (warmup contamination fix — addresses finding #3, RT-7), computes per-iteration step fractions from the remaining `iters=5` timed entries, then computes per-run mean fractions. Across R=3 replicates: computes between-run mean, std, and 95% CI for each step fraction and for x_space_pct. Produces `results/analysis.json` (structured) and `results/comparison_table.md` (human-readable). Handles the `y_heap`→`y_dist` key rename when reading historical baselines for context.

## Environment

**No custom environment needed.**

The project's existing toolchain is sufficient:

- **Rust**: rustc 1.96.0-nightly (23903d01c 2026-03-26) — present, builds with `--features cli,profiling`
- **Python**: 3.13.2 with numpy 2.2.6 — sufficient for .npy header reading and analysis
- **Hardware**: AMD Ryzen 7 9800X3D (8 cores / 16 threads), recorded in environment log

No environment.yml will be created. Prior experiments used conda environments to pin older Python/scipy versions for specific analysis needs; this experiment requires only numpy for .npy I/O and basic arithmetic, which the system Python satisfies.

## Implementation Phases

### Phase 1: Directory Structure and Prerequisite Gate

1. Create `research/2026-04-07-tw-merfish-step-timing/` and subdirectories (`scripts/`, `data/`, `results/`, `results/profiler/`)
2. Create symlinks in `data/` pointing to MERFISH and Gaussian fixture files
3. Create `scripts/record_environment.sh`:
   - Record git commit hash (`git rev-parse HEAD`)
   - Record `rustc --version`, `cargo --version`
   - Record CPU model (`grep 'model name' /proc/cpuinfo | head -1`)
   - Record core count (`nproc`)
   - Record `RAYON_NUM_THREADS` value or "unset"
   - Compute SHA256 of all 6 input .npy files
   - Write all to `results/environment.json`
4. Create `scripts/check_dimensions.py`:
   - For each .npy file in `data/`, read the header and extract shape and dtype
   - Compute d_x for each X-space file
   - Write structured output to `results/dimensions.json`
   - Print summary to stdout
5. Run `record_environment.sh` and `check_dimensions.py`
6. **Gate check**: Verify d_x was successfully read for all fixtures. Record actual d_x value. If d_x cannot be determined, stop and investigate.

**Deliverables**: `results/environment.json`, `results/dimensions.json`, populated `data/` symlinks.

### Phase 2: Build and Dry Run

1. Build: `cargo build --release --features cli,profiling --bin tw_profiler`
2. Record SHA256 of `target/release/tw_profiler` into `results/environment.json`
3. Create `scripts/run_profiler.sh`:
   - Accept parameters: `WARMUP=2`, `ITERS=5`, `REPS=3`
   - Define dataset configurations:
     - `merfish_10k`: `--x data/merfish_n10k_x.npy --y data/merfish_n10k_y.npy`
     - `merfish_50k`: `--x data/merfish_n50k_x.npy --y data/merfish_n50k_y.npy`
     - `gaussian_10k`: `--x data/gaussian_n10k_x.npy --y data/gaussian_n10k_y.npy`
   - For each dataset and replicate: invoke `tw_profiler` as a fresh process
   - Output naming: `results/profiler/${dataset}_rep${r}.json` and `_stderr.txt`
4. **Dry run**: Execute run_profiler.sh with `REPS=1` and `ITERS=1` on gaussian_10k only to verify:
   - Binary runs without error
   - JSON output contains `step_timing` with non-zero values
   - stderr capture file contains `[timing:...]` lines
   - Expected number of timing line sets (warmup + iters = 3 sets of 4 lines)

**Deliverables**: Built binary, `scripts/run_profiler.sh`, dry-run validation.

### Phase 3: Analysis Script

1. Create `scripts/analyze_fractions.py`:
   - Read all JSON files from `results/profiler/`
   - For each file:
     - Extract `step_timing` dict (keys: `x_dist`, `x_sort`, `y_dist`, `penalty`)
     - Strip first `WARMUP` entries from each vector (addresses finding #3)
     - Compute per-iteration step fractions: `step_ns / sum(all_step_ns) × 100`
     - Compute per-iteration x_space_pct: `(x_dist + x_sort) / sum(all) × 100`
     - Compute within-run mean and std for each metric
   - Group by dataset configuration
   - For each dataset: compute between-run mean, std, 95% CI from R=3 per-run means
   - Produce comparison table: MERFISH 10K vs Gaussian 10K vs MERFISH 50K
   - Include historical baseline (2026-04-06 `profiler_baseline_n10000.json`) for context only — clearly labeled as different collection conditions, mapping `y_heap`→`y_dist`
   - Write `results/analysis.json` (all structured data)
   - Write `results/comparison_table.md` (formatted table)
2. **Dry run**: Run analyze_fractions.py on dry-run output from Phase 2 to verify parsing and output format

**Deliverables**: `scripts/analyze_fractions.py`, validated output format.

### Phase 4: Full Experiment Execution

1. Execute `scripts/run_profiler.sh` with full parameters (`WARMUP=2`, `ITERS=5`, `REPS=3`)
   - Gaussian 10K: 3 invocations × 7 calls each ≈ fast (seconds per invocation)
   - MERFISH 10K: 3 invocations × 7 calls each ≈ moderate (depends on d_x)
   - MERFISH 50K: 3 invocations × 7 calls each ≈ slower (~25× the 10K runtime due to O(n²))
2. Execute `scripts/analyze_fractions.py` on full results
3. Review `results/comparison_table.md` for completeness

**Deliverables**: All 18 profiler JSON files (9 datasets × 2 files each), `results/analysis.json`, `results/comparison_table.md`.

## Execution Protocol

All commands are run from the experiment directory: `research/2026-04-07-tw-merfish-step-timing/`.

```bash
# Phase 1: Environment and dimensions
bash scripts/record_environment.sh
python3 scripts/check_dimensions.py
# GATE: verify dimensions.json shows valid d_x for all fixtures

# Phase 2: Build and dry run
cargo build --release --features cli,profiling --bin tw_profiler
# Record binary checksum (handled by record_environment.sh re-run after build)
bash scripts/record_environment.sh  # re-run to capture binary hash
# Dry run: single rep, single iter, gaussian only
REPS=1 ITERS=1 DATASETS="gaussian_10k" bash scripts/run_profiler.sh
python3 scripts/analyze_fractions.py --input-dir results/profiler --warmup 2 --dry-run
# Verify: JSON has non-zero step_timing, analysis parses correctly

# Phase 3: not a separate execution step — scripts already created above

# Phase 4: Full execution
REPS=3 ITERS=5 DATASETS="gaussian_10k merfish_10k merfish_50k" bash scripts/run_profiler.sh
python3 scripts/analyze_fractions.py --input-dir results/profiler --warmup 2 \
  --output-json results/analysis.json --output-table results/comparison_table.md
```

**Runtime estimates:**
- Gaussian 10K: ~5-10 seconds per invocation × 3 reps ≈ 30 seconds
- MERFISH 10K: ~10-60 seconds per invocation × 3 reps ≈ 3 minutes (d_x-dependent)
- MERFISH 50K: ~250-1500 seconds per invocation × 3 reps ≈ 15-75 minutes (O(n²) scaling)
- Total: ~20-80 minutes depending on d_x

## Analysis Plan

### Primary analysis: Step-fraction comparison

For each dataset configuration, report the mean step fractions (x_dist%, x_sort%, y_dist%, penalty%) and x_space_pct with 95% confidence intervals derived from R=3 independent process invocations. Present as a table:

| Dataset | d_x | x_dist% | x_sort% | y_dist% | penalty% | x_space_pct% | 95% CI |
|---------|-----|---------|---------|---------|----------|--------------|--------|

Compare MERFISH 10K against Gaussian 10K (fresh, same-session baseline). If CIs overlap, the difference is not detectable with this sample size. If CIs are separated, the difference is robust to between-run variance.

### Secondary analysis: Scale stability

Compare MERFISH 10K vs MERFISH 50K to assess whether step fractions shift with n. Under the O(n²) cost model, penalty's O(n²·k) term should grow relative to distance O(n²·d) terms only through k vs d ratio, which is constant across n. Any observed shift indicates cache effects, scheduling overhead, or other non-asymptotic factors.

### Contextual reference (not primary comparison)

The historical baseline (`research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_baseline_n10000.json`) is reported for context only, clearly labeled with its collection conditions (different binary version, --warmup 5 --iters 30, y_heap key). It is NOT used for the primary MERFISH vs. Gaussian comparison (addresses finding #2).

### What this analysis cannot do (addresses finding #10, RT-8)

MERFISH data differs from synthetic Gaussian in at least three confounded dimensions: (1) d_x (possibly 10 vs ~48), (2) distribution shape (gene expression vs. standard normal), (3) Y-space clustering structure (5000+ biological clusters vs. 8-blob Gaussian mixture). Any observed difference in step fractions is attributed to "MERFISH data geometry as a whole," not to any single mechanism. The theoretical FLOP-count relationship (x_dist share scales with d_x / (d_x + d_y)) is reported as a qualitative sanity check, not a falsifiable causal claim (addresses RT-6).

### Instrumentation overhead assessment (addresses RT-5)

Per-row instrumentation cost is `Instant::now()` (~20-30ns on x86) + `AtomicU64::fetch_add` (~5-10ns). For x_dist at d_x=10 with AVX2, per-row compute is O(n·d_x/8) FMA cycles ≈ microseconds at n=10K. Instrumentation overhead is <1% of per-row step time for all steps at n>=10K. Asymmetric overhead between steps is negligible and is not corrected for.

## Success Criteria

**Primary:**
- d_x for all MERFISH fixtures confirmed with actual values from .npy headers
- Step fractions measured for all 3 dataset configurations with R=3 independent replicate invocations each
- Comparison table produced showing MERFISH vs. fresh Gaussian baseline with between-run 95% CIs
- Qualitative verdict: does X-space compute share on MERFISH data exceed, match, or fall below the fresh Gaussian baseline?

**Secondary:**
- Scale stability assessed: MERFISH 10K vs 50K step fractions compared
- Between-run coefficient of variation (CV) for x_space_pct reported for all configurations

**What constitutes each outcome:**
- **Conclusive — X-space dominance confirmed:** x_space_pct on MERFISH 10K has mean >= 50% with lower bound of 95% CI > 45%. The qualitative conclusion from PR #238 extends to this MERFISH dataset.
- **Conclusive — X-space dominance not confirmed on this data:** x_space_pct on MERFISH 10K has mean < 45% with upper bound of 95% CI < 50%. X-space ANN may still be productive at larger n or higher d_x, but this specific dataset does not confirm the PR #238 conclusion.
- **Inconclusive:** 95% CI for x_space_pct on MERFISH 10K spans 45-55% (straddles the midpoint). Between-run variance is too large relative to the effect size. Would require more replicates or longer runs to resolve.

**Note (addresses finding #4, RT-2):** The 45% and 50% values above are interpretive guideposts, not pre-registered statistical thresholds. They were chosen after observing the 56.1% synthetic baseline. This experiment is exploratory — the goal is to estimate x_space_pct on MERFISH data with quantified uncertainty, not to perform a formal hypothesis test.

## Threats to Validity

### Internal

1. **Thread-aggregate vs. wall-clock (finding #1):** Thread-ns fractions are a proxy for optimization ROI. Steps with higher SIMD throughput contribute more thread-ns per wall-clock second. The proxy is valid for ranking steps by compute density, but may over-count well-parallelized steps relative to their wall-clock contribution. No correction is applied because per-step wall-clock timing is not available without profiler modifications.

2. **Warmup contamination (finding #3):** Mitigated by stripping the first `warmup` entries from each step_timing vector. Since AtomicU64 counters reset at the top of each `trustworthiness()` call (commit 578ea5b), each entry in the vector is independent. The ordering is deterministic: first `warmup` entries are warmup, last `iters` entries are timed.

3. **Within-run correlation (finding #5):** The R=3 independent process invocations provide between-run variance estimates. Within-run iterations (n=5) share process state (thread pool, TLB, thermal state) and are not treated as independent samples. Between-run means are the unit of analysis.

4. **Instrumentation overhead (RT-5):** ~30-40ns per-row overhead from `Instant::now()` + `fetch_add` is <1% of per-row step compute time at n>=10K. Asymmetric overhead between steps is negligible because all steps use the same instrumentation pattern (one timer start + one fetch_add per row per step).

5. **Profiling feature overhead on non-profiled steps:** The `#[cfg(feature = "profiling")]` blocks add code to the hot loop. This increases instruction cache pressure relative to a non-profiling build. Step fractions are measured with profiling enabled; absolute wall-clock times may differ from production builds. Fractions (ratios) are less affected than absolutes because the overhead is approximately proportional across steps.

### External

1. **Single dataset (finding #12):** MERFISH hypothalamic data has specific properties (gene panel size, PCA dimensionality, cluster structure) that may not represent other biological datasets (scRNA-seq, CITE-seq, spatial transcriptomics from other tissues). Results apply to this dataset only.

2. **Single hardware:** AMD Ryzen 7 9800X3D has specific cache hierarchy (96MB L3 3D V-Cache) and SIMD capabilities (AVX2). Step fractions may differ on Intel hardware, ARM, or CPUs with different cache sizes.

3. **Scale limitation (finding #6):** n=10K and n=50K are below production scale for some applications (n=100K-1M). The penalty step's relative share may decrease at larger n due to its O(n²·k) cost growing slower than O(n²·d_x) for x_dist when k < d_x. The n=50K measurement partially addresses this but does not cover production-scale n.

4. **k=15 only:** Different k values change the penalty step's fraction (O(n²·k)). A multi-k sweep was considered but excluded to keep the experiment focused. If penalty fraction is unexpectedly large, a follow-up k-sweep may be warranted.

## Estimated Resource Requirements

- **Compute time:** 20-80 minutes total (dominated by MERFISH 50K runs)
- **Disk space:** ~50 MB (9 JSON files + 9 stderr captures + analysis output; input data accessed via symlinks)
- **Dependencies:** None beyond existing toolchain (Rust nightly 1.96.0, Python 3.13.2 + numpy 2.2.6)
- **No network access required** — all data is local
