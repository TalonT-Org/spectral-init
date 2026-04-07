# Experiment Plan: y_heap Bottleneck Optimization in Trustworthiness Computation

## Motivation

The `trustworthiness()` function in `src/metrics.rs` spends an estimated **70.3% of total
parallel thread-work** in the `y_heap` step — a per-row `BinaryHeap<(u64, usize)>` that
scans all n Y-space points to find the k nearest neighbors in embedding space. This was
measured in `research/2026-04-05-tw-perf-rerun-clean` using `AtomicU64` step counters
(summed CPU thread-time, see §Dependent Variables for the time-base distinction).

Prior optimization work (PR #226/229) applied thread-local buffers and AVX2 to the X-space
distance step (`x_dist`). The current baseline already incorporates those changes; the `x_dist`
step now accounts for only ~13% of thread-work. The `y_heap` step was explicitly flagged as
the unresolved dominant cost in those reports.

The prior rerun-clean experiment also attempted to apply thread-local optimization to y_heap
and measured **2× wall-clock slowdown** (0.634 s vs 0.313 s baseline at n=10K). This failure
is a critical prior result; the experiment must establish why it failed before committing to
the same strategy.

This experiment tests four precisely isolated variants to answer: (a) is BinaryHeap
**allocation** the bottleneck (variant: `heap_reuse`), (b) does replacing the heap with a
flat buffer + introselect help or hurt (variant: `flat_partial`), and (c) does a 2D AVX2
SIMD distance kernel on top of the flat buffer deliver meaningful throughput improvement
(variant: `flat_simd`). Results directly determine which — if any — y_heap optimization
should be shipped and whether to escalate to a KD-tree approach (H3).

---

## Hypothesis

**Null hypothesis (H0):** Optimizing the y_heap step in `trustworthiness()` yields no
statistically significant wall-time speedup. Formally: the Criterion 95% CI for the
wall-time speedup ratio (T_baseline / T_flat_simd) at n=10K, k=15 contains 1.0.

**Alternative hypothesis (H1_alt):** The combined variant (`flat_simd`: thread-local flat
buffer + `select_nth_unstable_by` + 2D AVX2 distance kernel) produces a Criterion 95% CI
lower bound strictly greater than 1.0 at n=10K, k=15 — meaning the optimization is reliably
faster than baseline with α=0.05.

**Stretch target (post-hoc, exploratory):** H1_alt with point estimate ≥ 1.5×, corresponding
to eliminating ~50% of y_heap thread-work. This threshold is derived post-hoc from the 70.3%
profiling result (RT-1 decision, see §Red-Team Decisions). It serves as an ambitious
reference point but does not gate H0 acceptance or rejection. **Any CI LB > 1.0 constitutes
a positive primary result.**

---

## Red-Team Decisions

The following decisions are made explicitly in response to the revision guidance. Each is
documented as a conscious choice, not an oversight.

### RT-1: 1.5× Threshold — Post-Hoc

The 1.5× target was derived after observing the 70.3% profiling result (1/0.648 ≈ 1.54×
under the approximation that 50% y_heap reduction maps linearly to wall-clock). This is
**post-hoc**. The threshold is retained as an exploratory reference only; H0 is rejected
solely by whether CI LB > 1.0 for the primary DV. The experiment is **exploratory-to-confirmatory**: it produces a decision (ship / do not ship / escalate to H3) based on the
sign and magnitude of the observed speedup, not by comparing against a pre-committed threshold.

### RT-2: Asymmetric Implementation Effort — Intentional

The `flat_simd` variant receives a custom 2D AVX2 kernel; the baseline receives no analogous
tuning. This is intentional. The research question is: "what does adding this specific
optimization to the existing production code yield?" The measured speedup is the deployment
value, not an abstract algorithmic comparison. This limit on generalizability is declared in
§Threats to Validity.

### RT-3: Seed 42 — Conventional

Seed 42 is the project-wide convention (used in `sklearn_parity_synthetic`, fixture generation
metadata). It was not selected by exploring multiple seeds. Sensitivity to seed choice is
**not tested** in this experiment; it is declared as an uncontrolled variable in §Threats to
Validity. A follow-on experiment could run with seeds {42, 7, 123, 1337} if sensitivity
becomes a concern.

### RT-4: Profiling Instrumentation Separated from Criterion

The `AtomicU64` step counters (summed CPU thread-time per step) are gated on the `profiling`
Cargo feature. Criterion benchmarks are compiled and run **without** the `profiling` feature:
the measurement infrastructure imposes zero overhead on the timed variants. The profiler runs
(`run_profiler.sh`) are executed as a **separate invocation** compiled with `--features
profiling`. The two result types are never merged — wall-time ratios come from Criterion,
step fractions come from the profiler. (See §Dependent Variables for time-base definitions.)

### RT-5: 10-Sample Budget — Accepted with Escalation Rule

SamplingMode::Flat with sample_size=10 is accepted as the default for time budget reasons.
Pre-specified escalation rule: **if after 10 samples the CI lower bound ≤ 1.0 but the point
estimate ≥ 1.1×, re-run the primary comparison only (baseline vs flat_simd) with
sample_size=50**. If CI LB > 1.0 after escalation, the result is classified as a weak
positive (meaningful but modest improvement). If CI LB ≤ 1.0 after escalation, result is
inconclusive for H0; escalate to H3 (KD-tree).

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Implementation variant | baseline, heap_reuse, flat_partial, flat_simd | Isolate: malloc cost (baseline→heap_reuse), heap vs introselect (heap_reuse→flat_partial), SIMD contribution (flat_partial→flat_simd) |
| Input size n | 1 000, 5 000, 10 000 | Matches existing Criterion bench; n=10K is the primary measurement; smaller sizes aid extrapolation |

**Variant definitions:**

- **baseline**: Current production `trustworthiness()` from `src/metrics.rs` — exactly as
  shipped. Already has thread-local X-space buffers (COMB_DIST_X, COMB_INDICES) and AVX2 for
  x_dist. Uses `BinaryHeap::with_capacity(k+1)` allocated fresh per row for y_heap. No change.

- **heap_reuse**: Thread-local `BinaryHeap<(u64, usize)>` initialized once per thread with
  `with_capacity(k+1)`, then `clear()`-d (not re-allocated) at the start of each row.
  Identical logic to baseline; only the allocation pattern changes. Diagnostic variant to
  isolate malloc cost.

- **flat_partial**: Thread-local `Vec<f64>` (COMB_DIST_Y, size n) and `Vec<usize>`
  (COMB_INDICES_Y, size n) initialized once per thread, reused each row. All Y squared
  distances written to COMB_DIST_Y; COMB_INDICES_Y extended with 0..n; then
  `select_nth_unstable_by(k, ...)` (same introselect pattern as the existing `x_sort` step).
  Self-exclusion: set `COMB_DIST_Y[i] = f64::INFINITY` instead of a conditional branch.
  No SIMD — scalar distance loop.

- **flat_simd**: All of flat_partial, plus a dedicated `dist_sq_2d_avx2_batch` kernel that
  fills COMB_DIST_Y using 256-bit AVX2 registers processing 4 Y-rows per iteration. Specialized
  for d_y=2 (fixed stride 2, no tail loop for the 2-element case). Self-exclusion handled by
  overwriting COMB_DIST_Y[i] after the batch fill.

---

## Dependent Variables (Metrics)

### Primary Dependent Variable (Confirmatory)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Wall-time speedup ratio: T_baseline / T_flat_simd | dimensionless ratio | Criterion bootstrap point estimate + 95% CI at n=10K | NEW — no entry in src/metrics.rs |

This is the **sole confirmatory DV**. H0 is rejected if and only if the 95% CI lower bound
strictly exceeds 1.0 for this ratio, evaluated at n=10K, k=15, seed=42.

No Bonferroni or family-wise error correction is applied because there is only one
confirmatory comparison. α=0.05 is the nominal Type I error rate for this comparison only.

**Note on CI construction**: Criterion's CI is constructed over raw wall-time measurements
for each variant independently. The speedup ratio is computed as `mean_baseline / mean_variant`
from Criterion's bootstrap distributions. The ratio CI's nominal coverage may differ from 95%
(ratio of two random variables); this is a known limitation of Criterion's reporting and is
documented in results as measurement uncertainty.

### Secondary Dependent Variables (Exploratory, Uncorrected)

All secondary DVs are exploratory. No significance criterion is pre-specified; results are
reported descriptively to support causal attribution.

| Metric | Unit | Collection Method | Purpose |
|--------|------|-------------------|---------|
| Speedup ratio: T_baseline / T_heap_reuse | ratio | Criterion, n=10K | Isolates malloc cost |
| Speedup ratio: T_baseline / T_flat_partial | ratio | Criterion, n=10K | Tests heap vs introselect |
| SIMD contribution: T_flat_partial / T_flat_simd | ratio | Criterion, n=10K | Isolates SIMD distance gain |
| Speedup ratio at n=5K, n=1K | ratio | Criterion | Scaling signal |
| y_heap step fraction of total thread-work | % | Profiler (profiling feature) | Causal attribution only |
| Correctness: |Δ trustworthiness| vs baseline | absolute f64 | Unit test assertions | Exact kNN invariance |

**Step fraction time-base clarification (addresses revision guidance §2):**
The `AtomicU64` step counters in the `profiling` feature accumulate nanoseconds via
`fetch_add(Relaxed)` across all Rayon worker threads for each row. This produces a **summed
CPU thread-time** (not wall-clock elapsed time). At T threads, the total across all six
steps can be up to T× the wall-clock elapsed time. The step fraction `y_heap_ns / total_ns`
is therefore a measure of "what share of total parallel thread-work is spent in y_heap" — not
"what share of wall-clock time is spent in y_heap." These are equivalent only in a
single-threaded execution.

This metric is **NOT used as a confirmatory gate**. It supports causal attribution:
- If `flat_simd` reduces y_heap fraction AND is faster overall, it is consistent with genuine
  y_heap improvement.
- If `flat_simd` is faster but y_heap fraction does not decrease, another step may have slowed
  (cache eviction, layout change), which would be a confound to investigate.
- If y_heap fraction decreases but total wall-time does not improve, overhead was added
  elsewhere.

The upper-bound approximation for wall-time speedup from y_heap elimination:
`1 / (1 - 0.703) ≈ 3.37×`, under the assumption that thread-work fraction equals wall-clock
fraction (valid only for perfectly parallel, no-synchronization execution). This is an
approximation; actual achievable speedup depends on Rayon scheduling and memory bandwidth.
The 1.5× stretch target corresponds to ~50% y_heap thread-work reduction, consistent with
this approximation.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k | 15 | Existing bench standard; prior research used this value |
| d_y | 2 | SIMD kernel is specialized for d_y=2; results do not generalize to d_y>2 (declared threat) |
| d_x | 10 | Existing bench standard; activates the x_dist AVX2 path (d_x ≥ 10 gate) |
| Seed | 42 | Project convention; RT-3 decision |
| Rayon thread count | RAYON_NUM_THREADS=$(nproc), recorded in results | Controls scheduling jitter; without fixation, two machines with different core counts produce structurally incomparable ratios. Actual value recorded in hardware_profile.txt |
| RUSTFLAGS | `-C target-cpu=native` | Already set in `.cargo/config.toml`; records expanded ISA in hardware_profile.txt |
| Rust toolchain | nightly-2026-03-26 | Pinned via `research/2026-04-06-y-heap-bottleneck-optimization/rust-toolchain.toml`; codegen differs across nightly dates |
| Criterion sampling mode | SamplingMode::Flat | Eliminates adaptive sampling variance |
| Criterion sample_size | 10 (escalation to 50 per RT-5) | Time-bounded; escalation rule defined |
| Criterion measurement_time | 10 s per benchmark | Bounds total Criterion runtime; avoids indefinite measurement under Flat sampling |
| Criterion warm_up_time | 10 s | Matches existing bench convention |

---

## Inputs and Data

The experiment requires synthetic datasets with known properties: uniform distribution in
high-dimensional X-space (d_x=10), uniform 2D embedding Y-space (d_y=2). No real data is
needed. The same seed and generation script ensures consistent comparison across runs.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| gaussian_n1000_x.npy, gaussian_n1000_y.npy | gen_data.py, seed 42 | n=1000, d_x=10, d_y=2, float64 | Criterion warm-up and small-scale CI |
| gaussian_n5000_{x,y}.npy | gen_data.py, seed 42 | n=5000, d_x=10, d_y=2, float64 | Mid-scale Criterion |
| gaussian_n10000_{x,y}.npy | gen_data.py, seed 42 | n=10K, d_x=10, d_y=2, float64 | Primary measurement dataset |

**Data verification requirement:** After generation, `scripts/gen_data.py` must print — and
`scripts/run_criterion.sh` must verify before proceeding — that each .npy file:
- Has the correct shape (e.g., (10000, 10) for X, (10000, 2) for Y)
- Has dtype float64
- Has no NaN or Inf values (`np.isfinite(arr).all()`)
- Has non-degenerate value range (max − min > 0.01 per column)

The verification output is logged to `results/data_verification.txt`.

**Relationship to profiler data:** The profiler (`run_profiler.sh`) uses the same .npy files
via the `tw_profiler` binary's `--x / --y` flags. The Criterion bench generates data in-process
using `make_data(n, d_x, d_y, seed)` from `benches/y_heap_variants_bench.rs`. Both use seed
42 with numpy `default_rng(42)` (Python) and `SmallRng::seed_from_u64(42)` (Rust). These
are **statistically equivalent distributions** (uniform[0,1] per element) but are **not
bitwise identical**. Cross-validation between the two data sources is out of scope; they are
treated as independent draws from the same distribution.

---

## Prior Failure Analysis (Phase 0 Prerequisite)

**The prior rerun-clean experiment (2026-04-05) applied thread-local optimization to y_heap
and measured 2× slowdown** (thread_local variant: 0.634s; avx2_kernel: 0.634s; baseline:
0.313s at n=10K). Before implementing any variant, the implementer must:

1. Check whether the rerun-clean worktree branch still exists:
   `git branch -a | grep rerun-clean` or `git worktree list`

2. If accessible, read the actual `src/metrics.rs` from that worktree to understand exactly
   what the "thread_local" and "avx2_kernel" variants did.

3. Hypothesize the root cause of slowdown. Candidate explanations:
   - **Cache pressure**: A flat COMB_DIST_Y Vec of 80KB (n=10K × 8 bytes) written per row
     evicts the already-resident COMB_DIST_X (also 80KB) from L2 cache. With both X and Y
     flat buffers per thread, working set = 4 × 80KB = 320KB per thread, exceeding typical
     per-core L2 (256KB).
   - **Introselect on full-n buffer**: For the heap, only k+1=16 elements live in memory at
     once (L1-resident). Introselect on an 80KB buffer has worse temporal locality.
   - **Broken implementation**: Self-exclusion or tie-breaking was incorrect, causing extra
     work or a correctness branch.

4. Document the root cause hypothesis in `results/prior_failure_analysis.md` before
   proceeding to implementation. If the worktree is inaccessible, document that the root
   cause is unknown and proceed with the `heap_reuse` diagnostic variant as the primary
   diagnostic tool (its result will distinguish malloc-dominated from algorithm-dominated
   cost).

The `heap_reuse` variant is designed specifically to diagnose this: if `heap_reuse` is fast
(< 10% overhead vs baseline or faster) but `flat_partial` remains slow, the root cause is
cache pressure from the full-n distance buffer. This confirms the flat buffer approach is
architecturally wrong for n=10K and the experiment should escalate to H3 (KD-tree).

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-06-y-heap-bottleneck-optimization/
├── rust-toolchain.toml           # Pins nightly-2026-03-26
├── environment.yml               # Minimal Python env for gen_data.py + analysis
├── scripts/
│   ├── gen_data.py               # Generates gaussian_n{1K,5K,10K}_{x,y}.npy, seed=42
│   ├── run_criterion.sh          # Criterion benchmark runs (profiling feature OFF)
│   ├── run_profiler.sh           # Step-timing profiler runs (profiling feature ON)
│   ├── analyze_results.py        # Loads Criterion + profiler JSON, computes speedup ratios,
│   │                             #   plots, writes analysis_report.md
│   └── dry_run.sh                # End-to-end smoke test with n=1K, sample_size=3
├── data/
│   ├── .gitkeep
│   ├── gaussian_n1000_{x,y}.npy  # Generated by gen_data.py (gitignored via /data/)
│   ├── gaussian_n5000_{x,y}.npy
│   └── gaussian_n10000_{x,y}.npy
└── results/
    ├── .gitkeep
    ├── data_verification.txt      # Shape/dtype/finiteness verification log
    ├── prior_failure_analysis.md  # Phase 0: root cause of rerun-clean slowdown
    ├── hardware_profile.txt       # CPU model, core count, RAYON_NUM_THREADS, rustc version,
    │                             #   numpy version, RUSTFLAGS expansion
    ├── criterion/
    │   ├── .gitkeep
    │   ├── y_heap_variants_n1000.json    # Criterion JSON output, n=1000
    │   ├── y_heap_variants_n5000.json    # Criterion JSON output, n=5000
    │   └── y_heap_variants_n10000.json   # Criterion JSON output, n=10000
    ├── profiler/
    │   ├── .gitkeep
    │   ├── profiler_baseline_n10000.json
    │   ├── profiler_heap_reuse_n10000.json
    │   ├── profiler_flat_partial_n10000.json
    │   └── profiler_flat_simd_n10000.json
    └── analysis/
        ├── .gitkeep
        ├── analysis_report.md            # Speedup table, CI table, interpretation
        └── speedup_ratios.png            # Bar chart: variants × n values
```

**File descriptions:**

- `rust-toolchain.toml`: Single-file toolchain pin scoped to this experiment directory.
  Rust uses the nearest `rust-toolchain.toml` walking up the directory tree.
  ```toml
  [toolchain]
  channel = "nightly-2026-03-26"
  # rustc 1.96.0-nightly (23903d01c 2026-03-26) — matches prior tw research experiments
  ```

- `environment.yml`: Minimal Python environment. Does not activate the full `spectral-test`
  env (unnecessary overhead). The existing materialized `envs/spectral-test/` may be used
  directly if preferred (has numpy 2.2.6 and scipy 1.15.2 already).

- `scripts/gen_data.py`: Uses `numpy.random.default_rng(seed=42)`. Generates float64 arrays
  of shape `(n, d_x)` for X and `(n, d_y)` for Y. Saves as `.npy` (not `.npz`) to match
  the format expected by `tw_profiler --x / --y`. Prints verification summary.

- `benches/y_heap_variants_bench.rs` (created in project root `benches/`): Criterion bench
  with four benchmark groups (one per variant), each sweeping n ∈ {1K, 5K, 10K}. Separate
  groups ensure Criterion does not re-use warm caches across variants (W4 mitigation — each
  group starts from a cold state in its own `cargo bench --bench y_heap_variants_bench
  --bench-group <group>` invocation; see §Execution Protocol).

- `scripts/run_criterion.sh`: Invokes `cargo bench --bench y_heap_variants_bench` separately
  for each variant group (not all at once), with `RAYON_NUM_THREADS=$(nproc)` fixed.
  Redirects Criterion JSON output to `results/criterion/`. Compiled **without** `profiling`
  feature.

- `scripts/run_profiler.sh`: Builds `tw_profiler` with `--features cli,profiling` and runs
  it for each variant (via `--variant` flag added in Phase 3). 5 warmup, 30 timed iterations
  at n=10K only (profiler is wall-clock noisy at smaller n). Outputs to
  `results/profiler/profiler_{variant}_n10000.json`.

- `scripts/analyze_results.py`: Loads all Criterion JSON files; computes speedup ratio point
  estimates as `mean_baseline / mean_variant`; extracts Criterion's reported CI bounds;
  builds a markdown table and PNG bar chart. Loads profiler JSON files; computes step
  fractions as `step_ns / sum(all_step_ns)` per iteration; reports mean ± std. Labels step
  fraction clearly as "thread-work fraction" not "wall-clock fraction."

- `scripts/dry_run.sh`: Runs gen_data.py for n=1K only, then runs Criterion with
  `sample_size=3` for all variants at n=1K, then runs profiler with `--iters 2 --warmup 1`
  for baseline only. Verifies no crashes and that JSON files are produced.

---

## Environment

**Custom environment required for Python scripts only (gen_data.py, analyze_results.py).**
The Rust toolchain is handled by `rust-toolchain.toml`; no custom Rust environment is needed.

The existing local conda prefix at `envs/spectral-test/` (Python 3.11, numpy 2.2.6,
scipy 1.15.2, matplotlib 3.10) may be used directly, or a minimal environment may be created:

```yaml
name: y-heap-bench
channels:
  - conda-forge
dependencies:
  - python=3.11.*
  - numpy=2.2.*      # Data generation; version pinned for reproducibility
  - scipy=1.15.*     # t-distribution CI computation in analyze_results.py
  - matplotlib=3.10.*  # speedup_ratios.png
```

**Rationale for each dependency:**
- `numpy=2.2.*`: `.npy` file generation (`np.save`) and verification. Version pinned because
  numpy's default RNG output is stable within major versions but could differ across (e.g.,
  2.1 vs 2.2 if promotion rules changed).
- `scipy=1.15.*`: `scipy.stats.t.ppf` for 95% CI computation over profiler iterations.
- `matplotlib=3.10.*`: Plot generation. Non-critical; version only needs to match project
  convention.

**Rust toolchain:** nightly-2026-03-26 (from `rust-toolchain.toml` in experiment directory).
`cargo build` and `cargo bench` will use this toolchain automatically via rustup's
directory-walk lookup.

**Cargo.lock:** Not tracked in git (`.gitignore` lists `Cargo.lock`). The `Cargo.lock`
file present on disk at time of experiment execution should be recorded by copying it to
`results/Cargo.lock.snapshot` before running benchmarks. This enables reproducibility
auditing.

---

## Implementation Phases

### Phase 0: Prior Failure Investigation (Prerequisite)

1. Check for surviving worktree/branch from the rerun-clean experiment:
   ```bash
   git worktree list
   git branch -a | grep -i rerun
   ```
2. If found: read the `src/metrics.rs` from that worktree's `thread_local` and
   `avx2_kernel` variant implementations. Identify the exact code that produced 2× slowdown.
3. Document root cause hypothesis in `results/prior_failure_analysis.md`.
4. **Decision gate:** If root cause is confirmed as "cache pressure from 80KB COMB_DIST_Y",
   reconsider whether `flat_partial` is worth testing. It may still be tested as a negative
   control (confirming the diagnosis), but `heap_reuse` becomes the primary candidate.

### Phase 1: Directory Structure and Environment

1. Create the experiment directory:
   ```bash
   mkdir -p research/2026-04-06-y-heap-bottleneck-optimization/{scripts,data,results/{criterion,profiler,analysis}}
   touch research/2026-04-06-y-heap-bottleneck-optimization/data/.gitkeep
   touch research/2026-04-06-y-heap-bottleneck-optimization/results/{.gitkeep,criterion/.gitkeep,profiler/.gitkeep,analysis/.gitkeep}
   ```
2. Create `rust-toolchain.toml` in the experiment directory (content shown above).
3. Create `environment.yml` (content shown above).
4. If using existing `envs/spectral-test/` prefix, verify it is functional:
   ```bash
   envs/spectral-test/bin/python -c "import numpy; print(numpy.__version__)"
   ```

### Phase 2: Data Generation

1. Write `scripts/gen_data.py`. Requirements:
   - Use `numpy.random.default_rng(seed=42)`.
   - Accept `--out-dir` argument (default: `data/`).
   - Generate float64 arrays: X shape `(n, 10)`, Y shape `(n, 2)` for n ∈ {1000, 5000, 10000}.
   - Save as `gaussian_n{n}_{x,y}.npy` using `np.save()`.
   - Print verification summary: filename, shape, dtype, min, max, any NaN/Inf.
   - Write verification log to stdout (captured to `results/data_verification.txt` by caller).

2. Run and verify:
   ```bash
   python scripts/gen_data.py --out-dir data/ | tee results/data_verification.txt
   ```
   Verify the log shows correct shapes and no NaN/Inf before proceeding.

### Phase 3: Library Implementation

All changes are to `src/metrics.rs` and `Cargo.toml`. No other source files are modified.

**3a. Add `profiling` feature to `Cargo.toml`:**
```toml
[features]
testing = [...]
cli = [...]
profiling = []  # Enables AtomicU64 step counters in trustworthiness variants
```

**3b. Add step timing counters (gated on `profiling` feature):**

Inside `src/metrics.rs`, add a `pub mod step_timing` module with six `AtomicU64` statics
(same schema as rerun-clean worktree):
```rust
#[cfg(feature = "profiling")]
pub mod step_timing {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static X_DIST_NS:     AtomicU64 = AtomicU64::new(0);
    pub static X_SORT_NS:     AtomicU64 = AtomicU64::new(0);
    pub static X_KNN_SET_NS:  AtomicU64 = AtomicU64::new(0);
    pub static Y_HEAP_NS:     AtomicU64 = AtomicU64::new(0);
    pub static PENALTY_NS:    AtomicU64 = AtomicU64::new(0);
    pub fn reset() { /* set all to 0 with Ordering::Release */ }
    pub fn read() -> [u64; 5] { /* load all with Ordering::Acquire */ }
}
```

Wrap each step in the existing `trustworthiness()` with:
```rust
#[cfg(feature = "profiling")]
let t_step = std::time::Instant::now();
// ... step code ...
#[cfg(feature = "profiling")]
step_timing::Y_HEAP_NS.fetch_add(t_step.elapsed().as_nanos() as u64, Ordering::Relaxed);
```

**3c. Implement the three variant functions:**

Add three new public functions to `src/metrics.rs` (and expose them from `src/lib.rs`):

- `pub fn trustworthiness_heap_reuse(x, y, k) -> f64`: Identical to `trustworthiness()` but
  uses a thread-local `RefCell<BinaryHeap<(u64, usize)>>` initialized with
  `with_capacity(k+1)`. At the start of each row's y_heap step: borrow, call `heap.clear()`,
  then proceed with the same push/pop logic. The heap allocation is reused across rows within
  each thread.

- `pub fn trustworthiness_flat_partial(x, y, k) -> f64`: Adds two thread-local statics
  (`COMB_DIST_Y: RefCell<Vec<f64>>`, `COMB_INDICES_Y: RefCell<Vec<usize>>`). Per row:
  - `dist_y.clear(); dist_y.resize(n, f64::INFINITY);`
  - `indices_y.clear(); indices_y.extend(0..n);`
  - Fill `dist_y[j]` for all j ≠ i with scalar Y squared distance.
  - `indices_y.select_nth_unstable_by(k, |&a, &b| dist_y[a].partial_cmp(&dist_y[b]).unwrap_or(std::cmp::Ordering::Equal));`
  - knn_y_set = HashSet from `indices_y[..k]` (note: k, not k+1 — the self-exclusion is
    already handled by dist_y[i] = INFINITY, which is correctly excluded by introselect).
  - Continue with penalty step as in baseline.
  - **Important tie-breaking note:** Verify that this produces an identical knn_y_set to the
    BinaryHeap approach for all test cases. Run `t_tw_08_combined_matches_baseline` variant
    comparing flat_partial output to baseline. Adjust tie-breaking if needed.

- `pub fn trustworthiness_flat_simd(x, y, k) -> f64`: All of flat_partial, plus:
  - Detect AVX2 at runtime (same `is_x86_feature_detected!` pattern as existing code).
  - If AVX2 available and y is C-contiguous: call an `unsafe fn dist_sq_2d_avx2_batch`
    kernel that fills `dist_y[0..n]` using 256-bit registers. The kernel processes 4 rows
    per SIMD iteration (pack `(y[j][0], y[j+1][0], y[j+2][0], y[j+3][0])` into a 256-bit
    register, similarly for column 1, subtract `yi` broadcast, square, hadd). Handle
    n % 4 ≠ 0 tail with the existing scalar loop. Set `dist_y[i] = f64::INFINITY` after fill.
  - Fall back to scalar loop if AVX2 not available.
  - The SIMD kernel signature: `unsafe fn dist_sq_2d_avx2_batch(yi: &[f64], y_flat: &[f64], dists: &mut [f64], n: usize)` where `y_flat` is the raw row-major flat data pointer from `y.as_slice().unwrap()`.

All three variant functions have step counters under `#[cfg(feature = "profiling")]` wrapping
the y_heap step specifically.

**3d. Add correctness tests:**

In `src/metrics.rs` test module, add:
- `t_tw_09_heap_reuse_matches_baseline`: For n ∈ {20, 50, 100}, seed=99, k=5: assert
  `|trustworthiness_heap_reuse(x,y,k) - trustworthiness(x,y,k)| < 1e-12`.
- `t_tw_10_flat_partial_matches_baseline`: Same assertion for flat_partial.
- `t_tw_11_flat_simd_matches_baseline`: Same assertion for flat_simd (requires AVX2;
  gate with `#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]`).
- `t_tw_12_flat_variants_sklearn_parity`: Run `sklearn_parity_synthetic` fixture comparison
  for each variant (if fixture exists); assert `|Δ| < 1e-6`.

Run `cargo test --features testing` and confirm all pass before proceeding.

### Phase 4: Benchmark Infrastructure

**4a. Write `benches/y_heap_variants_bench.rs`:**

```rust
// Four benchmark groups, one per variant. Each group sweeps n ∈ {1K, 5K, 10K}.
// SamplingMode::Flat, sample_size(10), warm_up_time(10s), measurement_time(10s).
// No `profiling` feature — pure wall-clock measurement.
// Bench names: "y_heap_variants/baseline/n=1000", "y_heap_variants/heap_reuse/n=1000", etc.
// Each bench function calls black_box(trustworthiness_X(x.view(), y.view(), 15)).
// make_data function: use SmallRng::seed_from_u64(42), generate uniform [0,1] arrays.
```

Register in `Cargo.toml`:
```toml
[[bench]]
name = "y_heap_variants_bench"
harness = false
required-features = ["testing"]
```

**4b. Write `scripts/run_criterion.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAYON_NUM_THREADS=$(nproc)

# Record hardware profile
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS" > results/hardware_profile.txt
rustc --version >> results/hardware_profile.txt
lscpu | grep "Model name" >> results/hardware_profile.txt
echo "RUSTFLAGS=${RUSTFLAGS:-}" >> results/hardware_profile.txt
cp Cargo.lock results/Cargo.lock.snapshot 2>/dev/null || true

# Run each variant group in a SEPARATE cargo bench invocation (W4 cold-cache mitigation).
# Groups: baseline, heap_reuse, flat_partial, flat_simd
for VARIANT in baseline heap_reuse flat_partial flat_simd; do
    echo "=== Running variant: $VARIANT ==="
    cargo bench --bench y_heap_variants_bench \
        --features testing \
        -- "y_heap_variants/$VARIANT" \
        2>&1 | tee "results/criterion/criterion_${VARIANT}_output.txt"
done
# Criterion JSON is written to target/criterion/; copy to results/criterion/
for N in 1000 5000 10000; do
    for VARIANT in baseline heap_reuse flat_partial flat_simd; do
        cp "target/criterion/y_heap_variants/${VARIANT}/n=${N}/new/estimates.json" \
           "results/criterion/${VARIANT}_n${N}.json" 2>/dev/null || true
    done
done
```

**4c. Write `scripts/run_profiler.sh`:**

The existing `src/bin/tw_profiler.rs` does not have a `--variant` flag. The implementer
must add one. The flag accepts: `baseline`, `heap_reuse`, `flat_partial`, `flat_simd`.
The binary dispatches to the appropriate `trustworthiness_*` function.

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAYON_NUM_THREADS=$(nproc)

# Build profiler binary with profiling feature enabled
cargo build --release --features cli,profiling --bin tw_profiler

for VARIANT in baseline heap_reuse flat_partial flat_simd; do
    ./target/release/tw_profiler \
        --x data/gaussian_n10000_x.npy \
        --y data/gaussian_n10000_y.npy \
        --k 15 --iters 30 --warmup 5 \
        --variant "$VARIANT" \
        --output "results/profiler/profiler_${VARIANT}_n10000.json"
done
```

**4d. Write `scripts/analyze_results.py`:**

Loads Criterion JSON files (one per variant × n). Criterion stores estimates in
`target/criterion/.../estimates.json` with fields `{mean: {point_estimate, confidence_interval: {lower_bound, upper_bound}}, ...}`. Computes:
- Speedup ratio: `point_estimate_baseline / point_estimate_variant`
- Speedup CI: propagate from Criterion's mean CI (approximation: ratio CI is not exact)
- Outputs markdown table to `results/analysis/analysis_report.md`
- Outputs `speedup_ratios.png` bar chart

Loads profiler JSON files. Computes step fractions as `y_heap_ns / sum_of_all_step_ns`
across iterations; reports mean ± std. Labels clearly as "thread-work fraction."

### Phase 5: Dry Run

Execute `scripts/dry_run.sh`. Verify:
- gen_data.py produces n=1K files without error
- Criterion runs 3 samples for all variants at n=1K without crash
- Criterion produces JSON output files (not empty)
- Profiler produces JSON with non-zero `step_times_ns` for baseline (and zeros for
  other variants until profiling instrumentation is wired in each)
- At least one trustworthiness score is finite and in [0, 1]

Fix any issues before proceeding to full run.

---

## Execution Protocol

Execute in order. Each step must complete successfully before the next begins.

**Step 0:** Read `results/prior_failure_analysis.md` (Phase 0 output). Decide whether
to proceed with all four variants or to skip `flat_partial` based on root cause analysis.
Document the decision.

**Step 1: Hardware profiling**
```bash
cd research/2026-04-06-y-heap-bottleneck-optimization
python --version >> results/hardware_profile.txt
python -c "import numpy; print('numpy', numpy.__version__)" >> results/hardware_profile.txt
```

**Step 2: Data generation**
```bash
python scripts/gen_data.py --out-dir data/ | tee results/data_verification.txt
# Verify results/data_verification.txt shows correct shapes and no NaN/Inf
grep -i "nan\|inf\|error" results/data_verification.txt && echo "DATA ERROR" && exit 1
```

**Step 3: Correctness verification**
```bash
cargo test --features testing 2>&1 | grep -E "test .* (ok|FAILED)"
# All t_tw_09 through t_tw_12 must pass. If any fail, fix variant implementation before proceeding.
```

**Step 4: Criterion benchmark (wall-time)**
```bash
bash scripts/run_criterion.sh 2>&1 | tee results/criterion/run_log.txt
```
Each variant group runs in a separate process. Expected runtime: 4 variants × 3 n-values ×
(10s warm-up + 10s measurement + overhead) ≈ 15–20 minutes total.

**Step 5: Apply RT-5 escalation check**
Inspect Criterion output for `flat_simd` vs `baseline` at n=10K:
- If CI LB > 1.0: primary result is positive. No escalation needed.
- If CI LB ≤ 1.0 AND point estimate ≥ 1.1×: re-run with sample_size=50 for baseline and
  flat_simd at n=10K only. Append results to `results/criterion/`.
- If CI LB ≤ 1.0 AND point estimate < 1.1×: result is negative. H0 not rejected. Document.

**Step 6: Profiler run (step fractions)**
```bash
bash scripts/run_profiler.sh 2>&1 | tee results/profiler/run_log.txt
```
Expected runtime: 4 variants × 35 iterations × ~0.3s = ~42 seconds wall-clock (profiler
has per-row instrumentation overhead; actual time may be 2-3× longer than Criterion timing).

**Step 7: Analysis**
```bash
python scripts/analyze_results.py \
    --criterion-dir results/criterion/ \
    --profiler-dir results/profiler/ \
    --out-dir results/analysis/
```

---

## Analysis Plan

### Primary Analysis (Confirmatory)

Extract from Criterion JSON for `flat_simd` vs `baseline` at n=10K:
- Point estimate: `mean_baseline_ns / mean_flat_simd_ns`
- 95% CI lower bound (from Criterion's bootstrap CI on baseline mean and variant mean)
- Decision rule: CI LB > 1.0 → reject H0 (flat_simd is reliably faster)

This is the sole confirmatory test. α=0.05.

### Causal Attribution (Exploratory)

Using the four-variant design, compute the attribution decomposition:

| Comparison | Interpretation |
|------------|----------------|
| T_baseline / T_heap_reuse | Malloc cost contribution |
| T_heap_reuse / T_flat_partial | Heap logic vs introselect (cache pressure cost) |
| T_flat_partial / T_flat_simd | SIMD distance computation benefit |

If `T_baseline / T_heap_reuse >> 1`: malloc is a major cost; heap_reuse may be sufficient.
If `T_heap_reuse / T_flat_partial < 1` (flat_partial is SLOWER than heap_reuse): confirms
cache pressure from 80KB distance buffer; flat buffer approach is architecturally wrong for n=10K.
If `T_flat_partial / T_flat_simd > 1`: SIMD provides genuine arithmetic throughput benefit.

### Step Fraction Analysis (Exploratory)

For each variant, report: `y_heap_ns / (x_dist_ns + x_sort_ns + x_knn_set_ns + y_heap_ns + penalty_ns)`
as mean ± std over 30 profiler iterations. Label as "y_heap thread-work fraction."

Causal attribution check: compare the direction of step fraction change vs wall-time change.
Concordant (both decrease for flat_simd) → consistent with genuine y_heap improvement.
Discordant (fraction decreases but wall-time does not change) → overhead introduced elsewhere.

### Scaling Analysis (Exploratory)

Plot speedup ratios at n=1K, 5K, 10K. If speedup increases with n, the approach scales
sub-linearly in cost and KD-tree may not be needed. If speedup decreases or reverses, the
approach has cache-dependent behavior.

### Inconclusive Zone Interpretation

| Outcome | Classification | Action |
|---------|---------------|--------|
| flat_simd CI LB > 1.0, estimate ≥ 1.5× | Positive (strong) | Ship flat_simd implementation |
| flat_simd CI LB > 1.0, estimate 1.1–1.5× | Positive (weak) | Ship with profiler validation |
| flat_simd CI LB ≤ 1.0, estimate ≥ 1.1× (after escalation to 50 samples) | Weak positive | Escalate to H3 (KD-tree experiment) |
| flat_simd CI LB ≤ 1.0 after escalation | Negative | Escalate to H3 |
| heap_reuse fast, flat_partial slow | Cache pressure confirmed | Skip flat buffer approaches; go directly to H3 |

---

## Success Criteria

**Conclusive positive:** The Criterion 95% CI lower bound for `T_baseline / T_flat_simd` at
n=10K is strictly > 1.0 (the optimization is reliably faster than baseline with α=0.05), AND
all correctness tests pass (`|Δ trustworthiness| < 1e-12` vs baseline, sklearn parity < 1e-6).

**Conclusive negative:** The Criterion 95% CI for `T_baseline / T_flat_simd` at n=10K
straddles 1.0 even after escalation to 50 samples, AND `T_heap_reuse / T_baseline` is also
not significantly > 1.0 — confirming that neither malloc elimination nor flat buffer + SIMD
helps, and H3 (KD-tree) is the next investigation.

**Diagnostic positive:** heap_reuse is significantly faster than baseline (CI LB > 1.0)
but flat_partial is not — confirming malloc is the dominant cost. Ship heap_reuse as a
minimal change; SIMD kernel is not needed.

**Inconclusive:** CI overlaps 1.0 at 10 samples, estimate ≥ 1.1×, escalation to 50 samples
needed. The experiment budget was insufficient to resolve the signal; escalation is defined.

---

## Threats to Validity

### Internal

**Cache warm-state asymmetry (W4 analog):** The four-variant Criterion bench invokes all
groups in the same binary. A prior group's data may leave cache state that benefits a
subsequent group. **Mitigation:** Run each variant group as a separate `cargo bench`
invocation (separate processes; OS clears cache between invocations by eviction). If groups
are run in a single invocation, the warm-state ordering should be documented.

**Profiler instrumentation overhead in profiler runs:** The `AtomicU64` `fetch_add(Relaxed)`
calls add ~1–2 ns per step per row — negligible vs 300ms wall-clock — but interact
differently with H1's flat-buffer memory access pattern (more predictable) vs H2's SIMD
kernel (wider register state). Since profiling runs are separate from Criterion runs (RT-4
resolution), this overhead does NOT affect the primary DV.

**Tie-breaking divergence:** BinaryHeap and select_nth_unstable have different tie-breaking
behavior when two Y distances are equal. If any row has tied k-th and (k+1)-th nearest
neighbor distances, the two approaches may return different knn sets. **Mitigation:** test
t_tw_10 and t_tw_11 enforce `|Δ trustworthiness| < 1e-12` vs baseline across multiple seeds
and n values, catching any tie-breaking divergence.

**Prior failure mode recurrence:** If the flat-partial implementation repeats the error from
the rerun-clean worktree (cache pressure or implementation bug), it will be 2× slower rather
than faster. **Mitigation:** Phase 0 identifies the root cause before implementation; Phase 3
dry run catches gross regressions early.

**Correctness of AVX2 kernel for d_y=2:** The 2D batch kernel processes Y in groups of 4
rows. If n is not divisible by 4, the tail rows must be handled by the scalar path. An
off-by-one in the tail could produce wrong distances for the last 1–3 rows, producing
incorrect knn_y sets. **Mitigation:** t_tw_11 exercises n ∈ {20, 50, 100} (all non-multiples
of 4) and asserts result matches baseline within 1e-12.

### External

**d_y=2 specialization:** The AVX2 SIMD kernel is specialized for d_y=2 (2D embeddings).
UMAP typically produces 2D embeddings, so this covers the primary use case. However, results
do not generalize to d_y ≥ 3. A d_y=3 user would not benefit from `flat_simd`.

**n ≤ 10K scope:** The KD-tree (H3) advantage grows with n; at n=100K, H3 might be
decisively better than any flat-scan approach regardless of SIMD or allocation optimization.
This experiment cannot test that scale within the time budget. If n ≤ 10K results are
negative, it does not mean H3 is unnecessary at larger scale.

**Thread count sensitivity:** RAYON_NUM_THREADS is fixed to the test machine's core count
but not normalized across machines. A machine with 4 cores vs 32 cores will see different
Rayon scheduling overhead and different memory bandwidth contention for thread-local buffers.
Results are machine-specific. Hardware profile is recorded to enable cross-machine comparison.

**Seed sensitivity:** Only seed 42 is tested (RT-3 decision). Data layouts that are
atypically favorable for SIMD alignment in Y-space could inflate or deflate the SIMD
contribution. This is an uncontrolled source of variance across seeds.

---

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| Criterion wall-clock time (4 variants × 3 n values × 20s) | ~15–20 min |
| Criterion escalation (if needed: baseline + flat_simd only, 50 samples × 10s × 2) | +20 min |
| Profiler runs (4 variants × 35 iterations × ~0.3–0.6s) | ~5–10 min |
| Phase 0 investigation | ~30–60 min |
| Implementation (Phase 3) | ~2–4 hours |
| Total implementation + experiment | ~4–6 hours |
| Disk space (data/ + results/) | ~50 MB |
| Python dependencies | Uses existing `envs/spectral-test/` prefix (already materialized) |
