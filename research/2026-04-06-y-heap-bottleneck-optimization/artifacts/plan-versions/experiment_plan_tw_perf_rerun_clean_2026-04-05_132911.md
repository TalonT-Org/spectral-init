# Experiment Plan: Trustworthiness Performance Re-run (Clean Infrastructure)

## Motivation

The prior trustworthiness performance experiment (PR #224, branch `research-20260404-174030`)
produced four blocked or inconclusive hypotheses and seven identified measurement infrastructure
gaps. Results cannot be trusted at face value: the shipped combined speedup (~1.95×–2.15×) was
extrapolated across a cache-regime boundary, the per-step timing data is contaminated by ~6.25×
testing-feature overhead, the MERFISH H5 benchmark was never executed, and the partial-rank CI
is too wide to conclude anything at n=50K on synthetic data.

This experiment re-runs those measurements with clean infrastructure: isolated Criterion bench
binaries, adequate sample sizes, Holm-Bonferroni multiple-comparisons correction, zero-cost
step instrumentation separated from `testing` feature, direct n=100K measurement, fresh
Gaussian data, and rigorous CI construction — to produce findings that can be committed to
the production record without caveats.

**Decisions informed:** (1) whether the shipped combined speedup survives the cache-regime
boundary at production scale; (2) whether `trustworthiness_approx` is field-deployable on
structured biological data; (3) which algorithmic step is the next optimization target;
(4) whether wide partial-rank CI is data-distribution-dependent.

---

## Hypotheses

### H5 — MERFISH Subsampling Quality and Speedup

**H0:** `trustworthiness_approx` with m=5000 fails to deliver ≥5× median wall-clock speedup
over exact computation OR delivers median |T_approx − T_exact| ≥ 0.001 on MERFISH n=10K data.

**H1:** `trustworthiness_approx` with m=5000 delivers both ≥5× median speedup AND median
|T_approx − T_exact| < 0.001 (95% t-CI upper bound, t(0.975, df=9) ≈ 2.262) on MERFISH
n=10K data across 10 independent seeds.

**Scope:** n=10K, m=5000 (ratio m/n=0.5), k=15, MERFISH PCA-50 coordinates only.
Extrapolation to n=100K (m/n=0.05) requires additional measurement.

**Timing boundaries (R5):** `wall_exact_s` = wall-clock for ONE call to
`trustworthiness_baseline(x, y, k)` after ONE warm-up call; data already loaded to memory;
timing starts immediately before the call and ends immediately after the return.
`wall_approx_s` = same protocol for `trustworthiness_approx(x, y, k, m, seed)`.
Neither timing includes file I/O.

**Near-threshold triggers (pre-registered):**
- Speedup: median ∈ [4.5×, 5.5×] → flag as "borderline"; report per-seed breakdown;
  verdict requires majority (≥6/10) of trials ≥5× for H1 to hold.
- Quality: median |delta| ∈ [0.0008, 0.0012] → flag as "borderline"; report per-trial values.

**5× speedup mechanism (RT-5 documentation):** Theoretical subsampling speedup for the
exact outer loop is n/m = 2×. The additional factor likely comes from avoided rank-sort
(O(n log n) → O(1) per sampled row), smaller working set improving cache hit rate, and
sequential `SmallRng` being faster than parallel coordination for small m. The mechanism
is not fully characterized; this experiment will report the measured speedup and its
components without claiming a mechanistic explanation.

---

### H-100K — Criterion Speedup Validation at Production Scale

**H0 (null):** The Holm-corrected 95% CI lower bound for combined variant speedup at n=100K
is ≤ 1.5× baseline. (Null: cache-regime shift from L3-resident to memory-bandwidth-bound
between 50K and 100K absorbed most speedup; the extrapolated 1.95×–2.15× is an overestimate.)

**H1 (alternative):** The Holm-corrected 95% CI lower bound for combined speedup at n=100K
exceeds 1.5×. (Alternative: shipped speedup survives the cache boundary.)

**Scope (W7):** n=100K, Gaussian d=10, k=15, x86-64-v3 AVX2/FMA only. Results do not
generalize to other hardware, non-AVX2 builds, or other data distributions.

**1.5× threshold rationale (RT-1 — Accept):** The 1.5× threshold is a conservative lower
bound set 30% below the prior observed ~1.95× result. It is not derived from a concrete
deployment latency requirement. The experiment tests whether the speedup survives the
cache-regime transition, not whether it meets a specific user-facing SLA. This is documented
explicitly; a follow-up experiment against a latency requirement is recommended if H1 holds.

**Power:** n=63 samples per group, CV=15% (estimated from prior bench output), r=10%
relative effect size: ~80% power at Holm-corrected α/(m+1−1)=α/4=0.0125 for the first
comparison. **CV sensitivity (W1):** at CV=20%, n=63 yields ~65% power — marginal. If
post-run CV exceeds 20%, this is reported as a limitation and may require additional samples.
Minimum detectable effect at 80% power is r≈10% at CV=15%.

---

### H0/H1-clean — Per-Step CPU-Time Fraction (Reframed at d=10, R3 + R4)

This is an **estimation task, not a binary verdict.** The contaminated prior measurement
(~6.25× overhead from `#[cfg(feature="testing")]` eprintln! inside rayon closures) is
discarded.

**Reframing (R3 Option B):** The hypothesis is measured at n=100K, d=10 (Gaussian
benchmark regime), matching the Criterion bench data. The d=50 regime requires a separate
MERFISH profiling run not in scope here.

**Reframing (R4 Option A):** The metric is **CPU-time fraction**, not wall-clock fraction.
Atomic counters under the `profiling` feature accumulate CPU-nanoseconds summed across
all rayon threads. For data-parallel steps, CPU-ns fraction > wall-clock fraction (more
threads contribute); for serial steps, they are equal. This is physically meaningful as
a measure of compute work allocation, not latency.

**Estimation goal:** Report mean ± 95% CI for each of the 6 algorithmic steps as a fraction
of total CPU-ns, at n=100K, d=10, using a `profiling`-feature build with no `testing`
contamination.

**First-principles prediction at d=10:** Per row, `tw_x_dist` requires ~2n×d ≈ 2M FLOPs
(AVX2-accelerated), while `tw_y_heap` requires ~n × (2 + log k) ≈ 600K scalar heap ops.
The ~3.3× raw FLOP advantage for `tw_x_dist`, amplified by AVX2 throughput (8× float
multiplications per cycle), suggests `tw_x_dist` CPU-time fraction will exceed all others.
The margin is smaller than at d=50 (~12× FLOP advantage); `tw_y_heap` may account for
20–40% of CPU-ns at d=10.

**CI-ordering check (W3):** The ordering check (tw_x_dist CI > tw_y_heap CI) is
descriptive with no family-wise error control. It does not generate a p-value; it is
used to assess whether the first-principles prediction is directionally correct.

---

### H-partial-MERFISH — Partial-Rank CI Width on Structured Data (R1)

**H0:** The Criterion CI half-width for `partial_rank` speedup at n=50K on MERFISH PCA-50
data is ≥ 0.26 (same as the [1.10×, 1.62×] Gaussian half-width from the prior run on
contaminated data; this run uses a fresh, isolated Criterion baseline).

**H1:** The CI half-width is < 0.26, indicating that MERFISH PCA-50 distances have narrower
dynamic range than Gaussian, reducing `select_nth_unstable_by` pivot variance.

**Gaussian CI baseline:** The Gaussian half-width comparison uses the NEW isolated Criterion
run from this experiment (not contaminated prior data) at n=50K, to ensure comparability.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm variant | baseline, thread_local, partial_rank, avx2_kernel, combined | All 5 production candidates; matches research worktree functions |
| Input scale n (Criterion) | 1K, 5K, 10K, 25K, 50K, 100K | Full scaling curve; n=100K is primary claim; 50K enables H-partial-MERFISH |
| Dataset (Criterion benches) | Gaussian d=10 (fresh seed 2026) | Algorithm-level regression; cache-regime analysis |
| Dataset (MERFISH partial_rank bench) | MERFISH PCA-50 n=50K | Structured biological data; H-partial-MERFISH CI comparison |
| Dataset (H5) | MERFISH PCA-50 n=10K | Primary H5 hypothesis dataset |
| Subsampling count m (H5 sweep) | 500, 1000, 2000, 5000, 10000 | m=5000 is confirmatory gate; remainder is descriptive-only (W2) |
| Random seed (H5 trials) | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 | 10 seeds for reliable median |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| `wall_clock_speedup` | ratio (×) | `wall_exact_s / wall_approx_s` from `tw_approx_runner` JSON; median over 10 seeds | NEW |
| `delta_tw` | dimensionless | `|t_approx − t_exact|` from `tw_approx_runner` JSON field `delta`; mean ± 95% t-CI over 10 seeds (t(0.975, df=9) ≈ 2.262; R9) | NEW |
| `criterion_speedup_n100k` | ratio (×) | Bootstrap ratio CI from Criterion raw samples at n=100K; Holm-corrected over m=4 comparisons | NEW |
| `partial_rank_ci_half_width_gaussian` | ratio | Half-width of Criterion speedup CI for partial_rank vs baseline at n=50K, Gaussian | NEW |
| `partial_rank_ci_half_width_merfish` | ratio | Half-width of Criterion speedup CI for partial_rank vs baseline at n=50K, MERFISH | NEW |
| `tw_x_dist_cpu_fraction` | fraction [0,1] | Mean fraction of total CPU-ns in `tw_x_dist` step across 30 profiling iterations; ± 95% CI | NEW |
| `tw_y_heap_cpu_fraction` | fraction [0,1] | Mean fraction of total CPU-ns in `tw_y_heap` step; ± 95% CI | NEW |
| `tw_x_sort_cpu_fraction` | fraction [0,1] | Mean fraction in `tw_x_sort` (partial sort / select_nth_unstable_by); ± 95% CI | NEW |
| `tw_rank_scatter_cpu_fraction` | fraction [0,1] | Mean fraction in `tw_rank_scatter`; ± 95% CI | NEW |
| `tw_x_knn_set_cpu_fraction` | fraction [0,1] | Mean fraction in `tw_x_knn_set`; ± 95% CI | NEW |
| `tw_penalty_cpu_fraction` | fraction [0,1] | Mean fraction in `tw_penalty` accumulation; ± 95% CI | NEW |

**Canonical name note:** `src/metrics.rs` defines no performance-dimension metric constants.
All performance DVs are NEW with no canonical entry in `src/metrics.rs`. The existing metric
infrastructure covers Accuracy and Parity dimensions only (eigensolver residuals,
orthogonality, sklearn parity). No changes to `src/metrics.rs` are needed for this experiment's
performance measurements — all metrics are collected in Python analysis scripts.

**Metric is NEW — definitions required (no threshold entry needed for these performance metrics;
thresholds are pre-registered in the research plan, not in `src/metrics.rs`):**
- `wall_clock_speedup`: formula = `wall_exact_s / wall_approx_s`; unit = dimensionless ×; threshold = ≥5× (H5), >1.5× lower CI (H-100K)
- `delta_tw`: formula = `|t_approx − t_exact|`; unit = dimensionless; threshold = <0.001 (H5 quality gate)
- `criterion_speedup_*`: formula = `baseline_mean / variant_mean` from Criterion samples; unit = ×; threshold = lower CI > 1.5× (H-100K)
- `*_cpu_fraction`: formula = `step_cpu_ns / total_cpu_ns`; unit = fraction; no threshold (descriptive)

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighborhood size) | 15 | Matches prior experiment and integration tests |
| Gaussian dimensionality d | 10 | Matches prior experiment bench configuration |
| MERFISH dimensionality d | 50 (PCA components) | Fixed by `prepare_merfish.py` pipeline |
| Rayon thread count (R8) | Physical core count of benchmark machine; pinned via `rayon::ThreadPoolBuilder::new().num_threads(N_THREADS).build_global().unwrap()` at top of each bench `main()` and `tw_profiler` | Thread count directly affects parallelism and benchmark results; must be constant across variants |
| Rust toolchain | nightly-2026-03-26 (pinned in `rust-toolchain.toml`) | Matches prior experiment; codegen stability |
| Criterion: n < 100K sample_size | 100 | Explicit (R7); Criterion default but pinned for reproducibility |
| Criterion: n < 100K warm_up_time | 10s | Explicit (R7) |
| Criterion: n < 100K measurement_time | 60s | Explicit (R7) |
| Criterion: n=100K sampling_mode | SamplingMode::Flat | Required for long-running samples; same as prior |
| Criterion: n=100K sample_size | 63 | 80%-power at CV=15%, r=10%, Holm-corrected |
| Criterion: n=100K warm_up_time | 30s | Allow cache steady-state at 100K |
| Criterion: n=100K measurement_time | 1500s (25 min per variant) | 63 samples × ~20s/sample + overhead |
| Benchmark isolation | 1 variant per `[[bench]]` binary; 60s cool-down between runs | Eliminates thread-local statics contamination (root cause of prior bias) |
| H5 warm-up protocol | 1 warm-up call before timing; data pre-loaded | Matches `tw_approx_runner` implementation; excludes I/O |
| Profiling iterations | 30 timed + 5 warm-up (discarded) | Statistical stability for step-fraction CIs |
| Gaussian data seed | 2026 (fresh; RT-2 mitigate) | Fresh realization; see RT-2 documentation below |

---

## Inputs and Data

The experiment uses four data sources:

1. **MERFISH n=10K fixtures** — generated by `prepare_merfish.py` from
   `temp/merfish_100k/merfish_100k_expression.npz` (confirmed present, 24MB, all 5 NPZ
   artifacts exist). Script runs PCA(50) on first 10K rows of 100K subset. Output:
   `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` (10K × 50 f64),
   `merfish_n10k_y.npy` (10K × 2 f64). Currently missing — generated in Phase 3.
   NPZ key: `arr_0` (confirmed; both `generate_merfish_subset.py` and `prepare_merfish.py`
   use `list(npz.files)[0]` which resolves to `arr_0`).

2. **MERFISH n=50K fixture (R1 fix)** — sliced from the same 100K PCA-50 output as n=10K.
   Script: `scripts/prepare_merfish_50k.py` (new script added in Phase 4) slices
   `expression[:50000]` and `spatial[:50000]` from the `prepare_merfish.py` PCA output.
   Output: `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy`
   (50K × 50 f64), `merfish_n50k_y.npy` (50K × 2 f64). Generated in Phase 3.

3. **Gaussian benchmark data** — freshly generated with seed 2026 (RT-2 mitigation). Script:
   `scripts/gen_synthetic.py --seed 2026` in Phase 3. Generates n × {_x, _y} .npy pairs for
   all six n values (1K, 5K, 10K, 25K, 50K, 100K), d=10, randn. Output directory:
   `research/2026-04-05-tw-perf-rerun-clean/data/gaussian/`. The bench files reference
   this path via `CARGO_MANIFEST_DIR + "/research/2026-04-05-tw-perf-rerun-clean/data/gaussian/"`.

4. **Raw MERFISH H5AD** — `data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad` (2.0GB) — present but
   not directly used (already pre-processed into `temp/merfish_100k/`; no re-generation needed).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| MERFISH n=10K x/y | Generated from `temp/merfish_100k/` via `prepare_merfish.py` | 10K × 50 f64, 10K × 2 f64; real biological structure | H5 speedup/quality gate |
| MERFISH n=50K x/y | Generated same pipeline, sliced at row 50K (R1) | 50K × 50 f64, 50K × 2 f64 | H-partial-MERFISH Criterion bench |
| Gaussian n=1K–100K d=10 | Fresh `gen_synthetic.py --seed 2026` (RT-2) | randn; 6 scale points | H-100K and H-partial-MERFISH Criterion benches |

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder within the new git worktree:

```
research/2026-04-05-tw-perf-rerun-clean/
├── environment.yml                     # Python env spec (micromamba/conda)
├── scripts/
│   ├── gen_synthetic.py                # Generate fresh Gaussian data (seed 2026)
│   ├── prepare_merfish.py              # Copy from old worktree; generate MERFISH 10K X/Y
│   ├── prepare_merfish_50k.py          # NEW: slice MERFISH 50K X/Y from 100K PCA output
│   ├── prepare_data.sh                 # Orchestrate all data generation; verify inputs
│   ├── apply_phase1_changes.sh         # Document/verify Phase 1 Rust source changes
│   ├── run_h5.sh                       # H5 confirmatory: 10-seed + m-sweep (descriptive)
│   ├── run_criterion_clean.sh          # Per-variant Criterion run with 60s cool-down
│   ├── run_profiling_clean.sh          # tw_profiler at n=100K; profiling feature only
│   └── analyze_clean.py               # Bootstrap CI, Holm correction, t-CI, step fractions
├── data/
│   ├── gaussian/                       # gaussian_n{N}_{x,y}.npy for N in {1K,5K,10K,25K,50K,100K}
│   └── merfish/                        # merfish_n10k_{x,y}.npy, merfish_n50k_{x,y}.npy
├── results/
│   ├── h5/                             # h5_trial_seed{42..51}.json; h5_sweep_m{M}.json
│   ├── criterion/                      # criterion_output.json (JSON-lines from cargo criterion)
│   ├── step_timing/                    # gaussian_n100000_{variant}.json (30-iter clean profiling)
│   └── analysis/                       # analysis_report.md with all hypothesis verdicts
└── report.md                           # Final report (written by write-report skill)
```

Source code changes (Phase 1) live in the worktree root — not inside the experiment folder:

```
[worktree root]/
├── rust-toolchain.toml                 # NEW: pin nightly-2026-03-26
├── Cargo.toml                          # MODIFIED: add profiling feature; replace bench entries
├── src/
│   ├── metrics.rs                      # MODIFIED: add step_timing module (profiling feature)
│   └── bin/
│       └── tw_profiler.rs              # MODIFIED: fix atomic ordering; fix reset call site
└── benches/
    ├── tw_baseline_bench.rs            # NEW: isolated Criterion bench for trustworthiness
    ├── tw_thread_local_bench.rs        # NEW: isolated bench for trustworthiness_thread_local
    ├── tw_partial_rank_bench.rs        # NEW: isolated bench for trustworthiness_partial_rank
    ├── tw_avx2_bench.rs                # NEW: isolated bench for trustworthiness_avx2_kernel
    ├── tw_combined_bench.rs            # NEW: isolated bench for trustworthiness_combined
    └── tw_partial_rank_merfish_bench.rs # NEW: isolated bench loading MERFISH 50K (H-partial)
    # NOTE: trustworthiness_bench.rs (old single-binary) is REMOVED
```

---

## Environment

**Custom environment required.** The experiment requires Python for data preparation and
analysis, with packages beyond the standard scientific stack.

```yaml
name: tw-perf-rerun-clean
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2.6
  - scipy=1.15.2
  - scikit-learn=1.6.0       # PCA in prepare_merfish.py
  - statsmodels=0.14.6       # Holm-Bonferroni in analyze_clean.py
  - pip
```

**Rationale:** scikit-learn provides `sklearn.decomposition.PCA` used by `prepare_merfish.py`.
statsmodels provides `multipletests(pvals, method='holm')` for Holm-Bonferroni correction.
scipy provides `scipy.stats.t` for t-distribution CI (R9) and bootstrap utilities.
anndata and polars are NOT required since `temp/merfish_100k/` NPZ files already exist; if
the NPZ files must be regenerated, `generate_merfish_subset.py` requires `anndata` and
`polars` — install separately as needed.

**Rust toolchain:** Create `rust-toolchain.toml` in the new worktree root:

```toml
[toolchain]
channel = "nightly-2026-03-26"
profile = "minimal"
components = ["rustfmt", "clippy"]
```

**cargo-criterion:** NOT installed on the host system (`~/.cargo/bin/cargo-criterion` absent).
Install in Phase 0:
```bash
cargo install cargo-criterion
```
Required for `--message-format=json` stable Criterion output stream that `analyze_clean.py`
parses. `run_criterion_clean.sh` will verify the binary exists before proceeding.

---

## Implementation Phases

### Phase 0: Worktree Creation and Tooling Bootstrap

```bash
# Create new worktree branching from the research worktree
git worktree add \
  /home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean \
  -b research-20260405-tw-perf-rerun-clean \
  research-20260404-174030

# Install cargo-criterion (not present on host)
cargo install cargo-criterion
# Verify:
which cargo-criterion && echo "OK" || echo "FAIL: cargo-criterion not found"

# Create experiment skeleton
cd /home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean
mkdir -p research/2026-04-05-tw-perf-rerun-clean/{scripts,data/gaussian,data/merfish,results/{h5,criterion,step_timing,analysis}}

# Create micromamba environment
micromamba create -f research/2026-04-05-tw-perf-rerun-clean/environment.yml
micromamba activate tw-perf-rerun-clean
```

**Verify:** `python -c "import sklearn, statsmodels, scipy; print('OK')"` exits 0.

### Phase 1: Source Code Changes (6 changes; verified by `apply_phase1_changes.sh`)

**Change 1 — `rust-toolchain.toml`:**
Create at worktree root with content shown in Environment section.
Verify: `rustup show active-toolchain` prints `nightly-2026-03-26`.

**Change 2 — `Cargo.toml` features:**
Add `profiling` feature (no additional dependencies):
```toml
[features]
testing  = ["dep:serde"]
cli      = ["dep:ndarray-npy", "dep:pico-args", "dep:serde_json"]
profiling = []    # zero-cost step timing; no additional deps
```
Verify: `cargo check --features profiling` exits 0.

**Change 3 — `Cargo.toml` bench entries:**
Remove the existing `[[bench]] name = "trustworthiness_bench"` entry.
Add 6 isolated entries (5 Gaussian variants + 1 MERFISH partial_rank):
```toml
[[bench]]
name = "tw_baseline_bench"
harness = false

[[bench]]
name = "tw_thread_local_bench"
harness = false

[[bench]]
name = "tw_partial_rank_bench"
harness = false

[[bench]]
name = "tw_avx2_bench"
harness = false

[[bench]]
name = "tw_combined_bench"
harness = false

[[bench]]
name = "tw_partial_rank_merfish_bench"
harness = false
```
Delete `benches/trustworthiness_bench.rs`.
Verify: `cargo bench --no-run` lists all 6 new bench binaries.

**Change 4 — `src/metrics.rs` profiling instrumentation (R6 fix):**
Add a `step_timing` module under `#[cfg(feature = "profiling")]`. Use `Ordering::Release`
in `reset()` to guarantee writes are visible to rayon workers before iteration begins.
Use `Ordering::Acquire` in `read()` to guarantee all worker `fetch_add`s are visible after
the rayon scope completes. Workers use `fetch_add(Relaxed)` — acceptable because rayon's
internal scope/join provides a happens-before fence around the parallel work.

```rust
#[cfg(feature = "profiling")]
pub mod step_timing {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static X_DIST_NS:       AtomicU64 = AtomicU64::new(0);
    pub static X_SORT_NS:       AtomicU64 = AtomicU64::new(0);
    pub static RANK_SCATTER_NS: AtomicU64 = AtomicU64::new(0);
    pub static X_KNN_SET_NS:    AtomicU64 = AtomicU64::new(0);
    pub static Y_HEAP_NS:       AtomicU64 = AtomicU64::new(0);
    pub static PENALTY_NS:      AtomicU64 = AtomicU64::new(0);

    /// Call BEFORE each timed iteration (R6 fix: Release ordering).
    pub fn reset() {
        X_DIST_NS.store(0,       Ordering::Release);
        X_SORT_NS.store(0,       Ordering::Release);
        RANK_SCATTER_NS.store(0, Ordering::Release);
        X_KNN_SET_NS.store(0,    Ordering::Release);
        Y_HEAP_NS.store(0,       Ordering::Release);
        PENALTY_NS.store(0,      Ordering::Release);
    }

    /// Call AFTER rayon scope completes (R6 fix: Acquire ordering).
    pub fn read() -> [(&'static str, u64); 6] {
        [
            ("tw_x_dist",        X_DIST_NS.load(Ordering::Acquire)),
            ("tw_x_sort",        X_SORT_NS.load(Ordering::Acquire)),
            ("tw_rank_scatter",  RANK_SCATTER_NS.load(Ordering::Acquire)),
            ("tw_x_knn_set",     X_KNN_SET_NS.load(Ordering::Acquire)),
            ("tw_y_heap",        Y_HEAP_NS.load(Ordering::Acquire)),
            ("tw_penalty",       PENALTY_NS.load(Ordering::Acquire)),
        ]
    }
}
```

Instrument each step in `trustworthiness` (baseline variant only is sufficient for profiling)
with `Instant::now()` + `elapsed().as_nanos()` under `#[cfg(feature = "profiling")]` guards,
accumulated via `fetch_add(Relaxed)` on the corresponding static.

Verify: `cargo check --features profiling` exits 0; `cargo check` (no features) exits 0.

**Change 5 — `src/bin/tw_profiler.rs` reset call site (R6 fix):**
Move `step_timing::reset()` to BEFORE each timed iteration (not after):
```rust
// BEFORE (wrong — contamination from prior iteration):
let _ = variant_fn(x, y, k);
let step_readings = step_timing::read();
step_timing::reset();   // ← too late

// AFTER (correct — clean slate per iteration):
step_timing::reset();   // ← reset first
let _ = variant_fn(x, y, k);
let step_readings = step_timing::read();
```
Also ensure tw_profiler builds with `--features profiling` (not `--features testing`).
Verify: `cargo build --features cli,profiling --release` exits 0.

**Change 6 — Rayon thread count pinning in tw_profiler (R8):**
At the top of `tw_profiler.rs` `main()`, before any rayon work:
```rust
const N_THREADS: usize = 8;  // update to physical core count of benchmark machine
rayon::ThreadPoolBuilder::new()
    .num_threads(N_THREADS)
    .build_global()
    .unwrap();
```
Document actual physical core count in the hardware profile section of the report.

After all 6 changes: run `apply_phase1_changes.sh` which executes `cargo check --features cli,profiling` and `cargo test --features testing` and reports any failures.

### Phase 2: Bench File Creation

Create 6 bench files in `benches/`. Each follows the same pattern with these invariants:

- **Rayon thread pinning** (R8): `rayon::ThreadPoolBuilder::new().num_threads(N_THREADS).build_global().unwrap()` at top of `main()`, same `N_THREADS` constant as `tw_profiler.rs`.
- **Profiling feature excluded** (W8): bench files must NOT `#[cfg(feature = "profiling")]` the step timing — Criterion builds should not activate profiling. Verify with: `cargo criterion --bench tw_baseline_bench --no-run --features cli 2>&1 | grep -v "profiling"` (profiling must not appear).
- **Explicit Criterion parameters for all n values** (R7): set `sample_size`, `warm_up_time`, `measurement_time` for n < 100K group before the loop; override `sampling_mode`, `sample_size`, `warm_up_time`, `measurement_time` for n=100K after the loop.
- **Data paths**: load from `research/2026-04-05-tw-perf-rerun-clean/data/{gaussian,merfish}/` via `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(...)` — resolves the R2 / data-path issue since fresh data lives in the new experiment folder.

**Template (Gaussian variants — `tw_baseline_bench.rs`, repeated for each variant):**

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use ndarray_npy::read_npy;
use std::{path::PathBuf, time::Duration};

const N_THREADS: usize = 8;  // pin to physical core count

fn bench_tw_baseline(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build_global()
        .unwrap();

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-04-05-tw-perf-rerun-clean/data/gaussian");

    let mut group = c.benchmark_group("tw_baseline");
    // Explicit parameters for all n < 100K (R7)
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(60));

    for &n in &[1000usize, 5000, 10000, 25000, 50000] {
        let x: ndarray::Array2<f64> = read_npy(data_dir.join(format!("gaussian_n{n}_x.npy"))).unwrap();
        let y: ndarray::Array2<f64> = read_npy(data_dir.join(format!("gaussian_n{n}_y.npy"))).unwrap();
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| spectral_init::metrics::trustworthiness(x.view(), y.view(), 15))
        });
    }

    // n=100K override parameters (R7)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(63);
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(1500));
    {
        let x: ndarray::Array2<f64> = read_npy(data_dir.join("gaussian_n100000_x.npy")).unwrap();
        let y: ndarray::Array2<f64> = read_npy(data_dir.join("gaussian_n100000_y.npy")).unwrap();
        group.bench_with_input(BenchmarkId::new("baseline", 100000), &100000usize, |b, _| {
            b.iter(|| spectral_init::metrics::trustworthiness(x.view(), y.view(), 15))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tw_baseline);
criterion_main!(benches);
```

**MERFISH partial_rank bench (`tw_partial_rank_merfish_bench.rs`):**
Same structure but loads `merfish_n50k_x.npy` and `merfish_n50k_y.npy` from the merfish data
directory. Runs ONLY at n=50K (the H-partial-MERFISH measurement scale). Parameters:
`sample_size(100)`, `warm_up_time(10s)`, `measurement_time(60s)` — identical to the n=50K
parameters in the Gaussian benches. Includes BOTH `partial_rank` and `baseline` to compute
the speedup ratio. Two benchmark IDs: `"partial_rank_merfish/50000"` and `"baseline_merfish/50000"`.

Verify all 6 bench files compile: `cargo bench --no-run --features cli 2>&1 | grep -E "^(error|warning\[)"` — must show no errors.

### Phase 3: Data Preparation

Run from the new worktree root with the Python environment active.

**Step 3.1 — Fresh Gaussian data (RT-2 mitigation):**
```bash
micromamba activate tw-perf-rerun-clean
python research/2026-04-05-tw-perf-rerun-clean/scripts/gen_synthetic.py \
  --seed 2026 \
  --output-dir research/2026-04-05-tw-perf-rerun-clean/data/gaussian \
  --sizes 1000 5000 10000 25000 50000 100000 \
  --d 10
```
Verify: 12 files present (6 scales × 2 arrays), each `_x.npy` has shape `(N, 10)` float64.

**Step 3.2 — MERFISH n=10K fixtures:**
```bash
cd /home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean
python research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish.py \
  --npz-dir /home/talon/projects/spectral-init/temp/merfish_100k \
  --output-dir research/2026-04-05-tw-perf-rerun-clean/data/merfish \
  --n 10000
```
Verify: `merfish_n10k_x.npy` shape `(10000, 50)` float64; `merfish_n10k_y.npy` shape
`(10000, 2)` float64.

**Step 3.3 — MERFISH n=50K fixtures (R1 fix):**
```bash
python research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish_50k.py \
  --x-source research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy \
  --npz-dir /home/talon/projects/spectral-init/temp/merfish_100k \
  --output-dir research/2026-04-05-tw-perf-rerun-clean/data/merfish \
  --n 50000
```
`prepare_merfish_50k.py` runs PCA(50) on first 50K rows of the 100K expression NPZ (same
PCA pipeline as `prepare_merfish.py`), and saves `merfish_n50k_x.npy` (50K × 50 f64) and
`merfish_n50k_y.npy` (50K × 2 f64).

Verify: `merfish_n50k_x.npy` shape `(50000, 50)` float64.

**Step 3.4 — Verify bench data path resolution:**
```bash
cargo bench --bench tw_baseline_bench --no-run --features cli
# Should compile without missing-file errors; actual file loading happens at bench runtime
```

### Phase 4: Script Creation

Create all scripts in `research/2026-04-05-tw-perf-rerun-clean/scripts/`:

**`apply_phase1_changes.sh`** — verifies each Phase 1 change was applied:
- Checks `rust-toolchain.toml` exists and contains `nightly-2026-03-26`
- Checks `Cargo.toml` contains `profiling = []`
- Checks `benches/tw_baseline_bench.rs` exists; `benches/trustworthiness_bench.rs` absent
- Runs `cargo check --features cli,profiling`; `cargo check --features testing`
- Runs `cargo test --features testing --test test_trustworthiness` — parity assertion must pass

**`run_h5.sh`** — builds `tw_approx_runner` (no `testing` feature) and runs 10 seeds at m=5000
plus a descriptive m-sweep at m ∈ {500, 1000, 2000, 5000, 10000}:
```bash
#!/usr/bin/env bash
set -euo pipefail
PROJ=/home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean
EXP=$PROJ/research/2026-04-05-tw-perf-rerun-clean
cargo build --manifest-path $PROJ/Cargo.toml --release --features cli --no-default-features
RUNNER=$PROJ/target/release/tw_approx_runner

# Confirmatory: 10 seeds at m=5000
for SEED in 42 43 44 45 46 47 48 49 50 51; do
  $RUNNER \
    --x $EXP/data/merfish/merfish_n10k_x.npy \
    --y $EXP/data/merfish/merfish_n10k_y.npy \
    --k 15 --sample 5000 --seed $SEED \
    --output $EXP/results/h5/h5_trial_seed${SEED}.json
done

# Descriptive sweep (W2 — no inferential comparisons derived from this)
for M in 500 1000 2000 10000; do
  $RUNNER \
    --x $EXP/data/merfish/merfish_n10k_x.npy \
    --y $EXP/data/merfish/merfish_n10k_y.npy \
    --k 15 --sample $M --seed 42 \
    --output $EXP/results/h5/h5_sweep_m${M}.json
done
```

**`run_criterion_clean.sh`** — runs each variant in a separate `cargo criterion` invocation
with 60s cool-down between variants. Also runs the MERFISH partial_rank bench:
```bash
#!/usr/bin/env bash
set -euo pipefail
PROJ=/home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean
EXP=$PROJ/research/2026-04-05-tw-perf-rerun-clean
RESULTS=$EXP/results/criterion

# Verify cargo-criterion is available
which cargo-criterion || { echo "ERROR: cargo-criterion not installed"; exit 1; }

# Verify profiling feature NOT active in Criterion builds (W8)
cargo criterion --manifest-path $PROJ/Cargo.toml \
  --bench tw_baseline_bench --no-run --features cli 2>&1 \
  | grep -qi "profiling" && { echo "ERROR: profiling feature active in Criterion build"; exit 1; } \
  || echo "OK: profiling feature not active in Criterion build"

# Pre-registered re-run policy (RT-4):
# If any variant must be re-run due to system event (thermal throttle, background process),
# ALL variants must be re-run together. The first complete set is the primary dataset.

VARIANTS=(tw_baseline_bench tw_thread_local_bench tw_partial_rank_bench tw_avx2_bench tw_combined_bench)
for BENCH in "${VARIANTS[@]}"; do
  echo "Running $BENCH..."
  cargo criterion --manifest-path $PROJ/Cargo.toml \
    --bench $BENCH \
    --message-format=json \
    --features cli \
    -- 2>&1 >> $RESULTS/criterion_output.json
  echo "Cooling down 60s..."
  sleep 60
done

# MERFISH partial_rank bench (H-partial-MERFISH)
echo "Running tw_partial_rank_merfish_bench..."
cargo criterion --manifest-path $PROJ/Cargo.toml \
  --bench tw_partial_rank_merfish_bench \
  --message-format=json \
  --features cli \
  -- 2>&1 >> $RESULTS/criterion_merfish_output.json

# Cache warm-state check (W4 — mandatory, not optional):
# Re-run combined then baseline in reversed order; compare point estimates.
echo "Running cache warm-state check (reversed order)..."
for BENCH in tw_combined_bench tw_baseline_bench; do
  cargo criterion --manifest-path $PROJ/Cargo.toml \
    --bench $BENCH \
    --message-format=json \
    --features cli \
    -- 2>&1 >> $RESULTS/criterion_reversed_output.json
  sleep 60
done
```

**`run_profiling_clean.sh`** — builds `tw_profiler` with `cli,profiling` features (NOT `testing`):
```bash
#!/usr/bin/env bash
set -euo pipefail
PROJ=/home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean
EXP=$PROJ/research/2026-04-05-tw-perf-rerun-clean
cargo build --manifest-path $PROJ/Cargo.toml --release --features cli,profiling --no-default-features
PROFILER=$PROJ/target/release/tw_profiler

VARIANTS=(baseline thread_local partial_rank avx2_kernel combined)
for VARIANT in "${VARIANTS[@]}"; do
  $PROFILER \
    --x $EXP/data/gaussian/gaussian_n100000_x.npy \
    --y $EXP/data/gaussian/gaussian_n100000_y.npy \
    --k 15 --n-warmup 5 --n-iters 30 \
    --variant $VARIANT \
    --output $EXP/results/step_timing/gaussian_n100000_${VARIANT}.json
done
```

**`analyze_clean.py`** — full analysis with all statistical fixes applied:
- Reads `results/h5/h5_trial_seed*.json`; computes median speedup; constructs 95% CI
  using `t_dist.ppf(0.975, df=n-1)` with `n=10` (R9 fix: t-distribution, not z=1.96)
- Reads `results/criterion/criterion_output.json` (JSON-lines from `--message-format=json`);
  bootstraps ratio CIs per-variant; applies Holm-Bonferroni via `multipletests(pvals, method='holm')`
- Reads `results/step_timing/*.json`; computes step CPU-time fractions and 95% t-CIs
- Reports bootstrap fallback flag if raw sample data is unavailable (W5)
- Criterion reproducibility note (W6): "Exact CI values may differ across runs; only
  mean point estimates and CI widths within measurement variance are expected to be stable"
- Produces `results/analysis/analysis_report.md`

### Phase 5: Dry Run

**Objective:** Verify end-to-end pipeline works before committing to full Criterion runs.

```bash
# Dry run H5 (single seed only)
tw_approx_runner --x data/merfish/merfish_n10k_x.npy --y data/merfish/merfish_n10k_y.npy \
  --k 15 --sample 5000 --seed 42 --output results/h5/h5_dry_run.json
# Verify: JSON contains {delta, wall_exact_s, wall_approx_s}

# Dry run Criterion (baseline only, n=1K only, reduced sample_size=5)
# Temporarily override sample_size in tw_baseline_bench.rs, run, restore
cargo criterion --bench tw_baseline_bench --features cli \
  --message-format=json -- --bench "baseline/1000" 2>&1 | head -20
# Verify: JSON-lines output with 'typical' field

# Dry run profiling (baseline only, 2 iters)
tw_profiler --x data/gaussian/gaussian_n100000_x.npy --y data/gaussian/gaussian_n100000_y.npy \
  --k 15 --n-warmup 1 --n-iters 2 --variant baseline \
  --output results/step_timing/dry_run.json
# Verify: JSON contains step_timing fields (tw_x_dist, tw_x_sort, etc.)

# Verify analysis script runs on dry-run data
python scripts/analyze_clean.py --dry-run
# Verify: analysis_report.md produced (may show inconclusive due to minimal data)
```

Resolve any failures before proceeding to Phase 6.

### Phase 6: Full Execution

Run in order:
1. `bash scripts/run_h5.sh` (~5 min)
2. `bash scripts/run_criterion_clean.sh` (~3 hours; leave unattended)
3. `bash scripts/run_profiling_clean.sh` (~30 min)
4. `python scripts/analyze_clean.py` (~2 min)

---

## Execution Protocol

1. Activate environment: `micromamba activate tw-perf-rerun-clean`
2. Change to new worktree: `cd /home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean`
3. Confirm machine is dedicated (no competing processes): `uptime; nproc; free -h`
4. Record hardware profile in `results/analysis/hardware_profile.txt`: CPU model, L1/L2/L3 cache sizes, RAM, NUMA topology, `N_THREADS` value used
5. Run `bash research/2026-04-05-tw-perf-rerun-clean/scripts/run_h5.sh`
6. Run `bash research/2026-04-05-tw-perf-rerun-clean/scripts/run_criterion_clean.sh` (leave in dedicated terminal; do not suspend)
7. After Criterion completes, run `bash research/2026-04-05-tw-perf-rerun-clean/scripts/run_profiling_clean.sh`
8. Run `python research/2026-04-05-tw-perf-rerun-clean/scripts/analyze_clean.py`
9. Review `results/analysis/analysis_report.md` for verdict on each hypothesis

**Re-run policy (RT-4 — Accept):** Single-machine, single-user environment with manual
supervision. If any benchmark must be re-run due to a system event (thermal throttle,
background process spike), ALL 5 variants must be re-run together. The first complete set
of results is the primary dataset. Raw `criterion_output.json` is published in full.

---

## Analysis Plan

### H5 Analysis

- Load all 10 `h5_trial_seed*.json` files; extract `wall_exact_s`, `wall_approx_s`, `delta`
- Speedup: compute per-trial `wall_exact_s / wall_approx_s`; report median and range
- Quality: compute per-trial `|delta|`; report mean ± 95% t-CI (t(0.975, df=9) ≈ 2.262, n=10)
- Verdict: H1 holds iff median speedup ≥ 5× AND CI upper bound of |delta| < 0.001
- Descriptive m-sweep results reported separately with no inferential comparison (W2)

### H-100K Analysis

- Parse `criterion_output.json` (JSON-lines from `cargo criterion --message-format=json`):
  extract per-benchmark-run timing samples. Each JSON line with `"reason": "benchmark-complete"`
  contains `"typical": {"estimate": <ns>, ...}` and optionally raw iteration times.
- For each variant at n=100K: bootstrap 10,000 ratio samples (variant_time / baseline_time)
  from Criterion's raw sample pool; report 95% CI of the ratio
- Extract p-values for "combined speedup > 1.5×" using Welch's t-test on log-times;
  apply Holm-Bonferroni correction across m=4 variants via `multipletests(pvals, method='holm')`
- Bootstrap fallback (W5): if raw Criterion samples are unavailable (e.g., JSON only has
  aggregate CI bounds), use conservative CI from aggregate bounds and flag output with
  "FALLBACK CI: uncertainty is understated relative to bootstrap on raw samples"
- Cache warm-state check (W4): compare forward-order (baseline first) vs reversed-order
  (combined first) point estimates; flag if they differ by >5%
- Criterion reproducibility note (W6): state that exact CI values may differ across runs;
  only mean point estimates and CI widths are stable within measurement variance

### H-partial-MERFISH Analysis

- From Criterion Gaussian output at n=50K: compute partial_rank speedup CI half-width
  = (CI_upper - CI_lower) / 2
- From Criterion MERFISH output at n=50K: compute same half-width for MERFISH partial_rank
- Compare: H1 holds iff MERFISH half-width < Gaussian half-width from this experiment
- Report both CIs and their difference; note that the Gaussian CI uses fresh seed-2026 data

### H0/H1-clean Analysis

- Load all `results/step_timing/gaussian_n100000_baseline.json` (baseline variant only for
  step-fraction analysis; combined omits several steps)
- For each of the 30 iterations, compute step CPU-time fractions: `step_ns / total_ns`
  where `total_ns = sum of all 6 step counts`
- Report mean ± 95% t-CI for each step fraction (t(0.975, df=29) ≈ 2.045)
- CI-ordering check (W3 — descriptive only, no p-value): verify that `tw_x_dist` 95% CI
  lower bound exceeds `tw_y_heap` 95% CI upper bound; report "dominant" or "no clear dominant"
- If no step fraction mean exceeds 40% and all 6 fractions have overlapping CIs, report
  "No clear dominant step — optimization target is indeterminate at d=10"

---

## Success Criteria

- **H5 — Conclusive positive:** Median speedup ≥ 5× AND 95% t-CI upper bound of |delta| < 0.001 across 10 seeds. Supports H1: tw_approx is field-deployable on MERFISH n=10K.
- **H5 — Conclusive negative:** Median speedup < 5× OR median |delta| ≥ 0.001. Supports H0: subsampling does not meet the pre-registered joint criterion at m=5000.
- **H5 — Inconclusive:** Speedup in near-threshold range [4.5×, 5.5×] with mixed per-seed results. Report borderline verdict; do not deploy without additional measurement.
- **H-100K — Conclusive positive:** Holm-corrected 95% CI lower bound for combined speedup at n=100K exceeds 1.5×. Supports H1: extrapolated speedup survives cache-regime boundary.
- **H-100K — Conclusive negative:** Holm-corrected CI lower bound ≤ 1.5×. Supports H0: shipped claim needs revision.
- **H-100K — Inconclusive:** Post-hoc CV exceeds 20% (n=63 yields <65% power at CV=20%); report with power caveat and recommend additional samples.
- **H-partial-MERFISH — Conclusive positive:** MERFISH CI half-width < Gaussian half-width (both from this fresh experiment). Supports H1: structured data reduces pivot variance.
- **H-partial-MERFISH — Conclusive negative:** MERFISH CI half-width ≥ Gaussian half-width. Supports H0: CI width is data-distribution-independent; wide CI persists on MERFISH.
- **H0/H1-clean — Estimation result:** Report mean ± 95% CI for all 6 step fractions. Flag which step (if any) is dominant. No binary pass/fail verdict.
- **Infrastructure:** All 6 bench binaries run successfully; `analyze_clean.py` produces `analysis_report.md` with all verdicts populated.

---

## Threats to Validity

### Internal

- **Rayon thread pool global state:** `build_global()` can only be called once per process.
  If cargo runs multiple bench binaries in the same process (it does not — each `[[bench]]`
  is a separate binary), thread count could be wrong. Mitigated by separate binaries.
- **Step-timing overhead:** `fetch_add(Relaxed)` on atomic counters introduces ~2–5ns
  overhead per call. At n=100K with 6 steps, this adds ~100K × 6 × 3ns ≈ 1.8ms of bias
  per iteration (< 1% at ~200ms/iter). Reported as a known bias in the report.
- **Criterion RNG non-determinism (W6):** Criterion's internal sampling uses a non-user-seeded
  RNG. Exact CI values will differ across runs; only point estimates and CI widths are stable.
- **Fresh Gaussian data (RT-2):** Different random realization from prior experiment.
  Per RT-2 decision: the specific realization is not expected to affect speedup ratios for
  this algorithm; risk is low. Prior results are no longer directly comparable to new data.

### External

- **Hardware specificity (W7):** All performance results are valid only for the specific
  benchmark machine (x86-64-v3, AVX2/FMA, documented in hardware_profile.txt). Results
  on ARM, non-AVX2 x86, or different cache sizes may differ substantially.
- **Rust toolchain specificity:** Results are valid for nightly-2026-03-26 only. Stable Rust
  results may differ due to codegen changes. A follow-up stable-Rust measurement is required
  before publishing production performance claims.
- **n=10K MERFISH scope:** H5 quality/speedup result is valid at n=10K, m=5000 only.
  The n/m=0.5 ratio cannot be maintained at n=100K with the same m; separate measurement needed.

---

## Red-Team Decisions

### RT-1 — 1.5× threshold (H-100K) — Accept

**Decision: Accept.** The 1.5× threshold is a conservative lower bound on the prior ~1.95×
result. It is not derived from a deployment latency requirement. The experiment tests whether
the speedup survives the cache-regime transition, not whether it meets a specific user SLA.
This is documented explicitly; a follow-up experiment against a concrete latency requirement
is recommended if H1 holds.

### RT-2 — Data re-use (Gaussian fixtures) — Mitigate

**Decision: Mitigate.** Fresh Gaussian data generated with seed 2026 (not re-using prior
experiment's data). The shape and distribution remain identical (randn, d=10); only the
specific realization changes. Prior Criterion results are NOT directly comparable to this
run; do not mix old and new Criterion output in the same analysis.

### RT-3 — Asymmetric Criterion parameters — Resolved by R7

**Decision: Resolved.** R7 is implemented: all n values have explicitly pinned Criterion
parameters. Confidence is structurally equal across all scales.

### RT-4 — No re-run protocol — Accept

**Decision: Accept.** Single-machine, single-user environment with manual supervision.
Re-run policy: "If any benchmark must be re-run due to a system event, ALL 5 variants must
be re-run together. The first complete set of results is the primary dataset." Full raw
`criterion_output.json` is published to prevent selective reporting.

### RT-5 — H5 5× speedup unexplained mechanism — Accept

**Decision: Accept.** The 5× threshold is empirically observed from prior data. The mechanism
is not fully characterized; the experiment will report the measured speedup and its components
(step fractions from profiling) without claiming a mechanistic explanation. If measured speedup
differs substantially from 5×, the threshold will be revisited in the report.

---

## Estimated Resource Requirements

- **Compute time:** ~3 hours total
  - H5 confirmatory (10 seeds): ~5 min
  - Criterion benches (5 variants × ~21 min at n=100K + smaller scales): ~2 hours
  - MERFISH partial_rank bench: ~10 min
  - Cache warm-state check (2 variants reversed): ~20 min
  - Profiling (5 variants × 35 iters × ~20s): ~35 min
- **Disk space:** ~500 MB (Gaussian data ~100MB, MERFISH ~25MB, Criterion target artifacts ~300MB)
- **Dependencies:**
  - Rust nightly-2026-03-26 (via rustup/rust-toolchain.toml)
  - cargo-criterion (must install: `cargo install cargo-criterion`)
  - Python: numpy, scipy, scikit-learn, statsmodels (via environment.yml)
- **Machine requirements:** Dedicated benchmark machine (no competing workloads); x86-64-v3 with AVX2/FMA for AVX2 variant to be meaningful; ≥16 GB RAM for n=100K data
