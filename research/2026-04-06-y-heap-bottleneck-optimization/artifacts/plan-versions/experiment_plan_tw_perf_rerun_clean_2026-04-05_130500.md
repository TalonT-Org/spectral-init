# Experiment Plan: Trustworthiness Performance Re-run (Clean Infrastructure)

## Motivation

The prior trustworthiness performance experiment (PR #224, branch `research-20260404-174030`)
produced four blocked or inconclusive hypotheses and seven identified measurement infrastructure
gaps. Results cannot be trusted at face value: the shipped combined speedup (~1.95×–2.15×) was
extrapolated across a cache-regime boundary, the per-step timing data is contaminated by ~6.25×
testing-feature overhead, the MERFISH H5 benchmark was never executed, and the partial-rank CI is
too wide to conclude anything at n=50K on synthetic data.

This experiment re-runs those measurements with clean infrastructure — isolated bench binaries,
adequate sample sizes, zero-cost step instrumentation, and rigorous statistical corrections — to
produce findings that can be committed to the production record without caveats.

**Scope limitation:** All performance results are valid for `nightly-2026-03-26`
(`rustc 1.96.0-nightly, 23903d01c`, x86-64-v3 AVX2/FMA) only. Stable Rust results may differ
due to codegen differences. A follow-up measurement on stable Rust is required before publishing
production performance claims.

---

## Hypotheses

### H5 — MERFISH Subsampling Quality and Speedup

**H0:** `trustworthiness_approx` with m=5000 fails to deliver ≥5× wall-clock speedup over exact
computation OR delivers |T_approx − T_exact| ≥ 0.001 on MERFISH n=10K data.

**H1:** `trustworthiness_approx` with m=5000 delivers both ≥5× median speedup (over 10 seeds)
AND |T_approx − T_exact| < 0.001 (95% CI upper bound) on MERFISH n=10K data.

**Scope:** This quality guarantee is valid at n=10K, m=5000 (ratio m/n=0.5) only. Extrapolation
to n=100K (m/n=0.05) requires additional measurement.

**Near-threshold triggers (pre-registered):**

- Speedup: if median speedup ∈ [4.5×, 5.5×] → flag as "borderline"; report individual trial
  breakdown; verdict requires majority of 10 trials ≥5× for H1 to hold.
- Quality: if median |delta| ∈ [0.0008, 0.0012] → flag as "borderline"; report per-trial values.

---

### H-100K — Criterion Speedup Validation at Production Scale

**H0 (null — skeptical claim):** The combined algorithm CI lower bound at n=100K is ≤ 1.5×
baseline (null: cache-regime shift from L3-resident to memory-bandwidth-bound between 50K and
100K absorbed most speedup; the extrapolated 1.95×–2.15× claim is an overestimate).

**H1 (alternative):** The combined algorithm CI lower bound at n=100K exceeds 1.5× baseline
(alternative: shipped speedup survives the cache boundary; extrapolation was valid or conservative).

**Power:** n=63 samples per group, CV=15%, r=10% relative effect: ~80% power (β≈0.20) at
Holm-corrected α/4=0.0125 for the first comparison. Minimum detectable effect at 80% power is
r≈10% at this threshold.

**CV=15% provenance:** Estimated from baseline variance visible in
`research/2026-04-04-tw-perf-scaling/results/criterion/bench_output.txt` (prior Criterion run,
n=1K–50K on Gaussian data). This estimate may differ at n=100K due to memory-bandwidth variance;
actual CV will be re-estimated from clean data and reported.

---

### H0/H1-clean — Per-Step Timing Estimation (Reframed)

This is an **estimation task, not a binary verdict.** The contaminated prior measurement
(~6.25× overhead from `#[cfg(feature="testing")]` eprintln! inside rayon closures) is discarded.

**Estimation goal:** Report mean and 95% CI for each of the 6 algorithmic steps as a fraction of
total wall time, at n=100K, d=50 (MERFISH PCA-50 regime), using clean profiling build.

**First-principles prediction:** At d=50, `tw_x_dist` is predicted to be the dominant step
because per-row work is O(n·d) = O(5×10⁶) FLOP at n=100K, d=50, compared to
O(n·log(k)) ≈ O(4×10⁵) for `tw_y_heap` and O(n) ≈ O(10⁵) for `select_nth_unstable_by`
— a ~12× raw FLOP advantage before accounting for AVX2 throughput and cache effects.
This prediction will be tested against the measured fractions. A step is "dominant" if
its fraction mean minus margin-of-error exceeds all others.

No threshold-based pass/fail verdict is assigned; the fractions inform future optimization
targeting.

---

### H-partial-MERFISH — Partial-Rank CI Width on Structured Data

**H0:** The Criterion CI half-width for `partial_rank` speedup at n=50K on MERFISH data
is ≥ 0.26 (same as the [1.10×, 1.62×] half-width from synthetic Gaussian in the prior run).

**H1:** The CI half-width is < 0.26, indicating that MERFISH PCA-50 distances have narrower
dynamic range than Gaussian, reducing `select_nth_unstable_by` pivot variance.

**Gaussian baseline source:** The Gaussian CI will come from the NEW isolated Criterion run
in this experiment (not from the contaminated prior data), to ensure comparability.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm variant | baseline, thread_local, partial_rank, avx2_kernel, combined | All 5 production candidates; matches research worktree functions. `avx512_kernel` out of scope (stub/conditional on AVX-512 host). |
| Input scale n | 1K, 5K, 10K, 25K, 50K, 100K | Spans from L3-resident to memory-bandwidth-bound; n=100K is the primary production claim. |
| Dataset (Criterion) | Gaussian (d=10, k=15) | Used for criterion speedup benchmarks for H-100K and H-partial-MERFISH. |
| Dataset (H5) | MERFISH PCA-50 (d=50, n=10K) | Structured biological data; required for H5 hypothesis. |
| Subsampling count m (H5 sweep) | 500, 1000, 2000, 5000, 10000 | Range of subsampling ratios; m=5000 is the primary confirmatory gate. |
| Random seed (H5) | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 | 10 seeds for reliable median estimate (per R7). |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| `wall_clock_speedup` | ratio (×) | `wall_exact_s / wall_approx_s` from `tw_approx_runner` JSON; median over 10 seeds | NEW |
| `delta_tw` | dimensionless | `|t_approx − t_exact|` from `tw_approx_runner` JSON field `delta`; mean ± 95% CI over 10 seeds | NEW |
| `criterion_speedup_n100k` | ratio (×) | Bootstrap ratio CI from Criterion individual timing samples at n=100K; see Analysis Plan | NEW |
| `partial_rank_ci_half_width` | ratio | Half-width of Criterion speedup CI for partial_rank vs baseline at n=50K | NEW |
| `tw_x_dist_fraction` | fraction [0,1] | Mean fraction of total wall time in `tw_x_dist` step across 30 profiling iterations; ± 95% CI | NEW |
| `tw_y_heap_fraction` | fraction [0,1] | Mean fraction of total wall time in `tw_y_heap` step; ± 95% CI | NEW |
| `tw_x_sort_fraction` | fraction [0,1] | Mean fraction in `tw_x_sort` (partial sort / select_nth_unstable_by); ± 95% CI | NEW |
| `tw_rank_scatter_fraction` | fraction [0,1] | Mean fraction in `tw_rank_scatter`; ± 95% CI | NEW |
| `tw_x_knn_set_fraction` | fraction [0,1] | Mean fraction in `tw_x_knn_set`; ± 95% CI | NEW |
| `tw_penalty_fraction` | fraction [0,1] | Mean fraction in `tw_penalty` accumulation; ± 95% CI | NEW |

**Note on canonical names:** `src/metrics.rs` defines no performance-dimension metric constants.
All performance DVs are NEW (no canonical entry) and are computed entirely in analysis scripts.
The existing `metrics.rs` metric infrastructure covers Accuracy and Parity dimensions only.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighborhood size) | 15 | Matches prior experiment and integration tests |
| Gaussian dimensionality d | 10 | Matches prior experiment bench configuration |
| MERFISH dimensionality d | 50 (PCA components) | Fixed by `prepare_merfish.py` pipeline |
| MERFISH cell count | 10K (first 10K of 100K subset) | Fixed by `prepare_merfish.py`; H5 prerequisite |
| Random seed (Gaussian data) | Per-scale seed (existing data reused) | Existing data at old worktree data dir; not regenerated |
| Rust toolchain | nightly-2026-03-26 | Pinned in `rust-toolchain.toml`; matches prior experiment |
| Benchmark machine | Single dedicated host throughout all phases | Cross-variant comparisons require same hardware |
| Benchmark isolation | 1 variant per Criterion binary, 60s cool-down between runs | Eliminates thread-local contamination and thermal bias |
| Profiling iters | 30 (clean build) | Per R4; provides adequate statistical stability |
| Profiling warmup | 5 iterations (discarded) | Warm CPU caches before timing |

---

## Inputs and Data

The experiment uses three data sources:

1. **MERFISH n=10K fixtures** — generated by `prepare_merfish.py` from
   `temp/merfish_100k/merfish_100k_expression.npz` (confirmed present, 24MB, all 5 NPZ artifacts
   at `temp/merfish_100k/` exist as of 2025-04-04). The script runs PCA(50) on the first 10K
   cells of the 100K subset. Output: `tests/fixtures/merfish/merfish_n10k_x.npy` (10K × 50 f64),
   `tests/fixtures/merfish/merfish_n10k_y.npy` (10K × 2 f64). **Currently missing — must be
   generated in Phase 3.**

2. **Gaussian benchmark data** — already present at
   `research/2026-04-04-tw-perf-scaling/data/gaussian/` (n × {_x, _y} .npy pairs for all six
   n values: 1K, 5K, 10K, 25K, 50K, 100K; generated by `gen_synthetic.py` with randn; 18MB total).
   The new experiment references these files in-place; no regeneration needed.

3. **Raw MERFISH H5AD** — `data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad` (2.0GB) — present but
   not directly used by this experiment (already pre-processed into `temp/merfish_100k/`).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| MERFISH n=10K x/y | Generated from `temp/merfish_100k/` via `prepare_merfish.py` | 10K × 50 f64, 10K × 2 f64; real biological structure | H5 speedup/quality gate; H-partial-MERFISH |
| Gaussian n=100K d=10 | Existing at old worktree data dir | 100K × 10 f64; randn; fits well in L3 | H-100K primary Criterion bench |
| Gaussian n=1K–50K d=10 | Existing at old worktree data dir | Same generation; 1K/5K/10K/25K/50K scales | H-partial-MERFISH baseline CI comparison |

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder, created inside the new git worktree:

```
research/2026-04-05-tw-perf-rerun-clean/
├── environment.yml                   # Micromamba/conda environment spec
├── scripts/
│   ├── apply_phase1_changes.sh       # Documents and verifies Phase 1 source code changes
│   ├── prepare_data.sh               # Runs prepare_merfish.py; verifies all inputs present
│   ├── run_h5.sh                     # H5 confirmatory: 10-seed multi-trial tw_approx_runner
│   ├── run_criterion_clean.sh        # Criterion bench: isolated per-variant with 60s cool-down
│   ├── run_profiling_clean.sh        # tw_profiler: 30 iters, profiling feature (no testing)
│   └── analyze_clean.py             # Analysis: bootstrap CI, Holm correction, step fractions
├── data/
│   └── merfish/                      # merfish_n10k_x.npy, merfish_n10k_y.npy (from prepare_data.sh)
├── results/
│   ├── h5/                           # h5_trial_seed{42..51}.json
│   ├── criterion/                    # criterion_output.json (JSON-lines from cargo criterion)
│   ├── step_timing/                  # gaussian_n100000_{variant}.json (30-iter w/ per-step stats)
│   └── analysis/                     # analysis_report.md with verdicts
└── report.md                         # Final report (written by write-report skill)
```

Source code changes (Phase 1) live in the worktree `src/` and `benches/` directories — not inside
the experiment folder. The new bench files are:

```
benches/
├── tw_baseline_bench.rs              # NEW: isolated bench for trustworthiness
├── tw_thread_local_bench.rs          # NEW: isolated bench for trustworthiness_thread_local
├── tw_partial_rank_bench.rs          # NEW: isolated bench for trustworthiness_partial_rank
├── tw_avx2_bench.rs                  # NEW: isolated bench for trustworthiness_avx2_kernel
└── tw_combined_bench.rs              # NEW: isolated bench for trustworthiness_combined
```

The existing `trustworthiness_bench.rs` (all 5 groups in one binary) is **removed** and replaced
by the 5 isolated files above. It is the root cause of the thread-local contamination bias.

---

## Environment

**Custom environment required.** The experiment requires Python for data preparation and
analysis. The prior experiment's `environment.yml` is adapted with `statsmodels` added
(was missing) and all versions pinned exactly.

```yaml
name: tw-perf-rerun-clean
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2.6
  - scipy=1.15.2
  - scikit-learn=1.6.0
  - statsmodels=0.14.6
  - pip
# Rust toolchain: nightly-2026-03-26 (pinned in rust-toolchain.toml)
# cargo-criterion: install via `cargo install cargo-criterion` (not on conda)
```

**Rust toolchain:** `rust-toolchain.toml` must be created in the new worktree root (currently
missing from main; worktree pins `nightly-2026-03-26` — carry this forward):

```toml
[toolchain]
channel = "nightly-2026-03-26"
profile = "minimal"
components = ["rustfmt", "clippy"]
```

**cargo-criterion:** Must be installed via `cargo install cargo-criterion`. Confirmed NOT
installed on the host system. Required for `--message-format=json` stable Criterion output.

**Hardware profile (to be filled by implementer before publishing results):**
Document CPU model, L1/L2/L3 cache sizes (critical for cache-regime analysis), RAM, NUMA
topology, and whether the system is dedicated or shared. This allows reproducers to assess
hardware comparability when evaluating the H-100K cache-regime hypothesis.

---

## Implementation Phases

### Phase 1: Source Code Changes

Create new git worktree branching from `research-20260404-174030`:

```bash
git worktree add \
  /home/talon/projects/worktrees/research-20260405-tw-perf-rerun-clean \
  -b research-20260405-tw-perf-rerun-clean \
  research-20260404-174030
```

Apply the following 6 changes. Each change is precisely described so it can be scripted or
applied manually. `apply_phase1_changes.sh` verifies each change was applied correctly via
`cargo check`.

**Change 1 — `rust-toolchain.toml`:**
Create file at worktree root with content shown in Environment section above.

**Change 2 — `Cargo.toml` features:** Add `profiling` feature (zero-cost step timing without
eprintln!):

```toml
[features]
# existing features unchanged
testing = ["dep:serde"]
cli     = ["dep:ndarray-npy", "dep:pico-args", "dep:serde_json"]
profiling = []    # NEW: enables atomic step timing; no additional deps
```

**Change 3 — `Cargo.toml` bench entries:** Remove the existing single `trustworthiness_bench`
entry and replace with 5 isolated entries:

```toml
# Remove:
[[bench]]
name = "trustworthiness_bench"
harness = false

# Add (5 entries, one per variant):
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
```

**Change 4 — `src/metrics.rs` profiling instrumentation:** At the top of the file, add 6
atomic counters under the `profiling` feature flag:

```rust
#[cfg(feature = "profiling")]
mod step_timing {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static X_DIST_NS:    AtomicU64 = AtomicU64::new(0);
    pub static X_SORT_NS:    AtomicU64 = AtomicU64::new(0);
    pub static RANK_SCATTER_NS: AtomicU64 = AtomicU64::new(0);
    pub static X_KNN_SET_NS: AtomicU64 = AtomicU64::new(0);
    pub static Y_HEAP_NS:    AtomicU64 = AtomicU64::new(0);
    pub static PENALTY_NS:   AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        X_DIST_NS.store(0, Ordering::Relaxed);
        X_SORT_NS.store(0, Ordering::Relaxed);
        RANK_SCATTER_NS.store(0, Ordering::Relaxed);
        X_KNN_SET_NS.store(0, Ordering::Relaxed);
        Y_HEAP_NS.store(0, Ordering::Relaxed);
        PENALTY_NS.store(0, Ordering::Relaxed);
    }

    pub fn read() -> [(&'static str, u64); 6] {
        [
            ("tw_x_dist",       X_DIST_NS.load(Ordering::Relaxed)),
            ("tw_x_sort",       X_SORT_NS.load(Ordering::Relaxed)),
            ("tw_rank_scatter", RANK_SCATTER_NS.load(Ordering::Relaxed)),
            ("tw_x_knn_set",    X_KNN_SET_NS.load(Ordering::Relaxed)),
            ("tw_y_heap",       Y_HEAP_NS.load(Ordering::Relaxed)),
            ("tw_penalty",      PENALTY_NS.load(Ordering::Relaxed)),
        ]
    }
}
```

Inside the `trustworthiness` per-row parallel closure, wrap each step's body with
`#[cfg(feature = "profiling")]` timing blocks that use `Instant::now()` + `fetch_add(Relaxed)`:

```rust
// Example for tw_x_dist step:
#[cfg(feature = "profiling")]
let t0 = std::time::Instant::now();
// ... existing tw_x_dist code ...
#[cfg(feature = "profiling")]
step_timing::X_DIST_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
```

Apply the same pattern to all 6 steps in the `trustworthiness` function. Do NOT add to variant
functions (they share the inner steps through the combined function or have their own structure).
**Important:** The `testing` feature eprintln! calls remain unchanged; they are only active when
`testing` is in the feature set.

**Change 5 — `src/bin/tw_profiler.rs` step-timing collection:** In the research worktree
`tw_profiler.rs` (which has `--variant` flag), add after each timed iteration call (inside the
iteration loop, after the variant function returns):

```rust
#[cfg(feature = "profiling")]
{
    let step_readings = crate::metrics::step_timing::read();
    for (name, ns) in &step_readings {
        step_accum.entry(name).or_default().push(*ns);
    }
    crate::metrics::step_timing::reset();
}
```

Then in the JSON output, compute per-step mean/std/CI and emit:

```json
{
  "step_fractions": {
    "tw_x_dist": { "mean": 0.62, "std": 0.03, "ci_lower_95": 0.57, "ci_upper_95": 0.67 },
    ...
  }
}
```

Fractions computed as `step_ns / sum(all_steps_ns)` per iteration, then mean/std/CI across
iterations. Also emit a `no_op_overhead_ns` field: time a zero-work loop to measure
`fetch_add` overhead per step; subtract if > 0.5% of average step time.

**Change 6 — New bench files:** Create 5 bench files in `benches/`. Each file follows this
template (example for `tw_baseline_bench.rs`):

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use ndarray::Array2;
use spectral_init::trustworthiness;
use std::time::Duration;

fn load_gaussian(n: usize) -> (Array2<f64>, Array2<f64>) {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-04-04-tw-perf-scaling/data/gaussian");
    let x = ndarray_npy::read_npy(base.join(format!("gaussian_n{n}_x.npy"))).unwrap();
    let y = ndarray_npy::read_npy(base.join(format!("gaussian_n{n}_y.npy"))).unwrap();
    (x, y)
}

fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("tw_baseline");
    for &n in &[1000usize, 5000, 10000, 25000, 50000] {
        let (x, y) = load_gaussian(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| trustworthiness(x.view(), y.view(), 15))
        });
    }
    // n=100K: Flat mode, 63 samples, extended timing
    {
        let (x, y) = load_gaussian(100000);
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(63);
        group.warm_up_time(Duration::from_secs(30));
        group.measurement_time(Duration::from_secs(1500));
        group.bench_with_input(BenchmarkId::from_parameter(100000usize), &100000usize, |b, _| {
            b.iter(|| trustworthiness(x.view(), y.view(), 15))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_baseline);
criterion_main!(benches);
```

Adapt for each variant (`trustworthiness_thread_local`, `trustworthiness_partial_rank`,
`trustworthiness_avx2_kernel`, `trustworthiness_combined`). The `ndarray-npy` crate must be
added as a `dev-dependency` if not already present.

**Verification:**

```bash
cargo check --features cli,profiling --release
cargo check --features cli,testing --release
cargo bench --bench tw_baseline_bench --no-run
```

All must pass before proceeding.

---

### Phase 2: Environment Setup

```bash
# From new worktree root:
cargo install cargo-criterion          # Install once per host; ~5 min build time
micromamba env create -f research/2026-04-05-tw-perf-rerun-clean/environment.yml
micromamba activate tw-perf-rerun-clean
python -c "import statsmodels; import scipy; import sklearn; print('env OK')"
```

Create experiment directory structure:

```bash
mkdir -p research/2026-04-05-tw-perf-rerun-clean/{scripts,data/merfish,results/{h5,criterion,step_timing,analysis}}
```

---

### Phase 3: Data Preparation

Run `prepare_data.sh` which performs:

1. **MERFISH source verification:**

```bash
MERFISH_SRC="/home/talon/projects/spectral-init/temp/merfish_100k"
if [ ! -f "$MERFISH_SRC/merfish_100k_expression.npz" ]; then
  echo "ERROR: MERFISH 100K source not found at $MERFISH_SRC"
  echo "All 5 NPZ files should be present (expression, spatial, labels, section_ids, meta.json)."
  echo "If missing, generate via: python tests/visual_eval/generate_merfish_subset.py \\"
  echo "  --n-cells 100000 --output-dir temp/merfish_100k"
  exit 1
fi
echo "MERFISH source OK: $(ls $MERFISH_SRC/*.npz | wc -l) NPZ files found"
```

2. **Generate MERFISH n=10K x/y fixtures:**

```bash
# From worktree root:
micromamba run -n tw-perf-rerun-clean \
  python research/2026-04-04-tw-perf-scaling/scripts/prepare_merfish.py
# Outputs: tests/fixtures/merfish/merfish_n10k_x.npy (10K×50 f64)
#          tests/fixtures/merfish/merfish_n10k_y.npy (10K×2 f64)
```

3. **Copy MERFISH fixtures to experiment data dir:**

```bash
cp tests/fixtures/merfish/merfish_n10k_{x,y}.npy \
   research/2026-04-05-tw-perf-rerun-clean/data/merfish/
```

4. **Verify Gaussian data:**

```bash
for N in 1000 5000 10000 25000 50000 100000; do
  F="research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n${N}_x.npy"
  [ -f "$F" ] || { echo "MISSING: $F"; exit 1; }
  echo "OK: $F"
done
```

5. **Verify shape assertions:**

```bash
micromamba run -n tw-perf-rerun-clean python -c "
import numpy as np
x = np.load('research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy')
y = np.load('research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy')
assert x.shape == (10000, 50), f'Expected (10000,50), got {x.shape}'
assert y.shape == (10000, 2),  f'Expected (10000,2), got {y.shape}'
print('MERFISH fixtures OK:', x.shape, y.shape)
"
```

---

### Phase 4: Dry Run

Verify end-to-end pipeline with minimal inputs before committing to full runs:

```bash
# Build binaries without testing feature
cargo build --release --features cli,profiling

# Smoke test: tw_profiler at n=1K, 3 iters, baseline variant
./target/release/tw_profiler \
  --x research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n1000_x.npy \
  --y research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n1000_y.npy \
  --k 15 --iters 3 --warmup 1 --variant baseline \
  --output research/2026-04-05-tw-perf-rerun-clean/results/step_timing/dry_run.json
cat research/2026-04-05-tw-perf-rerun-clean/results/step_timing/dry_run.json

# Smoke test: tw_approx_runner on MERFISH
cargo build --release --features cli
./target/release/tw_approx_runner \
  --x research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy \
  --y research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy \
  --k 15 --sample 5000 --seed 42 \
  --output research/2026-04-05-tw-perf-rerun-clean/results/h5/dry_run.json
cat research/2026-04-05-tw-perf-rerun-clean/results/h5/dry_run.json

# Smoke test: Criterion at n=1K, 3 samples
cargo criterion --bench tw_baseline_bench --message-format=json 2>/dev/null | head -5
```

Expected: all three produce valid JSON output with no panics. Proceed only if dry run passes.

---

### Phase 5: H5 Confirmatory (MERFISH Subsampling)

Build `tw_approx_runner` WITHOUT `testing` feature:

```bash
cargo build --release --features cli --no-default-features
```

Run 10-seed multi-trial confirmatory:

```bash
# run_h5.sh
RESULT_DIR="research/2026-04-05-tw-perf-rerun-clean/results/h5"
X="research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy"
Y="research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy"

for SEED in 42 43 44 45 46 47 48 49 50 51; do
  echo "Running seed $SEED..."
  ./target/release/tw_approx_runner \
    --x "$X" --y "$Y" \
    --k 15 --sample 5000 --seed "$SEED" \
    --output "$RESULT_DIR/h5_trial_seed${SEED}.json"
done

# Compute summary
micromamba run -n tw-perf-rerun-clean python -c "
import json, glob, numpy as np
results = [json.load(open(f)) for f in sorted(glob.glob('$RESULT_DIR/h5_trial_seed*.json'))]
speedups = [r['wall_exact_s'] / r['wall_approx_s'] for r in results]
deltas = [abs(r['delta']) for r in results]
print(f'Speedup: median={np.median(speedups):.2f}x, IQR=[{np.percentile(speedups,25):.2f}, {np.percentile(speedups,75):.2f}]')
print(f'|delta|: mean={np.mean(deltas):.6f}, 95CI_upper={np.mean(deltas)+1.96*np.std(deltas)/len(deltas)**0.5:.6f}')
print(f'Near-threshold speedup: {4.5 <= np.median(speedups) <= 5.5}')
print(f'Near-threshold delta:   {0.0008 <= np.mean(deltas) <= 0.0012}')
" | tee "$RESULT_DIR/h5_summary.txt"
```

Run subsampling sweep for secondary analysis (m ∈ {500, 1K, 2K, 5K, 10K}):

```bash
for M in 500 1000 2000 5000 10000; do
  micromamba run -n tw-perf-rerun-clean python \
    research/2026-04-04-tw-perf-scaling/scripts/subsampling_sweep.py \
    --x "$X" --y "$Y" --m "$M" --seed 99 --n-trials 5 \
    --output "$RESULT_DIR/sweep_m${M}.json"
done
```

---

### Phase 6: Criterion Benchmark (Isolated, n=1K–100K)

Build release (no testing, no profiling):

```bash
cargo build --release --features cli
```

Run per-variant with isolation:

```bash
# run_criterion_clean.sh
OUT="research/2026-04-05-tw-perf-rerun-clean/results/criterion"

# Optional: sudo cpupower frequency-set -g performance  (if available)

for BENCH in tw_baseline_bench tw_thread_local_bench tw_partial_rank_bench tw_avx2_bench tw_combined_bench; do
  echo "=== Running $BENCH ==="
  cargo criterion --bench "$BENCH" --message-format=json 2>/dev/null \
    >> "$OUT/criterion_output.json"
  echo "Cooling down (60s)..."
  sleep 60
done
echo "Criterion complete. Lines: $(wc -l < "$OUT/criterion_output.json")"
```

**Total estimated runtime:** n=1K–50K ≈ 15–30 min per variant × 5 + cool-downs.
n=100K ≈ 21 min (baseline) + 10.5 min (combined) + intermediate variants + cool-downs.
**Full Criterion phase: ~3.5–4 hours total.**

---

### Phase 7: Clean Step Timing (profiling Feature)

Build with `profiling` feature, without `testing`:

```bash
cargo build --release --features "cli,profiling" --no-default-features
```

Run per-variant at n=100K with cool-down:

```bash
# run_profiling_clean.sh
OUT="research/2026-04-05-tw-perf-rerun-clean/results/step_timing"
X_100K="research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n100000_x.npy"
Y_100K="research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n100000_y.npy"

for VARIANT in baseline thread_local partial_rank avx2_kernel combined; do
  echo "=== Profiling $VARIANT ==="
  ./target/release/tw_profiler \
    --x "$X_100K" --y "$Y_100K" \
    --k 15 --iters 30 --warmup 5 \
    --variant "$VARIANT" \
    --output "$OUT/gaussian_n100000_${VARIANT}.json"
  echo "Cooling down (60s)..."
  sleep 60
done
```

**Expected output per JSON:** `{ variant, n, iters: [f64 × 30], mean_s, std_s, warmup,
step_fractions: { tw_x_dist: {mean, std, ci_lower_95, ci_upper_95}, ... } }`

**Estimated runtime:** ~1 min × 30 iters × 5 variants + cool-downs ≈ 2–2.5 hours.

---

### Phase 8: Analysis

```bash
micromamba run -n tw-perf-rerun-clean python \
  research/2026-04-05-tw-perf-rerun-clean/scripts/analyze_clean.py \
  --results-dir research/2026-04-05-tw-perf-rerun-clean/results \
  --output research/2026-04-05-tw-perf-rerun-clean/results/analysis/analysis_report.md
```

`analyze_clean.py` performs the operations described in the Analysis Plan below.

---

## Execution Protocol

Execute phases in order. Each phase has a pass/fail gate; do not proceed if a gate fails.

1. `scripts/apply_phase1_changes.sh` — verify all 6 code changes; runs `cargo check`
2. Phase 2 environment setup — verify Python packages importable
3. `scripts/prepare_data.sh` — all shape assertions pass
4. **Dry run** — 3 JSON outputs produced without errors
5. `scripts/run_h5.sh` — produces 10 trial JSONs + summary; check for panics
6. `scripts/run_criterion_clean.sh` — produces `criterion_output.json`; verify JSON is valid
7. `scripts/run_profiling_clean.sh` — produces 5 step-timing JSONs
8. `scripts/analyze_clean.py` — produces `analysis_report.md`

---

## Analysis Plan

### Holm-Bonferroni Correction Family (pre-registered, m=4)

The following table is committed before any data collection. Rankings are assigned at analysis
time (sort by p-value ascending, compare to Holm-adjusted threshold):

| DV | In Holm family? | Holm threshold | Test |
|----|----------------|----------------|------|
| `wall_clock_speedup` (H5) | No — deterministic threshold gate | — | Median ≥5× |
| `delta_tw` (H5) | No — deterministic threshold gate | — | 95% CI upper bound <0.001 |
| `criterion_speedup_n100k` (baseline vs partial_rank) | **Yes** | α/(5−rank) | Bootstrap CI lower bound vs 1.5× |
| `criterion_speedup_n100k` (baseline vs thread_local) | **Yes** | α/(5−rank) | Bootstrap CI lower bound vs 1.5× |
| `criterion_speedup_n100k` (baseline vs avx2_kernel) | **Yes** | α/(5−rank) | Bootstrap CI lower bound vs 1.5× |
| `criterion_speedup_n100k` (baseline vs combined) | **Yes** | α/(5−rank) | Bootstrap CI lower bound vs 1.5× |
| `partial_rank_ci_half_width` | No — informational comparison only | — | Width < 0.26 |
| Step fractions (H0/H1-clean) | No — estimation task, no test | — | Report distribution |

**Holm procedure:** Sort the 4 Criterion comparisons by p-value ascending. Compare p_k ≤
α/(m+1−k) where m=4, k=1,2,3,4. Use `statsmodels.stats.multitest.multipletests(pvals, method='holm')`.

### Bootstrap CI for Criterion Speedup Ratio (R8)

After running `cargo criterion`, parse Criterion's on-disk timing data:

```python
import json, numpy as np
from pathlib import Path

def load_criterion_samples(target_dir, bench_name, n):
    """Load individual timing samples from Criterion's on-disk data."""
    # Criterion stores raw samples in target/criterion/<group>/<bench>/new/
    crit_dir = Path(target_dir) / "criterion" / bench_name / str(n) / "new"
    estimates_file = crit_dir / "estimates.json"
    # Try raw sample file first (Criterion internal format)
    sample_file = crit_dir / "raw.csv"  # older format
    if sample_file.exists():
        import csv
        with open(sample_file) as f:
            return [float(r[0]) for r in csv.reader(f) if r]
    # Fallback: use Criterion CI bounds with conservative ratio CI
    with open(estimates_file) as f:
        est = json.load(f)
    mean = est["mean"]["point_estimate"]
    ci_lo = est["mean"]["confidence_interval"]["lower_bound"]
    ci_hi = est["mean"]["confidence_interval"]["upper_bound"]
    return mean, ci_lo, ci_hi

def bootstrap_ratio_ci(baseline_samples, variant_samples, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        b = rng.choice(baseline_samples, len(baseline_samples))
        v = rng.choice(variant_samples, len(variant_samples))
        ratios.append(np.mean(b) / np.mean(v))
    return np.percentile(ratios, [2.5, 97.5]), np.mean(baseline_samples) / np.mean(variant_samples)

# If individual samples unavailable, use conservative propagation:
# speedup_lower = baseline_ci_lower / combined_ci_upper
# speedup_upper = baseline_ci_upper / combined_ci_lower
```

### H5 Analysis

```python
# From h5_trial_seed{42..51}.json
speedups = [r['wall_exact_s'] / r['wall_approx_s'] for r in results]
deltas = [abs(r['delta']) for r in results]
median_speedup = np.median(speedups)
delta_ci_upper = np.mean(deltas) + 1.96 * np.std(deltas, ddof=1) / np.sqrt(len(deltas))

h5_verdict = (
    "H1 SUPPORTED" if median_speedup >= 5.0 and delta_ci_upper < 0.001
    else "BORDERLINE" if (4.5 <= median_speedup <= 5.5 or 0.0008 <= np.mean(deltas) <= 0.0012)
    else "H0 NOT REJECTED"
)
```

### Step Fraction Analysis

```python
# For each variant JSON:
fracs_per_iter = []
for i, iteration_step_ns in enumerate(per_iter_step_readings):
    total = sum(iteration_step_ns.values())
    fracs_per_iter.append({k: v/total for k, v in iteration_step_ns.items()})

# Per-step CI across 30 iterations:
for step in step_names:
    vals = [f[step] for f in fracs_per_iter]
    mean, std = np.mean(vals), np.std(vals, ddof=1)
    ci = (mean - 1.96*std/np.sqrt(len(vals)), mean + 1.96*std/np.sqrt(len(vals)))

# H0/H1-clean: report fractions; note whether tw_x_dist CI lower bound exceeds all other steps
```

---

## Success Criteria

| Hypothesis | Conclusive Positive | Conclusive Negative | Inconclusive |
|-----------|---------------------|---------------------|--------------|
| **H5** | Median speedup ≥5× AND 95% CI upper bound of \|delta\| < 0.001 | Median speedup < 4.5× OR 95% CI lower bound of \|delta\| > 0.002 | Either metric in near-threshold window AND borderline flag raised |
| **H-100K** | CI lower bound > 1.5× for combined vs baseline (reject H0) | CI upper bound < 1.5× for combined (extrapolation confirmed overstated) | CI spans 1.5× (cannot distinguish H0 from H1 at current sample size) |
| **H0/H1-clean** | `tw_x_dist` fraction CI lower bound exceeds all other steps' CI upper bounds (unambiguous dominant step) | Any other step's CI lower bound exceeds `tw_x_dist` CI upper bound | Overlapping CIs among top 2 steps (no clear dominant step identified) |
| **H-partial-MERFISH** | CI half-width on MERFISH < 0.26 (Gaussian CI half-width from new isolated run) | CI half-width on MERFISH ≥ 0.26 (pivot variance is data-independent) | Half-width comparison underpowered if new Gaussian baseline CI differs significantly |
| **Infrastructure** | All 7 measurement gaps addressed (isolated benches, clean profiling, n=100K Criterion, Holm correction, multi-seed H5, bootstrap CI, toolchain pin) | Any gap unaddressed → mark as remaining limitation in report | — |

---

## Threats to Validity

### Internal

- **Profiling overhead:** `fetch_add(Relaxed)` atomic ops in the rayon parallel closure add
  ~1–5 ns per step call. At n=100K rows, 6 steps, total overhead ≈ 6M × 3 ns ≈ 18 ms per
  iteration (vs ~10–20 s total). Mitigation: measure no-op overhead; subtract if > 0.5%.

- **Rayon thread scheduling:** Atomic accumulation sums work across all rayon threads; if threads
  are not balanced, per-step fractions reflect load imbalance as well as algorithmic cost.
  Mitigation: report `mean ± CI` across 30 iterations; note thread count in results.

- **MERFISH single-dataset:** H5 quality claim (|delta| < 0.001) is validated only on one
  structured biological dataset at one scale (n=10K, m=5000). Do NOT claim generality to
  arbitrary biological data. Scope statement: "valid for MERFISH-like structured PCA-50 data."

- **Cache warm state at n=100K:** Even with 60s cool-down, L3 cache may retain some data from
  prior bench variant. Mitigation: separate binaries eliminate thread-local contamination; 60s is
  sufficient for thermal recovery on most modern CPUs. Residual bias assessed by comparing
  first-variant results across repeated runs if time permits.

- **Nightly toolchain:** All speedup ratios are valid for nightly-2026-03-26 only.
  Codegen differences on stable Rust may shift speedups by 5–15%.

### External

- **Hardware generalizability:** Results are specific to the test machine's cache hierarchy and
  memory subsystem. The cache-regime hypothesis (H-100K) is particularly hardware-dependent.
  Reproducers must document and compare hardware profiles.

- **Dataset generalizability:** Gaussian d=10 may not represent all production use cases.
  MERFISH PCA-50 is one biological domain; other high-dimensional data types (scRNA-seq, spatial)
  may show different pivot variance behavior (H-partial-MERFISH).

- **Extrapolation to larger n:** Results at n=100K cannot be directly extrapolated to n=250K+
  without measurement, as memory-bandwidth effects increase nonlinearly.

---

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| Disk space | ~500 MB (Criterion HTML reports, step timing JSONs, MERFISH data copies) |
| Phase 5 (H5) runtime | ~2–5 min (tw_approx_runner at n=10K, 10 trials × 2 runs each) |
| Phase 6 (Criterion) runtime | ~3.5–4 hours (5 variants × n=1K–100K, 63 samples at n=100K, 60s cool-downs) |
| Phase 7 (tw_profiler) runtime | ~2–2.5 hours (5 variants × 30 iters × ~2 min at n=100K + cool-downs) |
| **Total active compute** | **~6–7 hours** |
| cargo-criterion install | ~5 min (first time only) |
| conda env creation | ~3 min |
| Additional dependencies | `cargo-criterion` (Rust), `statsmodels=0.14.6` (Python, already installed on host) |
