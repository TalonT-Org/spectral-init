# Experiment Plan: Trustworthiness Performance Re-run — Clean Measurement

**Experiment directory:** `research/2026-04-05-tw-perf-rerun-clean/`  
**Worktree:** `/home/talon/projects/worktrees/research-20260404-174030/`  
**Scope report:** `.autoskillit/temp/scope/scope_tw_perf_rerun_unanswered_2026-04-05_122253.md`

---

## Motivation

PR #224 (branch `research-20260404-174030`) shipped a trustworthiness performance experiment
whose results were compromised by seven measurement infrastructure defects: benchmark isolation
failure, testing-feature overhead contamination (~6.25×), missing Criterion JSON output,
dead analysis scaffolding, absent toolchain pin, insufficient sample size, and unimplemented
per-step timing capture. The experiment left four hypotheses either N/A or inconclusive:

- **H5** (MERFISH subsampling): blocked by missing data — now unblocked (`temp/merfish_100k/*.npz`
  confirmed present)
- **H-100K**: production speedup at n=100K was extrapolated from n=50K across a
  cache-regime boundary (13% overestimate possible); never directly measured
- **H0/H1-clean**: per-step timing contaminated by eprintln! overhead; true step-fraction
  breakdown unknown
- **H-partial-MERFISH**: partial_rank showed wide CI [1.10×–1.62×] on synthetic Gaussian;
  whether this is data-distribution-dependent is unknown

This experiment re-runs all four measurements with a clean infrastructure and fixes all seven
gaps. Results will either validate the shipped combined algorithm's performance claims or
identify where they need revision.

---

## Hypotheses

**H5 — Subsampling quality and speedup on MERFISH:**  
Null (H0): `trustworthiness_approx` with m=5000 on MERFISH n=10K delivers neither ≥5×
wall-clock speedup nor |T_approx − T_exact| < 0.001.  
Alternative (H1): `trustworthiness_approx` delivers ≥5× speedup AND |delta| < 0.001 on
structured MERFISH data.

**H-100K — Direct n=100K Criterion speedup:**  
Null (H0): The combined variant's Criterion CI at n=100K is entirely above 1.5× (speedup
claim holds; extrapolation was not overestimated).  
Alternative (H1): The combined speedup CI at n=100K is lower than the extrapolated 1.95×–2.15×
(algorithm has crossed into memory-bandwidth-bound regime, reducing AVX2 benefit).

**H0/H1-clean — Per-step time fractions at n=100K, d=50:**  
Null (H0): No single step dominates (all steps contribute < 50% of wall time).  
Alternative (H1): X-dist (`tw_x_dist`) alone accounts for > 50% of wall time at d=50,
making AVX2-kernel acceleration the single highest-leverage optimization target.

**H-partial-MERFISH — Partial_rank CI width on structured data:**  
Null (H0): The partial_rank variant's Criterion CI half-width at n=10K on MERFISH PCA-50
is ≥ the half-width on synthetic Gaussian at the same n (CI width is data-independent).  
Alternative (H1): CI half-width is narrower on MERFISH PCA-50 (structured distance
distribution reduces `select_nth_unstable_by` pivot variance).

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Input data distribution | Synthetic Gaussian, MERFISH PCA-50 | H5 and H-partial-MERFISH require real structured data |
| Sample size n | 10K (H5, H-partial-MERFISH), 100K (H-100K, H0/H1) | Pre-registered protocol; 100K is the production-scale target |
| Variant | baseline, thread_local, partial_rank, avx2_kernel, combined | Full coverage of research worktree variants |
| Subsampling m | 5000 (H5 confirmatory) | Pre-registered gate value |

---

## Dependent Variables (Metrics)

All five metrics are **NEW** — no canonical names exist in `src/metrics.rs`, which has zero
Performance-dimension constants. The metrics below must be tracked in experiment-local JSON
output files and analyzed by `scripts/analyze.py`. They are not added to `src/metrics.rs`
(that would be a source-code change outside scope of this experiment).

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| `wall_clock_speedup` | ratio (×) | `tw_approx_runner` JSON field: `wall_exact_s / wall_approx_s` | NEW |
| `delta_tw` | absolute score difference | `tw_approx_runner` JSON field: `t_approx - t_exact` | NEW |
| `criterion_speedup_100k` | ratio (×) | `cargo criterion --message-format=json` → `criterion_output.json`; computed as `baseline_mean_ns / variant_mean_ns` at n=100K | NEW |
| `per_step_fraction` | ratio 0–1, per step | `tw_profiler --features profiling` step_timing JSON; computed as `step_ns / total_ns` per step label | NEW |
| `partial_rank_ci_half_width` | ratio (×) | Criterion JSON `change.confidence_interval` half-width = `(upper_bound − lower_bound) / 2` at n=10K | NEW |

**Thresholds (pre-registered in scope, not in code):**
- `wall_clock_speedup ≥ 5.0` — H5 gate
- `|delta_tw| < 0.001` — H5 quality gate
- `criterion_speedup_100k CI ⊂ [1.5×, ∞)` — H-100K conclusive positive
- `per_step_fraction[tw_x_dist] > 0.5` — H0/H1 conclusive positive
- `partial_rank_ci_half_width_merfish < partial_rank_ci_half_width_gaussian` — H-partial-MERFISH

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (nearest neighbors) | 15 | Pre-registered protocol; matches existing parity fixtures |
| Rust toolchain | `nightly-2026-03-26` (rustc 1.96.0) | Active toolchain; matches original measurement run |
| Random seed (H5) | 42 | Pre-registered; sealed confirmatory gate |
| Random seeds (bench data) | Deterministic per-group (SmallRng seed=0) | Reproducibility |
| PCA components (MERFISH) | 50 | Matches prepare_merfish.py output |
| CPU features | AVX2/FMA (x86-64-v3 baseline; runtime dispatch in code) | Production hardware profile |
| Criterion warm_up_time | 30s at n=100K, 10s at n≤50K | Allow JIT/CPU ramp-up before measurement |
| Statistical α | 0.05 with Holm-Bonferroni for m=4 comparisons | Controls FWER at 5% |

---

## Inputs and Data

### H5: MERFISH 10K

The MERFISH 100K NPZ source files are **confirmed present** at
`/home/talon/projects/spectral-init/temp/merfish_100k/`:
- `merfish_100k_expression.npz` — key `arr_0`, shape (100K, 1122), float32
- `merfish_100k_spatial.npz` — key `arr_0`, shape (100K, 2), float32

`tests/fixtures/merfish/` does not yet exist; `prepare_merfish.py` will execute the full
PCA path on first run (sklearn PCA-50 on first 10K rows).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| `merfish_n10k_x.npy` | `prepare_merfish.py` from `temp/merfish_100k/*.npz` | (10K, 50) float64, PCA-50 of MERFISH expression | H5 exact/approx timing; H-partial-MERFISH Criterion |
| `merfish_n10k_y.npy` | `prepare_merfish.py` | (10K, 2) float64, spatial coords | H5 input |
| Gaussian synthetic | Inline `make_data(n, d, seed)` in bench | n=100K, d=50, Normal(0,1) | H-100K, H0/H1-clean |
| Gaussian synthetic (existing) | Pre-generated `.npy` in worktree `data/gaussian/` | n=1K–50K | H-partial-MERFISH Gaussian baseline |

### Power analysis for Criterion sample size

At CV=15%, target effect r=10%, α=0.05 with Holm-Bonferroni correction for m=4 comparisons
(first threshold α/m = 0.0125), the required sample size is:

```
n_samples = ceil(15.68 × (0.15/0.10)^2 × (0.05/0.0125)) = ceil(15.68 × 2.25 × 4) ≈ 141
```

At n=100K, each combined sample ≈ 5–10s; 141 samples ≈ 12–24 min per variant, ≈ 1–2 hours
total. For the plan we use **sample_size = 100** as a practical compromise (default Criterion
value), which achieves 80% power at r ≈ 11.6% given CV=15%. This is explicitly documented
as the chosen trade-off.

---

## Experiment Directory Layout

All artifacts for this experiment live in the **research worktree**:
`/home/talon/projects/worktrees/research-20260404-174030/research/2026-04-05-tw-perf-rerun-clean/`

```
research/2026-04-05-tw-perf-rerun-clean/
├── environment.yml                     # Extended tw-perf-scaling env + statsmodels
├── rust-toolchain.toml                 # Pins nightly-2026-03-26
├── scripts/
│   ├── setup.sh                        # Install cargo-criterion, create fixture dirs
│   ├── prepare_data.sh                 # Run prepare_merfish.py, verify outputs
│   ├── run_h5.sh                       # Build tw_approx_runner (no testing), run H5
│   ├── run_criterion_clean.sh          # Isolated bench binaries, n=100K, JSON output
│   ├── run_profiling_clean.sh          # Build with profiling feature, n=100K step timing
│   ├── run_merfish_criterion.sh        # Criterion on MERFISH 10K for partial_rank CI
│   └── analyze.py                      # Collect results, Holm-Bonferroni, emit tables
├── data/
│   └── merfish/                        # merfish_n10k_x.npy, merfish_n10k_y.npy (symlinks or copies)
├── results/
│   ├── h5/                             # h5_result.json from tw_approx_runner
│   ├── criterion/                      # criterion_output.json (JSON-lines from cargo criterion)
│   ├── step_timing/                    # Clean profiling JSONs (no testing feature overhead)
│   ├── merfish_criterion/              # criterion_output_merfish.json
│   └── analysis/
│       ├── hypothesis_verdicts.json    # Per-hypothesis PASS/FAIL/INCONCLUSIVE
│       └── ranked_recommendations.md  # Final table
└── report.md                           # Final report (written by write-report skill)
```

**Script descriptions:**

- `setup.sh`: Checks for `cargo-criterion` binary; installs via `cargo install cargo-criterion`
  if absent. Creates `tests/fixtures/merfish/` directory at worktree root. Creates
  `rust-toolchain.toml` at experiment directory level.

- `prepare_data.sh`: Runs `python scripts/prepare_merfish.py` (from research worktree's
  existing scripts). Copies outputs to `data/merfish/`. Verifies shapes
  (expects `(10000, 50)` and `(10000, 2)`).

- `run_h5.sh`: Builds `tw_approx_runner` without testing feature:
  `cargo build --release --features cli --no-default-features --bin tw_approx_runner`.
  Runs against MERFISH 10K data with k=15, m=5000, seed=42. Writes to
  `results/h5/h5_result.json`. Idempotent — overwrites if run again (not a sealed gate).

- `run_criterion_clean.sh`: Runs `cargo criterion --bench tw_baseline_bench
  --bench tw_partial_rank_bench --bench tw_combined_bench --message-format=json` with
  n=100K included. Redirects JSON-lines output to `results/criterion/criterion_output.json`.
  Separate bench binary per variant for isolation (see Infrastructure Changes below).

- `run_profiling_clean.sh`: Builds `tw_profiler` with `profiling` feature:
  `cargo build --release --features cli,profiling --bin tw_profiler`. Runs at n=100K,
  d=50 Gaussian, k=15, --iters 5 --warmup 2 for all 5 variants. Writes clean
  `results/step_timing/gaussian_n100000_{variant}.json` files.

- `run_merfish_criterion.sh`: Runs `cargo criterion --bench tw_merfish_bench
  --message-format=json`. MERFISH bench is a new single-binary bench that reads
  `data/merfish/merfish_n10k_x.npy` and benchmarks `partial_rank` vs `combined` at n=10K.
  Writes to `results/merfish_criterion/criterion_output_merfish.json`.

- `analyze.py`: Reads all `results/*/` JSON artifacts. Computes:
  - H5: `wall_clock_speedup = wall_exact_s / wall_approx_s`, `delta_tw`
  - H-100K: Criterion CI for combined vs baseline at n=100K
  - H0/H1: Per-step fractions summed to 100% from profiling JSON
  - H-partial-MERFISH: CI half-width ratio (MERFISH / Gaussian) for partial_rank
  Applies `statsmodels.stats.multitest.multipletests(pvals, method='holm')` for H-100K
  and H-partial-MERFISH multi-comparison correction. Writes `results/analysis/` outputs.

---

## Environment

**Custom environment required.**

The existing `tw-perf-scaling` environment.yml (Python 3.11, numpy 2.2, scipy 1.15,
scikit-learn 1.8) is extended with `statsmodels` for Holm-Bonferroni correction. The
`anndata`/`polars`/`h5py` heavy MERFISH generation stack is NOT needed — `temp/merfish_100k/`
is already populated.

The critical Rust-side blocker is `cargo-criterion`, which is not installed. It must be
installed before any Criterion benchmarks can run.

```yaml
name: tw-perf-rerun-clean
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2
  - scipy=1.15
  - scikit-learn=1.8
  - statsmodels>=0.14        # Holm-Bonferroni via multipletests()
  - pip
```

`rust-toolchain.toml` for the experiment directory:

```toml
[toolchain]
channel = "nightly-2026-03-26"
profile = "minimal"
components = ["rustfmt", "clippy"]
```

---

## Infrastructure Changes Required in Research Worktree

These changes are **source-code modifications** to the worktree at
`/home/talon/projects/worktrees/research-20260404-174030/`. They are prerequisites for the
experiment and are part of Phase 1 of the implementation plan.

### Change 1: Add `profiling` Cargo feature (`Cargo.toml`)

Add to the `[features]` section:
```toml
profiling = []
```

This feature gate is distinct from `testing` (which enables serde + eprintln!). The
`profiling` feature will enable only `std::time::Instant` step-timing snapshots — no eprintln!
calls, no serde dependency, no output to stderr.

### Change 2: Add per-step Instant instrumentation to variants (`src/metrics.rs`)

In each of the 5 exact trustworthiness variants (baseline, thread_local, partial_rank,
avx2_kernel, combined), wrap the timing-sensitive steps with:

```rust
#[cfg(feature = "profiling")]
let t0 = std::time::Instant::now();
// ... step code ...
#[cfg(feature = "profiling")]
STEP_TIMINGS.with(|v| v.borrow_mut().push(("tw_x_dist", t0.elapsed())));
```

Use a thread-local `RefCell<Vec<(&'static str, Duration)>>` declared once per variant
function. Steps to instrument: `tw_x_dist`, `tw_x_sort`, `tw_rank_scatter` (baseline only),
`tw_x_knn_set` (baseline only), `tw_y_heap`, `tw_penalty`. A `pub fn drain_step_timings()`
function (cfg-gated) drains the thread-local into a `HashMap<String, f64>` of mean
microseconds across all rows.

### Change 3: Update `tw_profiler.rs` to emit profiling step_timing

When compiled with `profiling` feature, after each timed iteration, call
`drain_step_timings()` and accumulate per-step totals. After all iterations, compute mean
per-step fractions and write them into the JSON output alongside the existing `mean_s`/`std_s`
fields:

```json
{
  "variant": "baseline",
  "n": 100000,
  "mean_s": 95.05,
  "std_s": 1.2,
  "step_fractions": {
    "tw_x_dist": 0.62,
    "tw_x_sort": 0.08,
    "tw_rank_scatter": 0.04,
    "tw_x_knn_set": 0.11,
    "tw_y_heap": 0.13,
    "tw_penalty": 0.02
  }
}
```

### Change 4: Split benchmark into isolated binaries (`Cargo.toml` + new bench files)

Add 5 separate `[[bench]]` entries (plus 1 for MERFISH):

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
name = "tw_merfish_bench"
harness = false
required-features = ["cli"]   # needs ndarray-npy for .npy loading
```

Each bench file (`benches/tw_baseline_bench.rs`, etc.) contains a single `criterion_group!`
with one variant function. Add `n = 100000` to the N_SIZES array. Set
`sample_size(100)` and `warm_up_time(Duration::from_secs(30))` at n=100K groups.

`benches/tw_merfish_bench.rs` reads `data/merfish/merfish_n10k_x.npy` (path via env var
`MERFISH_DATA_DIR`) using `ndarray-npy`, benchmarks `partial_rank` and `combined` at n=10K.

### Change 5: Add `ndarray-npy` as dev-dependency (`Cargo.toml`)

```toml
[dev-dependencies]
ndarray-npy = "0.9"
```

Required for `tw_merfish_bench.rs` to load MERFISH `.npy` files.

---

## Implementation Phases

### Phase 0: Environment Setup

**Goal:** Install `cargo-criterion`; create output directories.

```bash
cd /home/talon/projects/worktrees/research-20260404-174030

# Install cargo-criterion (blocking prerequisite for all Criterion work)
cargo install cargo-criterion

# Create new experiment directory structure
mkdir -p research/2026-04-05-tw-perf-rerun-clean/{scripts,data/merfish,results/{h5,criterion,step_timing,merfish_criterion,analysis}}

# Write environment.yml and rust-toolchain.toml
# (content as specified in Environment section above)
```

Verify: `cargo criterion --version` returns successfully.

### Phase 1: Infrastructure Changes to Research Worktree

**Goal:** Fix all 7 measurement gaps before any measurement.

**1a. Toolchain pin** — Create `research/2026-04-05-tw-perf-rerun-clean/rust-toolchain.toml`
with `channel = "nightly-2026-03-26"`.

**1b. Add `profiling` Cargo feature** — Edit `Cargo.toml`: add `profiling = []` to `[features]`.

**1c. Add per-step Instant instrumentation** — Edit `src/metrics.rs` in the research worktree.
For each of the 5 variant functions, add the thread-local `STEP_TIMINGS` RefCell and
Instant snapshots at each step boundary (gated by `#[cfg(feature = "profiling")]`).
Add `pub fn drain_step_timings() -> HashMap<String, f64>` behind the same gate.

Verify instrumentation compiles cleanly:
```bash
cargo build --release --features cli,profiling --bin tw_profiler
```

**1d. Split bench binaries** — Create `benches/tw_baseline_bench.rs`,
`benches/tw_thread_local_bench.rs`, `benches/tw_partial_rank_bench.rs`,
`benches/tw_avx2_bench.rs`, `benches/tw_combined_bench.rs`, `benches/tw_merfish_bench.rs`.
Each file is ~30 lines: imports, `make_data()`, single `criterion_group!`, `criterion_main!`.
Add `ndarray-npy = "0.9"` to `[dev-dependencies]` in `Cargo.toml`.
Add corresponding `[[bench]]` entries to `Cargo.toml`.

Verify all bench binaries compile:
```bash
cargo build --benches --release
```

**1e. Update tw_profiler.rs** — Add `drain_step_timings()` call after each timing iteration
and accumulate into a step_fractions map; emit in JSON output when `profiling` feature is
active.

### Phase 2: Data Preparation

**Goal:** Produce MERFISH 10K X/Y fixtures needed by H5 and H-partial-MERFISH.

```bash
cd /home/talon/projects/worktrees/research-20260404-174030/research/2026-04-04-tw-perf-scaling

# Activate Python environment
micromamba activate tw-perf-rerun-clean  # or tw-perf-scaling (same deps except statsmodels)

# Run prepare_merfish.py — reads temp/merfish_100k/*.npz (confirmed present),
# writes tests/fixtures/merfish/merfish_n10k_x.npy and merfish_n10k_y.npy
python scripts/prepare_merfish.py

# Verify output shapes
python -c "import numpy as np; x=np.load('../../tests/fixtures/merfish/merfish_n10k_x.npy'); print(x.shape)  # expect (10000, 50)"
```

Then copy/symlink to new experiment's data directory:
```bash
cp /home/talon/projects/worktrees/research-20260404-174030/tests/fixtures/merfish/merfish_n10k_x.npy \
   /home/talon/projects/worktrees/research-20260404-174030/research/2026-04-05-tw-perf-rerun-clean/data/merfish/
cp /home/talon/projects/worktrees/research-20260404-174030/tests/fixtures/merfish/merfish_n10k_y.npy \
   /home/talon/projects/worktrees/research-20260404-174030/research/2026-04-05-tw-perf-rerun-clean/data/merfish/
```

### Phase 3: H5 Confirmatory Run

**Goal:** Answer H5 (MERFISH subsampling speedup and quality gate).

```bash
cd /home/talon/projects/worktrees/research-20260404-174030

# Build tw_approx_runner WITHOUT testing feature (prevents ~6.25× timing inflation)
cargo build --release --features cli --no-default-features --bin tw_approx_runner

# Run H5 confirmatory
./target/release/tw_approx_runner \
  --x research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy \
  --y research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy \
  --k 15 \
  --sample 5000 \
  --seed 42 \
  --output research/2026-04-05-tw-perf-rerun-clean/results/h5/h5_result.json

# Spot-check result
python -c "import json; d=json.load(open('research/2026-04-05-tw-perf-rerun-clean/results/h5/h5_result.json')); print(f'speedup={d[\"wall_exact_s\"]/d[\"wall_approx_s\"]:.2f}x, delta={d[\"delta\"]:.6f}')"
```

Expected output: `wall_exact_s`, `wall_approx_s`, `t_exact`, `t_approx`, `delta`, `n`, `m`, `seed`.

### Phase 4: Clean n=100K Criterion Benchmark (H-100K)

**Goal:** Direct measurement of combined speedup at n=100K with proper isolation and
adequate sample size.

```bash
cd /home/talon/projects/worktrees/research-20260404-174030

# Run all isolated bench binaries; collect JSON-lines to criterion_output.json
# Note: --message-format=json routes to stdout; redirect to file
cargo criterion \
  --bench tw_baseline_bench \
  --bench tw_partial_rank_bench \
  --bench tw_combined_bench \
  --message-format=json \
  2>/dev/null \
  > research/2026-04-05-tw-perf-rerun-clean/results/criterion/criterion_output.json

# Runtime estimate: ~2–3 hours for all groups at n=100K with sample_size=100
# Can run baseline+combined only (2 variants) for a focused 1-hour run
```

For CI: `cargo criterion` reads `sample_size` from the bench's `BenchmarkConfig`. With
`sample_size(100)` at n=100K, expect ~10–20 minutes per variant binary.

### Phase 5: Clean Per-Step Timing (H0/H1)

**Goal:** Measure per-step time fractions at n=100K, d=50 without testing-feature overhead.

```bash
cd /home/talon/projects/worktrees/research-20260404-174030

# Build tw_profiler with profiling feature (NOT testing — avoids eprintln! inflation)
cargo build --release --features cli,profiling --bin tw_profiler

# Generate synthetic n=100K, d=50 test data (reuse existing gen_synthetic.py or inline)
python research/2026-04-04-tw-perf-scaling/scripts/gen_synthetic.py \
  --n 100000 --d 50 --seed 0 \
  --out-x research/2026-04-05-tw-perf-rerun-clean/data/gaussian_n100k_d50_x.npy \
  --out-y research/2026-04-05-tw-perf-rerun-clean/data/gaussian_n100k_d50_y.npy

# Run profiling for each variant
for VARIANT in baseline thread_local partial_rank avx2_kernel combined; do
  ./target/release/tw_profiler \
    --x research/2026-04-05-tw-perf-rerun-clean/data/gaussian_n100k_d50_x.npy \
    --y research/2026-04-05-tw-perf-rerun-clean/data/gaussian_n100k_d50_y.npy \
    --k 15 --iters 5 --warmup 2 \
    --variant $VARIANT \
    --output research/2026-04-05-tw-perf-rerun-clean/results/step_timing/gaussian_n100000_${VARIANT}.json
done
```

Runtime estimate: ~5 iterations × 5 variants × ~95s/iter = ~40 minutes.

**Note:** gen_synthetic.py may need an update to support `--d` argument for non-default
dimensionality. If it only generates d=10 (original bench dimensionality), add a
`--d` CLI flag or write an inline Python one-liner:
```python
import numpy as np
np.save('x.npy', np.random.default_rng(0).standard_normal((100000, 50)))
np.save('y.npy', np.random.default_rng(1).standard_normal((100000, 2)))
```

### Phase 6: MERFISH Criterion Benchmark (H-partial-MERFISH)

**Goal:** Measure partial_rank CI half-width on MERFISH data vs Gaussian baseline.

```bash
cd /home/talon/projects/worktrees/research-20260404-174030

# Set data path for tw_merfish_bench.rs to find MERFISH .npy files
export MERFISH_DATA_DIR=research/2026-04-05-tw-perf-rerun-clean/data/merfish

cargo criterion \
  --bench tw_merfish_bench \
  --message-format=json \
  2>/dev/null \
  > research/2026-04-05-tw-perf-rerun-clean/results/merfish_criterion/criterion_output_merfish.json
```

The Gaussian baseline CI at n=10K is available from the new isolated bench run or from the
existing `results/criterion/bench_output.txt` in the prior experiment.

### Phase 7: Dry Run

Before Phase 4–6 full runs, execute a minimal dry run to validate the pipeline end-to-end:

```bash
# Quick smoke: n=1000 only, sample_size=5
cargo criterion --bench tw_baseline_bench -- --n 1000 2>/dev/null \
  > research/2026-04-05-tw-perf-rerun-clean/results/criterion/dry_run.json

# Verify JSON-lines are parseable
python -c "
import json
with open('research/2026-04-05-tw-perf-rerun-clean/results/criterion/dry_run.json') as f:
    for line in f:
        obj = json.loads(line)
        print(obj.get('id', '?'), obj.get('mean', {}).get('estimate', '?'))
"
```

### Phase 8: Analysis

```bash
cd /home/talon/projects/worktrees/research-20260404-174030/research/2026-04-05-tw-perf-rerun-clean

python scripts/analyze.py \
  --h5-result results/h5/h5_result.json \
  --criterion-json results/criterion/criterion_output.json \
  --step-timing-dir results/step_timing/ \
  --merfish-criterion-json results/merfish_criterion/criterion_output_merfish.json \
  --output-dir results/analysis/
```

`analyze.py` emits:
- `results/analysis/hypothesis_verdicts.json` — JSON with PASS/FAIL/INCONCLUSIVE per hypothesis
- `results/analysis/ranked_recommendations.md` — Markdown table with all metrics and verdicts

---

## Execution Protocol

1. Activate the `tw-perf-rerun-clean` conda environment
2. Run `scripts/setup.sh` — installs cargo-criterion, creates directories
3. Apply Phase 1 infrastructure changes to the research worktree (code edits)
4. Run `scripts/prepare_data.sh` — generates MERFISH 10K fixtures
5. Run `scripts/run_h5.sh` — H5 confirmatory (fast, ~5 minutes)
6. Run `scripts/run_profiling_clean.sh` — per-step timing (fast, ~40 minutes)
7. Run `scripts/run_merfish_criterion.sh` — MERFISH CI benchmark (medium, ~30 minutes)
8. Run `scripts/run_criterion_clean.sh` — full n=100K Criterion (slow, ~2–3 hours; can
   schedule overnight)
9. Run `scripts/analyze.py` — collect and report all results

Steps 6, 7, and 8 can run in any order after step 4 and the Phase 1 changes.

---

## Analysis Plan

### H5 (MERFISH subsampling)

From `results/h5/h5_result.json`:
```python
speedup = result["wall_exact_s"] / result["wall_approx_s"]
delta = abs(result["delta"])
verdict = "PASS" if speedup >= 5.0 and delta < 0.001 else "FAIL"
```
No statistical test needed — single pre-registered measurement.

### H-100K (Criterion speedup at n=100K)

Parse `results/criterion/criterion_output.json` (newline-delimited JSON objects).
For each object with `"id"` containing `"100000"`:
- Extract `mean["estimate"]` (nanoseconds) for baseline and combined
- Compute `speedup = baseline_mean_ns / combined_mean_ns`
- Extract `confidence_interval["lower_bound"]` and `upper_bound` from `change` field

Apply Holm-Bonferroni for 4 pairwise comparisons (baseline vs each of: thread_local,
partial_rank, avx2_kernel, combined). Use `statsmodels.stats.multitest.multipletests`.

Verdict: PASS if combined CI lower_bound > 1.5×; INCONCLUSIVE if CI overlaps 1.5×;
FAIL if CI upper_bound < 1.5×.

### H0/H1-clean (per-step fractions)

From each `results/step_timing/gaussian_n100000_{variant}.json`:
```python
fracs = result["step_fractions"]
dominant = max(fracs, key=fracs.get)
verdict = "H1 supported" if fracs.get("tw_x_dist", 0) > 0.5 else "H0 not rejected"
```
Report the full fraction breakdown as a table in the analysis output.

### H-partial-MERFISH (CI width comparison)

From `results/merfish_criterion/criterion_output_merfish.json` and the n=10K entries in
`results/criterion/criterion_output.json`:
```python
gaussian_hw = (gaussian_ci_upper - gaussian_ci_lower) / 2  # in speedup ratio units
merfish_hw  = (merfish_ci_upper  - merfish_ci_lower)  / 2
verdict = "H1 supported" if merfish_hw < gaussian_hw else "H0 not rejected"
```

Apply Holm-Bonferroni correction if this test is part of the multi-comparison family.

---

## Success Criteria

| Hypothesis | Conclusive positive | Conclusive negative | Inconclusive |
|-----------|---------------------|---------------------|--------------|
| **H5** | `speedup ≥ 5×` AND `\|delta\| < 0.001` | Either condition fails | Both conditions fail but by < 10% margin |
| **H-100K** | Combined CI lower_bound > 1.5×, CI overlaps [1.8×–2.1×] | CI upper_bound < 1.5× | CI straddles 1.5× threshold |
| **H0/H1-clean** | `tw_x_dist fraction > 0.5` for baseline | `tw_x_dist fraction < 0.3` (heap dominates) | No step > 40% |
| **H-partial-MERFISH** | MERFISH CI half-width < Gaussian CI half-width (p < 0.05 Holm) | No difference (p > 0.05) | CI estimates have overlapping error ranges |

---

## Threats to Validity

### Internal

- **Profiling feature overhead:** Thread-local RefCell borrow + Vec push in the `profiling`
  feature adds a small overhead per step. The step_fractions are relative, so absolute
  times are not needed — but if overhead is non-uniform across steps, fractions will be
  biased. Mitigation: measure overhead with a no-op step and subtract if > 1% of total.

- **Bench binary isolation is incomplete:** Even with separate binaries, OS-level context
  (CPU caches, frequency scaling) persists across sequential bench runs in the same session.
  Mitigation: run each bench binary in a separate `cargo criterion` invocation with a
  1-minute pause between variants.

- **MERFISH n=10K is too small for stable CI at d=50:** At n=10K the `select_nth_unstable_by`
  call processes n=10K elements, which may not exhibit the same cache behavior as n=50K.
  The hypothesis was originally stated at n=50K. Mitigation: document that the test is at
  n=10K (limited by fixture availability); add a note that n=50K MERFISH would require
  extending `prepare_merfish.py` to generate more cells.

- **H-100K time per sample:** If combined at n=100K takes > 20s per sample, 100 samples
  = 33 minutes per variant binary. Wall-clock variance under load may increase CV beyond
  the assumed 15%. Mitigation: measure CV on first 20 samples; abort and document if
  CV > 25%.

- **Single H5 trial:** The H5 gate uses seed=42, one trial. A single measurement is
  sufficient for a go/no-go decision but cannot quantify variance. If the result is near
  the threshold, run 5 seeds and report the distribution.

### External

- **Nightly toolchain specificity:** Results are tied to `nightly-2026-03-26`. A different
  nightly may produce different LLVM optimizations. Production Rust is stable, not nightly.
  Mitigation: document the toolchain pin; this experiment is measurement-only, not shipping
  code.

- **Synthetic Gaussian data at d=50:** The per-step timing (H0/H1) uses d=50 Gaussian data
  to match the MERFISH PCA dimensionality, but the actual MERFISH data has different
  numeric range and co-variance structure. AVX2 kernel performance may differ on real data.

- **MERFISH 10K vs production:** The H5 result at n=10K, m=5000 proves the approximation
  quality on 10K structured cells. Behavior at n=100K with m=5000 (2×–3× higher subsampling
  ratio improvement expected) is not covered by this experiment.

---

## Estimated Resource Requirements

| Phase | Wall-clock | Disk |
|-------|------------|------|
| Phase 0: Setup | < 5 min | ~20 MB (cargo-criterion) |
| Phase 1: Code edits | < 30 min | negligible |
| Phase 2: Data prep | ~5 min (PCA on 10K cells) | ~15 MB (.npy files) |
| Phase 3: H5 run | ~5 min | < 1 MB |
| Phase 4: n=100K Criterion | 2–3 hours (5 variants × 100 samples × ~10–20s) | ~50 MB |
| Phase 5: Step timing | ~40 min (5 variants × 5 iters × ~95s) | < 5 MB |
| Phase 6: MERFISH Criterion | ~30 min (2 variants × 100 samples × ~1s at n=10K) | < 5 MB |
| Phase 7: Dry run | ~5 min | < 1 MB |
| Phase 8: Analysis | < 2 min | < 1 MB |
| **Total** | **~4–5 hours** | **~100 MB** |

**Dependencies:**
- `cargo-criterion` (must install via `cargo install cargo-criterion`)
- `statsmodels>=0.14` (Python; must add to conda env)
- `ndarray-npy = "0.9"` (Rust dev-dep; must add to Cargo.toml)
- All other deps already present in existing research worktree
