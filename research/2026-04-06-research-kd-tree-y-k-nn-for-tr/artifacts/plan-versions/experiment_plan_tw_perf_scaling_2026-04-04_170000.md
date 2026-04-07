# Experiment Plan: Trustworthiness Performance — Scaling Analysis and Optimization Evaluation

**Revision note:** This plan supersedes
`.autoskillit/temp/plan-experiment/experiment_plan_tw_perf_scaling_2026-04-04_161626.md`.
Four STOP-trigger findings from evaluation dashboard
`evaluation_dashboard_tw-perf-scaling_2026-04-04_163836.md` have been addressed.
See revision guidance at
`.autoskillit/temp/resolve-design-review/revision_guidance_tw-perf-scaling_2026-04-04_164114.md`
for per-finding rationale.

---

## Motivation

The `trustworthiness()` function in `src/metrics.rs` is the dominant runtime
bottleneck beyond n=100K cells. At n=250K it was killed after 7+ minutes at
805% CPU utilization. Without a profiled breakdown of where time is actually
spent, any optimization is a guess. This experiment characterizes the per-step
scaling profile, evaluates six specific algorithmic and SIMD alternatives, and
produces a ranked GO/NO-GO recommendation grounded in measured wall-clock
speedups at the practical target scale of 100K–250K cells.

The results directly inform the implementation decision: which optimizations to
ship in `ComputeMode::RustNative`, which to mark NO-GO, and what approximate
parity threshold to document for the subsampling code path.

**Reported speedup ratios (≥1.5× GO threshold) are measured relative to the
current unmodified production `trustworthiness()` function as-shipped. They
represent an upper bound on real-world gains if the baseline itself received
additional optimization.** (RT-3 disclosure)

This plan supersedes the prior plan which received a STOP verdict on three
grounds (STOP-1: cold-start allocation confound; STOP-2: post-hoc sampling
calibration; STOP-3: vacuous parity gate on Gaussian data). All three are
addressed below, together with four additional findings (RT-1 through RT-4)
from the second review-design pass.

---

## Hypothesis

**Null hypothesis (H0):** No single optimization achieves a measured speedup
greater than 2× at n=100K under warm-start conditions with ≥3 iterations, and
row subsampling at the pre-registered m=5000 produces |T_approx − T_exact| >
0.001 on MERFISH-structured data.

**Alternative hypothesis (H1_combined):** At least one exact optimization
achieves ≥2× speedup at n=100K (warm), AND row subsampling at the
pre-registered m=5000 produces |T_approx − T_exact| ≤ 0.001 on
MERFISH-derived structured data.

Individual falsifiable sub-hypotheses:

- **H1** X-sort is the dominant wall-clock step (≥40% of total) at n≥10K, and
  partial-rank computation achieves the largest single-step speedup.
- **H2** Thread-local Vec reuse provides 5–15% throughput improvement at
  n=5K–50K (Criterion ratio, 95% CI).
- **H3** Manual AVX2 for the X-distance inner loop yields < 2× over the current
  auto-vectorized baseline at d=10 f64 (confirmed by `cargo asm` + benchmark).
- **H4** AVX-512 for the distance kernel provides < 20% improvement over AVX2
  at d=10 due to 62.5% register utilization and potential frequency throttling.
- **H5** Row subsampling at the pre-registered m=5000 produces
  |T_approx − T_exact| ≤ 0.001 on MERFISH-derived structured data (n=10K
  slice, PCA-reduced to 50 dims).
- **H6** The combined exact optimization — unconditionally defined as the
  simultaneous application of `thread_local` + `partial_rank` + `avx2_kernel`
  — achieves 3–6× end-to-end speedup at n=100K (warm). **This composition is
  fixed before any individual variant results are examined.** The `combined`
  variant always implements all three components; no component is omitted based
  on individual measurements. (RT-1 fix)

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Dataset size n | 1K, 5K, 10K, 50K, 100K, (250K with timeout guard) | Span the pre-bottleneck to bottleneck regime |
| Optimization variant | baseline, thread_local, partial_rank, avx2_kernel, combined, approx_m5000 | Each addresses a distinct hypothesis |
| Sample size m (subsampling only) | 500, 1000, 2000, 5000, 10000 | Sweep to characterize error bound; m=5000 pre-registered (RT-2) |
| Data type | Gaussian synthetic (d=10), MERFISH-derived structured (d=50) | Gaussian = throughput characterization; structured = parity validation |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Per-step wall-clock fraction | % of total row time | `#[cfg(feature="testing")]` timing in `trustworthiness()`, stderr JSON parse | NEW — no entry in `src/metrics.rs` |
| Criterion throughput ratio | dimensionless (speedup factor) | `criterion::BenchmarkGroup`, `BenchmarkId`, `throughput` | NEW |
| Criterion mean ± CI | ms | Criterion HTML report + extracted JSON | NEW |
| Wall-clock at large n | seconds (mean ± std, ≥3 warm iters) | `tw_profiler` binary with `--iters 5 --warmup 2` | NEW |
| Trustworthiness parity (exact) | absolute | `|T_rust − T_sklearn|`, verified via Python reference | Threshold `1e-6` in `tests/integration/test_trustworthiness.rs`; no canonical MetricResult entry |
| Subsampling deviation | absolute | `|T_approx(m) − T_exact(n)|` on MERFISH-derived data | NEW — no canonical entry; approximate threshold `< 0.001` proposed |
| AVX2 auto-vectorization status | boolean | `cargo asm` instruction inspection | NEW — qualitative |

**NEW metrics requiring definition before finalization:**
- `tw_step_fraction`: formula = `step_wall_us / total_row_wall_us`; unit = %; no threshold (diagnostic)
- `tw_speedup_ratio`: formula = `baseline_mean_ms / variant_mean_ms`; unit = ×; GO threshold ≥ 1.5×
- `tw_approx_deviation`: formula = `|T_approx − T_exact|`; unit = absolute; GO threshold ≤ 0.001 on structured data

Note: `src/metrics.rs` contains `MetricResult` / `AssessmentReport` / `ExperimentMetrics` structs
gated behind `#[cfg(feature = "testing")]`, covering eigensolver quality metrics. These structs are
not used by `trustworthiness()` and no trustworthiness-specific entries exist. The above three NEW
metrics must be added to the catalog before the experiment is finalized.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k (neighborhood size) | 15 | Matches existing sklearn parity fixture and production default |
| X dimensionality (synthetic) | d_x = 10 | Matches complexity analysis in scope report; controls per-step FLOP count |
| Y dimensionality | d_y = 2 | Standard UMAP 2D embedding output |
| RNG seed | 42 | Reproducibility for synthetic datasets |
| Rayon thread count | system default (≥8) | Not pinned; document actual count in results |
| Rust toolchain | stable (current) | No nightly features needed |
| target-cpu | native (from `.cargo/config.toml`) | AVX2+FMA available; document in results |
| Criterion measurement time | 10s per benchmark | Sufficient for stable estimates at n≤50K |
| Warm iterations (large n) | 2 discarded warmup + 5 measured | Addresses STOP-1 cold-start allocation confound |
| Synthetic data sizing | Fixed at n values sized ≥2× L3 cache (n≥50K @ d=10 f64 = ~80MB per n-vector) to ensure consistent cache pressure across all runs; baseline binary compiled and frozen before any variant is measured | Prevents cache-pressure confound in H6 combined benchmark; ensures baseline and variants see equivalent memory pressure (RT-1 fix) |
| Baseline optimization effort | None — current production `trustworthiness()` as-shipped; no PGO, parallelism tuning, or code changes applied | Speedups are measured relative to production code, not a theoretical performance floor; reported ratios are upper bounds on real-world improvement over any further-optimized baseline (RT-3 fix) |
| Combined variant composition | Unconditionally: thread_local + partial_rank + avx2_kernel, defined before any Phase 3 measurements | Prevents Goodhart exploitation by post-hoc composition selection (RT-1 fix) |

---

## Inputs and Data

The experiment requires both synthetic datasets (to control n and d, and
characterize throughput without structured-data confounds) and MERFISH-derived
structured data (to validate subsampling parity on realistic biological data,
addressing STOP-3). Gaussian synthetic results are labeled **throughput
characterization only** and cannot be used as the parity GO gate.

**Pre-registration — STOP-2 fix:**
The production subsampling value m=5000 is fixed here, in this document, before
any measurement is run. The fraction scan over m={500, 1000, 2000, 5000, 10000}
is conducted as an exploratory secondary analysis only. The GO/NO-GO criterion
for H5 is evaluated at m=5000 exclusively, on MERFISH data, before the sweep
is run.

**Pre-registration — RT-1 fix:**
The `combined` variant composition is unconditionally fixed as
`thread_local + partial_rank + avx2_kernel` regardless of individual variant
outcomes. This is registered here before any Phase 3 measurements are run.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| `gaussian_n{N}_x.npy` / `gaussian_n{N}_y.npy` | Generated: `np.random.RandomState(42).randn(n, d).astype(np.float64)` | Gaussian i.i.d., n ∈ {1K,5K,10K,50K,100K}, d_x=10, d_y=2 | Throughput benchmarks (H1–H4, H6) — NOT parity gate |
| `structured_n{N}_x.npy` / `structured_n{N}_y.npy` | Generated: `make_blobs(n, centers=8, n_features=10, random_state=42)` + 2D PCA for Y | 8-cluster structure, n ∈ {1K,5K,10K}, d_x=10, d_y=2 | Secondary subsampling parity validation on non-Gaussian data |
| `merfish_n10k_x.npy` / `merfish_n10k_y.npy` | `temp/merfish_100k/merfish_100k_expression.npz["arr_0"]` first 10K rows; X PCA-reduced to 50 dims | Biological structure, n=10K, d_x=50, d_y=2 | Primary MERFISH parity gate for H5 (STOP-3 fix) |
| `tw_parity.npz` | Existing: `tests/fixtures/tw_parity/tw_parity.npz` (keys: `X`, `Y`, `k`, `sklearn_score`) | n=200, d_x=10, d_y=2, k=15, sklearn reference score | Exact parity regression check (n=200) |

**MERFISH preparation rationale:** The full 100K expression matrix
(`merfish_100k_expression.npz`) is 1122-dimensional. Computing T_exact at
n=100K, d_x=1122 is prohibitively expensive (~112× more than d=10). Using a
10K slice PCA-reduced to 50 dims (standard UMAP preprocessing) preserves the
biological cluster structure — the property that makes STOP-3 relevant — while
making T_exact tractable. This scope limitation must be documented in the
report.

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-04-tw-perf-scaling/
├── scripts/
│   ├── gen_synthetic.py            # Generate Gaussian + structured synthetic .npy datasets
│   ├── prepare_merfish.py          # Extract + PCA-reduce MERFISH 10K slice to .npy
│   ├── sklearn_reference.py        # Compute sklearn T(k) for all datasets → results/sklearn_scores.json
│   ├── run_profiling.sh            # Drive tw_profiler across n values, capture step-timing JSON
│   ├── run_h5_confirmatory.sh      # H5 confirmatory gate: m=5000 MERFISH only; seals h5_confirmatory_result.json (RT-2)
│   ├── run_subsampling_sweep.sh    # Exploratory m-sweep; must run AFTER run_h5_confirmatory.sh completes (RT-2)
│   ├── run_criterion.sh            # Drive cargo bench; includes assembly identity check (RT-4)
│   ├── subsampling_sweep.py        # Python driver for subsampling sweep
│   ├── dry_run.sh                  # Minimal end-to-end smoke test before full run
│   └── analyze_results.py          # Parse all results/ JSON → recommendation table + plots
├── data/
│   ├── gaussian/                   # gaussian_n{1k,5k,10k,50k,100k}_{x,y}.npy
│   ├── structured/                 # structured_n{1k,5k,10k}_{x,y}.npy
│   └── merfish/                    # merfish_n10k_{x,y}.npy
├── results/
│   ├── step_timing/                # JSON per n: {n, steps[], total_mean_us, total_std_us}
│   ├── criterion/
│   │   ├── criterion_output.txt    # Raw cargo bench stdout
│   │   ├── summary.json            # Extracted speedup ratios per variant per n
│   │   └── binary_identity_check.txt  # ASM diff: clean build vs --features testing (RT-4)
│   ├── subsampling/
│   │   ├── h5_confirmatory_result.json  # Sealed GO/NO-GO at m=5000 on MERFISH (RT-2)
│   │   ├── merfish_n10k.json            # Full m-sweep on MERFISH (secondary)
│   │   ├── structured_n10k.json         # Full m-sweep on structured synthetic (secondary)
│   │   └── gaussian_n10k.json           # Full m-sweep on Gaussian (informational only)
│   ├── asm_inspection.txt          # cargo asm output for distance inner loop (H3 evidence)
│   ├── sklearn_scores.json         # sklearn T(k) reference values for all datasets
│   └── recommendation_table.md    # Final ranked GO/NO-GO table
└── report.md                       # Written by write-report skill
```

**Source-tree artifacts** (canonical Cargo locations — added by the implementer):

| File | Location | Purpose |
|------|----------|---------|
| `trustworthiness_bench.rs` | `benches/` | Criterion baseline + variant benchmarks at n=1K–50K |
| `tw_profiler.rs` | `src/bin/` | Warm-start per-step timing binary; JSON output to stdout |
| New `[[bench]]` entry | `Cargo.toml` | Register `trustworthiness_bench` — **no** `required-features` (RT-4) |
| New `[[bin]]` entry | `Cargo.toml` | Register `tw_profiler`, `required-features = ["testing", "cli"]` |
| Per-step timing in `trustworthiness()` | `src/metrics.rs` | `#[cfg(feature="testing")]`-gated `Instant` + `eprintln!` at each of the 6 steps |

---

## Environment

**No custom environment needed.**

The `spectral-test` conda environment (at `tests/environment.yml`) already
provides all required Python packages: `python=3.11`, `numpy=2.2`,
`scikit-learn=1.8`, `scipy=1.15`, `matplotlib=3.10`, and `ndarray-npy`
I/O compatibility. The Rust toolchain with `cargo`, `criterion=0.5`, and
`target-cpu=native` (from `.cargo/config.toml`) covers all benchmark needs.

Activate with: `micromamba activate spectral-test`

No `environment.yml` will be created for this experiment.

---

## Implementation Phases

### Phase 0: Crate Instrumentation (prerequisite — must land before any measurement)

Modifies `src/metrics.rs`, `src/bin/` (new file), and `Cargo.toml`.
**This phase must be complete before any Phase 1+ work runs.**

**0a. Per-step timing in `trustworthiness()`** (`src/metrics.rs`):

Add `#[cfg(feature = "testing")]` timing guards around each of the 6 inner-loop
steps, matching the exact pattern from `src/solvers/mod.rs` (using
`std::time::Instant` captures + `eprintln!("[timing:...]")`). The six steps
and their tags:

```
[timing:tw_step_1] X pairwise distances (us)
[timing:tw_step_2] X full sort (us)
[timing:tw_step_3] rank scatter (us)
[timing:tw_step_4] X k-NN set build (us)
[timing:tw_step_5] Y streaming heap (us)
[timing:tw_step_6] penalty accumulation (us)
[timing:tw_total]  total per-call (us)
```

**RT-4 critical constraint:** These timing guards must be positioned in
`trustworthiness()` such that they are NOT inside the inner loop being measured
by Criterion benchmarks. If instrumentation cannot be isolated outside the hot
path, the instrumented steps must live in a separate wrapper function
`trustworthiness_instrumented()` that calls the clean production
`trustworthiness()` internally. The clean function must remain unmodified for
Criterion use.

**0b. `tw_profiler` binary** (`src/bin/tw_profiler.rs`):

CLI interface: `tw_profiler --x X.npy --y Y.npy [--k 15] [--iters 5] [--warmup 2] --output path.json`

Behavior:
- Load X, Y from `.npy` via `ndarray_npy::read_npy`
- Run `warmup` iterations (discard timing — STOP-1 fix)
- Run `iters` measured iterations, capturing stderr timing lines
- Compute mean ± std per step across measured iterations
- Report `rayon::current_num_threads()` in output
- Output JSON to `--output` path and stdout:
  ```json
  {
    "n": 10000, "d_x": 10, "d_y": 2, "k": 15, "iters": 5,
    "rayon_threads": 8,
    "steps": [
      {"step": 1, "name": "x_dist", "mean_us": ..., "std_us": ..., "fraction": ...},
      ...
    ],
    "total_mean_us": ..., "total_std_us": ...
  }
  ```

**0c. `Cargo.toml` additions:**

```toml
# RT-4: trustworthiness_bench must NOT have required-features = ["testing"]
# The testing feature adds Instant guards inside the hot path; benchmarks must
# measure clean production code.
[[bench]]
name = "trustworthiness_bench"
harness = false

[[bin]]
name = "tw_profiler"
path = "src/bin/tw_profiler.rs"
required-features = ["testing", "cli"]
```

**Verification:** `cargo build --release --features testing,cli --bin tw_profiler` succeeds;
`cargo bench --bench trustworthiness_bench` succeeds **without** `--features testing`.

---

### Phase 1: Directory Structure and Data Generation

Create the research directory tree:

```bash
mkdir -p research/2026-04-04-tw-perf-scaling/{scripts,data/{gaussian,structured,merfish},results/{step_timing,criterion,subsampling}}
```

**1a. `scripts/gen_synthetic.py`:**

Generates Gaussian and `make_blobs` structured datasets. For each n in
{1000, 5000, 10000, 50000, 100000}:
- `X = np.random.RandomState(42).randn(n, 10).astype(np.float64)` → `data/gaussian/gaussian_n{n}_x.npy`
- `Y = np.random.RandomState(42).randn(n, 2).astype(np.float64)` → `data/gaussian/gaussian_n{n}_y.npy`

For n in {1000, 5000, 10000}:
- `X, _ = make_blobs(n_samples=n, centers=8, n_features=10, random_state=42).astype(np.float64)` → `data/structured/structured_n{n}_x.npy`
- `Y = PCA(n_components=2, random_state=42).fit_transform(X).astype(np.float64)` → `data/structured/structured_n{n}_y.npy`

Print shapes and dtypes for each file as verification.

**1b. `scripts/prepare_merfish.py`:**

```python
import numpy as np
from sklearn.decomposition import PCA

X_full = np.load("temp/merfish_100k/merfish_100k_expression.npz")["arr_0"]  # (100000, d)
Y_full = np.load("temp/merfish_100k/merfish_100k_spatial.npz")["arr_0"]     # (100000, 2)
X10k = X_full[:10000].astype(np.float64)
Y10k = Y_full[:10000].astype(np.float64)
X10k_pca = PCA(n_components=50, random_state=42).fit_transform(X10k)       # (10000, 50)
np.save("research/2026-04-04-tw-perf-scaling/data/merfish/merfish_n10k_x.npy", X10k_pca)
np.save("research/2026-04-04-tw-perf-scaling/data/merfish/merfish_n10k_y.npy", Y10k)
print(f"X shape: {X10k_pca.shape}, Y shape: {Y10k.shape}")
```

Print shapes and PCA explained variance ratio (first 10 components) as
verification that biological structure is preserved.

**1c. `scripts/sklearn_reference.py`:**

Compute `sklearn.manifold.trustworthiness(X, Y, n_neighbors=15)` for each
dataset (Gaussian n=1K–100K, structured n=1K–10K, MERFISH n=10K). Write
results to `results/sklearn_scores.json`:
```json
{
  "gaussian_n1000":   {"T_sklearn": 0.9..., "k": 15},
  "gaussian_n10000":  {"T_sklearn": 0.9..., "k": 15},
  ...
  "merfish_n10000":   {"T_sklearn": 0.9..., "k": 15}
}
```

This file is the ground truth for all subsequent parity checks. It must be
written and committed before running any Rust benchmarks.

**Verification:** All `.npy` files load without error; all sklearn scores output
to JSON. Separately verify `tw_parity.npz` fixture: confirm Rust
`trustworthiness` on `tests/fixtures/tw_parity/tw_parity.npz` (keys: `X`, `Y`,
`sklearn_score`) produces `|T_rust − sklearn_score| < 1e-6`.

---

### Phase 2: Baseline Characterization

After Phases 0 and 1 complete, run the profiler to establish per-step fractions
and the Criterion baseline.

**2a. `scripts/run_profiling.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"
BIN="cargo run --release --features testing,cli --bin tw_profiler --"

for N in 1000 5000 10000 50000 100000; do
  echo "=== Profiling n=$N ==="
  $BIN \
    --x "$RESEARCH_DIR/data/gaussian/gaussian_n${N}_x.npy" \
    --y "$RESEARCH_DIR/data/gaussian/gaussian_n${N}_y.npy" \
    --k 15 --iters 5 --warmup 2 \
    --output "$RESEARCH_DIR/results/step_timing/gaussian_n${N}.json"
done

# MERFISH profiling (d_x=50 — note: slower per step due to higher dimensionality)
$BIN \
  --x "$RESEARCH_DIR/data/merfish/merfish_n10k_x.npy" \
  --y "$RESEARCH_DIR/data/merfish/merfish_n10k_y.npy" \
  --k 15 --iters 5 --warmup 2 \
  --output "$RESEARCH_DIR/results/step_timing/merfish_n10k.json"
```

Optional n=250K (Gaussian): add with `--warmup 2 --iters 3` wrapped in
`timeout 300`.

**2b. `benches/trustworthiness_bench.rs`:**

Criterion benchmark structure:

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spectral_init::trustworthiness;

fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("tw_baseline");
    for &n in &[1_000usize, 5_000, 10_000, 50_000] {
        let x = make_gaussian_data(n, 10, 42);  // returns Array2<f64>
        let y = make_gaussian_data(n, 2, 43);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| trustworthiness(x.view(), y.view(), 15))
        });
    }
    group.finish();
}

// Placeholder groups for each variant (filled in Phase 3):
// bench_thread_local, bench_partial_rank, bench_avx2_kernel, bench_combined

criterion_group!(benches, bench_baseline);
criterion_main!(benches);
```

This bench uses only the public `trustworthiness` API, no `testing` feature
required.

**2c. `scripts/run_criterion.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"

# RT-4: Verify the benchmarked binary uses clean production code, not the
# instrumented variant. Run before recording any speedup ratios.
echo "=== Assembly identity check (RT-4) ==="
cargo rustc --release --bench trustworthiness_bench -- --emit=asm 2>/dev/null
CLEAN_ASM=$(grep -c "vmovupd\|vfmadd\|ymm" target/release/deps/*.s 2>/dev/null || echo "0")

cargo rustc --release --bench trustworthiness_bench --features testing -- --emit=asm 2>/dev/null
TESTING_ASM=$(grep -c "vmovupd\|vfmadd\|ymm" target/release/deps/*.s 2>/dev/null || echo "0")

echo "Clean build AVX2 instruction count: $CLEAN_ASM" | tee "$RESEARCH_DIR/results/criterion/binary_identity_check.txt"
echo "Testing-feature build AVX2 instruction count: $TESTING_ASM" | tee -a "$RESEARCH_DIR/results/criterion/binary_identity_check.txt"
echo "Match: $([ "$CLEAN_ASM" = "$TESTING_ASM" ] && echo YES || echo NO)" \
  | tee -a "$RESEARCH_DIR/results/criterion/binary_identity_check.txt"

# Run benchmarks without --features testing
cargo bench --bench trustworthiness_bench 2>&1 \
  | tee "$RESEARCH_DIR/results/criterion/criterion_output.txt"

# Extract per-variant summary JSON via jq or Python helper
python3 "$RESEARCH_DIR/scripts/extract_criterion_summary.py" \
  --criterion-dir target/criterion/ \
  --output "$RESEARCH_DIR/results/criterion/summary.json"
```

**AVX2 auto-vectorization inspection (H3):**

```bash
cargo asm spectral_init::metrics::trustworthiness 2>&1 | head -150 \
  > research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt
# Fallback if cargo-asm unavailable:
# cargo rustc --release -- --emit asm 2>/dev/null
# grep -A 30 "vmovupd\|vfmadd\|ymm" target/release/deps/*.s | head -150 \
#   >> research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt
```

The file must contain evidence of whether AVX2 instructions (`vmovupd`,
`vfmadd213pd`, `ymm` registers) appear in the distance inner loop — the H3
GO/NO-GO gate.

---

### Phase 3: Optimization Implementations and H5 Confirmatory Gate

Implementations are gated on Phase 2 profiling results per the following rules:
- **Thread-local buffers (H2):** Always implement (zero correctness risk, pure throughput).
- **Partial-rank X (H1):** Implement only if X-sort step ≥ 40% of total wall-clock at n=50K.
- **Manual AVX2 kernel (H3):** Implement only if `asm_inspection.txt` shows NO AVX2
  instructions in the distance loop. If auto-vectorized, mark NO-GO immediately.
- **AVX-512 kernel (H4):** Implement only if H3 benchmark shows ≥ 20% AVX2 gain over scalar.
- **Row subsampling (H5):** Always implement (needed for 250K+ regardless).
- **Combined variant (H6):** Always implement as thread_local + partial_rank + avx2_kernel,
  unconditionally, regardless of individual measurement outcomes. (RT-1 fix)

**3a. Thread-local Vec reuse** (modifies `src/metrics.rs`):

Replace the three per-row `Vec::new()` allocations with `thread_local!`
`RefCell<Vec>` buffers:

```rust
thread_local! {
    static DIST_X: RefCell<Vec<(f64, usize)>> = RefCell::new(Vec::new());
    static RANK_X: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}
// In the closure:
DIST_X.with(|buf| {
    let mut dist_x = buf.borrow_mut();
    dist_x.clear(); dist_x.resize(n, (0.0, 0));
    // ... use dist_x ...
});
```

Add `bench_thread_local` group using a new public function
`trustworthiness_thread_local(x, y, k)` or a feature-gated variant.
Verify all 5 unit tests + sklearn parity fixture pass.

**3b. Partial-rank X computation** (conditional on X-sort ≥ 40%):

Replace `sort_unstable_by(total_cmp)` + full rank scatter with:
1. `select_nth_unstable_by` at position k (O(n) average).
2. For each Y-neighbor index j, count X-distances strictly less than
   `dist_x[j]` via a single O(n) linear scan.

Mandatory correctness gate before any benchmarking:
- `|T_partial_rank − T_exact| == 0` on n=200 sklearn fixture.
- All 5 unit tests pass.
- Include a synthetic case with intentional X-distance ties.

**3c. Row subsampling variant** (new public function):

```rust
pub fn trustworthiness_approx(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
    sample: usize,
    seed: u64,
) -> f64
```

Randomly sample `sample` row indices without replacement (seeded via
`rand::SeedableRng`), run exact per-row computation only on those rows.

Also add `src/bin/tw_approx_runner.rs` CLI binary with `--x, --y, --k,
--sample, --seed, --output` args, registered in `Cargo.toml` with
`required-features = ["testing", "cli"]`.

**3d. H5 Confirmatory Gate — must run before m-sweep (RT-2 fix):**

Create `scripts/run_h5_confirmatory.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# H5 CONFIRMATORY GATE — must run and seal result before run_subsampling_sweep.sh
# Pre-registered value: m=5000 on MERFISH data (n=10K, d=50)
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"

python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/merfish/merfish_n10k_x.npy" \
  --y "$RESEARCH_DIR/data/merfish/merfish_n10k_y.npy" \
  --n-exact 10000 \
  --sample-sizes "5000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/h5_confirmatory_result.json"

echo "H5 confirmatory result sealed at: $RESEARCH_DIR/results/subsampling/h5_confirmatory_result.json"
echo "Commit this file to git before running run_subsampling_sweep.sh"
```

**Phase ordering constraint (RT-2):**
`run_h5_confirmatory.sh` must complete and its output file committed to git
as a sealed artifact before `run_subsampling_sweep.sh` is invoked. The H5
GO/NO-GO verdict is taken exclusively from `h5_confirmatory_result.json`.
The subsequent m-sweep is exploratory secondary analysis with no bearing on
the H5 GO/NO-GO decision.

---

### Phase 4: Subsampling Deviation Sweep (secondary — runs after Phase 3d)

**Pre-registered production value: m = 5000.** Fixed before this sweep runs.

**4a. `scripts/run_subsampling_sweep.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
RESEARCH_DIR="research/2026-04-04-tw-perf-scaling"

# Verify the confirmatory gate is already sealed
if [ ! -f "$RESEARCH_DIR/results/subsampling/h5_confirmatory_result.json" ]; then
  echo "ERROR: h5_confirmatory_result.json not found. Run run_h5_confirmatory.sh first." >&2
  exit 1
fi

# Primary validation: MERFISH-derived structured data (STOP-3 fix)
echo "=== Subsampling sweep: MERFISH (n=10K, d=50) — secondary analysis ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/merfish/merfish_n10k_x.npy" \
  --y "$RESEARCH_DIR/data/merfish/merfish_n10k_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/merfish_n10k.json"

# Secondary: structured synthetic (non-Gaussian, non-biological)
echo "=== Subsampling sweep: structured synthetic (n=10K, d=10) — secondary ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/structured/structured_n10000_x.npy" \
  --y "$RESEARCH_DIR/data/structured/structured_n10000_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/structured_n10k.json"

# Informational only — NOT used for GO gate; documents STOP-3 finding
echo "=== Subsampling sweep: Gaussian (n=10K, d=10) — informational ONLY ==="
python3 "$RESEARCH_DIR/scripts/subsampling_sweep.py" \
  --x "$RESEARCH_DIR/data/gaussian/gaussian_n10000_x.npy" \
  --y "$RESEARCH_DIR/data/gaussian/gaussian_n10000_y.npy" \
  --n-exact 10000 \
  --sample-sizes "500,1000,2000,5000,10000" \
  --n-trials 5 \
  --output "$RESEARCH_DIR/results/subsampling/gaussian_n10k.json"
```

The script exits with an error if `h5_confirmatory_result.json` is not present,
enforcing the Phase 3d ordering constraint.

`scripts/subsampling_sweep.py` calls `tw_approx_runner` via subprocess at each m
value with 5 independent seeds, collecting `{m, T_exact, T_approx_mean,
T_approx_std, delta_mean, delta_max}` and writing a JSON result file.

---

### Phase 5: Dry Run

Before committing to full Criterion sweeps (30+ minutes):

1. Run `tw_profiler` at n=1K: verify JSON output format, all 6 steps present,
   fractions sum to ~100%.
2. Run Criterion at n=1K only with shortened time (`CRITERION_BENCH_TIME=2s`):
   confirm no panics, output is parseable.
3. Run `run_h5_confirmatory.sh` with `--n-trials 1` (for speed): confirm JSON
   output, delta is finite.
4. Check `asm_inspection.txt` has content and contains meaningful assembly.
5. Verify all unit tests pass: `cargo test --features testing`.
6. Verify `cargo bench --bench trustworthiness_bench` compiles and runs
   **without** `--features testing`.

Create `scripts/dry_run.sh` that automates steps 1–6 in order.

---

## Execution Protocol

Run in order after all implementations are complete:

```bash
# 0. Environment
micromamba activate spectral-test
cd /home/talon/projects/spectral-init

# 1. Build all instrumented binaries (verify Phase 0)
cargo build --release --features testing,cli
# Also verify bench compiles without testing feature (RT-4):
cargo build --release --bench trustworthiness_bench

# 2. Generate all datasets (Phase 1)
python3 research/2026-04-04-tw-perf-scaling/scripts/gen_synthetic.py
python3 research/2026-04-04-tw-perf-scaling/scripts/prepare_merfish.py
python3 research/2026-04-04-tw-perf-scaling/scripts/sklearn_reference.py

# 3. Dry run (Phase 5) — abort if anything fails
bash research/2026-04-04-tw-perf-scaling/scripts/dry_run.sh

# 4. Baseline characterization (Phase 2)
bash research/2026-04-04-tw-perf-scaling/scripts/run_profiling.sh
bash research/2026-04-04-tw-perf-scaling/scripts/run_criterion.sh   # ~30 min
# Inspect binary_identity_check.txt before proceeding:
cat research/2026-04-04-tw-perf-scaling/results/criterion/binary_identity_check.txt
cat research/2026-04-04-tw-perf-scaling/results/asm_inspection.txt  # inspect H3 gate

# 5. Gate checks from Phase 2 results:
#    - X-sort < 40% → skip partial_rank implementation (but still include in combined)
#    - asm_inspection shows AVX2 already → mark H3 NO-GO; avx2_kernel in combined = no-op wrapper
#    Note: combined variant is ALWAYS implemented regardless (RT-1)

# 6. Implement optimizations (Phase 3a–3c)
#    cargo test --features testing  # verify parity after each

# 7. H5 Confirmatory Gate — MUST run before m-sweep (RT-2)
bash research/2026-04-04-tw-perf-scaling/scripts/run_h5_confirmatory.sh
git add research/2026-04-04-tw-perf-scaling/results/subsampling/h5_confirmatory_result.json
git commit -m "seal H5 confirmatory result (m=5000 MERFISH)"

# 8. Run optimization benchmarks (after Phase 3 implementations land)
bash research/2026-04-04-tw-perf-scaling/scripts/run_criterion.sh   # includes variant groups

# 9. Subsampling sweep (Phase 4) — secondary analysis; confirmatory gate must be sealed
bash research/2026-04-04-tw-perf-scaling/scripts/run_subsampling_sweep.sh

# 10. Analysis and recommendation table
python3 research/2026-04-04-tw-perf-scaling/scripts/analyze_results.py \
  --results-dir research/2026-04-04-tw-perf-scaling/results/ \
  --output research/2026-04-04-tw-perf-scaling/results/recommendation_table.md
```

---

## Analysis Plan

**Step 1 — Per-step dominance (H1):** Load all `results/step_timing/gaussian_n*.json`.
For each n, compute `fraction = step_mean_us / total_mean_us`. Plot step fractions
as stacked bar chart vs n. H1 fails if X-sort fraction < 40% at n≥50K.

**Step 2 — Criterion speedup ratios (H2, H3, H6):** For each variant group, compute
`speedup = baseline_mean / variant_mean` at each n. Include 95% confidence intervals
from Criterion's bootstrap. H2 passes if speedup ∈ [1.05, 1.15] for thread-local.
H6 passes if combined speedup ≥ 3.0 at n=100K.

**Step 3 — AVX2 gate (H3):** Manual inspection of `asm_inspection.txt`. If AVX2
instructions present: mark H3 "NO-GO — auto-vectorization confirmed, no manual
SIMD needed." If absent: run manual AVX2 implementation and compare Criterion means.

**Step 4 — AVX-512 gate (H4):** Only if H3 shows ≥ 20% manual AVX2 gain. Otherwise
mark H4 "NO-GO — ceiling too low for AVX-512 to matter."

**Step 5 — Subsampling deviation (H5, MERFISH primary):** The H5 GO/NO-GO verdict
is taken exclusively from `results/subsampling/h5_confirmatory_result.json` (sealed
in Phase 3d, before any sweep is run):
- If `delta_max < 0.001` at m=5000: H5 PASS — GO for subsampling in production.
- If `delta_max ≥ 0.001` at m=5000: H5 FAIL — report empirical threshold from
  smallest m where `delta_max < 0.001`, or mark NO-GO if no such m ≤ 10000.

The subsequent m-sweep (`merfish_n10k.json`) is secondary/exploratory only. Compare
MERFISH errors to Gaussian to confirm STOP-3 finding: Gaussian errors should be
near-zero by concentration, MERFISH errors will be structurally different.

**Step 6 — Recommendation table:** Produce `results/recommendation_table.md`:

| Approach | Speedup (n=100K) | CI 95% | Scaling change | Implementation LOC | Parity impact | GO/NO-GO | Rationale |
|---|---|---|---|---|---|---|---|
| Thread-local buffers | {measured}× | ± | None | ~30 | Zero (exact) | {decision} | ... |
| Partial-rank X | {measured}× | ± | O(n²) → O(n²) | ~60 | Zero if correct | {decision} | ... |
| Manual AVX2 kernel | {measured}× | ± | None | ~40 | Zero (bit-identical) | {decision} | ... |
| Combined | {measured}× | ± | O(n²) → O(n²) | ~120 | Zero (exact) | {decision} | ... |
| Row subsampling m=5000 | ~50× (est) | N/A | O(n²)→O(mn) | ~40 | New: ≤0.001 | {decision} | ... |

No multiple-comparison correction required for primary GO/NO-GO decisions (each
is evaluated against a single pre-specified criterion). The exploratory m-sweep
(H5 secondary) is reported descriptively only.

---

## Success Criteria

- **Conclusive positive:** Per-step profiling shows X-sort ≥ 40% wall-clock at
  n≥50K (H1 supported), AND at least one exact optimization achieves ≥ 1.5×
  Criterion speedup with non-overlapping CI (H2 or H6 supported), AND MERFISH
  subsampling at pre-registered m=5000 produces `delta_max < 0.001` (H5 supported).

- **Conclusive negative:** X-sort < 40% at all measured n (H1 fails; partial-rank
  is NO-GO), AND Criterion speedup for all exact variants ≤ 1.2× (H2/H6 fail),
  AND MERFISH `delta_max ≥ 0.001` at m=5000 (H5 fails). Report the true dominant
  bottleneck revealed by profiling and recommend further investigation.

- **Inconclusive:** Per-step fractions are within 10 percentage points of each
  other at n≥50K (no dominant step), or Criterion runs do not converge (std > 50%
  of mean), or MERFISH dataset preparation fails. Document the specific gap.

---

## Threats to Validity

### Internal

- **Allocation confound (mitigated — STOP-1):** All large-n timing uses ≥2 warm
  iterations discarded before measurement. The cold-start page-fault cost is
  explicitly excluded from reported speedup. Criterion handles this automatically;
  `tw_profiler` implements it manually.

- **Post-hoc sample fraction selection (mitigated — STOP-2, RT-2):** m=5000 is
  pre-registered in this document. The confirmatory test `run_h5_confirmatory.sh`
  runs and seals its result before the m-sweep. The GO decision for H5 uses only
  the confirmatory result; the sweep is exploratory secondary analysis.

- **Vacuous Gaussian parity gate (mitigated — STOP-3):** The H5 GO gate uses
  MERFISH-derived PCA-50 data (biological cluster structure). Gaussian results are
  collected but labeled throughput-characterization-only and are not used for any
  GO/NO-GO decision.

- **Goodhart exploitation of H6 (mitigated — RT-1):** The combined variant
  composition (thread_local + partial_rank + avx2_kernel) is unconditionally fixed
  before any Phase 3 measurements. It cannot be post-hoc adjusted to fit individual
  results.

- **Evaluation collision in Criterion benchmarks (mitigated — RT-4):** The
  `trustworthiness_bench` target carries no `required-features = ["testing"]`.
  The assembly identity check in `run_criterion.sh` confirms the benchmarked binary
  is identical to the clean production build before any speedup ratio is recorded.

- **Baseline comparator framing (disclosed — RT-3):** Speedups are relative to
  current unmodified production code. They are upper bounds on real-world gains if
  the baseline were further optimized.

- **MERFISH scope limitation:** T_exact is computed at n=10K (not n=100K) due to
  the d_x=1122 compute cost. Whether subsampling error at n=10K generalizes to
  n=100K on MERFISH data is not directly tested. This must be documented in the
  report.

- **Partial-rank tie handling:** Equal X-distances must be broken consistently with
  `sort_unstable_by(total_cmp)` behavior. Unit test coverage must include a
  synthetic case with intentional ties.

### External

- **Hardware specificity:** AVX2 and AVX-512 results are specific to CI target
  (`x86-64-v3` via `target-cpu=native`). Machines without AVX2 fall back to scalar;
  the speedup table must document CPU features detected at benchmark time.

- **d_x=10 synthetic benchmarks:** Step fractions at d_x=10 differ from d_x=50
  (MERFISH, PCA-reduced) — Step 1 (distance computation) will be proportionally
  larger at higher d. The profiler must also run on the MERFISH d=50 dataset to
  characterize this.

- **Rayon thread count:** Results depend on system thread count. All JSON outputs
  must include `rayon::current_num_threads()`.

---

## Estimated Resource Requirements

- **Disk:** ~500 MB (synthetic data for n=100K, Criterion HTML, results JSON)
- **Criterion benchmarks (n=1K–50K, 5 variants × 10s each):** ~10 min
- **Large-scale profiling (n=100K, 5 warm iters):** ~40–60 min (exact wall-clock unknown)
- **Optional n=250K run:** potentially killed; wrap with `timeout 600s`
- **Subsampling sweep (n=10K × 3 datasets × 5 m values × 5 trials):** ~30 min
- **Python data generation + sklearn reference:** ~5 min
- **Total:** ~2–3 hours compute
