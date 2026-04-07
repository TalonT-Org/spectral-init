# Experiment Plan: KD-tree Y k-NN for Trustworthiness at n ≥ 50K

## Motivation

`trustworthiness()` uses an O(n²) brute-force scan to find the k nearest
neighbors in Y-space. At n=50K the flat_simd implementation takes ~3.55s
wall-clock; at n=100K ~13.6s. These runtimes make `trustworthiness()` unusable
as a live metric in iterative tuning loops. Replacing the inner Y k-NN scan with
an `ImmutableKdTree` query would drop Y k-NN from O(n²) to O(n k log n) at the
cost of a one-time O(n log n) build. This experiment measures whether that
trade-off is favorable at n ≥ 50K, finds the crossover point below which
brute-force remains preferable, and determines whether an adaptive dispatch
should be shipped.

The decision this experiment informs: ship an adaptive dispatch in
`trustworthiness()` that uses kiddo ImmutableKdTree when n ≥ T_cross and d_y == 2,
falling back to flat_simd otherwise.

---

## Research Design Rationale (Pre-Specification Notice)

All thresholds, metric definitions, success criteria, and the conjunctive success
structure are specified here, before any data is collected, and are not to be
revised after `run_criterion.sh` is first executed. This section documents the
pre-specification rationale so that the conjunction cannot be mistaken for a
post-hoc design.

**Why ≥ 5× at n=50K (H1 primary threshold)?**
Theoretical KD-tree speedup at d=2 is O(n / k log n) vs O(n). At n=50K, k=15:
ratio = 50K / (15 × log₂(50K)) = 50000 / (15 × 15.6) = 214×. Practical constant
factors (cache misses, tree traversal overhead, Rayon task granularity, kiddo's
priority queue) compress this to an estimated 50–100× wall-clock speedup.
A ≥ 5× threshold is deliberately conservative — it is the minimum improvement
that would make `trustworthiness()` practical at n=50K in an interactive loop.

**Why ≥ 10× at n=100K?**
The O(n k log n) vs O(n²) gap widens linearly in n. If the 5× threshold is met at
n=50K, a 10× threshold at n=100K merely requires the same theoretical gain to
persist — it is a consistency check, not a new claim. These two thresholds together
constitute a single intersection hypothesis (H1 ∧ H1'), not 30 independent tests.

**Why conjunctive "all five conditions" for conclusive positive?**
The shipping decision requires: (a) it is fast enough to matter, (b) it does not
regress on real inputs, (c) the crossover is stable enough to set a hard threshold.
These are not independent hypotheses — a KD-tree that fails correctness is not
shippable regardless of its speed. The conjunction reduces familywise error rate
relative to testing each claim separately; its conservatism is appropriate here
because false positives (shipping a broken or marginal implementation) are costlier
than false negatives (deferring the optimization).

**Primary DV declaration (W3):**
The primary dependent variable is `tw_kdtree_total_speedup` — total wall-clock
speedup including tree build cost — at n=50K, measured by Criterion wall-clock time.
Query-only speedup is a secondary DV. This matches H1's language ("wall-time
improvement") and the deployment context (a single `trustworthiness()` call incurs
the build cost every time).

---

## Hypothesis

**Null hypothesis (H0):** Building an `ImmutableKdTree<f64, u32, 2, 32>` over the
Y embedding and querying it for k-NN in parallel provides ≤ 2× total wall-time
speedup over flat_simd at n=50K with d_y=2, k=15.

**Alternative hypothesis (H1):** The KD-tree path provides ≥ 5× total wall-time
speedup over flat_simd at n=50K AND ≥ 10× at n=100K, while preserving
|T_kdtree − T_brute_force| < 1e-12 on all test cases.

**H2 (crossover):** There exists a crossover threshold T_cross ∈ [1K, 50K] below
which flat_simd is faster than KD-tree. T_cross is identified empirically from
Criterion wall-clock measurements at n ∈ {1K, 5K, 10K, 50K, 100K}.

**H3 (tie-breaking):** Equidistant-at-rank-k points are sufficiently rare in both
uniform random and Gaussian-cluster 2D embeddings that |T_kdtree − T_brute_force|
< 1e-12 holds on all tested inputs.

**H4 (amortization):** At n ≥ 50K, ImmutableKdTree build cost ≤ 10% of the total
KD-tree path wall time (build + all n queries), confirming that a single
`trustworthiness()` call is dominated by query cost, not build cost.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm path | `flat_simd`, `kdtree` | The two candidates being compared |
| n (dataset size) | 1,000 · 5,000 · 10,000 · 50,000 · 100,000 | Spans the crossover region and the target regime |
| Data distribution | uniform random, Gaussian clusters | Uniform = adversarial (no spatial structure); clusters = realistic UMAP embedding |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name | Status |
|--------|------|-------------------|----------------|--------|
| Total wall-time speedup at n=50K | dimensionless ratio | Criterion wall-clock, flat_simd / kdtree | `tw_kdtree_total_speedup_50k` | NEW |
| Total wall-time speedup at n=100K | dimensionless ratio | Criterion wall-clock, flat_simd / kdtree | `tw_kdtree_total_speedup_100k` | NEW |
| KD-tree build time | ms | Explicit `Instant::now()` inside bench function, captured per sample | `tw_kdtree_build_ms` | NEW |
| KD-tree query-only speedup | dimensionless ratio | `[timing:y_dist]` (flat_simd profiler) / `[timing:y_kdtree_query]` (kdtree profiler) | `tw_kdtree_query_speedup` | NEW |
| Correctness: |T_kdtree − T_brute_force| | dimensionless | Existing `t_tw_08` + `t_tw_10` + new `t_tw_11_kdtree_matches_baseline` | `tw_correctness_delta` | NEW path in existing infra |
| Build cost fraction | % | `tw_kdtree_build_ms / tw_kdtree_total_ms` | `tw_kdtree_build_fraction` | NEW |

**NEW metric formulas and thresholds:**
- `tw_kdtree_total_speedup_50k` = Criterion mean time (flat_simd, n=50K) / Criterion mean time (kdtree, n=50K); threshold for H1: ≥ 5.0
- `tw_kdtree_build_fraction` = build_ms / total_ms; threshold for H4: ≤ 10%
- `tw_correctness_delta`: inherits the existing threshold of < 1e-12 from `t_tw_08` / `t_tw_10`

**Correlated DV acknowledgment (W11):** `tw_kdtree_total_speedup` is derived from
`tw_flat_simd_total_ms` and `tw_kdtree_total_ms`, which are also reported as primary
DVs. This correlation is structural and expected. No separate statistical correction
is applied to the derived ratio; the ratio is the primary decision variable, and the
two component DVs are reported as supporting evidence.

**Separation of total vs query-only speedup (R2):** These are distinct metrics
collected by different mechanisms (Criterion for total; profiler atomics for
step-level). The `tw_kdtree_total_speedup` metric places build cost in the
denominator (correct for single-call deployment). `tw_kdtree_query_speedup` isolates
the query algorithm. Both are reported; the shipping decision is based on total.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| d_y (embedding dimension) | 2 | KD-tree path is restricted to d_y=2 (kiddo uses const-generic K; compile-time 2D specialization) |
| d_x (input dimension) | 10 | Matches existing bench; x-side computation unchanged |
| k (neighborhood size) | 15 | Matches existing bench and UMAP default |
| RNG seed | 42 (uniform), 99 (Gaussian) | Reproducibility |
| Rayon thread count (RT6) | Fixed to `RAYON_NUM_THREADS` recorded at run time | Thread count is recorded in results JSON and scoped into all conclusions |
| kiddo leaf size | 32 (const generic) | Default recommended by kiddo documentation; not empirically tuned against this workload (RT2) |
| Criterion sample size | 10 samples | Matches existing bench configuration |
| Criterion warm_up_time | 10s per group | Not optimized for either path — applies equally to both (RT5) |
| Criterion benchmark group order | flat_simd first, kdtree second | KD-tree inherits cache warmup from brute-force run; this is a conservative bias against KD-tree and is documented as such (W8) |
| Benchmark binary compilation | Both paths compiled into the same binary via runtime dispatch (`use_kdtree: bool` argument) | Eliminates dead-code elimination asymmetry (R1) |
| Profiling atomic isolation | Each variant runs as a separate process invocation | Eliminates cross-group atomic accumulation (R3, R4) |

---

## Inputs and Data

The experiment uses two data conditions to avoid a narrow conclusion from a single
distribution (RT3):

1. **Uniform random** (adversarial): U(0,1) independent draws for both X and Y.
   No spatial clustering in Y → KD-tree cannot exploit density; this is the worst
   case for tree pruning efficiency. If KD-tree is fast here, it is fast everywhere.

2. **Gaussian clusters** (realistic): Y is sampled from a mixture of 8 isotropic
   Gaussian clusters with σ=0.3, center grid [0,3]². X is uniform. This simulates a
   realistic UMAP 2D embedding with cluster structure. KD-tree pruning is most
   effective here.

Both conditions use the same n values and k=15. A negative result on uniform data
is necessary but not sufficient to reject KD-tree; if KD-tree loses only on uniform
but wins strongly on clusters, the adaptive dispatch is still shippable (documented
in analysis plan, RT3 accepted).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| uniform_n{1k,5k,10k,50k,100k}_x.npy | Generated by gen_data.py | n×10, U(0,1) | Input X for uniform condition |
| uniform_n{1k,5k,10k,50k,100k}_y.npy | Generated by gen_data.py | n×2, U(0,1) | Input Y for uniform condition |
| gauss_n{1k,5k,10k,50k,100k}_x.npy | Generated by gen_data.py | n×10, U(0,1) | Input X for Gaussian condition |
| gauss_n{1k,5k,10k,50k,100k}_y.npy | Generated by gen_data.py | n×2, 8-cluster Gaussian mixture | Input Y for Gaussian condition |

**Data manifest source_type tags (W9):**
All 20 files: `source_type: generated` — produced by `scripts/gen_data.py` from
fixed seeds; not gitignored (stored in `research/` which is in-repo). Verification:
assert shape, assert no NaN/Inf, assert dtype=float64. Full verification covers all
20 files (not a subset).

**Re-use of existing data:** The 6 existing `.npy` files in
`research/2026-04-06-y-heap-bottleneck-optimization/data/` (n={1K,5K,10K}, uniform)
can be symlinked or copied into `data/` for the n ≤ 10K uniform condition to avoid
regeneration, but re-generation is also acceptable since seeds are identical.

---

## Experiment Directory Layout

All experiment artifacts live in one self-contained folder:

```
research/2026-04-07-kdtree-y-knn-trustworthiness/
├── environment.yml             # Python env for gen_data.py and analyze_results.py
├── rust-toolchain.toml         # Pin nightly channel
├── scripts/
│   ├── gen_data.py             # Generate uniform + Gaussian .npy files for all n
│   ├── run_criterion.sh        # Run Criterion bench (3 repetitions), collect JSONs
│   ├── run_profiler.sh         # Run tw_profiler per variant (separate processes)
│   └── analyze_results.py     # Compute speedup ratios, CV, crossover, render plots
├── data/
│   ├── uniform_n1000_{x,y}.npy
│   ├── uniform_n5000_{x,y}.npy
│   ├── uniform_n10000_{x,y}.npy
│   ├── uniform_n50000_{x,y}.npy
│   ├── uniform_n100000_{x,y}.npy
│   ├── gauss_n1000_{x,y}.npy
│   ├── gauss_n5000_{x,y}.npy
│   ├── gauss_n10000_{x,y}.npy
│   ├── gauss_n50000_{x,y}.npy
│   └── gauss_n100000_{x,y}.npy
├── results/
│   ├── criterion/
│   │   ├── flat_simd_{uniform,gauss}_n{1k,5k,10k,50k,100k}_rep{1,2,3}.json
│   │   └── kdtree_{uniform,gauss}_n{1k,5k,10k,50k,100k}_rep{1,2,3}.json
│   ├── profiler/
│   │   ├── flat_simd_n{1k,5k,10k,50k,100k}_{uniform,gauss}.json
│   │   └── kdtree_n{1k,5k,10k,50k,100k}_{uniform,gauss}.json
│   └── analysis/
│       ├── analysis_report.md
│       ├── speedup_by_n.png
│       ├── build_fraction_by_n.png
│       └── crossover_summary.json
└── report.md                   # Final report (written by /write-report)
```

**File descriptions:**
- `gen_data.py`: numpy-only data generator. Outputs 20 `.npy` files. Verifies each
  file (shape, dtype, NaN/Inf check) and writes a verification manifest JSON.
- `run_criterion.sh`: iterates over (variant, distribution, n, rep) tuples; copies
  `target/criterion/<group>/estimates.json` to `results/criterion/`. Records
  `RAYON_NUM_THREADS` (or system default) into a `results/run_metadata.json`.
- `run_profiler.sh`: for each (variant, distribution, n) combination, invokes
  `cargo run --bin tw_profiler --features profiling,cli -- ...` as a **separate
  process** so profiling atomics start at zero. Parses stderr `[timing:*]` lines
  into JSON objects in `results/profiler/`.
- `analyze_results.py`: reads all Criterion JSONs and profiler JSONs; computes DVs;
  flags high-CV measurements (>10%); identifies crossover; writes `analysis_report.md`
  and PNG plots.

---

## Environment

**Python environment required** for `gen_data.py` and `analyze_results.py`.

```yaml
name: kdtree-y-knn-bench
channels:
  - conda-forge
dependencies:
  - python=3.11.*
  - numpy=2.2.*
  - scipy=1.15.*
  - matplotlib=3.10.*
```

Rationale: numpy for data generation and `.npy` I/O; scipy for Gaussian mixture
sampling; matplotlib for speedup plots. No scikit-learn required — sklearn parity is
handled by the existing `#[ignore]` integration test suite using its own fixture.

**Rust environment:** Project's existing nightly toolchain is sufficient.
- `cargo-criterion v1.1.0` is already installed.
- `target-cpu=native` is already set via `.cargo/config.toml`.
- `kiddo v5` with `rayon` feature is added to `[dev-dependencies]` in `Cargo.toml`
  during Phase 1. It is a dev-dependency only (bench and test scope); it does not
  affect the library's public dependency surface.

```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly-2026-03-26"
```
(Matches the pin used in the prior research worktree for consistency.)

---

## Implementation Phases

### Phase 1: Repository Setup

1. Create `research/2026-04-07-kdtree-y-knn-trustworthiness/` and all
   subdirectories (`scripts/`, `data/`, `results/criterion/`, `results/profiler/`,
   `results/analysis/`).
2. Create `environment.yml` and `rust-toolchain.toml` as specified above.
3. Add `kiddo` to `Cargo.toml` under `[dev-dependencies]`:
   ```toml
   kiddo = { version = "5", features = ["rayon"] }
   ```
4. Run `cargo build --tests --features testing` to verify the dependency resolves
   cleanly and there are no ndarray version conflicts. Run `cargo tree -p kiddo` and
   record the resolved version.

**Verification:** `cargo build` exits 0.

### Phase 2: Data Generation

1. Write `scripts/gen_data.py`:
   - Generates `uniform_n{n}_{x,y}.npy` for n ∈ {1000, 5000, 10000, 50000, 100000}
     using `np.random.default_rng(42).random((n, d))`.
   - Generates `gauss_n{n}_{x,y}.npy` for same n: Y from 8-cluster Gaussian mixture
     (centers on [0,3]² grid, σ=0.3, balanced clusters) using `rng(99)`; X is uniform.
   - After each file is written, verifies: shape, dtype==float64, no NaN, no Inf.
   - Writes `data/manifest.json` with filename, shape, source_type="generated",
     seed, and verification pass/fail for all 20 files.
2. Activate the micromamba environment and run `python scripts/gen_data.py` from
   the research directory.

**Verification:** `manifest.json` shows all 20 files with `verified: true`.

### Phase 3: KD-tree Implementation in Metrics

Modify `src/metrics.rs` to add the KD-tree code path. Do **not** change the public
API of `trustworthiness()`. The implementation strategy is:

**3a. Add new profiling atomics** (inside `trustworthiness()`, under
`#[cfg(feature = "profiling")]`, alongside the existing `Y_DIST_NS`):
```rust
static Y_KDTREE_BUILD_NS: AtomicU64 = AtomicU64::new(0);  // tree construction only
static Y_KDTREE_QUERY_NS: AtomicU64 = AtomicU64::new(0);  // all n queries total
```
Emit these in the profiling output block as:
```
[timing:y_kdtree_build] {ns}
[timing:y_kdtree_query] {ns}
```
These are separate from `Y_DIST_NS`, which continues to label the flat_simd path
as `[timing:y_dist]` (R7).

**3b. Add runtime-dispatch helper** `trustworthiness_inner(x, y, k, use_kdtree: bool)`
(private). This function contains all the per-row logic with a branch:
- `use_kdtree == false`: existing flat_simd path (unchanged)
- `use_kdtree == true`: kiddo path (new)

Both paths are compiled into the same binary regardless of which is called. Dead-code
elimination cannot remove either path because the branch is on a runtime `bool` (R1).

**3c. Public function** `trustworthiness()` calls `trustworthiness_inner(x, y, k, false)`.
This ensures zero behavioral change to the production path.

**3d. KD-tree path implementation:**
```rust
// Before Rayon loop — timed with Y_KDTREE_BUILD_NS:
use kiddo::{ImmutableKdTree, SquaredEuclidean};
let build_start = Instant::now();
let points: Vec<[f64; 2]> = (0..n).map(|i| [y[[i, 0]], y[[i, 1]]]).collect();
let tree: Arc<ImmutableKdTree<f64, u32, 2, 32>> =
    Arc::new(ImmutableKdTree::new_from_slice(&points));
// (record build ns in Y_KDTREE_BUILD_NS)

// Inside Rayon loop for row i — timed with Y_KDTREE_QUERY_NS per row:
let results = tree.nearest_n::<SquaredEuclidean>(
    &[y[[i, 0]], y[[i, 1]]],
    NonZero::new(k + 1).unwrap(),
);
let knn_y_indices: Vec<usize> = results
    .into_iter()
    .filter(|nb| nb.item as usize != i)   // self-exclusion
    .take(k)
    .map(|nb| nb.item as usize)
    .collect();
// (penalty loop unchanged)
```

**3e. Add correctness test** `t_tw_11_kdtree_matches_baseline` in `src/metrics.rs`,
following the pattern of `t_tw_08`:
- n=50, X(50×6), Y(50×2), seed=123, k ∈ {3, 7}
- Calls `trustworthiness_inner(x, y, k, true)` and asserts
  `|T_kdtree − T_brute_force| < 1e-12`.

**Verification:** `cargo test t_tw_11 --features testing` passes.

**3f. Verify existing tests still pass** for the unmodified production path:
`cargo test t_tw_08 t_tw_10 --features testing`

### Phase 4: Benchmark Extension

Modify `benches/trustworthiness_bench.rs` to add KD-tree variant groups:

For each (distribution, n) pair, add a Criterion group named
`kdtree_{distribution}_n{n}` alongside the existing `flat_simd_{distribution}_n{n}`.
Each group calls `trustworthiness_inner(x.view(), y.view(), k, use_kdtree)` where
`use_kdtree` is `true` or `false`.

The bench function also captures build time separately using `Instant::now()` in a
`black_box`-wrapped pre-measurement step, storing the build duration in the group's
user data or printing to stderr in a parseable format:
```
[bench:build_ms] <f64>
```
One build-time measurement per sample (not amortized over iterations).

Add one additional n value: n=75,000 as a held-out crossover validation point (RT8).
This n was not used in defining the hypothesis thresholds and provides an independent
check on whether the crossover identified from {1K, 5K, 10K, 50K, 100K} is stable.

**Verification:** `cargo bench --bench trustworthiness_bench --features testing
  -- flat_simd_uniform_n1000 --profile-time 5` runs without error.

### Phase 5: Profiler Extension

Modify `src/bin/tw_profiler.rs` to accept a `--variant flat_simd|kdtree` argument.
When `--variant kdtree` is specified, it calls `trustworthiness_inner(x, y, k, true)`
and emits `[timing:y_kdtree_build]` and `[timing:y_kdtree_query]` to stderr.

**Verification:** `cargo run --bin tw_profiler --features profiling,cli -- --n 1000
  --variant flat_simd --iters 5` and `--variant kdtree` both emit valid timing lines.

### Phase 6: Dry Run

Execute the full pipeline at small scale (n=1000 only, 2 Criterion samples, 3
profiler iterations, one repetition) to confirm end-to-end correctness:

```bash
# From research directory:
python scripts/gen_data.py --n-max 1000       # or rely on full gen, use n=1000 files
bash scripts/run_criterion.sh --dry-run        # n=1000 only, 2 samples
bash scripts/run_profiler.sh --dry-run         # n=1000 only, 3 iters
python scripts/analyze_results.py --dry-run   # reads n=1000 results only
```

Verify:
- All Criterion JSONs exist with non-zero `point_estimate`
- All profiler JSONs exist with `[timing:y_dist]` (flat_simd) and
  `[timing:y_kdtree_build]` + `[timing:y_kdtree_query]` (kdtree)
- `analysis_report.md` is generated without errors
- Correctness test `t_tw_11` continues to pass

---

## Execution Protocol

All commands assume the micromamba environment `kdtree-y-knn-bench` is active for
Python steps, and a shell in the project root for Rust steps.

**Step 1 — Generate data (once):**
```bash
cd research/2026-04-07-kdtree-y-knn-trustworthiness
micromamba run -n kdtree-y-knn-bench python scripts/gen_data.py
```
Verify `data/manifest.json` shows all 20 files verified.

**Step 2 — Run correctness tests:**
```bash
cargo test t_tw_08 t_tw_10 t_tw_11 --features testing 2>&1 | tee results/correctness.log
```
All three tests must pass before proceeding. If any fail, halt.

**Step 3 — Criterion benchmarks (3 repetitions, all n and distributions):**
```bash
bash research/2026-04-07-kdtree-y-knn-trustworthiness/scripts/run_criterion.sh
```
This script:
- Sets `RAYON_NUM_THREADS` to the system CPU count (via `nproc`) and exports it
- Records `RAYON_NUM_THREADS`, Rust toolchain version, and timestamp into
  `results/run_metadata.json`
- Runs `cargo criterion --bench trustworthiness_bench --features testing`
  three times sequentially (rep 1, 2, 3)
- After each run, copies `target/criterion/*/estimates.json` to
  `results/criterion/<group_name>_rep{1,2,3}.json`

Expected wall time: n=50K bench takes ~35s/sample × 10 samples × 2 variants ×
2 distributions × 3 reps ≈ 70 minutes. n=100K: ~140s/sample — estimated total
bench time: ~8–10 hours. Run overnight or in tmux.

**Step 4 — Profiler step-level measurements (per variant, separate processes):**
```bash
bash research/2026-04-07-kdtree-y-knn-trustworthiness/scripts/run_profiler.sh
```
This script iterates over (variant ∈ {flat_simd, kdtree}) × (n ∈ {1K,5K,10K,50K,100K})
× (distribution ∈ {uniform, gauss}) and for each combination invokes a **fresh
cargo run** (separate process):
```bash
RAYON_NUM_THREADS=$NUM_THREADS cargo run --bin tw_profiler \
  --features profiling,cli --release -- \
  --n $N --variant $VARIANT --dist $DIST --iters 30 \
  2> stderr_tmp.txt
```
Parses `stderr_tmp.txt` for `[timing:*]` lines, computes mean and std_dev across
iterations, and writes `results/profiler/${variant}_n${n}_${dist}.json`.

**Profiler variance reporting (W5):** Each profiler JSON includes per-iteration
timing arrays (not just means), enabling post-hoc CV computation for each step.

**Step 5 — Analysis:**
```bash
micromamba run -n kdtree-y-knn-bench python \
  research/2026-04-07-kdtree-y-knn-trustworthiness/scripts/analyze_results.py
```

**Step 6 — Record all run outcomes (RT4):**
All runs are reported regardless of outcome. The script writes `results/run_log.json`
listing every (variant, n, distribution, rep) tuple with status (completed/failed/
high-variance). No selective reporting.

---

## Analysis Plan

### Primary analysis (H1): Total wall-time speedup at n=50K and n=100K

For each n ∈ {50K, 100K} and each distribution ∈ {uniform, gauss}:
1. Take the median of the three Criterion `point_estimate` values (reps 1, 2, 3) as
   the primary estimate for each (variant, n, dist) cell.
2. Compute CV = std(three reps) / mean(three reps) for each cell.
3. Compute `tw_kdtree_total_speedup_N` = median_flat_simd / median_kdtree.
4. Check whether the 95% CI of the flat_simd group and the 95% CI of the kdtree
   group overlap (Criterion's built-in CI). Non-overlap indicates a statistically
   distinguishable difference.

**High-variance policy (W6 unified):**
- CV > 10%: flag cell; attempt one re-run of that specific (variant, n, dist) cell.
- CV > 10% after re-run: mark cell as "high variance"; exclude from primary
  speedup ratio but include in report with raw values.
- If both n=50K AND n=100K primary speedup estimates are high-variance: declare
  inconclusive (cannot test H1).

### Crossover analysis (H2)

For the uniform distribution (adversarial, conservative for KD-tree):
- Plot `tw_kdtree_total_speedup` vs log(n) for n ∈ {1K, 5K, 10K, 50K, 75K, 100K}.
- The crossover point T_cross is the n at which speedup = 1.0 (interpolated
  linearly between adjacent measured n values where the sign flips).
- Crossover variance: T_cross from rep 1 vs rep 2 vs rep 3 (W4). Report the range.
  If max(T_cross_reps) / min(T_cross_reps) > 2×, flag crossover as unstable.
- Conclusions about T_cross are explicitly scoped to "description of tested n values,
  not a predictive crossover estimate" — no extrapolation beyond n=100K (RT8).

Note: n=75K is the held-out validation point (RT8). The crossover estimate from
{1K, 5K, 10K, 50K, 100K} is used to predict whether n=75K falls above or below the
crossover; the actual n=75K speedup is then checked against the prediction.

### Build cost analysis (H4)

For each (n, distribution): compute `tw_kdtree_build_fraction` from profiler output
(`[timing:y_kdtree_build]` / (`[timing:y_kdtree_build]` + `[timing:y_kdtree_query]`)).
Verify H4: build fraction ≤ 10% at n=50K and n=100K.

Report `tw_kdtree_build_ms` as an absolute number to assess amortization for
single-call deployment.

### Tie-breaking (H3)

Report `|T_kdtree − T_brute_force|` from `t_tw_11` on the small correctness inputs.
For the benchmark-scale inputs, correctness is checked indirectly: run the benchmark
with n=1K (where both paths are exercised) and verify that the printed T scores match
to < 1e-8 (a relaxed check appropriate for the seeded benchmark data, where brute-force
reference is not available at n=50K scale).

### Allocation asymmetry acknowledgment (W7)

The flat_simd path reuses thread-local `Vec` buffers with no per-query heap allocation.
kiddo's `nearest_n` uses a heap priority queue (estimated 1–3 heap allocations per
query). This structural asymmetry is reported as a confound: if the speedup is
measured at ~2–3× instead of the expected ≥5×, heap allocation pressure may explain
part of the gap. The analysis will note whether allocation profiling (e.g., via
`DHAT` or `jemalloc` stats) would be warranted in a follow-up.

### d_y scope (W13)

The KD-tree path is gated on `d_y == 2`. Document that all production UMAP calls
default to d_y=2, so the adaptive flag benefits the dominant use case. For d_y ≠ 2,
the fall-back to flat_simd is silent and correct. No empirical bound on the fraction
of d_y ≠ 2 calls is needed for the shipping decision; the flag is simply a no-op for
non-2D embeddings.

### Profiler RT1 asymmetry acknowledgment

The KD-tree path receives decomposed profiling (`build` + `query` separately).
The flat_simd path receives a single `[timing:y_dist]` label (fill + introselect
combined). This asymmetry is accepted and documented: flat_simd profiling
decomposition is not symmetrized because it would require restructuring the existing
hot path. Any performance advantage from targeted profiler-visible optimization
(RT1) applies equally to both paths once production code is shipped.

---

## Success Criteria

**Conclusive positive (ship the KD-tree adaptive dispatch):**
All five of the following conditions are met simultaneously:
1. `tw_kdtree_total_speedup_50k` ≥ 5.0 on uniform distribution (primary test)
2. `tw_kdtree_total_speedup_100k` ≥ 10.0 on uniform distribution
3. `t_tw_08`, `t_tw_10`, `t_tw_11` all pass with `|ΔT| < 1e-12`
4. Crossover T_cross variance ≤ 2× across 3 reps (stable enough to set adaptive threshold)
5. `tw_kdtree_build_fraction` ≤ 10% at both n=50K and n=100K

**Conclusive negative (do not ship KD-tree; flat_simd is sufficient at these scales):**
- `tw_kdtree_total_speedup_50k` ≤ 2.0 on both distributions
- (H0 is retained; the expected KD-tree gain at d=2 does not materialize in practice)

**Informative inconclusive (follow-up recommended):**
- `tw_kdtree_total_speedup_50k` ∈ (2.0, 5.0) — above H0 but below H1;
  suggests the tree is faster but allocation pressure or cache effects limit gain;
  recommend profiling with DHAT before shipping
- Correctness failures (`|ΔT| ≥ 1e-12`) — investigate tie-breaking; may require
  index-stable comparison in kiddo query post-processing before shipping
- High-variance measurements in both primary cells — re-run on a quiet machine

**Mandated reporting (RT4):** All run outcomes are included in `results/run_log.json`
and summarized in `report.md`, including inconclusives and failures. No selective
omission.

---

## Threats to Validity

### Internal

**Allocation asymmetry (W7):** kiddo allocates a heap priority queue per query;
flat_simd reuses thread-local Vecs. The KD-tree speedup may be partially offset by
allocator contention under Rayon. If speedup falls below expectation on uniform data,
this is the first candidate confounder.

**Cache warming bias (W8):** Criterion runs flat_simd first, then kdtree in the same
process. The kdtree group benefits from warm OS file caches and may benefit from
residual CPU cache state. This is a conservative bias against the KD-tree path
(the tree brings its own new working set that must displace whatever flat_simd cached);
net direction depends on working set sizes.

**Rayon thread count variability (RT6):** Speedup is measured at a fixed
`RAYON_NUM_THREADS` (recorded in `run_metadata.json`). If the system has background
load, Rayon's work-stealing efficiency varies. Conclusions are scoped to the recorded
thread count.

**Floating-point non-associativity (W12):** `trustworthiness()` uses Rayon's parallel
reduction for the penalty sum. Non-deterministic reduction ordering may produce
`|ΔT|` differences at the 1e-12–1e-15 level even between two identical code paths.
The 1e-12 correctness tolerance is inherited from the existing test suite (`t_tw_08`,
`t_tw_10`), which was established under the same Rayon reduction structure. If `t_tw_11`
fails by a margin consistent with floating-point non-associativity (e.g., |ΔT| =
2e-13), this is not a KD-tree correctness failure — it is a Rayon reduction artifact.
The analysis plan documents this distinction.

**Leaf-size tuning risk (RT2):** Leaf size 32 is the kiddo-recommended default. It
has not been empirically tuned against n=50K/k=15 in this codebase. If results are
weaker than expected, leaf-size sensitivity is the first follow-up to investigate.

### External

**Generalizability to d_y > 2:** KD-tree pruning efficiency degrades with dimension.
All conclusions are scoped to d_y=2 only, which is the UMAP default and the
experimentally tested condition.

**Data distribution:** Results on uniform random and Gaussian clusters bound
real-world performance from two extremes. Pathological cases (e.g., all points
collinear, degenerate clusters with duplicates) are not tested. The adaptive dispatch
falls back to flat_simd for all d_y ≠ 2 inputs, which eliminates correctness risk
for untested distributions; performance risk is limited to d_y=2 non-clustered inputs.

**Machine specificity:** Results depend on CPU cache hierarchy, AVX2 availability,
and Rayon thread count. Conclusions include the hardware configuration
(`run_metadata.json`). The speedup ratio should be directionally correct on any
x86_64 AVX2 machine but the crossover threshold may shift.

---

## Estimated Resource Requirements

| Item | Estimate |
|------|---------|
| Criterion bench time (all n, all variants, 3 reps) | 8–12 hours (run overnight) |
| Profiler time (all n, all variants, 30 iters) | 2–4 hours |
| Disk space for data (20 × n=100K × 10 and 2 doubles) | ~200 MB |
| Disk space for Criterion results | ~5 MB |
| Build time (add kiddo, recompile) | ~2–5 minutes |
| Implementation time (Phases 1–5) | 4–6 hours |
| Analysis (Python script + plots) | 30–60 minutes |
