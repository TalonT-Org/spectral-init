# Experiment Plan: X-space Distance Kernel SIMD Optimization (AVX-512 / Looped AVX2 / Cache Tiling)

## Motivation

`x_dist` accounts for 58.9% of total `trustworthiness` runtime on MERFISH 10K (n=10,000, d_x=50), yet
the current `dist_sq_avx2` kernel covers only 8 of 50 dimensions with SIMD — the remaining 84% fall
to scalar arithmetic. This experiment measures whether replacing the two-fixed-load structure with a
properly looped SIMD kernel (AVX2 and/or AVX-512) meets the ≥1.5× total trustworthiness speedup
target, and whether cache tiling provides additive benefit on AMD Ryzen 7 9800X3D (96 MB 3D V-Cache).

The results inform whether to ship the looped kernel to `src/metrics.rs` and whether to invest in the
more complex tiling path.

> **Data note:** The scope report asserts "all four MERFISH fixture files present." Feasibility
> analysis found **no MERFISH fixture files** anywhere in the repo. All benchmarks use synthetic
> `make_data(n, d_x=50, d_y=2, seed=42)` data with matching dimensions (n=10,000 or n=50,000).
> This is sufficient to test the hypothesis because `dist_sq_*` performance depends on d_x and n,
> not on the specific floating-point values.

---

## Hypothesis

**Null hypothesis (H0):** Replacing the two-fixed-load `dist_sq_avx2` with a properly looped SIMD
kernel does not improve total `trustworthiness` wall-clock time by more than 1.5× on synthetic
n=10,000, d_x=50 data.

**Alternative hypothesis (H1):** A looped SIMD kernel (AVX2 or AVX-512) that processes all d_x
elements in register-width chunks reduces x_dist step time by ≥4× and total trustworthiness time
by ≥1.5× on n=10,000, d_x=50. This follows from: (a) 84% of FLOP currently in scalar, (b) AVX2
FMA throughput ~4–8× scalar f64 on Zen 5, implying total speedup ≈ 1/(0.411 + 0.589/5) ≈ 1.9×.

**Secondary hypotheses:**

| Label | Claim |
|-------|-------|
| H2 | AVX-512 (8-wide ZMM) adds ≥1.2× additional speedup over looped AVX2 at d_x=50 |
| H3 | Cache tiling improves x_dist time by ≤10% on this machine (96 MB V-Cache eliminates LLC pressure) |
| H4 | Block-triangular symmetry exploitation is infeasible (memory cost ≥ 800 MB at n=10K) |

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| SIMD kernel variant | `current` (2-fixed-load AVX2), `avx2_looped` (4-wide YMM loop), `avx512_looped` (8-wide ZMM loop) | Tests H1 and H2 — SIMD width vs. scalar tail coverage |
| Cache tile size | `none` (current), `64`, `128`, `256` (rows per tile) | Tests H3 — L2 tiling on V-Cache hardware |
| Input n | 1,000, 5,000, 10,000, 50,000 | Scaling behavior; n=10,000 is primary MERFISH target |
| Input d_x | 10 (regression), 50 (primary) | Matches the d_x where current kernel is most broken |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Total trustworthiness wall-clock time | ms / call | `cargo bench --bench trustworthiness_bench` Criterion JSON | NEW — no threshold; track speedup ratio |
| x_dist step time | ns | `tw_profiler` binary (`--features profiling`) JSON output, field `step_timing.x_dist` | NEW — `X_DIST_NS` atomic accumulator in `src/metrics.rs` |
| Trustworthiness score parity (sklearn gate) | absolute delta | `cargo test --features testing -- --ignored sklearn_parity_synthetic` | NEW — inline 1e-6 threshold in test; no named constant `TW_PARITY_THRESHOLD` in `src/metrics.rs` |
| `dist_sq_*` kernel microbenchmark time | ns / call | New Criterion microbench group in `benches/dist_sq_bench.rs` | NEW — not currently benchmarked |

**Notes on "NEW" metrics:**
- `TW_PARITY_THRESHOLD = 1e-6` should be added to `src/metrics.rs` before finalizing production
  changes (existing `1e-6` constants in that file are for eigensolver quality, not trustworthiness).
- `dist_sq_avx2` and `dist_sq_avx512_looped` are currently `unsafe fn` with private visibility. The
  microbenchmark requires either making them `pub(crate)` or placing the bench in the same module
  via `mod benches` inside `src/metrics.rs`.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| d_y | 2 | Matches existing benchmarks; y_dist is not being optimized |
| k (trustworthiness neighbors) | 15 | Matches existing `trustworthiness_bench.rs` |
| RNG seed | 42 | Reproducibility |
| Thread count | default Rayon (8 threads on 9800X3D) | Reflects intended production use |
| Compiler flags | `target-cpu=native` (existing `.cargo/config.toml`) | Ensures AVX-512 is available at compile time |

---

## Inputs and Data

No external fixtures are required. The existing `make_data(n, d_x, d_y, seed)` function in
`benches/trustworthiness_bench.rs` generates uniformly distributed f64 arrays of arbitrary shape.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| Synthetic 1K×50 | `make_data(1_000, 50, 2, 42)` | n=1,000, d_x=50, d_y=2 | Warm-up / regression at low n |
| Synthetic 5K×50 | `make_data(5_000, 50, 2, 42)` | n=5,000, d_x=50, d_y=2 | Scaling midpoint |
| Synthetic 10K×50 | `make_data(10_000, 50, 2, 42)` | n=10,000, d_x=50, d_y=2 | Primary MERFISH-scale target |
| Synthetic 50K×50 | `make_data(50_000, 50, 2, 42)` | n=50,000, d_x=50, d_y=2 | Scaling upper bound |
| Synthetic 1K×10 | `make_data(1_000, 10, 2, 42)` | n=1,000, d_x=10, d_y=2 | Regression: existing d_x=10 path unaffected |
| TW parity 200×50 | `scripts/gen_tw_parity_50d.py` | n=200, d_x=50, sklearn score | Correctness gate at d_x=50 |

For the correctness gate, a new fixture `data/tw_parity_50d.npz` must be generated by extending
`tests/visual_eval/generate_tw_fixture.py`. The Python environment at `envs/spectral-test/` (Python
3.11 + sklearn) is already installed.

---

## Experiment Directory Layout

```
research/2026-04-08-x-dist-simd-avx512/
├── scripts/
│   ├── gen_tw_parity_50d.py       # Generate sklearn parity fixture at d_x=50
│   ├── run_baseline.sh            # Criterion bench + profiler for baseline kernels
│   ├── run_optimized.sh           # Criterion bench + profiler after kernel changes
│   └── analyze.py                 # Parse JSON results, compute speedup table
├── data/
│   └── tw_parity_50d.npz          # sklearn parity fixture at n=200, d_x=50
├── results/
│   ├── baseline_criterion.json    # Criterion output (current kernel)
│   ├── baseline_profiler.json     # tw_profiler step timing (current kernel)
│   ├── avx2_looped_criterion.json
│   ├── avx2_looped_profiler.json
│   ├── avx512_looped_criterion.json
│   ├── avx512_looped_profiler.json
│   ├── tiled_criterion.json
│   ├── tiled_profiler.json
│   └── correctness.json           # Parity test results per kernel variant
└── report.md                      # Final report (written by /write-report)
```

**Script descriptions:**

- `gen_tw_parity_50d.py` — Extends `tests/visual_eval/generate_tw_fixture.py` to output
  `n=200, d_x=50` data and sklearn trustworthiness score to `data/tw_parity_50d.npz`.
- `run_baseline.sh` — Runs `cargo bench --bench trustworthiness_bench` with the current kernel;
  captures Criterion JSON to `results/baseline_criterion.json`. Also runs `tw_profiler` at n=10K,
  d_x=50 (once the bench is extended) and saves to `results/baseline_profiler.json`.
- `run_optimized.sh` — Same commands, parameterised by kernel variant name (passed as `$1`);
  saves to `results/${1}_criterion.json` and `results/${1}_profiler.json`.
- `analyze.py` — Reads all result JSONs, computes per-variant speedup ratios (wall-clock and
  x_dist step), outputs Markdown table.

---

## Environment

**No custom environment needed.**

The experiment runs entirely within the existing Rust toolchain. Specifics:
- Criterion is already a dev-dependency (`criterion = "0.5"` in `Cargo.toml`).
- `.cargo/config.toml` sets `rustflags = ["-C", "target-cpu=native"]`, which exposes `avx512f` at
  compile time on the Ryzen 9800X3D — no `RUSTFLAGS` override needed.
- AVX-512 intrinsics (`_mm512_loadu_pd`, `_mm512_fmadd_pd`, `_mm512_reduce_add_pd`) are in
  `core::arch::x86_64` (stable Rust, no new crate dependencies).
- `profiling` feature and `tw_profiler` binary are already registered in `Cargo.toml`.
- Python is only needed for `gen_tw_parity_50d.py` (correctness fixture); `envs/spectral-test/`
  (Python 3.11 + sklearn) is already installed in-repo.

No `environment.yml` will be created.

---

## Implementation Phases

### Phase 1: Directory Structure and Benchmark Extension

**Goal:** Create the experiment folder and extend `trustworthiness_bench.rs` to cover d_x=50,
so baseline and optimized measurements are apples-to-apples.

Files to create:
- `research/2026-04-08-x-dist-simd-avx512/` and all subdirectories
- `research/2026-04-08-x-dist-simd-avx512/scripts/gen_tw_parity_50d.py`
- `research/2026-04-08-x-dist-simd-avx512/scripts/run_baseline.sh`
- `research/2026-04-08-x-dist-simd-avx512/scripts/run_optimized.sh`
- `research/2026-04-08-x-dist-simd-avx512/scripts/analyze.py`

Files to modify:
- `benches/trustworthiness_bench.rs` — add a new `trustworthiness_d50` benchmark group calling
  `make_data(n, 50, 2, 42)` for `n` in `{1_000, 5_000, 10_000, 50_000}`. Keep the existing
  `trustworthiness` group (d_x=10) untouched for regression.

Commands to verify Phase 1 is functional:
```bash
cargo bench --bench trustworthiness_bench --no-run --features testing
```

### Phase 2: Fixture Generation and Correctness Gate

**Goal:** Generate `data/tw_parity_50d.npz` and verify the *current* kernel passes the correctness
gate at d_x=50 (establishes the correctness baseline before any kernel changes).

Steps:
1. Run `scripts/gen_tw_parity_50d.py` (activate `envs/spectral-test/` first):
   ```bash
   source envs/spectral-test/bin/activate
   python research/2026-04-08-x-dist-simd-avx512/scripts/gen_tw_parity_50d.py
   ```
2. Copy the generated fixture to `tests/fixtures/tw_parity/tw_parity_50d.npz`.
3. Add `#[ignore]` test `sklearn_parity_50d` to `tests/integration/test_trustworthiness.rs` that
   loads `tw_parity_50d.npz` and asserts `|rust - sklearn| < 1e-6`.
4. Run the test to confirm baseline passes:
   ```bash
   cargo test --features testing -- --ignored sklearn_parity_50d
   ```

### Phase 3: Baseline Measurement

**Goal:** Capture wall-clock and step-timing numbers for the current (broken) kernel at d_x=50.

Steps:
1. Run Criterion baseline and save JSON:
   ```bash
   cargo bench --bench trustworthiness_bench --features testing \
     -- trustworthiness_d50 2>&1 | tee research/2026-04-08-x-dist-simd-avx512/results/baseline_criterion.json
   ```
   Or use `cargo bench ... -- --save-baseline baseline` and read from `target/criterion/`.
2. Run `tw_profiler` at n=10,000 (requires extending `tw_profiler` to accept n and d_x as CLI args,
   or hard-coding a MERFISH-scale run):
   ```bash
   cargo run --bin tw_profiler --features "cli profiling" -- \
     --n 10000 --d-x 50 --d-y 2 --k 15 --warmup 3 --iters 10 \
     > research/2026-04-08-x-dist-simd-avx512/results/baseline_profiler.json
   ```
3. Record: total wall-clock time (Criterion median), x_dist step ns (profiler mean), x_dist fraction.

### Phase 4: Looped AVX2 Kernel Implementation

**Goal:** Replace the two-fixed-load `dist_sq_avx2` in `src/metrics.rs` with a properly looped
4-wide YMM implementation that covers all d_x elements.

Changes to `src/metrics.rs`:
- Rename current `dist_sq_avx2` to `dist_sq_avx2_looped` (or add new function and update dispatch).
- Loop structure:
  ```rust
  #[target_feature(enable = "avx2,fma")]
  unsafe fn dist_sq_avx2_looped(xi: &[f64], xj: &[f64]) -> f64 {
      let n = xi.len();
      let mut acc = _mm256_setzero_pd();
      let mut k = 0usize;
      while k + 4 <= n {
          let a = _mm256_loadu_pd(xi.as_ptr().add(k));
          let b = _mm256_loadu_pd(xj.as_ptr().add(k));
          let d = _mm256_sub_pd(a, b);
          acc = _mm256_fmadd_pd(d, d, acc);
          k += 4;
      }
      // Reduce 256-bit to scalar
      let lo = _mm256_castpd256_pd128(acc);
      let hi = _mm256_extractf128_pd(acc, 1);
      let sum128 = _mm_add_pd(lo, hi);
      let sum64 = _mm_hadd_pd(sum128, sum128);
      let mut result = _mm_cvtsd_f64(sum64);
      // Scalar tail (0–3 elements)
      while k < n { result += (xi[k] - xj[k]).powi(2); k += 1; }
      result
  }
  ```
- Update the dispatch guard: replace existing `dist_sq_avx2` call site with `dist_sq_avx2_looped`.
- Add kernel microbenchmarks: either add `benches/dist_sq_bench.rs` or expose kernels as `pub(crate)`
  for cross-module bench access. The bench should time `dist_sq_avx2_looped(xi, xj)` in a loop at
  d_x=50 with 1,000 iterations (Criterion).

Verify:
```bash
cargo test --features testing -- --ignored sklearn_parity_50d
cargo test --features testing
```

### Phase 5: AVX-512 Kernel Implementation

**Goal:** Add `dist_sq_avx512_looped` (8-wide ZMM) and wire it ahead of `dist_sq_avx2_looped` in
the runtime dispatch chain.

Changes to `src/metrics.rs`:
- Add `static AVX512_AVAILABLE: OnceLock<bool>` initialized by `is_x86_feature_detected!("avx512f")`.
- Add `dist_sq_avx512_looped`:
  ```rust
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[target_feature(enable = "avx512f")]
  unsafe fn dist_sq_avx512_looped(xi: &[f64], xj: &[f64]) -> f64 {
      let n = xi.len();
      let mut acc = _mm512_setzero_pd();
      let mut k = 0usize;
      while k + 8 <= n {
          let a = _mm512_loadu_pd(xi.as_ptr().add(k));
          let b = _mm512_loadu_pd(xj.as_ptr().add(k));
          let d = _mm512_sub_pd(a, b);
          acc = _mm512_fmadd_pd(d, d, acc);
          k += 8;
      }
      let sum512 = _mm512_reduce_add_pd(acc);
      let mut result = sum512;
      while k < n { result += (xi[k] - xj[k]).powi(2); k += 1; }
      result
  }
  ```
- Update dispatch: `avx512f (runtime) → avx2+fma (runtime) → scalar`.
- Add `dist_sq_avx512_looped` to the microbenchmark group.

Verify:
```bash
cargo test --features testing -- --ignored sklearn_parity_50d
```

### Phase 6: Measurement of SIMD Kernels

**Goal:** Capture wall-clock and step-timing for `avx2_looped` and `avx512_looped` variants.
Run `scripts/run_optimized.sh avx2_looped` and `scripts/run_optimized.sh avx512_looped`.

Outputs:
- `results/avx2_looped_criterion.json`
- `results/avx2_looped_profiler.json`
- `results/avx512_looped_criterion.json`
- `results/avx512_looped_profiler.json`

If the looped AVX2 already satisfies the ≥1.5× total speedup criterion, AVX-512 measurement is
still valuable for documentation but not required for a "pass" verdict.

### Phase 7: Cache Tiling (Conditional)

**Goal:** Test H3 — whether L2 tile-blocking of the x_dist inner loop provides measurable
improvement on the 96 MB V-Cache machine.

**Trigger condition:** Only execute Phase 7 if the SIMD kernels alone do NOT achieve ≥1.5× total
speedup (i.e., if x_dist speedup is limited by memory latency rather than arithmetic throughput).

Change to `src/metrics.rs` x_dist inner loop:
```rust
const TILE: usize = 128; // 128 rows × 50 dims × 8 bytes = 51.2 KB, fits in L2
let n = dist_x.len();
let mut j_base = 0;
while j_base < n {
    let j_end = (j_base + TILE).min(n);
    for j in j_base..j_end {
        dist_x[j] = dist_sq(xi, xj_row(j));  // dispatch as before
    }
    j_base = j_end;
}
```
Add `.with_min_len(TILE)` to the outer Rayon `par_iter()` to prevent work-stealing below tile
granularity.

Outputs:
- `results/tiled_criterion.json`
- `results/tiled_profiler.json`

Test all three tile sizes (64, 128, 256) and report in `results/`.

---

## Execution Protocol

After all implementation phases are complete, run the full experiment in order:

```bash
# 1. Generate parity fixture
source envs/spectral-test/bin/activate
python research/2026-04-08-x-dist-simd-avx512/scripts/gen_tw_parity_50d.py

# 2. Baseline (current kernel)
# Revert src/metrics.rs to the original two-fixed-load dist_sq_avx2
# OR use git stash / branch to isolate states
bash research/2026-04-08-x-dist-simd-avx512/scripts/run_baseline.sh

# 3. Correctness (baseline)
cargo test --features testing -- --ignored sklearn_parity_50d 2>&1 \
  >> research/2026-04-08-x-dist-simd-avx512/results/correctness.json

# 4. Apply avx2_looped changes
bash research/2026-04-08-x-dist-simd-avx512/scripts/run_optimized.sh avx2_looped
cargo test --features testing -- --ignored sklearn_parity_50d 2>&1 \
  >> research/2026-04-08-x-dist-simd-avx512/results/correctness.json

# 5. Apply avx512_looped changes
bash research/2026-04-08-x-dist-simd-avx512/scripts/run_optimized.sh avx512_looped
cargo test --features testing -- --ignored sklearn_parity_50d 2>&1 \
  >> research/2026-04-08-x-dist-simd-avx512/results/correctness.json

# 6. (If needed) Apply tiling
bash research/2026-04-08-x-dist-simd-avx512/scripts/run_optimized.sh avx512_tiled

# 7. Analyze
python research/2026-04-08-x-dist-simd-avx512/scripts/analyze.py \
  research/2026-04-08-x-dist-simd-avx512/results/ \
  > research/2026-04-08-x-dist-simd-avx512/results/summary.md
```

**Branch strategy:** Use a feature branch (`exp/x-dist-simd-avx512`) to isolate `src/metrics.rs`
changes during the experiment. Results are committed to the `research/` folder on `main` regardless
of whether the optimization is shipped.

---

## Analysis Plan

For each kernel variant, compute:

1. **x_dist speedup** = `baseline_profiler.x_dist_mean_ns / variant_profiler.x_dist_mean_ns`
2. **Total wall-clock speedup** = `baseline_criterion.median_ms / variant_criterion.median_ms` at n=10,000, d_x=50
3. **Amdahl projection check**: If x_dist = 58.9% of runtime and measured x_dist speedup = S,
   predicted total speedup = `1 / (0.411 + 0.589 / S)`. Compare predicted vs. measured total
   speedup to validate the step-timing fraction.
4. **AVX-512 marginal gain** = `avx2_looped_median / avx512_looped_median`
5. **Tiling marginal gain** = `avx512_looped_median / avx512_tiled_median` (if Phase 7 executed)
6. **Correctness** = `|rust_score - sklearn_score|` for each variant (must be < 1e-6)

Statistical note: Criterion provides median and 95% CI. Use median for speedup ratios. Report
whether 95% CIs overlap to assess whether differences are within noise.

---

## Success Criteria

- **Conclusive positive (H1 supported):** Criterion benchmark shows ≥1.5× speedup in total
  trustworthiness time at n=10,000, d_x=50 for `avx2_looped` or `avx512_looped` vs. current
  kernel, with non-overlapping 95% CIs. All correctness gates pass (`|Δ| < 1e-6`). Ship the
  kernel to production.

- **Conclusive negative (H0 supported):** Speedup < 1.5× even with `avx512_looped`, despite x_dist
  step speedup ≥4×. This would indicate another step has risen to become the new bottleneck
  (x_sort, y_dist, or penalty). No kernel change to ship; investigate next bottleneck.

- **H2 verdict (AVX-512 marginal gain):** AVX-512 marginal gain ≥1.2× over looped AVX2 → ship
  AVX-512 as the primary dispatch path. Gain < 1.05× → ship AVX2 only (simpler, portable, no
  AVX-512 dependency).

- **H3 verdict (tiling marginal):** Tiling gain < 5% → skip tiling in production (V-Cache
  hypothesis confirmed). Gain > 10% → add tiling with `TILE=128` as a quality improvement.

- **Inconclusive:** If CPU frequency scaling or thermal throttling causes high variance in Criterion
  results (CV > 15%), the experiment is inconclusive. Mitigate by pinning thread affinity and
  running at fixed governor frequency (`sudo cpupower frequency-set --governor performance`).

---

## Threats to Validity

### Internal

- **Micro-benchmark warmup state:** The Criterion benchmark runs many iterations; the first few may
  warm the cache in a way that doesn't reflect real-world use. Mitigate by using Criterion's
  built-in warmup and the `--warm-up-time` parameter.
- **CPU frequency scaling:** AMD Zen 5 boosts aggressively. If the CPU clocks differently across
  variants (e.g., due to thermal state), speedup ratios are confounded. Mitigate by running
  baseline and variants in the same session with consistent load.
- **Rayon work-stealing interference:** Thread migration can perturb L1/L2 residency between Rayon
  tasks. This affects cache tiling results more than pure SIMD throughput. Use `.with_min_len(TILE)`
  to mitigate.
- **Compiler auto-vectorization of scalar tail:** `rustc` with `target-cpu=native` may already
  auto-vectorize the scalar tail loop in the current `dist_sq_avx2`. If so, the baseline is not
  "pure scalar" — the measured speedup of the looped kernel will be smaller than the theoretical
  maximum. This must be checked by inspecting generated assembly (`cargo asm` or `objdump`).
- **AVX-512 frequency downclocking on Intel CPUs:** Not applicable here (AMD Zen 5 does not throttle
  for AVX-512), but relevant context if results are ever replicated on Intel hardware.

### External

- **Hardware specificity:** The 96 MB 3D V-Cache is not present on most x86-64 servers or cloud
  instances. Cache-tiling conclusions are specific to this machine. On a typical server CPU (8–32 MB
  L3), tiling at n=10K/50K would show larger improvement.
- **d_x generalizability:** Results are measured at d_x=50. The looped AVX2 kernel is beneficial for
  any d_x ≥ 8; at d_x ≤ 8, the current two-fixed-load kernel is already full-SIMD. The dispatch
  threshold `d_x >= 10` should be reviewed (lowering it to `d_x >= 4` is safe post-optimization).
- **Rayon thread count:** Results measured on 8 threads. At higher thread counts (e.g., cloud
  instances), L2 pressure increases and tiling benefits may be larger.

---

## Estimated Resource Requirements

| Resource | Estimate |
|----------|----------|
| Wall-clock time (all measurements) | ~45–90 minutes (Criterion at n=50K is slow) |
| Disk space | < 50 MB (Criterion HTML reports, JSON results) |
| RAM | < 4 GB (n=50K, d_x=50, f64: ~20 MB per matrix) |
| New Rust dependencies | None |
| Python (fixture generation only) | < 5 seconds; `envs/spectral-test/` already present |
| Git branches | 1 feature branch (`exp/x-dist-simd-avx512`) for `src/metrics.rs` changes |
