# Scope Report: X-space Distance Kernel Optimization (AVX-512, Cache Tiling, Blocked Computation)

## Research Question

Can x_dist throughput be improved 2× or more via wider SIMD (AVX-512), cache-aware tiling, or blocked
computation — while remaining exact and O(n²)? The primary target is MERFISH 10K (n=10,000, d_x=50)
where x_dist currently consumes 58.9% of total trustworthiness runtime.

---

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| SIMD coverage | Current `dist_sq_avx2` covers only elements 0–7 via 2 fixed loads; elements 8–49 fall to scalar (84% of work at d_x=50) | Exact speedup from fixing the loop structure (empirical measurement needed) |
| AVX-512 availability | Confirmed: `avx512f`, `avx512dq`, `avx512bw`, `avx512vl` all present on AMD Ryzen 7 9800X3D (Zen 5) in WSL2 | Whether compiler emits optimal code with `target_feature(enable = "avx512f")` on Zen 5 |
| Cache hierarchy | L1d = 48 KB, L2 = 1 MB (per core), L3 = 96 MB (3D V-Cache) | Per-core LLC miss rate under the current Rayon workload |
| 3D V-Cache impact | MERFISH 10K X matrix = 3.9 MB (fits in V-Cache); MERFISH 50K X matrix = 20 MB (fits in V-Cache) | Whether the V-Cache eliminates the LLC contention problem described in the issue |
| ComputeMode | `metrics.rs` has no `ComputeMode` parameter — SIMD dispatch is purely compile-time cfg + runtime `is_x86_feature_detected!` | N/A — the issue's ComputeMode constraint does not apply to this code path |
| Thread-local buffer layout | Each Rayon thread holds a single `Vec<f64>` of length n for `dist_x`; it is overwritten per row — no n×n matrix is stored | — |
| Symmetry exploitation feasibility | Would require O(n²) memory for a full distance matrix (800 MB at n=10K, 20 GB at n=50K) — infeasible | Whether a streaming write-back to both positions in a temporary row-pair buffer is practical |
| Criterion benchmark baseline | `trustworthiness_bench.rs` measures n=1k/5k/50k at d_x=10 only — not d_x=50 | Baseline wall-clock time for MERFISH 10K (d_x=50) in Criterion |
| Existing MERFISH fixtures | **Correction:** This scope report originally stated "All four files present and correct sizes," but subsequent feasibility analysis found no MERFISH fixture files in the repository. All benchmarks were executed on synthetic `make_data(n, d_x=50, d_y=2, seed=42)` data. The original claim was incorrect. | — |

---

## Prior Art in Codebase

### `dist_sq_avx2` — Current SIMD Distance Kernel (`src/metrics.rs:408–437`)

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2(xi: &[f64], xj: &[f64]) -> f64
```

**Structure:** Exactly two `_mm256_loadu_pd` loads cover elements 0–3 (`a0/b0`) and 4–7 (`a1/b1`).
FMA accumulation into a 256-bit register. 128-bit split reduction + `_mm_hadd_pd` + scalar extract.
Scalar tail loop `for i in 8..n` handles all remaining elements.

**Impact at d_x=50:** 8 of 50 elements (16%) use SIMD; 42 of 50 (84%) fall through to scalar.
The kernel was designed for d_x ~ 8–10 (synthetic benchmark dimensions) and is structurally mismatched
to the MERFISH workload.

**Call site guard:** Dispatched only when `d_x >= 10` and `use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.

### `dist_sq_2d_avx2_batch` — Y-space 2D Batch Kernel (`src/metrics.rs:449–483`)

A separate AVX2 batch kernel for d_y=2 embeddings. Broadcasts query point into a 256-bit register,
processes two target points per SIMD iteration via `_mm256_hadd_pd`. This demonstrates the "proper loop"
pattern already exists in the codebase for Y-space.

### AVX-512 Bench Kernel (`benches/simd_spmv_exp.rs:235–296`)

`spmv_avx512_gather` uses `_mm512_i32gather_pd` (8-wide scatter-gather) for SpMV. This is the only
AVX-512 code in the repository and is **bench-only** — not in production `src/`. It establishes that
the project is comfortable with AVX-512 intrinsics and that the pattern of bench-first validation before
shipping to production is established.

### Step-Profiler Infrastructure (`src/metrics.rs`, feature = "profiling")

Four `AtomicU64` accumulators (`X_DIST_NS`, `X_SORT_NS`, `Y_DIST_NS`, `PENALTY_NS`), each step
bracketed with `Instant::now()`. The `tw_profiler` binary (`src/bin/tw_profiler.rs`) wraps this into
a JSON harness with warmup+iteration control and `--stderr-capture` for parsing timing output.

### Existing Research Establishing x_dist Dominance

- **`research/2026-04-08-tw-merfish-step-timing/README.md`**: Confirms x_dist = 58.9% of total runtime
  on MERFISH 10K (d_x=50). Explicitly recommends "SIMD-optimized high-dimensional distance kernels
  (AVX-512 for d=50), cache-aware tiling, or blocked distance computation" as next steps.
- **`research/2026-04-06-y-heap-bottleneck-optimization/`**: Prior optimization shipped the
  `dist_sq_2d_avx2_batch` Y-space kernel (introselect + 2D AVX2 batch), achieving ~2× total speedup
  at n=10K. The x_dist problem is the direct analog for X-space.
- **`research/2026-04-07-kdtree-y-knn-trustworthiness/`**: KD-tree for Y-space is 36–44% slower —
  establishes that algorithmic alternatives to brute-force for Y-space are exhausted.

### Criterion Benchmark Coverage Gap

`benches/trustworthiness_bench.rs` measures n ∈ {1K, 5K, 50K} with **d_x=10 only**. There is no
benchmark for d_x=50 or MERFISH data. Adding MERFISH fixtures to the benchmark is a prerequisite for
measuring the optimization.

---

## External Research

### AVX-512 f64 in Stable Rust

- `_mm512_loadu_pd` and `_mm512_fmadd_pd` are available on stable Rust via `core::arch::x86_64`.
  Both require `#[target_feature(enable = "avx512f")]` or compile-time `avx512f` feature.
- Runtime detection: `is_x86_feature_detected!("avx512f")` — same pattern as the existing AVX2
  dispatch. Caching in `std::sync::OnceLock<bool>` is recommended for hot paths.
- `std::simd` (portable_simd) remains **nightly-only** as of early 2026, with no committed stable
  date. The 2025 H1 Rust Project Goal for SIMD multiversioning is in design phase, not yet in nightly.
- For stable Rust targeting AVX-512 f64: raw intrinsics via `core::arch::x86_64` is the recommended
  approach for this project (already using this pattern for AVX2).
- The `simsimd` crate provides battle-tested AVX-512 distance kernels for f64 as a C-backed library,
  but adds an external dependency that may not be warranted given the project already uses raw
  intrinsics and the task is straightforward.

### Cache Tiling

Standard L2 tile-blocking for pairwise distance:
- Choose tile size `B` such that 2 × B × d_x × 8 bytes ≤ L2 capacity
- At d_x=50, f64: each row is 400 bytes. L2=1 MB → tile of ~1200 rows fits comfortably.
- Multi-level blocking: register tile (8×8 for AVX-512), L1 tile, L2 tile.
- **Key caveat for this machine**: The 96 MB L3 3D V-Cache is specifically designed for high
  bandwidth random-access workloads. MERFISH 10K X matrix (3.9 MB) fits entirely in L3. MERFISH 50K
  (20 MB) also fits. Cache tiling improvements may be marginal compared to SIMD width fixes.
- Rayon interaction: work-stealing can migrate tasks between cores, invalidating L1/L2. Use
  `.with_min_len(TILE)` to prevent stealing below tile granularity.

### Triangular Rayon Parallelism

- Standard approach: enumerate n*(n-1)/2 pairs as a flat linear index `k`, map to `(i,j)` via
  O(1) inverse formula. Balances Rayon work perfectly.
- Writing both symmetric positions requires `unsafe` raw pointer writes to a flat `Vec<f64>`.
  The invariant (unique `(i,j)` pairs per task) makes this sound.
- **Memory infeasibility**: For n=10K: n×n×8 bytes = 800 MB. For n=50K: 20 GB. A full precomputed
  distance matrix is impractical. Symmetry exploitation in a streaming (per-row) model is complex
  and requires a different algorithmic approach than the current design.

Sources: Rust std docs for `_mm512_*`, Shnatsel "State of SIMD in Rust 2025", rust-project-goals
2025h1, TVM block matmul docs, gendignoux.com Rayon optimization blog, SimSIMD GitHub.

---

## Technical Context

### Data Flow in `trustworthiness()`

```
(0..n).into_par_iter()              ← Rayon outer parallel loop
  |
  ├── COMB_DIST_X: RefCell<Vec<f64>>   thread-local, length n, reused per row i
  ├── COMB_INDICES: RefCell<Vec<usize>>
  ├── COMB_DIST_Y: RefCell<Vec<f64>>
  └── COMB_INDICES_Y: RefCell<Vec<usize>>

Per row i:
  Step 1 x_dist:   dist_x[j] = dist(xi, xj) for j in 0..n     ← bottleneck
  Step 2 x_sort:   introselect on dist_x → k nearest X neighbors
  Step 3 y_dist:   dist_y[j] = dist(yi, yj) for j in 0..n
  Step 4 y_sort:   introselect on dist_y → k nearest Y neighbors
  Step 5 penalty:  count Y-neighbors not in X-neighborhood, weight by rank
```

**Critical observation**: `dist_x` is a 1D `Vec<f64>` of length n, **not** a 2D n×n matrix.
Each row i writes its own buffer, which is discarded after use. There is no shared distance matrix.
This means symmetry exploitation (computing dist(i,j) once for use in both row i and row j) requires
a fundamental restructuring — precomputing an n×n matrix before the Rayon loop, which is memory-
prohibitive at scale.

### SIMD Dispatch Architecture

```
compile-time: #[cfg(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")]
  + runtime: is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
  + d_x >= 10
    → dist_sq_avx2 (2 fixed loads + scalar tail)
else
    → scalar fallback
```

**No `ComputeMode` involvement**: The `trustworthiness` function signature is `fn trustworthiness(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64`. It takes no `ComputeMode` parameter. The issue description's constraint about `ComputeMode::RustNative` / `PythonCompat` does not apply here — SIMD dispatch is governed only by CPU feature flags. New kernels should follow the same runtime-detection dispatch pattern.

### Key Performance Numbers (from prior research)

| Metric | Value |
|--------|-------|
| x_dist share of runtime (MERFISH 10K, d_x=50) | 58.9% |
| x_dist share of runtime (synthetic, d_x=10) | 33.5% |
| Current SIMD coverage at d_x=50 | 8/50 elements = 16% |
| Current scalar tail at d_x=50 | 42/50 elements = 84% |

### Hardware Context (Benchmark Machine)

| Property | Value |
|----------|-------|
| CPU | AMD Ryzen 7 9800X3D (Zen 5, 8C/16T) |
| AVX-512 | avx512f, dq, bw, vl, ifma, cd, vbmi, vbmi2, vnni, bf16, bitalg, vpopcntdq, vp2intersect |
| L1d | 48 KB |
| L2 | 1 MB per core |
| L3 | 96 MB (3D V-Cache) |
| MERFISH 10K X matrix | 3.9 MB — fits in L3 V-Cache |
| MERFISH 50K X matrix | 20 MB — fits in L3 V-Cache |

**The 3D V-Cache materially changes the cache analysis.** The issue description assumes L3 spilling
at n=10K–50K, but the 96 MB V-Cache contains the entire MERFISH dataset. LLC miss-rate driven tiling
may yield less improvement than expected on this specific hardware.

---

## Hypotheses

**H1 (High confidence): Looped AVX2 kernel provides substantial speedup**
> A `dist_sq_avx2_looped` variant that processes all d_x elements in 4-wide chunks (with a ≤3-element
> scalar tail) will reduce x_dist time by ~4–6× compared to the current 2-fixed-load implementation
> at d_x=50. This follows from: (a) 84% of elements currently fall to scalar, (b) AVX2 FMA throughput
> is ~4–8× scalar f64 throughput on Zen 5.

**H2 (High confidence): AVX-512 adds an additional ~1.5–2× over looped AVX2 for d_x=50**
> At d_x=50, AVX-512 (8-wide, ZMM registers) needs 6 full loads + 2-element scalar tail vs AVX2's
> 12 loads + 2-element tail. The additional benefit is: (a) halved load count, (b) ZMM register
> pressure reduction. On Zen 5 (which has first-class AVX-512 support unlike early Intel implementations
> that incur frequency throttling), the speedup should be consistent.

**H3 (Moderate confidence): Cache tiling yields marginal improvement on this specific hardware**
> Given the 96 MB 3D V-Cache on the Ryzen 9800X3D, the entire MERFISH X matrix fits in the effective
> LLC. Cache miss-driven tiling improvements, while real on typical hardware (256 KB L2, 8 MB L3),
> may be below measurement noise on this machine. If confirmed, tiling is a "correctness-preserving
> quality-of-life improvement" but not the primary speedup source.

**H4 (Low confidence): Symmetry exploitation is impractical for this algorithm**
> The current per-row streaming design requires O(n) memory per row, not O(n²) total. Exploiting
> dist(i,j)==dist(j,i) requires either a full n×n matrix (800 MB at n=10K — impractical) or a
> complex block-triangular restructuring with unsafe concurrent writes. The FLOP savings (50%) do
> not justify the memory and complexity cost.

**H5 (High confidence): Combined looped AVX2 + AVX-512 (fallback chain) will meet the ≥1.5× total
trustworthiness improvement target on MERFISH 10K**
> If x_dist is 58.9% of runtime and a looped SIMD kernel achieves 5× speedup on x_dist, the total
> speedup is: 1 / (1 - 0.589 + 0.589/5) ≈ 1 / (0.411 + 0.118) = 1.89× total. This exceeds the 1.5×
> target even with conservative assumptions about SIMD speedup.

---

## Proposed Investigation Directions

### Direction 1: Looped SIMD Kernel (AVX2 + AVX-512 fallback chain) — **Recommended Primary**

**Scope:** Replace `dist_sq_avx2` with a properly looped implementation:
1. `dist_sq_avx512_looped` (if avx512f): 8-wide ZMM loop, ≤7-element scalar tail
2. `dist_sq_avx2_looped` (always): 4-wide YMM loop, ≤3-element scalar tail
3. Runtime dispatch: `avx512f → avx2_looped → scalar`

**Measurement:** Criterion micro-benchmark for `dist_sq_*` in isolation at d_x=50 (not just d_x=10).
Then full trustworthiness benchmark on MERFISH 10K and 50K fixtures.

**Trade-offs:**
- Low code complexity (incremental change to existing kernel)
- No memory footprint change
- No algorithmic restructuring
- Immediate 5–10× speedup on x_dist is plausible
- AVX-512 may be overkill if looped AVX2 already satisfies the ≥1.5× total target

**Implementation sketch:**
```rust
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
    // reduce ZMM to scalar (sum 8 f64 lanes)
    let sum512 = _mm512_reduce_add_pd(acc);
    // scalar tail
    let mut tail = sum512;
    while k < n { tail += (xi[k] - xj[k]).powi(2); k += 1; }
    tail
}
```

### Direction 2: Cache Tiling of the X-Matrix Inner Loop

**Scope:** Block the `for j in 0..n` loop into tiles of size `TILE` (start with 64 or 128 rows).
Process all `j` in tile `[j_base..j_base+TILE]` before moving to the next tile. The tiled X data
stays L2-resident across the d_x inner loop.

**Measurement:** Use `perf stat -e cache-references,cache-misses` before and after. Or use the
step profiler (feature = "profiling") to isolate x_dist step time.

**Trade-offs:**
- L2 benefit likely small on this machine (96 MB V-Cache serves as effective large L3)
- Does not reduce FLOP count
- Adds ~10 lines of loop restructuring, minimal complexity
- Should be combined with Direction 1 (tiling alone without SIMD fix leaves most FLOPs as scalar)
- **Low ROI on the Ryzen 9800X3D specifically; may matter on cloud/server hardware**

**Tile size formula:** `TILE * d_x * 8 bytes ≤ L2 / 4 = 256 KB` → at d_x=50: `TILE ≤ 655` rows.
A tile of 128–256 rows is a good starting point.

### Direction 3: Block-Triangular Computation (Symmetry + Tiling)

**Scope:** Restructure the outer Rayon loop from row-parallel to block-row-parallel. For each
i-block, compute both the lower triangle (j < i_base) and the diagonal block (j in i-block).
Store a local distance matrix block and scatter to the appropriate rows via `unsafe` pointer writes.

**Trade-offs:**
- Reduces FLOPs by ~50%
- Requires full n×n distance matrix in memory (impractical at n≥10K due to memory cost)
- OR a two-pass approach: (1) compute triangle, (2) copy to transpose position — but this requires
  writing to rows owned by other threads (requires `unsafe` or a mutex per row)
- High implementation complexity
- For the step-local use case in `trustworthiness`, the matrix is not stored — symmetry would
  require precomputing and caching the entire distance matrix before the penalty-computation pass
- **Not recommended** without a memory feasibility analysis confirming n×n fits in available RAM

---

## Success Criteria

1. **Primary:** Criterion benchmark showing ≥1.5× improvement in total `trustworthiness` time on
   MERFISH 10K (d_x=50) vs baseline. Baseline must be established with d_x=50 fixtures (currently
   missing from `trustworthiness_bench.rs`).
   
2. **Alternative:** Step-profiler (`tw_profiler` or `feature = "profiling"`) showing ≥2× improvement
   in `x_dist` step time on MERFISH 10K, if total improvement is below 1.5× due to other bottlenecks
   rising to fill the gap.

3. **Correctness gate:** `|rust_score − sklearn_score| < 1e-6` (existing sklearn parity test in
   `tests/integration/test_trustworthiness.rs`). The bit-exact trustworthiness score must not
   change beyond floating-point associativity differences (which typically contribute < 1e-12 for
   this kernel, not near the 1e-6 threshold).

4. **Coverage:** Measurements on MERFISH fixtures (primary) and synthetic d_x=10 (secondary, for
   regression).

5. **Baseline requirement:** Add MERFISH 10K (d_x=50) fixture loading to `trustworthiness_bench.rs`
   before any optimization, to establish an apples-to-apples comparison.

---

## Metric Context

The `trustworthiness` function in `src/metrics.rs` touches two of the three canonical quality dimensions:

| Metric | Dimension | Current Threshold | Location | Notes |
|--------|-----------|------------------|----------|-------|
| `trustworthiness` sklearn parity | Accuracy / Parity | `< 1e-6` | `tests/integration/test_trustworthiness.rs:39` | Primary correctness gate for any x_dist change |
| Trustworthiness Criterion benchmark | Performance | No hard threshold (measurement-only) | `benches/trustworthiness_bench.rs` | Only measures d_x=10; must be extended to d_x=50 |
| Step profiler `x_dist` step time | Performance | No hard threshold | `src/metrics.rs` (feature = "profiling") + `src/bin/tw_profiler.rs` | 58.9% share on MERFISH 10K established by issue #248 |
| `RSVD_QUALITY_THRESHOLD` | Accuracy (eigensolver) | `1e-2` | `src/metrics.rs:21` | **Not applicable** — this is for spectral init, not trustworthiness |
| `DENSE_EVD_QUALITY_THRESHOLD` | Accuracy (eigensolver) | `1e-6` | `src/metrics.rs:26` | **Not applicable** |

**Gaps in canonical metric coverage:**
- No performance pass/fail threshold for trustworthiness wall-clock time. Experiment should track
  relative improvement (speedup ratio) rather than an absolute budget.
- No canonical metric for "x_dist step fraction of total runtime" — this must be measured via the
  `profiling` feature flag and `tw_profiler` binary, not via Criterion alone.
- The `ComputeMode::RustNative` / `PythonCompat` gating present in the eigensolver pipeline does NOT
  apply to `trustworthiness` — this function has a single implementation path, and all SIMD optimizations
  are unconditionally dispatched based on CPU feature availability.
