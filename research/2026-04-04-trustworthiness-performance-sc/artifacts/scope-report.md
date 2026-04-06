# Scope Report: Trustworthiness Performance — Scaling Analysis and Optimization Evaluation

## Research Question

What optimization approaches for the `trustworthiness()` function in `src/metrics.rs` are worth fully implementing, and what are their measured speedups, implementation complexities, and scaling-law impacts at the practical target scales of 100K–1M cells?

The existing O(n²) row-parallel implementation is the dominant bottleneck beyond 100K cells: at 250K it was killed after 7+ minutes at 805% CPU. The research must experimentally characterize the per-step scaling profile, evaluate algorithmic and SIMD alternatives, and produce a ranked GO/NO-GO recommendation grounded in measured wall-clock speedups — not theoretical ones.

---

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| **Algorithm structure** | 6-step per-row: X-dist (O(n·d)), X-sort (O(n log n)), rank-scatter (O(n)), X-kNN-set (O(k)), Y-heap (O(n log k)), penalty (O(k)) | Fraction of wall-clock each step consumes at 10K/50K/100K |
| **Asymptotic complexity** | O(n² log n) total; O(n) peak memory per thread via streaming heap | Whether sort or distance computation dominates in practice |
| **Parallelism** | Rayon `into_par_iter` over rows; confirmed 805% CPU utilization at 250K | Scaling efficiency (strong/weak scaling) — whether Amdahl limits are hit |
| **Memory allocation** | 3 fresh Vec allocations per row (`dist_x`, `rank_x`, `knn_x_set`); no thread-local reuse | Allocation overhead fraction of total wall-clock |
| **SIMD status** | No SIMD on distance kernels; LLVM auto-vectorization with `-C target-cpu=x86-64-v3` (AVX2+FMA) is possible | Whether auto-vectorization is actually triggered; measured AVX2 vs scalar speedup at d=10 |
| **AVX-512 infrastructure** | Bench-only AVX-512 kernel exists in `benches/simd_spmv_exp.rs` for SpMV (8-wide gather); CI targets x86-64-v3 (AVX2 baseline) | Whether AVX-512 provides meaningful gains for d=10 f64 distance (register utilization unknown) |
| **Benchmarks** | No Criterion benchmark for trustworthiness at any scale; no wall-clock timing in `src/metrics.rs` | Wall-clock at n=1K, 5K, 10K, 50K, 100K — none have been measured |
| **sklearn parity** | Parity threshold: absolute deviation `< 1e-6`; verified via `tests/integration/test_trustworthiness.rs` (n=200, k=15 fixture) | Whether parity holds after any approximation; effect of subsampling on structured (non-Gaussian) data |
| **X k-NN reuse** | UMAP graph (`mapper.graph_`) and trustworthiness are entirely separate pipelines; no shared precomputed state | Whether UMAP adjacency structure is accurate enough for the required X-rank computation |
| **Sampling error bounds** | Row-subsampling expected deviation O(1/sqrt(m)) for Gaussian data; no published tight constant for structured manifold data | Error on MERFISH-type data (structured, high dynamic range); whether sklearn parity < 1e-6 survives subsampling |
| **k-d tree at d=10** | Curse of dimensionality typically limits k-d trees at d≥8; `kiddo` ImmutableKdTree is best available in Rust | Break-even n where k-d tree beats brute force at d=10 (uncharacterized) |
| **HNSW approximation** | HNSW gives O(log n) query; 95–99% recall@15 typical | Recall on MERFISH embeddings; impact on trustworthiness parity when Y-kNN recall < 100% |
| **Partial select for X** | `partial_sort` k=15 on n=10K ≈ 12x faster than full sort; but trustworthiness needs full ranks for Y-neighbors | Whether a rank-estimation approach (select_nth_unstable threshold + linear count) is correct and matches sklearn output |
| **Prior experiment state** | One prior scope report (2026-04-04_122311) and experiment plan exist; plan received STOP verdict from review-design on 3 critical grounds | — |

---

## Prior Art in Codebase

### Implementation (`src/metrics.rs:397-457`)
The production trustworthiness function is complete and correct. Key structural facts:
- Rayon parallel outer loop over `n` rows (no shared mutable state)
- X-space: brute-force pairwise distances → `sort_unstable_by(total_cmp)` → rank scatter → `HashSet` k-NN
- Y-space: streaming `BinaryHeap<(u64, usize)>` using `f64::to_bits()` for order-preserving comparison; capacity `k+1`; O(n log k)
- Three `Vec` allocations per row: `dist_x` (n), `rank_x` (n), `knn_x_set` (k) — no thread-local buffer reuse
- Formula: `T(k) = 1 − 2/(n·k·(2n−3k−1)) · Σᵢ Σ_{j∈U_i(k)} (r(i,j)−k)`

### CLI Binary (`src/bin/trustworthiness.rs`)
32-line wrapper: `pico_args` CLI, `ndarray-npy` `.npy` I/O, calls `spectral_init::trustworthiness`. No wall-clock output, no timing flags.

### SIMD Infrastructure (`src/operator.rs`)
Production AVX2+FMA SpMV kernel (`spmv_avx2_gather_inner`, lines 98–145): 4-wide `_mm256_i32gather_pd` + `_mm256_fmadd_pd` with horizontal reduction. Pattern is directly adaptable for the distance kernel (no gather needed at d=10 — just two `_mm256_loadu_pd` loads + `_mm256_sub_pd` + `_mm256_fmadd_pd` with 2 full 4-wide passes and a scalar tail for the 10th element). Runtime dispatch via `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.

Bench-only AVX-512 kernel (`benches/simd_spmv_exp.rs`): `_mm512_i32gather_pd` 8-wide gather for SpMV. Demonstrates the AVX-512 pattern but not applicable to the distance kernel without adaptation.

Solver timing pattern (`src/solvers/mod.rs`): `#[cfg(feature = "testing")]` guards around `Instant::now()` / `eprintln!("[timing:level_N] {}µs", ...)` at each solver level — the exact pattern needed for per-step trustworthiness timing.

### Tests
- `tests/integration/test_trustworthiness.rs`: One `#[ignore]` sklearn parity test at n=200, k=15; threshold `< 1e-6` absolute.
- 5 unit tests in `src/metrics.rs`: perfect preservation, [0,1] interval, 4-point/k=1 formula check, k-boundary.
- No performance or timing tests anywhere for trustworthiness.

### Benchmarks
No Criterion benchmark for `trustworthiness` at any scale. The `benches/` directory covers SpMV, LOBPCG, rSVD, full eigensolver pipeline — nothing for metrics.

### Prior Research
- `.autoskillit/temp/scope/scope_trustworthiness_perf_scaling_2026-04-04_122311.md`: Prior scope with 6 hypotheses (H1–H6) covering AVX2, partial select, subsampling, k-d tree, thread-local buffers. Similar coverage to this report.
- `.autoskillit/temp/plan-experiment/experiment_plan_trustworthiness_perf_scaling_2026-04-04_123526.md`: 8-phase plan including `benches/trustworthiness_bench.rs`, `src/bin/tw_step_profiler.rs`, `src/bin/tw_large_scale.rs`, Python parity script, Python analysis script. Code sketches for all optimization variants.
- `.autoskillit/temp/review-design/evaluation_dashboard_trustworthiness-perf-scaling_2026-04-04_150000.md`: **STOP verdict** with three critical findings (detail in Hypotheses section below).

### MERFISH Data
Real data arrays available in `temp/merfish_100k/` per review-design finding — the prior experiment plan did not schedule their use, which was flagged as a critical gap.

---

## External Research

### sklearn Trustworthiness (Reference Implementation)
sklearn materializes an n×n `dist_X` matrix and an n×n `inverted_index` matrix — O(n²) space. Your implementation's O(n) per-thread memory is a significant improvement. sklearn's Y-space uses `NearestNeighbors` (brute force, BALL_TREE, or KD_TREE depending on d and n). X-space is always full `np.argsort`. Parity threshold tested at `< 1e-6` absolute.

### Partial Sort
`partial_sort` crate benchmarks (n=10K, k=20): 5.36 µs vs ~65 µs for full sort — **~12x speedup**. However: (1) k=15 on n=10K is the X-sort case; (2) trustworthiness needs full ranks for all Y-neighbors, not just the top-k, so a naive `partial_sort` cannot replace the X full sort without a rank-estimation workaround. The workaround: use `select_nth_unstable` to find the k-th threshold value in X-distances, then do a linear scan to count ranks for the ≤15 Y-neighbors that exceed rank k. Note: stdlib `select_nth_unstable` had a known quadratic worst case (issue #102451), addressed by PR #107522 (Median of Medians fallback merged ~2023).

### SIMD for Distance at d=10
SimSIMD, simd-euclidean, and similar libraries all report their gains for d≥32 ("2–8x for vectors ≥ 32 elements"). At d=10 f64 with 4 doubles per AVX2 register: manual SIMD provides at most ~2x over scalar, and LLVM with `-C target-cpu=x86-64-v3` often auto-vectorizes the `sum((a-b)^2)` loop at this width already. The SIMD gain for the inner product kernel is bounded by the data width, not the implementation quality. The actual benefit must be measured — the theoretical maximum is modest.

For AVX-512 at d=10: only 2 full 8-wide passes (d=10 requires 2×8=16 slots; pad 6 with zero), with poor register fill (62.5% utilization for the first pass). In practice, AVX-512 gains at narrow d are often negative on some microarchitectures (frequency throttling, port contention) — empirical measurement is essential.

### k-NN Data Structures
- **kiddo ImmutableKdTree**: Best-in-class Rust k-d tree, const-generic d, compile-time specialization for d=10. At d≥8, k-d trees approach O(n) per query due to curse of dimensionality — break-even with brute force typically at n~10K–50K for d=10. No published benchmark at d=10 for this library.
- **HNSW (hnsw_rs, hnswlib-rs)**: O(log n) query, 95–99% recall@15 typical. Useful for approximating Y-kNN. Does NOT help with X-ranks (still need full distance scan for rank computation unless approximation is accepted).
- **TopOMetry (Python, pyNNDescent)**: Uses approximate k-NN for trustworthiness computation; 80–100% recall; trustworthiness impact on structured data uncharacterized.

### Row Subsampling
No published formal error bounds on trustworthiness row-subsampling for manifold-structured data. Standard variance argument: E[|T_sub - T_exact|] = O(1/sqrt(m)) but with unknown constant. Practically, m=1000 is cited as sufficient for n=10K on Gaussian data in UMAP literature. On MERFISH data (concentrated, structured), variance may be higher or lower — empirical measurement required. Critical: sklearn parity < 1e-6 will NOT survive subsampling; a different (approximate) parity threshold must be established.

---

## Technical Context

### Per-Row Operation Profile (Asymptotic)

| Step | Operation | Complexity | Allocations |
|------|-----------|------------|-------------|
| 1 | X pairwise distances | O(n · d_x) = O(10n) | Vec<(f64,usize)> length n |
| 2 | X full sort | O(n log n) | in-place |
| 3 | Rank scatter | O(n), random writes | Vec<usize> length n |
| 4 | X k-NN set | O(k) | HashSet<usize> cap k |
| 5 | Y streaming heap | O(n · d_y + n log k) = O(n · 2 + n log 15) | BinaryHeap cap k+1 |
| 6 | Penalty accumulation | O(k) | none |

At n=100K: Steps 1+2 dominate (O(n log n) = O(100K · 17) ≈ 1.7M ops per row × 100K rows = 170G total). Steps 3+5 are O(n) = O(100K) per row = 10G total. Steps 4+6 are negligible.

Total floating-point ops at n=100K: ~1.2×10¹² (rough; Steps 1+5 dominate FLOP count, Steps 2+3 dominate cache traffic). Memory per thread: 2 × n × 16 bytes = 2 × 100K × 16 = ~3.2 MB per row (fits in L3 but not L1/L2 — cache misses in rank scatter).

### Known Instrumentation Gap
No timing instrumentation exists in `src/metrics.rs`. Before any optimization can be validated, a `#[cfg(feature = "testing")]`-gated instrumentation layer (matching the pattern in `src/solvers/mod.rs`) must be added to measure wall-clock per step. Without this, speedup claims for individual steps are unmeasurable.

### Pipeline Execution Context
At n=100K: total pipeline ≈ 683s. Trustworthiness wall-clock contribution within that total is unknown (no prior measurement). At n=250K: 7+ minutes of continuous CPU before kill. The n=250K case is the design target — the experiment must characterize behavior at that scale, not just ≤100K.

---

## Hypotheses

The three STOP findings from the prior experiment plan's review-design are reproduced here because they constrain hypothesis testing methodology:

**STOP-1 (Cold-start allocation confound):** Single-iteration timing at large n conflates OS page-fault cost with algorithmic time. Any timing at n≥25K must use at least 3 warm iterations; std must be reported to distinguish noise from signal.

**STOP-2 (Post-hoc sampling calibration):** The sampling fraction scan over {0.05, 0.10, 0.20, 0.50} must be pre-registered (fixed before benchmarking) and validated on held-out data (MERFISH real arrays), not on the same synthetic data used for timing benchmarks.

**STOP-3 (Vacuous parity gate on Gaussian data):** At n≥10K, d=10, subsampling errors on Gaussian data are near-zero by concentration of measure — the parity gate must be evaluated on MERFISH-type structured data from `temp/merfish_100k/`.

---

**H1 — X-sort is the dominant wall-clock step, and partial-rank computation yields the largest single-step speedup.**

*Rationale:* O(n log n) sort with cache-hostile rank scatter over n=100K elements is expected to dominate. Partial-rank computation (find threshold via `select_nth_unstable`, scan ≤15 Y-neighbors) would reduce this to O(n) selection + O(15) scans — potentially 10–20x for the sort step alone.
*Falsifiable:* Step profiling at n=10K/50K/100K; if X-sort < 30% of total, this hypothesis fails.
*Risk:* Correctness of partial-rank workaround requires careful tie-handling to preserve sklearn parity < 1e-6.

**H2 — Thread-local buffer reuse provides 5–15% throughput improvement.**

*Rationale:* At n=100K with 8+ threads, the 3 fresh Vec allocations per row (2 × n-length Vecs = ~3.2 MB per row) generate ~100K × 3 = 300K allocator calls per call. Reusing `thread_local!` buffers eliminates the per-row alloc/dealloc cost.
*Falsifiable:* Criterion benchmark with and without thread-local reuse at n=5K–50K.
*Risk:* Low — purely mechanical change; zero correctness risk.

**H3 — AVX2 SIMD for the X-distance inner loop yields < 2x over the current auto-vectorized baseline at d=10.**

*Rationale:* d=10 f64 = 80 bytes = 2 full AVX2 passes (8 doubles, 8 doubles, 2 scalar tail, or rearranged as 2×4 + 2 scalar). LLVM with `-C target-cpu=x86-64-v3` very likely auto-vectorizes the `sum((a-b)^2)` loop. Manual intrinsics would produce identical instructions. If auto-vectorization is confirmed (via `cargo asm`), manual SIMD provides no additional benefit.
*Falsifiable:* Compare `cargo asm` output (auto-vectorized) vs manual intrinsics benchmark; if no instruction difference, manual SIMD is NO-GO.

**H4 — AVX-512 for the distance kernel provides marginal or negative benefit at d=10 due to poor register fill (62.5% utilization) and potential frequency throttling.**

*Rationale:* 8-wide AVX-512 at d=10 requires 2 full passes but only 10/16 = 62.5% register utilization. On many Intel microarchitectures, AVX-512 triggers core frequency reduction ("downclocking") that can negate the wider lane advantage. Creative tricks (multi-point blocking: process 2 rows of X simultaneously to fill all 8 lanes) could improve utilization to 100% — but complexity is high.
*Falsifiable:* Direct benchmark of scalar vs AVX2 vs AVX-512 (scalar, 2-point block) for the distance kernel at d=10 f64. If AVX-512 < 20% over AVX2, NO-GO.

**H5 — Row subsampling at m=5000 provides ≥5x speedup with < 0.001 absolute deviation from exact on MERFISH-type data.**

*Rationale:* T(k) is a row-mean; subsampling m rows is exact on those rows and uses a Monte Carlo estimate for the rest. At m=5000, expected error ~O(1/sqrt(5000)) ≈ 0.014 — but this is for Gaussian data. On MERFISH structured data, variance may differ. The sklearn parity < 1e-6 gate will not survive subsampling (it's ~1000x tighter than the expected error).
*Falsifiable:* Compute T(k) at m={500, 1000, 2000, 5000, 10000} vs exact on `temp/merfish_100k/` X and a known Y embedding. Report |T_sub - T_exact| at each m. Pre-register m=5000 as the proposed production value before running the benchmark.
*Risk:* Requires establishing a new approximate parity threshold; the existing `< 1e-6` test is inapplicable.

**H6 — A combined optimization (thread-local buffers + partial-rank X + auto-vectorized distance) achieves 3–6x end-to-end speedup at n=100K with no correctness change, and remains O(n²) asymptotically.**

*Rationale:* These three approaches are composable, all exact (no approximation), and each targets a different bottleneck. Their combined effect is subadditive (less than 2x × 5x × 2x = 20x due to Amdahl's law) but likely in the 3–6x range.
*Falsifiable:* Criterion benchmark at n=5K–50K and wall-clock at n=100K for the combined variant.

---

## Proposed Investigation Directions

### Direction 1 — Baseline Profiling First (Phase 0, mandatory precondition)

Before any optimization is implemented, add `#[cfg(feature = "testing")]`-gated per-step timing to `trustworthiness()`, matching the `src/solvers/mod.rs` pattern (6 `Instant`s with `eprintln!("[timing:tw_step_N] {}µs", ...)` guards). Create a standalone timing binary (no Criterion overhead) that runs warm iterations (≥3) and reports per-step wall-clock fractions at n=10K, 50K, 100K — and if possible, n=250K (with a timeout guard). Use MERFISH real data from `temp/merfish_100k/` for n=100K measurements.

**Why first:** Every optimization hypothesis (H1–H6) depends on knowing which steps dominate. Without per-step profiling, speedup claims for individual steps are unmeasurable and the ranking of optimization priorities cannot be grounded.

**Tradeoffs:** Small initial implementation cost (< 50 lines), enables all subsequent work.

### Direction 2 — Exact Optimizations (Phase 1, composable)

After profiling confirms the bottleneck distribution, implement the exact (no-approximation) optimizations in order of expected impact:

1. **Thread-local Vec reuse** (H2): Wrap `dist_x` and `rank_x` in `thread_local! { static ... }` `RefCell<Vec>`. Reuse across rows by clearing and resizing. Zero correctness risk; pure throughput gain.

2. **Partial-rank X computation** (H1): If X-sort is ≥40% of wall-clock: replace `sort_unstable_by` + full rank scatter with:
   - `select_nth_unstable_by` to partition at position k (O(n) average)
   - Linear scan of only the ≤k Y-neighbor indices to compute their ranks via partial-sort or counting
   - Correctness verification: confirm `|T_partial - T_exact| = 0` on all 5 unit tests + sklearn parity fixture at n=200

3. **Manual AVX2 distance kernel** (H3): Only implement if `cargo asm` confirms auto-vectorization is NOT occurring. The kernel is simple: 2 `_mm256_loadu_pd` loads + `_mm256_sub_pd` + `_mm256_mul_pd` + horizontal reduction; scalar tail for d%4=2. Gate behind `#[cfg(all(target_arch = "x86_64", ...))]` + runtime `is_x86_feature_detected!` dispatch, same pattern as `operator.rs`.

**Tradeoffs:** These are exact — no new parity concern. Combined, they are expected to achieve H6's 3–6x range. If X-sort < 40%, prioritize thread-local buffers only.

### Direction 3 — Approximate Optimization (Phase 2, separate validation path)

Implement row subsampling as a separate code path (not the default):

```rust
pub fn trustworthiness_approx(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize, 
                               sample: usize, seed: u64) -> f64
```

Validate against MERFISH real data from `temp/merfish_100k/` at m={500, 1000, 2000, 5000, 10000} rows. Establish an empirical approximate parity threshold (`< 0.001` is a reasonable initial target, orders of magnitude looser than `< 1e-6`). Report |T_approx - T_exact| and |T_approx - T_sklearn| separately.

Speedup: O(m·n) for sampled rows + O(n log n) sort within each sampled row → total O(m·n log n). At m=5000, n=250K: ~5000×250K = 1.25B operations vs 250K×250K = 62.5B — **50x reduction**.

**Tradeoffs:** Requires a new approximate parity gate (< 1e-6 inapplicable). Subsampling error is data-dependent; MERFISH-specific error must be empirically established. This is appropriate for the 250K+ regime where exact computation is killed.

---

## Success Criteria

The research is conclusive if it delivers:

1. **Per-step wall-clock fractions** at n=10K, 50K, 100K (and 250K if feasible) — confirming or refuting H1's claim that X-sort dominates.

2. **Measured speedup** for each exact optimization (H2 thread-local, H1 partial-rank, H3 auto-vectorize) as a Criterion ratio with confidence intervals at n=5K–50K.

3. **A GO/NO-GO decision** on AVX-512 (H4), backed by instruction-level comparison (`cargo asm`) and direct benchmark, not theoretical argument.

4. **A measured deviation table** for row subsampling (H5) on `temp/merfish_100k/` MERFISH data at pre-registered sample fractions, reporting |T_approx - T_exact| and whether it is structurally bounded.

5. **A ranked recommendation table** with: approach name, measured speedup range, scaling law change (yes/no), implementation complexity (LOC estimate), GO/NO-GO, and rationale.

---

## Metric Context

### Trustworthiness Parity (Accuracy)
- **Metric**: `|T_rust - T_sklearn|`
- **Threshold**: `< 1e-6` (absolute)
- **Defined in**: `tests/integration/test_trustworthiness.rs`
- **Scope**: Exact implementations only. Any approximate implementation (row subsampling, HNSW Y-kNN) requires a separate, explicitly documented threshold.
- **Gap**: No canonical `MetricResult` struct entry for trustworthiness parity — only an `assert!` in the integration test. The `AssessmentReport` framework (`src/metrics.rs`, `#[cfg(feature = "testing")]`) covers eigensolver metrics, not trustworthiness.

### Trustworthiness Performance (Performance)
- **Metric**: Wall-clock seconds at specified n (n=100K, n=250K)
- **Threshold**: None defined — no performance regression threshold exists in the codebase
- **Gap**: No Criterion benchmark for trustworthiness; no wall-clock assertion anywhere. A performance threshold (e.g., "n=100K must complete in ≤ T seconds on the CI hardware") must be defined as part of this research's output.

### Quality Dimension Mapping
| Optimization | Dimension | New threshold needed? |
|---|---|---|
| Thread-local buffers (exact) | Performance | No — parity preserved exactly |
| Partial-rank X (exact) | Performance + Accuracy | Verify `|delta| = 0` exactly |
| AVX2 distance kernel (exact) | Performance | No — bit-identical output |
| AVX-512 distance kernel (exact) | Performance | No — bit-identical output |
| Row subsampling (approximate) | Performance + Accuracy | YES — new approximate threshold required |
| HNSW Y-kNN (approximate) | Performance + Accuracy | YES — recall-dependent threshold required |

The `< 1e-6` sklearn parity threshold applies only to exact implementations. Approximate implementations are a new category not currently covered by any canonical metric in the codebase.
