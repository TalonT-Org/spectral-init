# Scope Report: KD-tree Y k-NN for trustworthiness at n ≥ 50K

## Research Question

Does building an `ImmutableKdTree` over the Y embedding enable O(k log n) per-row
nearest-neighbor queries that beat the O(n) brute-force scan in `trustworthiness()`
at n ≥ 50K, and at what crossover point does the tree's build cost amortize?

---

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| Current behavior | Y k-NN is O(n) per row (O(n²) total): flat Vec distance fill + `select_nth_unstable_by`; AVX2 2D batch kernel for `d_y=2` | How large the y_dist fraction of wall time grows at n=50K-100K relative to n=10K |
| Performance | `flat_simd` timings: 7.9ms (n=1K), 37.7ms (n=5K), 145ms (n=10K); older `combined` variant at n=50K ≈ 4.87s, n=100K ≈ 21s | Actual `flat_simd` wall time at n=50K (bench present but not run post-PR #237 merge); exact build + query cost for kiddo at these sizes |
| Scaling | O(n²) confirmed from three-point regression; extrapolation: n=50K ≈ 3.6s, n=100K ≈ 14.5s | Whether SIMD memory-bandwidth effects shift the O(n²) constant enough to change the crossover |
| Edge cases | Self-exclusion currently via `dist_y[i] = f64::INFINITY`; tie-breaking by `total_cmp().then(a.cmp(&b))`; correctness oracle test at `|ΔT| < 1e-12` | Whether KD-tree tie-breaking with equidistant rank-k neighbors can produce a different (but equally valid) neighbor set, and whether that changes T |
| Prior work | Scope report in `research/2026-04-06-y-heap-bottleneck-optimization/artifacts/scope-report.md` evaluated KD-tree as H3; deferred it for n≥50K follow-up; no implementation exists | Whether the earlier scope report quantified expected speedup or recommended a specific crate version |
| Dependency choice | `kiddo` v5.3.0 (2026-03-19) is the leading exact k-NN crate for Rust; `kd-tree` v0.6.2 is simpler; `bosque` limited to 3D; `fnntw` archived | kiddo ImmutableKdTree overhead for construction at n=10K (the possible regression threshold) |
| Correctness | All existing unit tests (`t_tw_08`, `t_tw_10`) require `|ΔT| < 1e-12` vs `trustworthiness_brute_force`; kiddo queries are exact (not ANN) | Whether tree-returned distances are bit-identical to scalar `(a-b)²+(c-d)²` or have rounding differences that could shift borderline k-th neighbors |

---

## Prior Art in Codebase

### Current Y k-NN implementation (production, shipped PR #237)

**File:** `src/metrics.rs`, lines 587–650

The Y-space k-NN block sits inside the Rayon `into_par_iter()` loop over rows `i`:

```rust
COMB_DIST_Y.with(|dy_cell| {
    COMB_INDICES_Y.with(|iy_cell| {
        let mut dist_y    = dy_cell.borrow_mut();
        let mut indices_y = iy_cell.borrow_mut();
        dist_y.clear(); dist_y.resize(n, 0.0f64);
        // AVX2 2D batch or scalar fill:
        unsafe { dist_sq_2d_avx2_batch(yi_slice, y_flat, n, &mut dist_y); }
        dist_y[i] = f64::INFINITY; // self-exclusion
        indices_y.clear(); indices_y.extend(0..n);
        indices_y.select_nth_unstable_by(k, |&a, &b|
            dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b)));
        // knn_y_indices = indices_y[..k]
    });
});
```

Thread-local `Vec<f64>` (COMB_DIST_Y) and `Vec<usize>` (COMB_INDICES_Y) avoid per-row heap allocation. The batch kernel `dist_sq_2d_avx2_batch` processes 2 target points per cycle via `_mm256_hadd_pd`.

### Step-level profiler data (n=10K, 30 iterations)

| Step | flat_simd (ms/iter) | % of wall time |
|------|----------------------|----------------|
| y_dist | 521.75 | **40%** |
| x_dist | 602.32 | 46% |
| x_sort | 457.75 | 35% |
| penalty | 306.86 | 23% |

Note: percentages sum >100% because they are per-step wall-clock (parallel rows overlap). At n=10K the y_dist step (the target for KD-tree) is ~40% of overall time; x_dist (brute-force, untouched) is the dominant single step.

### Criterion bench

**File:** `benches/trustworthiness_bench.rs`

Covers n ∈ {1_000, 5_000, 50_000} with `d_x=10, d_y=2, k=15`. No KD-tree variant is benchmarked. The n=50,000 test point exists and can be extended for KD-tree variant comparison.

### Correctness tests

`t_tw_08_combined_matches_baseline` (line 1178) and `t_tw_10_self_exclusion_never_in_knn` (line 1224) both assert `|ΔT| < 1e-12` vs `trustworthiness_brute_force`. These are the correctness oracle tests; any KD-tree implementation must pass them.

### Prior research scoping

The research directory `research/2026-04-06-y-heap-bottleneck-optimization/` contains:
- `artifacts/scope-report.md` — explicitly listed KD-tree as **H3 direction**, evaluated `kiddo`, `nabo`, `FNNTW`, `kd-tree`, `rstar` as candidates, noted expected 3–10× speedup at n=10K and 10–30× at n=100K, deferred to a follow-up for n≥50K workloads
- `results/criterion/` — Criterion JSON for four variants up to n=10K
- `results/profiler/` — Step-level timing at n=10K, 30 iters

### No existing KD-tree infrastructure

Comprehensive search confirms zero KD-tree, ball-tree, HNSW, R-tree, or ANN crate references anywhere in `Cargo.toml`, `Cargo.lock`, or any `.rs` file.

---

## External Research

### Crate landscape (as of 2026-04-07)

| Crate | Version | Status | Exact k-NN | Sync | Rayon build |
|-------|---------|--------|-----------|------|-------------|
| `kiddo` | 5.3.0 (2026-03-19) | Active, 498K dl/month | Yes | Yes (ImmutableKdTree) | Yes (feature) |
| `kd-tree` | 0.6.2 (2025-11-16) | Active, 43K dl/month | Yes | Yes (immutable) | Yes (`par_build_by_ordered_float`) |
| `bosque` | active | Active | Yes | Yes | Yes | 3D only — unusable |
| `fnntw` | 0.4.1 (2023) | **Archived** | Yes | Yes | Yes | Do not use |

### kiddo v5 ImmutableKdTree

- Const-generic dimensions: `ImmutableKdTree<f64, u32, 2, 32>` specializes at compile time for 2D
- Eytzinger stem ordering → better cache locality vs. pointer-chasing trees
- `nearest_n::<SquaredEuclidean>(&[x, y], NonZero::new(k+1).unwrap())` returns exact k+1 neighbors ordered by distance (include self + filter)
- Thread safety: `Sync + Send` (explicit bounds on `A: Sync + Send`, `T: Sync + Send`) → safe to wrap in `Arc` and share across Rayon `par_iter` threads
- Rayon feature for parallel construction; queries via external `par_iter` over query points

### KD-tree dimensionality and n-crossover

- **d=2 is the optimal regime.** Scikit-learn's brute-force auto-switch threshold is d>15; at d=2 (UMAP output), pruning efficiency is maximal.
- **n-based crossover (d=2):** KD-tree breaks even with brute-force around n≈1K–10K depending on implementation. At n≥50K with d=2 and exact k-NN (k=15), KD-tree is definitively faster.
- **Complexity:** Tree build O(n log n); per-query O(k log n); total for n queries: O(n k log n). At n=100K, k=15: ~25.5M node visits vs 10B distance computations for brute-force.

### Estimated wall times (100K points, d=2, k=15)

| Approach | Build | n queries | Total (serial) | Total (8 Rayon threads) |
|----------|-------|-----------|----------------|------------------------|
| flat_simd (extrapolated) | — | ~14.5s | ~14.5s | ~1.8s |
| kiddo ImmutableKdTree | ~5–10ms | ~300–500ms | ~310ms | ~50–70ms |

Sources: kiddo benchmark webapp (Ryzen 5900X), FNNTW paper (AMD EPYC, 100K nodes, 1M queries @ ~22ms), scikit-learn k-d tree docs.

---

## Technical Context

### What a KD-tree integration replaces

The natural replacement unit is the entire `COMB_DIST_Y.with(...)` block (lines 587–650), replacing:
1. Distance fill (`dist_sq_2d_avx2_batch` or scalar loop) → eliminated
2. `indices_y.extend(0..n)` + `select_nth_unstable_by(k, ...)` → eliminated

A KD-tree call returns `k` nearest neighbor indices directly. The penalty loop (`for &j in &indices_y[..k]`) is **unchanged** — it only needs a slice of k neighbor indices.

### Self-exclusion requirement

Current approach: set `dist_y[i] = f64::INFINITY` before introselect. KD-tree approach: query for `k+1` neighbors and drop the entry where index == i (distance 0). This requires the tree to store original indices as the item type (e.g., `u32`). kiddo's `NearestNeighbour<f64, u32>` result struct carries both distance and item — compatible.

### Tie-breaking requirement

Current comparator: `dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b))` — stable by index for equal distances. kiddo returns results ordered by distance but does not guarantee index-based tie-breaking. **If two points are equidistant from query i at rank k, kiddo and the current brute-force may select different k-th neighbors.** However, since T is a sum over all rows, and equidistant ties are measure-zero in floating-point practice, this is unlikely to cause `|ΔT| ≥ 1e-12` failures — but must be verified empirically.

### Thread-local buffers

`COMB_DIST_Y` and `COMB_INDICES_Y` exist solely to avoid per-row Vec allocations in the current brute-force path. A KD-tree implementation allocates its own query scratch internally (kiddo uses a small fixed-size priority queue per query). The thread-locals can be removed for the Y path once the KD-tree is used.

### ComputeMode interaction

`trustworthiness()` has no `ComputeMode` parameter and is not gated by it. A KD-tree implementation would follow the same pattern: no `ComputeMode` gating. The CLAUDE.md `ComputeMode::RustNative` gate rule applies to eigensolver-chain divergences from Python UMAP behavior; trustworthiness is a post-hoc metric with no Python reference path.

### Profiling feature

If `feature = "profiling"` is enabled, the `Y_DIST_NS` atomic accumulator tracks the y_dist step. This label would need updating to `Y_KNN_NS` or similar if the step name changes, but this is cosmetic.

### Dependency addition

Adding `kiddo` as an optional dependency (e.g., `kiddo = { version = "5", features = ["rayon"], optional = true }`) gated behind a `kdtree` feature would let users opt in without imposing a compile-time cost on the default build. Alternatively, it can be an unconditional dependency since it has no C build dependencies.

---

## Hypotheses

**H1 (primary):** Building `ImmutableKdTree<f64, u32, 2, 32>` once over the Y embedding and querying it in the Rayon parallel loop will reduce the Y k-NN step from O(n²) to O(n k log n), yielding ≥5× wall-time improvement at n=50K and ≥10× at n=100K relative to `flat_simd`, while preserving `|ΔT| < 1e-12` correctness.

**H2 (crossover):** The flat_simd brute-force path outperforms KD-tree at n ≤ some threshold T_cross (likely 5K–15K) due to AVX2 sequential memory access vs. KD-tree pointer-chasing; an adaptive dispatch (brute-force below T_cross, KD-tree above) achieves optimal performance across all n.

**H3 (tie-breaking):** Equidistant-at-rank-k points are rare in practice for UMAP 2D embeddings, and the KD-tree's unspecified tie-breaking order will produce the same T value within `1e-12` of the brute-force result on all tested inputs.

**H4 (build cost amortization):** At n≥50K, the one-time `ImmutableKdTree` build cost (O(n log n), estimated ~5–25ms at n=50K) is dominated by query savings, making it beneficial even for a single `trustworthiness()` call.

---

## Proposed Investigation Directions

### Direction 1: Direct kiddo integration with adaptive dispatch (recommended)

Add `kiddo` v5 as an optional dependency. In `trustworthiness()`:
- If `n >= N_THRESHOLD` and `d_y == 2`: build `ImmutableKdTree<f64, u32, 2, 32>` before the Rayon loop, share it via `Arc`, query `nearest_n(k+1)` per row, filter self.
- Else: use existing `flat_simd` path.

Benchmark at n ∈ {1K, 5K, 10K, 50K, 100K} to find T_cross empirically. Run correctness tests (`t_tw_08`, `t_tw_10`) with the KD-tree path active.

**Trade-offs:** Adds a crate dependency; correctness test at small n validates tie-breaking behavior; complexity is localized to one block in `trustworthiness()`.

### Direction 2: d_y-agnostic KD-tree path

Current `flat_simd` is specialized for `d_y == 2`. kiddo's const-generic API requires a compile-time `K` — can't do `K=d_y` at runtime. Options:
- Match on common d_y values (2, 3) with compile-time specialization for each
- Use `kd-tree` crate which handles runtime dimensions via slice-based API (less optimized but more flexible)
- Restrict KD-tree path to `d_y == 2` only (matching the current AVX2 path's scope)

Restricting to `d_y == 2` is simplest and covers the dominant UMAP use case; it also matches where the biggest gain is (2D is the optimal KD-tree regime).

**Trade-offs:** Restricting to d_y=2 leaves d_y=3 unoptimized; full generalization requires a match ladder or dynamic dispatch crate.

### Direction 3: Benchmark-only study (no code change)

Before implementing, run the existing `trustworthiness_bench` at n=50K with the current `flat_simd` to get the true post-PR-#237 baseline (the existing research data is pre-PR-#237 `combined` variant). This confirms the extrapolated 3.6s estimate and quantifies the actual problem size. Then implement H1 with empirical build+query timing separation.

**Trade-offs:** Provides ground truth for the cost-benefit analysis at the cost of one benchmark run.

---

## Success Criteria

A conclusive answer to the research question requires:

1. **Correctness:** KD-tree path passes `t_tw_08_combined_matches_baseline` and `t_tw_10_self_exclusion_never_in_knn` with `|ΔT| < 1e-12` on all test cases.
2. **Performance at n=50K:** KD-tree path shows ≥5× wall-time speedup over `flat_simd` at n=50K.
3. **Performance at n=100K:** KD-tree path shows ≥10× wall-time speedup over extrapolated `flat_simd` at n=100K.
4. **Crossover quantified:** Benchmark data at n ∈ {1K, 5K, 10K, 50K, 100K} identifies the n at which KD-tree becomes faster than brute-force, enabling adaptive dispatch.
5. **Build cost measured:** Separate timing of tree construction vs. total query time to confirm amortization assumption (H4).

---

## Metric Context

`trustworthiness()` does not use `ComputeMode` and does not appear in the eigensolver Accuracy/Parity assessment pipeline in `test_metrics_assess.rs`. The applicable quality dimensions and their thresholds for this research are:

| Metric | Quality Dimension | Threshold | Source |
|--------|------------------|-----------|--------|
| `|T_kdtree − T_brute_force|` | **Correctness** | `< 1e-12` | `t_tw_08`, `t_tw_10` in `src/metrics.rs` tests |
| `|T_rust − T_sklearn|` | **Parity** | `< 1e-6` | `tests/integration/test_trustworthiness.rs` (currently `#[ignore]`) |
| y_knn wall time at n=50K | **Performance** | No canonical threshold; goal is ≥5× speedup vs flat_simd | Derived from PR #237 baseline |
| y_knn wall time at n=100K | **Performance** | No canonical threshold; goal is ≥10× speedup vs extrapolated flat_simd | Derived from extrapolation |

**Gap:** No canonical `Performance` metric with a hard threshold exists for `trustworthiness()` computation time. The experiment plan should define explicit performance pass/fail criteria (e.g., "n=50K total wall time ≤ 500ms with 8 Rayon threads").

The eigensolver thresholds (`DENSE_EVD_QUALITY_THRESHOLD = 1e-6`, `LOBPCG_QUALITY_THRESHOLD = 2e-5`, etc.) in `src/metrics.rs` lines 16–62 are **not relevant** to this research — they govern eigenpair residuals in `solve_eigenproblem`, not trustworthiness k-NN.
