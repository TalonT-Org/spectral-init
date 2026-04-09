# Scope Report: Sub-sampled Trustworthiness Error/Speed Trade-off

## Research Question

What is the empirical error/speed trade-off when computing trustworthiness on a random subset of points instead of the full dataset? Specifically: at what subsample sizes does |T_sampled - T_exact| remain acceptable, and what speedup does each size yield?

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| Current behavior | Exact `trustworthiness(x, y, k)` in `src/metrics.rs`: O(n²·d_x + n²·d_y + n²·k), brute-force pairwise distances, Rayon parallel outer loop, AVX2+FMA SIMD for d_x≥10, introselect partial ranking. Validated ±1e-6 against sklearn. | No sub-sampled variant exists in `src/`. The only prior attempt (`tw_approx_runner` in a research worktree) produced systematically biased results (|ΔT|≈0.47) — likely a normalization bug, not a real signal about sub-sampling accuracy. |
| Performance | MERFISH n=10K d_x=50: ~3.6s wall time (x_dist=58.9%, x_sort=9.4%, y_dist=21.5%, penalty=10.2%). Profile stable across n=10K and n=50K. Gaussian d_x=10 is ~2× faster per-n. | Wall-clock scaling for sub-sampled computation on MERFISH at m={500,1K,2K,5K,10K} is unmeasured. Whether speedup is linear in m/n (Approach A) or quadratic (m/n)² (Approach B) depends on implementation choice. |
| Edge cases | Gaussian random embeddings give T≈0.5 (no structure to preserve). The prior H5 experiment on Gaussian data is uninformative about real-data sub-sampling accuracy because (a) the implementation was buggy and (b) Gaussian data has uniform local structure, making per-point contributions nearly identical. | How sub-sampling accuracy behaves on MERFISH data (non-uniform cluster structure, varying local density) is completely unknown. Whether cluster boundary points contribute disproportionately to T variance is unknown. |
| Prior work | H5 in `2026-04-05-tw-perf-rerun-clean`: tested `tw_approx_runner` on Gaussian n=40K, 10 seeds, m∈{500,1K,2K,5K,10K}. Got |ΔT|≈0.47 at all m values (474× over 0.001 threshold). Also 9% *slower* than exact at m=5000. MERFISH gate never ran (data absent from worktree). | Whether the H5 failure was a normalization bug (likely: T_approx≈1.0 vs T_exact≈0.5 across all m) or a fundamental problem with the sub-sampling approach. A correct re-implementation on MERFISH data has never been attempted. |

## Prior Art in Codebase

### Trustworthiness Implementation
- **`src/metrics.rs:518-765`** — Production exact trustworthiness. Fully optimized: thread-local scratch buffers, introselect partial ranking, AVX2+FMA SIMD dispatch (x_dist for d_x≥10, y_dist for d_y=2), Rayon parallel outer loop. Re-exported from `src/lib.rs:214`.
- **`src/bin/trustworthiness.rs`** — CLI wrapper (`--x`, `--y`, `--k`), feature `cli`.
- **`src/bin/tw_profiler.rs`** — Profiling harness with `--iters`, `--warmup`, JSON output, feature `cli,profiling`.

### Sub-sampling Attempts
- **No `trustworthiness_approx` function exists** anywhere in `src/`. Designed in multiple experiment plans but never shipped.
- **`tw_approx_runner`** binary existed in the `research-20260405-tw-perf-rerun-clean` worktree with `--sample` and `--seed` flags. Never merged. Its results showed systematic bias (T_approx≈1.0 across all m while T_exact≈0.5), strongly suggesting a normalization error — likely using the full n in the denominator while computing KNN/ranks within the m-point subset only.

### Research History (Trustworthiness Thread)
1. **`2026-04-04-tw-perf-scaling`** — First TW benchmark. Combined variant: 2.04× at n=50K. H5 (sub-sampling) blocked by absent MERFISH data.
2. **`2026-04-05-tw-perf-rerun-clean`** — Clean rerun. H5 result: |ΔT|=0.474, 9% slower — definitively rejected on Gaussian data. MERFISH gate never reached.
3. **`2026-04-06-y-heap-bottleneck-optimization`** — Shipped flat_simd (BinaryHeap→Vec+introselect+AVX2). ~2× total speedup.
4. **`2026-04-07-kdtree-y-knn-trustworthiness`** — KD-tree for Y-space. Definitive negative: flat_simd 36–44% faster at all n. DO NOT SHIP. Recommended sub-sampling and ANN-in-X as future directions.
5. **`2026-04-08-tw-merfish-step-timing`** — MERFISH step profiling. x_dist=58.9% on MERFISH d_x=50 vs. 33.5% on Gaussian d_x=10. X-space dominates.

### MERFISH Fixtures (Verified on Disk)
All at `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`:
| File | Shape | Dtype | Size |
|------|-------|-------|------|
| `merfish_n10k_x.npy` | (10000, 50) | f64 | 4.0 MB |
| `merfish_n10k_y.npy` | (10000, 2) | f64 | 160 KB |
| `merfish_n50k_x.npy` | (50000, 50) | f64 | 20.0 MB |
| `merfish_n50k_y.npy` | (50000, 2) | f64 | 800 KB |

### Parity Fixtures
- `tests/fixtures/tw_parity/tw_parity_50d.npz` — Active CI test (`sklearn_parity_50d`), validates AVX2 looped kernel.

## External Research

### Sub-sampled Trustworthiness in the Literature
- **No formal methodology** exists for sub-sampled trustworthiness. UMAP creator Leland McInnes acknowledged trustworthiness "is not tractable for large data sets" (GitHub Issue #6) and only compares on small datasets.
- **sklearn has no sub-sampling parameter.** Callers must manually subsample before calling. sklearn's implementation is O(n²d) time, O(n²) memory.
- **The "10K → ~1% error" claim has no published source.** It is not in the UMAP paper, UMAP documentation, or any identified publication. Likely folk wisdom or informal community guideline.

### Statistical Theory
- **CLT:** Variance of subsample estimate scales as σ²/m. Standard error ∝ σ/√m.
- **Hoeffding bound** (contributions bounded in [0,1]):
  - P(|T_sub - T_pop| > ε) ≤ 2·exp(-2mε²)
  - For ε=0.01, 99% confidence: m ≥ ~26,500 (worst case)
  - For ε=0.001, 99% confidence: m ≥ ~2,650,000 (impractical — but this is worst case; real variance is much smaller)
- **Practical implication:** 5K–10K points may suffice for ~1% error if empirical variance is well below theoretical maximum. This must be measured, not assumed.

### Stratified vs. Uniform Sampling
- No published comparison for trustworthiness specifically.
- General theory: stratification reduces variance when stratum-level statistics vary substantially. For trustworthiness, this would matter if cluster boundary points contribute disproportionately.
- **Uniform sampling is the defensible default** for unsupervised settings (no labels).

### Alternative Approaches
- Approximate KNN (FAISS, HNSWlib) for the inner loops — but the bottleneck is the full pairwise distance matrix, not KNN lookup.
- Saturn coefficient (PeerJ 2026) — simpler distance-preservation metric but still O(n²) naively.
- Point sub-sampling remains the only approach that changes the fundamental O(n²) scaling.

## Technical Context

### Architecture
The trustworthiness computation lives entirely in `src/metrics.rs`. The algorithm:

```
T(k) = 1 − (2 / (n·k·(2n−3k−1))) · Σᵢ Σ_{j ∈ U_i(k)} (r(i,j) − k)
```

For each of n rows (parallel via Rayon):
1. **X-dist** (O(n·d_x)): Compute squared Euclidean distance to all n points. Uses `dist_sq_avx2_looped` for d_x≥10.
2. **X-sort** (O(n) average): Introselect to find k-nearest in X. Build `HashSet<usize>` for O(1) membership.
3. **Y-dist** (O(n·d_y)): Compute distances to all n points. Uses `dist_sq_2d_avx2_batch` for d_y=2.
4. **Y-sort + penalty** (O(n) + O(k·n) worst): Introselect for k-nearest in Y. For each Y-neighbor not in X-NN set, linear scan to compute rank r(i,j).

Total: O(n²·d_x + n²·d_y + n²·k). Memory: O(n) per thread (scratch buffers), O(1) global (no distance matrix materialized).

### Two Distinct Sub-sampling Approaches

**Approach A — Row sub-sampling (query subset, full distances):**
Sample m query points. For each query point, compute distances to ALL n points in both spaces. KNN and ranks computed against full population. Denominator uses m·k·(2n−3k−1). This is an unbiased estimator of exact T. Cost: O(m·n·d_x + m·n·d_y + m·n·k). Speedup: ~n/m (linear).

**Approach B — Full sub-sampling (subset embedding):**
Take m points, compute distances only within those m points. KNN and ranks are within the subsample. Denominator uses m·k·(2m−3k−1). This computes trustworthiness *of the sub-embedding*, which is a different quantity than T of the full embedding. Cost: O(m²·d_x + m²·d_y + m²·k). Speedup: ~(n/m)² (quadratic).

**Critical distinction:** The failed H5 experiment likely used Approach B with the normalization denominator from Approach A (mixed semantics), producing the systematic 0.47 bias. A correct experiment must clearly choose one approach and implement it consistently.

**Recommendation for this research:** Test Approach A first (unbiased estimator, simpler theory). Test Approach B as a secondary comparison — it gives larger speedups but measures a subtly different quantity.

### What Code Changes Are Needed for This Experiment
1. A Python experiment script that loads MERFISH `.npy` fixtures.
2. Computes exact T via sklearn (ground truth).
3. For each subsample size m, draws random indices, extracts sub-matrices `X[idx]` and `Y[idx]`, computes T on the sub-matrices (Approach B), and separately computes the row-subsampled estimator (Approach A).
4. Repeats across multiple seeds, records |ΔT|, std, and wall-clock time.

No Rust code changes needed — this is a Python-only characterization experiment. The Rust implementation changes would follow only if the results justify adding a `trustworthiness_approx` to `src/metrics.rs`.

## Hypotheses

**H1 (Row sub-sampling accuracy):** For Approach A (query subset, full distances) on MERFISH n=10K, |T_sub - T_exact| < 0.01 at m ≥ 2000 (20% subsample) with std < 0.005 across 10 seeds. *Falsifiable by:* measuring mean and std of |ΔT| across seeds.

**H2 (Full sub-sampling accuracy):** For Approach B (subset embedding) on MERFISH n=10K, |T_sub - T_exact| < 0.01 at m ≥ 5000 (50% subsample) with std < 0.005. Approach B measures a different quantity and will have larger systematic bias than Approach A. *Falsifiable by:* measuring ΔT and checking whether the bias is constant or random.

**H3 (Speed scaling):** Wall-clock speedup is approximately linear in n/m for Approach A and quadratic (n/m)² for Approach B. On MERFISH n=50K → m=5K, Approach A gives ~10× speedup, Approach B gives ~100× speedup. *Falsifiable by:* measuring wall times.

**H4 (Variance scaling):** Standard deviation of the sub-sampled estimator scales as C/√m for both approaches, consistent with CLT. *Falsifiable by:* fitting std vs. m on a log-log plot and checking slope ≈ -0.5.

**H5 (MERFISH vs. synthetic):** The error profile on MERFISH (real manifold with non-uniform density) differs from synthetic Gaussian data — specifically, variance is higher on MERFISH due to heterogeneous cluster structure. *Falsifiable by:* comparing std at matched m on both data types.

**H6 (Unverified claim):** The "10K subsample from n=100K gives ~1% error and ~100× speedup" claim from PR #238 holds only for Approach B, and only if the data has sufficiently uniform structure. On MERFISH data, the error at m=10K may exceed 1%. *Falsifiable by:* extrapolating the measured error curve to n=100K.

## Proposed Investigation Directions

### Direction 1 — Python-only characterization (Recommended)

Use sklearn as the ground truth. Write a Python experiment script that:
1. Loads MERFISH n=10K and n=50K fixtures.
2. Computes exact T(k=15) via `sklearn.manifold.trustworthiness` (ground truth).
3. For each m ∈ {250, 500, 1000, 2000, 5000, 7500, 10000 (50K only), 25000 (50K only)}:
   - Draw 10 random seeds.
   - Approach A: subsample m query rows, compute T with `sklearn.manifold.trustworthiness(X[idx], Y[idx], k=15)`.
   - Record |ΔT|, std, wall time.
4. Fit variance scaling model.
5. Generate error/speed table and plots.

**Trade-offs:** Python is slower but uses the validated sklearn implementation. No risk of normalization bugs. No Rust code changes. Can use sklearn's NearestNeighbors for Approach A if needed.

**Note:** Approach A (row sub-sampling with full distances) is not directly expressible via `sklearn.manifold.trustworthiness(X[idx], Y[idx], k)` — that function computes T on the subsample (Approach B). To implement Approach A, a custom Python trustworthiness that takes the full X/Y but only iterates over m query rows is needed. However, for pragmatic purposes, Approach B is what practitioners actually do (slice then compute), so characterizing Approach B's accuracy is the directly actionable result.

### Direction 2 — Rust implementation + characterization

Add `trustworthiness_subsampled(x, y, k, m, seed)` to `src/metrics.rs` implementing Approach A. Write Rust benchmark comparing exact vs. subsampled on MERFISH fixtures.

**Trade-offs:** More engineering effort. Risk of repeating the normalization bug. Benefit: measures speedup in the production implementation with SIMD, rayon, etc.

### Direction 3 — Combined: Python characterization → Rust implementation

Python experiment first (Direction 1) to establish the accuracy baseline. If results are promising, implement the optimal approach in Rust (Direction 2) as a follow-up.

**Trade-offs:** More total work, but de-risks the Rust implementation. Recommended if the research is expected to lead to a shipped feature.

## Success Criteria

1. **Error/speed table** across ≥4 subsample sizes on MERFISH n=10K and n=50K data.
2. **Variance estimates** with ≥5 seeds per subsample size.
3. **Recommended default subsample size** with quantitative justification (target: |ΔT| < 0.01 with 95% confidence).
4. **Clear verdict** on whether the "~1% error at 10K subsample" claim holds on MERFISH data.
5. **Identification of the correct sub-sampling approach** (A vs. B) for the use case of monitoring embedding quality during large-scale runs.
6. **Statement on whether MERFISH error profile differs from synthetic** (H5).

## Metric Context

### Canonical Metrics in `src/metrics.rs`

The following metrics from `src/metrics.rs` are relevant to this research:

| Metric | Quality Dimension | Threshold | Relevance |
|--------|------------------|-----------|-----------|
| `trustworthiness` (exact) | Accuracy (embedding quality) | ±1e-6 vs sklearn (CI parity gate) | The function being sub-sampled |
| `max_eigenpair_residual` | Accuracy | `DENSE_EVD_QUALITY_THRESHOLD = 1e-6` (L0/L5), `LOBPCG_QUALITY_THRESHOLD = 2e-5` (L1/L3), `RSVD_QUALITY_THRESHOLD = 1e-2` (L4) | Not directly relevant; eigensolver accuracy |
| `sign_agnostic_max_error` | Parity | Used in parity assessment | Not directly relevant |

### Gaps

1. **No `trustworthiness_approx` function** — does not exist in `src/metrics.rs`.
2. **No `TW_SUBSAMPLING_ACCURACY_THRESHOLD` constant** — the 0.001 threshold from prior research plans exists only in markdown, not as a named constant.
3. **No test for sub-sampling accuracy** — `test_trustworthiness.rs` only tests exact sklearn parity.
4. **No performance regression gate** — no wall-clock assertion for trustworthiness in CI.
5. The visual eval Python pipeline uses a looser `|ΔT| < 0.01` threshold for trustworthiness comparison (per `temp/metrics-standardization-report.md` §2.2), which is a more realistic target for sub-sampling than the 0.001 from prior plans.
