# Scope Report: y_heap Bottleneck Optimization in Trustworthiness Computation

## Research Question

The trustworthiness computation at n=10K, k=15 spends **70.3% of CPU time** in the `y_heap` step — a brute-force O(n) scan per point to find k-nearest neighbors in Y-space (embedding) using a `BinaryHeap<(u64, usize)>`. The question is: **Is there a faster way to implement the y_heap step that is both correct (exact kNN, not approximate) and thread-safe for use in a Rayon parallel loop?**

---

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| Current behavior | BinaryHeap max-heap of size k, O(n log k) per row, scalar distance, per-row allocation via `BinaryHeap::with_capacity(k+1)` | Whether the heap *allocation* or the heap *push/pop overhead* or the *memory scan itself* is the dominant cost within the 70.3% |
| Performance | y_heap = 70.3% of total CPU at n=10K, k=15 (95% CI [70.3%, 70.4%]); x_dist = 13.0% (already SIMD-optimized); penalty = 6.4% | How y_heap scales at n=50K, n=100K; breakdown of time *within* y_heap between allocation, arithmetic, heap ops |
| Edge cases | k < n/2 enforced; self-point excluded via `if j == i { continue; }`; ties broken by index order in introselect (not in y_heap) | Whether tie-breaking in y_heap matters for trustworthiness correctness; NaN-safety of `to_bits()` trick is implicitly assumed |
| Prior work | thread_local (1.54×), avx2 (1.49×), partial_rank (not faster) were all measured for *x_dist* in PR #226/229. No optimization has been applied to y_heap. | Whether select_nth_unstable on a flat Y-distance buffer matches or exceeds heap performance for k=15 |
| Algorithm alternatives | KD-tree (kiddo, nabo) viable at n=10K, d=2; SIMD batch for 2D distances possible; select_nth_unstable already used for X | KD-tree build cost amortization at n=10K vs n=100K; exact wall-time ratio of tree vs brute-force at these scales |
| Measurement infrastructure | `tw_profiler` binary measures step-level CPU fractions; `trustworthiness_bench.rs` has Criterion at n={1K,5K,50K} | No y_heap-specific Criterion benchmark exists; no multi-variant comparison bench exists yet |

---

## Prior Art in Codebase

### Existing Implementation (`src/metrics.rs:501-507`)

```rust
let mut heap: BinaryHeap<(u64, usize)> = BinaryHeap::with_capacity(k + 1);
for j in 0..n {
    if j == i { continue; }
    let d: f64 = yi.iter().zip(y.row(j).iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
    heap.push((d.to_bits(), j));
    if heap.len() > k { heap.pop(); }
}
```

Key properties:
- `(u64, usize)` encodes distance bits + point index. Since squared distances ≥ 0.0, IEEE 754 bit ordering is preserved — correct max-heap over distances without `Ord` on f64.
- `BinaryHeap::with_capacity(k+1)` is allocated per-row, per-thread (not reused).
- No SIMD dispatch. No thread-local buffer reuse.

### Thread-Local Pattern for X-dist (`src/metrics.rs:455-490`)

```rust
thread_local! {
    static COMB_DIST_X:  RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
    static COMB_INDICES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}
```

Used by the x_dist and x_sort steps:
1. Buffer cleared + resized to `n` once per row (no malloc after first row per thread).
2. `COMB_DIST_X` stores all X squared distances; `COMB_INDICES` holds `0..n` permuted by introselect.
3. This pattern is the established idiom for per-row scratch buffers in this codebase.

The pattern is **not applied to y_heap**. `COMB_DIST_X` is reused for the penalty rank scan, but Y distances are computed inline and discarded.

### Existing AVX2 Kernel (`src/metrics.rs:386-409`)

- 2-load AVX2 path (`_mm256_loadu_pd`, `_mm256_fmadd_pd`) covering the first 8 elements; scalar tail for remainder.
- Gated on `target_feature = "avx2,fma"` (compile-time) and `d_x >= 10` (runtime).
- **Only used for X-space distances.** Y-space distances are always scalar.

### Prior Research (PR #226/229 — now merged, research directory not yet on disk)

From git history:
- **thread_local alone**: 1.54× speedup at n=10K (CI [1.47, 1.62])
- **avx2 alone**: 1.49× speedup at n=10K (CI [1.42, 1.56])
- **combined** (thread_local + avx2): 1.03× — inconclusive due to W4 cache warm-state anomaly
- **partial_rank** (introselect replacing full sort for rank computation): 0.978× — confirmed NOT faster
- **approximate trustworthiness (H5)**: decisively rejected (|Δ|=0.474926, ~474× over threshold of 0.001)

All prior optimization work was for x_dist. The report explicitly flagged y_heap as the unresolved dominant cost.

### Existing Benchmarks

- `benches/trustworthiness_bench.rs`: Criterion at n={1000, 5000, 50000}, d_x=10, d_y=2, k=15. Flat sampling, sample_size=10.
- `src/bin/tw_profiler.rs`: Step-level CPU profiler producing JSON with step fractions. Used in PR #229 to generate the 70.3% measurement.
- No variant-comparison benchmarks for y_heap alternatives exist.

### No Spatial Index Code

Comprehensive search confirms: zero KD-tree, ball-tree, HNSW, R-tree, or ANN crate references anywhere in the source. kNN is entirely hand-rolled stdlib.

---

## External Research

### KD-Tree vs Brute Force in 2D at n=10K

**Key finding**: KD-trees do beat brute force in 2D at n=10K, k=15, but the margin is smaller than at higher k or higher d.

- For d < 20, KD-tree query is O(log n) average vs O(n) brute force. At n=10K the theoretical ratio is ~770×, but practical constant factors reduce this to ~3-10× for k=15.
- `sdd/kd-tree-comparison` webapp (https://sdd.github.io/kd-tree-comparison-webapp/) has Criterion benchmarks covering multiple Rust kNN crates at various n and d values.
- At n=10K, `kiddo` achieves ~3-10× speedup over brute force for k=1 in 2D. For k=15 the ratio shrinks because the query must guarantee k neighbors, visiting more tree nodes.
- **Practical threshold for KD-tree to win in 2D**: n > 500–1000 for single queries. For all-pairs (n=10K queries on n=10K points), the build cost is easily amortized.

### Rust kNN Crates

| Crate | Type | Notes |
|---|---|---|
| **kiddo** | KD-tree | Best benchmarked in Rust for low-d. Immutable (Sync-safe after build). |
| **nabo** | KD-tree | Rust port of libnabo C++; explicitly designed for "low-dimensional spaces"; parallel build |
| **FNNTW** | KD-tree | "Fastest Nearest Neighbor in the West"; uses `unsafe get_unchecked`, tcmalloc, parallel build |
| **kd-tree** | KD-tree | Ergonomic; Rayon+nalgebra support; "not the fastest but very usable" |
| **rstar** | R*-tree | General spatial index; higher overhead than KD-tree for pure kNN |

**Thread safety**: All KD-tree crates above support concurrent queries on a shared immutable tree (read-only after build). This matches the Rayon parallelism model in `trustworthiness`.

### select_nth_unstable vs BinaryHeap for k-NN

- **BinaryHeap (current)**: O(n log k) per row. For k=15: ~3.9 comparisons per element = ~39K heap ops per row at n=10K.
- **select_nth_unstable**: O(n) average (O(n) worst-case since Rust PR #107522 added Median of Medians fallback). Requires a flat distance buffer of size n.
- **Practical**: For k=15 the log(k) factor is small (~4). The heap's advantage is no extra allocation; the introselect's advantage is better cache behavior on a flat array + no pointer indirections.
- **Recommendation from external benchmarks**: For k << n (k=15, n=10K), the heap is competitive. The primary win from switching to introselect is *allocation elimination* (no per-row `BinaryHeap::with_capacity`), not algorithmic complexity.
- The `partial_sort` crate benchmarks (n=10K, limit=20): partial_sort = 5.4 µs, nth_select = 8.6 µs, full heap sort = 242 µs. But these are on pre-filled arrays; the allocation cost of the current BinaryHeap is on top of this.

### SIMD for 2D Distance Computation

- For d_y=2 with f64, each distance is: `(a0-b0)² + (a1-b1)²` — 5 operations.
- AVX2 registers hold 4 f64 values (256-bit). You can pack **4 Y-row pairs** per register, compute 4 squared distances per SIMD group.
- The Y matrix at n=10K, d_y=2 is 160KB (fits in L2 cache). The bottleneck is register throughput, not memory bandwidth.
- Current scalar loop uses `iter().zip().map().sum()` — LLVM auto-vectorizes this to SSE2 (128-bit, 2 f64) but not AVX2 (256-bit). An explicit AVX2 kernel could process 4 rows per cycle instead of 2.
- **Expected gain**: 2× from 128→256-bit, up to 4× for a well-tuned implementation.

### Source References

- https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/ — KD-tree vs brute force at n=10K, d=2
- https://sdd.github.io/kd-tree-comparison-webapp/ — Criterion benchmarks for Rust kNN crates
- https://github.com/sdd/kiddo — kiddo crate
- https://github.com/enlightware/nabo-rs — nabo crate
- https://lib.rs/crates/fnntw — FNNTW crate
- https://github.com/sundy-li/partial_sort — partial_sort Rust crate benchmarks
- https://github.com/rust-lang/rust/pull/107522 — Median of Medians fallback for select_nth_unstable
- https://padas.oden.utexas.edu/static/papers/sc15nn.pdf — "Performance Optimization for the K Nearest-Neighbor Kernel on x86 Architectures" (SC'15)
- https://blog.cloudflare.com/computing-euclidean-distance-on-144-dimensions/ — SIMD distance batching

---

## Technical Context

### Call Stack and Data Layout

```
trustworthiness(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64
  ├── runtime AVX2 detection: use_avx2 = is_x86_feature_detected!("avx2") && ...
  ├── thread_local: COMB_DIST_X: RefCell<Vec<f64>>, COMB_INDICES: RefCell<Vec<usize>>
  └── (0..n).into_par_iter().map(|i| -> f64 { ... }).sum()
       ├── COMB_DIST_X/COMB_INDICES.with(borrow_mut) → dist_x, indices
       ├── [x_dist] fill dist_x[0..n] with ‖xi-xj‖² (AVX2 if use_avx2 && d_x>=10)
       ├── [x_sort] indices.select_nth_unstable_by(k, ...)  ← O(n) introselect
       ├── [x_knn_set] HashSet::from(indices[..=k] - {i})  ← O(k)
       ├── [y_heap] BinaryHeap<(u64,usize)> scan all j    ← O(n log k), UNOPTIMIZED
       └── [penalty] for j in heap: rank scan over dist_x ← O(n) per miss
```

### y_heap Step Detail (`src/metrics.rs:501-507`)

Input: `yi: ArrayView1<f64>` (2 elements for d_y=2), `y: ArrayView2<f64>` (n×2, row-major).

Per iteration:
1. Compute `d = (yi[0]-y[j][0])² + (yi[1]-y[j][1])²` — scalar, 5 FP ops.
2. `d.to_bits()` — zero-cost bitcast, preserves IEEE 754 ordering for non-negative f64.
3. `heap.push((bits, j))` — O(log k) heap up-sift.
4. `if heap.len() > k { heap.pop(); }` — O(log k) heap down-sift when over capacity.

Total per row: n iterations × (5 FP + bitcast + ~2 heap ops × log(15) ≈ 8 comparisons) = n × ~25 ops.
At n=10K: ~250K ops per row, 10K rows = ~2.5B ops total — all scalar, no SIMD, with heap pointer chasing.

### Why y_heap Dominates (70.3%)

Compared to x_dist (13%):
- x_dist uses SIMD: processes 8 f64 per cycle for d_x=10. Effective throughput: ~10× faster per element.
- y_heap is scalar: 5 FP ops per distance — BUT these are already trivially fast. The overhead is the **BinaryHeap pointer indirection and allocation** per row.
- y_heap has no buffer reuse: `BinaryHeap::with_capacity(k+1)` allocates on the heap per row. For n=10K rows × (Rayon threads), this is thousands of small allocations.
- The `heap.pop()` when len > k is a conditional branch + heap restructure — branch prediction fails approximately k/n of the time (k=15 pops per n=10K pushes).

The issue description confirms: "The cost is iteration + heap maintenance, not arithmetic."

### Rayon Parallelism Constraint

The outer loop is `into_par_iter()` over 0..n. Any shared structure must be:
- **Built before** the parallel section (immutable once built)
- **Read-only** during the parallel section (Sync)
- Thread-local scratch buffers (`thread_local!`) are the established pattern for mutable per-row state

KD-tree crates (kiddo, nabo) produce immutable trees after build, which are Sync — compatible with Rayon concurrent queries.

### ComputeMode: Not Applicable

`ComputeMode` (PythonCompat/RustNative) controls eigensolver routing only. `trustworthiness()` has no ComputeMode parameter and no conditional branches based on it.

---

## Hypotheses

**H1 (Thread-local Y-dist buffer + select_nth_unstable):**
Replacing the `BinaryHeap` with a thread-local `Vec<f64>` (reusing the COMB_DIST_X pattern) for Y distances and using `select_nth_unstable_by(k, ...)` will yield a **20–50% reduction** in y_heap time at n=10K by eliminating per-row allocation and log(k) heap maintenance. No new dependencies. Correctness is straightforward (same introselect pattern as x_sort).

*Falsification condition:* Criterion shows y_heap time ≤ 5% faster with the thread-local approach, or the thread-local buffer adds cross-row interference.

**H2 (2D-specialized SIMD batch for Y-distances):**
An explicit AVX2 kernel for d_y=2 that processes 4 Y-points simultaneously (packing 4 × 2 f64 per 256-bit register pair) will yield a **1.5–3× speedup** over scalar for the distance arithmetic within y_heap, on top of any heap-replacement gain.

*Falsification condition:* Criterion shows < 20% speedup from the SIMD kernel alone (indicating arithmetic is not the bottleneck, only iteration overhead is).

**H3 (KD-tree pre-build):**
Building a 2D KD-tree (kiddo or nabo) from Y before the parallel loop and querying it for all n rows will yield a **3–10× speedup** on the y_heap step at n=10K and **10–30× at n=100K**, due to O(log n) query complexity in 2D vs O(n) brute force.

*Falsification condition:* Criterion shows KD-tree approach ≤ 2× faster than brute force at n=10K (indicating tree overhead dominates at this scale), or the build cost is not amortized within a single trustworthiness call.

**H4 (Combined H1 + H2 = primary recommended experiment):**
The combination of thread-local Y-dist buffer (H1) + 2D SIMD distance kernel (H2) yields a **2–4× speedup** on y_heap with zero new dependencies, mapping to **~35% total wall-time improvement** (from the 70.3% × 0.5 formula in the issue).

*Falsification condition:* The combined approach does not exceed 1.3× speedup or shows W4-style cache anomaly from the combined baseline.

---

## Proposed Investigation Directions

### Direction 1: Thread-local flat buffer + select_nth_unstable for Y (Recommended Primary)

**Approach**: Add `COMB_DIST_Y: RefCell<Vec<f64>>` thread-local. Fill it with all n Y squared distances (same scalar or SIMD loop as currently, but into a flat Vec). Then call `select_nth_unstable_by(k, ...)` on a `COMB_INDICES_Y: RefCell<Vec<usize>>` to find the k-NN, instead of using a `BinaryHeap`.

**Why this first**: Mirrors the established pattern, requires zero new dependencies, eliminates per-row allocation, and the introselect path is already proven to work for x_sort. The gap is that x_dist had SIMD+thread_local applied together; y_heap has neither.

**Trade-offs**:
- **Pro**: No new dependency. Safe. Same algorithmic correctness guarantee as current (exact kNN via comparison).
- **Pro**: Eliminates `BinaryHeap::with_capacity(k+1)` allocation per row (all n=10K rows × threads).
- **Con**: Does not reduce the O(n) iteration count (same n distances computed). Won't help if iteration is the bottleneck.
- **Con**: Adds a second index permutation buffer for Y (memory: n × 8 bytes = 80KB at n=10K per thread — fits in L2).

**Expected speedup**: 20–50% reduction in y_heap time based on the x_dist thread_local result (1.54× total, x_dist fraction was ~13%). For y_heap which is 70.3%, even 20% y_heap reduction → ~14% total improvement.

### Direction 2: 2D SIMD batch kernel for Y-distances

**Approach**: Add a dedicated `dist_sq_2d_avx2_batch` kernel that processes 4 Y-points simultaneously using `_mm256_sub_pd` + `_mm256_mul_pd` + horizontal add. Build on the existing `dist_sq_avx2` pattern but specialized for d_y=2 (fixed stride, no tail loop needed).

**Why this second**: The x_dist AVX2 kernel (1.49×) was for d_x=10 with 8-element loads. For d_y=2, a different kernel structure is needed. The Y matrix is 160KB at n=10K (L2-resident), so arithmetic throughput is the limiting factor once allocation is removed.

**Trade-offs**:
- **Pro**: 2–4× on arithmetic throughput within y_heap. No new dependency.
- **Con**: More complex implementation than H1. Only benefits when arithmetic is the bottleneck (which it may not be if heap maintenance dominates).
- **Con**: Requires careful handling of the `if j == i { continue; }` skip in a batched context.

**Expected speedup**: 1.5–2.5× on y_heap time, mapping to ~35–52% total improvement.

### Direction 3: KD-tree (kiddo or nabo)

**Approach**: Add `kiddo` or `nabo` as a dependency. Build a 2D KD-tree from Y before the `into_par_iter()` loop. Query `tree.k_nearest(yi, k)` for each row `i` inside the parallel loop.

**Trade-offs**:
- **Pro**: Algorithmically superior at large n. At n=100K, O(n log n) build + O(n log n) queries vs O(n²) brute force.
- **Pro**: Well-tested crates with established Criterion benchmarks.
- **Con**: New dependency (kiddo: ~3MB, C-FFI-free but significant crate). Requires vetting for correctness and version pinning.
- **Con**: Tree build is sequential (must finish before parallel query). Build cost at n=10K is ~O(n log n) ≈ 130K ops — trivial, but at n=100K it becomes ~1.3M ops.
- **Con**: At n=10K specifically, brute force + SIMD may match or beat a tree (tree traversal overhead, cache misses in tree nodes).
- **Con**: kiddo/nabo return approximate neighbors for some configurations — must verify exact-kNN mode.

**Expected speedup**: 3–10× at n=10K, 10–30× at n=100K. **Highest ceiling but highest implementation risk and dependency cost.**

**Recommendation for experiment budget**: Run Directions 1 and 2 as the primary experiment (no new dependencies, 3-hour budget is feasible). Run Direction 3 as a stretch goal at n ≤ 10K only, or leave for a follow-up research cycle. The issue says Criterion benchmarks must stay at n ≤ 10K, so the KD-tree advantage at n=100K cannot be statistically validated within budget anyway.

---

## Success Criteria

1. **Primary**: A Criterion benchmark at n=10K, k=15 shows ≥ 1.5× speedup on the trustworthiness wall-time measurement with 95% CI below 1.0 (i.e., new variant is faster with high confidence).
2. **Correctness**: All existing trustworthiness unit tests pass (`t_tw_01` through `t_tw_07`). The sklearn parity test (`test_trustworthiness::sklearn_parity_synthetic`) passes with `|Δ| < 1e-6`.
3. **Scalability**: Single-run smoke tests at n=50K and n=100K show proportional wall-time reduction (not worse than current at large scale).
4. **No new correctness risk**: The exact kNN set is identical to the current implementation for all test cases (deterministic, same tie-breaking).
5. **Stretch**: The `tw_profiler` step-level fraction for y_heap drops from 70.3% to ≤ 40% (indicating the improvement is genuine, not a measurement artifact).

---

## Metric Context

### Applicable Metrics from `src/metrics.rs`

The research question touches only the **Performance** quality dimension.

| Metric | Quality Dimension | Function | Current Threshold | Applies Here? |
|--------|-----------------|----------|------------------|--------------|
| `trustworthiness` | Performance | `trustworthiness(x, y, k) -> f64` | No pass/fail threshold; it is the *measurement target* | Yes — the function being optimized |
| `max_eigenpair_residual` | Accuracy | `max_eigenpair_residual(...)` | Dense: 1e-6; LOBPCG: 1e-5; rSVD: 1e-2 | No — eigensolver metric, not relevant |
| `orthogonality_error` | Accuracy | `orthogonality_error(...)` | 1e-8 | No |
| `sign_agnostic_max_error` | Parity | `sign_agnostic_max_error(...)` | 5e-3 | No |
| `subspace_gram_det` | Parity | `subspace_gram_det(...)` | 0.95 | No |

### Correctness Constraint (Not a Threshold, But Binding)

The sklearn parity test (`test_trustworthiness.rs::sklearn_parity_synthetic`) asserts `|rust_T − sklearn_T| < 1e-6`. Any optimization that changes the kNN set (even one neighbor for one point) would change the score and could violate this bound. **Exact kNN is non-negotiable.**

### Gap

There is no canonical "performance threshold" for how fast `trustworthiness()` must run. The research outcome will be evaluated by Criterion speedup ratio and step-level profiler fractions, not by an absolute timing threshold. No metric in `src/metrics.rs` captures wall-time or throughput. This is a measurement gap — the performance dimension has one function (`trustworthiness`) but no defined service level.
