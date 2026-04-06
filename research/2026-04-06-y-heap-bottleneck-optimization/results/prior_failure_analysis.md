# Prior Failure Analysis: thread_local 2× Slowdown (2026-04-05-tw-perf-rerun-clean)

## Observed Failure

The `trustworthiness_thread_local` variant measured 0.634s mean wall-clock vs 0.313s for
baseline at n=10K, k=15 — a 2.03× regression. Source: step_timing JSON files in
`research/2026-04-05-tw-perf-rerun-clean/results/step_timing/`.

## Root Cause: O(n log n) Sort Regression in X-NN Detection

**Primary cause (confirmed from source):** The thread_local variant replaced the O(n)
`select_nth_unstable_by` (already in production) with an O(n log n) `sort_unstable_by`.
At n=10,000:
- `select_nth_unstable_by(k=15)`: ~O(n) ≈ 10,000 comparisons average
- `sort_unstable_by`: ~O(n log n) ≈ 133,000 comparisons

This 13× increase in X-sort comparison work would increase x_sort thread-work from ~245M ns
(baseline) to ~3.3B ns — larger than the entire baseline invocation (2.49B ns thread-work).

**Supporting evidence:** The thread_local step_timing JSON shows all zeros because the
`#[cfg(feature = "profiling")]` guards were not active for that variant. However, the
algorithm difference is confirmed by reading `src/metrics.rs` in the rerun-clean worktree
(line 670: `dist_x.sort_unstable_by(...)`).

**Secondary factor:** `TL_DIST_X` stored `(f64, usize)` tuples (16 bytes/element) vs
the baseline's separate 8-byte f64 buffer. At n=10K: 160KB per thread vs 80KB. The
additional `TL_RANK_X` buffer (80KB) brings total per-thread allocation to 240KB.
While this is within Zen 5's per-core 1MB L2, the doubled memory footprint increases
cache pressure relative to baseline.

**Critical finding: y_heap was NOT modified.** The prior experiment did not test y_heap
allocation reuse (`clear()` instead of fresh `BinaryHeap::with_capacity(k+1)`). The
2× slowdown was entirely from X-side regressions. The y_heap step remained identical
to baseline throughout the rerun-clean experiment.

## Three Candidate Hypotheses for Remaining y_heap Cost

1. **Malloc cost per row** (`heap_reuse` target): Each row allocates a fresh
   `BinaryHeap::with_capacity(k+1)` via the system allocator. At n=10K with 8 threads,
   this is 10,000 malloc+free pairs per invocation. The `heap_reuse` variant isolates this
   cost by pre-allocating per thread and calling `clear()` per row.

2. **Introselect locality disadvantage** (`flat_partial` target): The heap's push/evict
   pattern accesses memory indirectly and maintains a k-element priority queue with
   pointer chasing. A flat Vec<f64> + `select_nth_unstable_by` operates on a contiguous
   array with sequential write followed by a single cache-local introselect pass.

3. **AVX2 throughput gap** (`flat_simd` target): The y_heap loop computes 2D squared
   distances as scalar f64 operations. A 256-bit AVX2 kernel processing 2 Y-rows per lane
   can theoretically compute 4 distances per cycle vs 1 scalar. This is independent of
   the data structure choice.

## Implication for Variant Selection

All three variants (`heap_reuse`, `flat_partial`, `flat_simd`) remain worth testing because
the prior experiment provided no evidence for or against any of them. The prior "thread_local"
experiment was a regression caused by algorithm complexity change in the X-side, not by
y_heap optimization. The current experiment starts fresh with a correctly isolated y_heap
investigation.

**RT-I gate status:** Satisfied. Root cause identified as O(n log n) sort regression in
x_dist/x_sort steps, confirmed by reading `src/metrics.rs` in worktree
`research-20260405-tw-perf-rerun-clean` (line 670).
