# Implementation Plan: groupB — Y-Heap Bottleneck Optimization Variants

## Summary

Implements all Rust code changes for the heap bottleneck experiment:

1. `profiling = []` feature flag in `Cargo.toml`
2. Step-timing instrumentation (`x_dist`, `x_sort`, `y_heap`, `penalty`) gated on `#[cfg(feature = "profiling")]` inside `trustworthiness()` and all three variant functions
3. Three variant functions in `src/metrics.rs`: `trustworthiness_heap_reuse`, `trustworthiness_flat_partial`, `trustworthiness_flat_simd`
4. A `dist_sq_2d_avx2_batch` kernel (d_y=2 specialization) for the flat_simd variant
5. Extended `lib.rs` exports so variant functions are accessible under `--features cli,profiling`
6. `--variant` flag in `src/bin/tw_profiler.rs` dispatching to any of the four implementations
7. Correctness tests in `src/metrics.rs` asserting `|ΔT| < 1e-12` for all variants across 7 data scenarios

`cargo test --features testing` must pass all new and existing tests before this group is considered complete.

---

## Proposed Architecture

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 45, 'rankSpacing': 55, 'curve': 'basis'}}}%%
flowchart TB
    %% CLASS DEFINITIONS %%
    classDef cli fill:#1a237e,stroke:#7986cb,stroke-width:2px,color:#fff;
    classDef stateNode fill:#004d40,stroke:#4db6ac,stroke-width:2px,color:#fff;
    classDef handler fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff;
    classDef phase fill:#6a1b9a,stroke:#ba68c8,stroke-width:2px,color:#fff;
    classDef newComponent fill:#2e7d32,stroke:#81c784,stroke-width:2px,color:#fff;
    classDef output fill:#00695c,stroke:#4db6ac,stroke-width:2px,color:#fff;
    classDef detector fill:#b71c1c,stroke:#ef5350,stroke-width:2px,color:#fff;
    classDef terminal fill:#1a237e,stroke:#7986cb,stroke-width:2px,color:#fff;

    START([START: tw_profiler invocation])

    subgraph CLI ["● tw_profiler.rs — CLI Layer"]
        direction TB
        VarFlag["★ --variant flag<br/>━━━━━━━━━━<br/>baseline | heap_reuse<br/>flat_partial | flat_simd"]
        Dispatch{"★ variant match<br/>━━━━━━━━━━<br/>select fn to call"}
    end

    subgraph Baseline ["trustworthiness() — Baseline"]
        direction LR
        B_XD["● x_dist<br/>━━━━━━━━━━<br/>COMB_DIST_X fill<br/>AVX2 dispatch"]
        B_XS["● x_sort<br/>━━━━━━━━━━<br/>select_nth_unstable_by(k)<br/>knn_x_set HashSet"]
        B_YH["● y_heap<br/>━━━━━━━━━━<br/>BinaryHeap::with_capacity(k+1)<br/>per-row alloc+fill+pop"]
        B_P["● penalty<br/>━━━━━━━━━━<br/>rank accumulation u64<br/>row penalty sum"]
        B_XD --> B_XS --> B_YH --> B_P
    end

    subgraph HeapReuse ["★ trustworthiness_heap_reuse()"]
        direction LR
        HR_XD["x_dist (identical)"]
        HR_XS["x_sort (identical)"]
        HR_YH["★ y_heap reuse<br/>━━━━━━━━━━<br/>thread_local BinaryHeap<br/>clear() — no realloc"]
        HR_P["penalty (identical)"]
        HR_XD --> HR_XS --> HR_YH --> HR_P
    end

    subgraph FlatPartial ["★ trustworthiness_flat_partial()"]
        direction LR
        FP_XD["x_dist (identical)"]
        FP_XS["x_sort (identical)"]
        FP_YH["★ y_flat introselect<br/>━━━━━━━━━━<br/>thread_local Vec dist+idx<br/>select_nth_unstable_by(k)"]
        FP_P["penalty (identical)"]
        FP_XD --> FP_XS --> FP_YH --> FP_P
    end

    subgraph FlatSimd ["★ trustworthiness_flat_simd()"]
        direction LR
        FS_XD["x_dist (identical)"]
        FS_XS["x_sort (identical)"]
        FS_K{"AVX2 + d_y==2<br/>+ standard_layout?"}
        FS_AVX["★ dist_sq_2d_avx2_batch<br/>━━━━━━━━━━<br/>2 pts/iter YMM registers<br/>_mm256_hadd_pd"]
        FS_SC["scalar fallback"]
        FS_YH["★ y_flat introselect<br/>━━━━━━━━━━<br/>select_nth_unstable_by(k)"]
        FS_P["penalty (identical)"]
        FS_XD --> FS_XS --> FS_K
        FS_K -->|"yes"| FS_AVX --> FS_YH
        FS_K -->|"no"| FS_SC --> FS_YH
        FS_YH --> FS_P
    end

    subgraph Profiling ["★ #[cfg(feature = profiling)] instrumentation"]
        TIM["eprintln!(\"[timing:{step}] {ns}\")<br/>━━━━━━━━━━<br/>Instant::now() + elapsed().as_nanos()<br/>emitted once per fn call after parallel section"]
    end

    subgraph Output ["Output"]
        JSON["profiler JSON output<br/>━━━━━━━━━━<br/>step_timing map<br/>parsed by parse_step_timing()"]
    end

    COMPLETE([COMPLETE])

    START --> VarFlag --> Dispatch
    Dispatch -->|"baseline"| Baseline
    Dispatch -->|"heap_reuse"| HeapReuse
    Dispatch -->|"flat_partial"| FlatPartial
    Dispatch -->|"flat_simd"| FlatSimd
    Baseline --> Profiling
    HeapReuse --> Profiling
    FlatPartial --> Profiling
    FlatSimd --> Profiling
    Profiling --> JSON --> COMPLETE

    class START,COMPLETE terminal;
    class VarFlag newComponent;
    class Dispatch newComponent;
    class B_XD,B_XS,B_YH,B_P handler;
    class HR_XD,HR_XS,HR_YH,HR_P handler;
    class FP_XD,FP_XS,FP_YH,FP_P handler;
    class FS_XD,FS_XS,FS_YH,FS_P handler;
    class FS_K stateNode;
    class FS_AVX,FS_SC newComponent;
    class HR_YH,FP_YH newComponent;
    class TIM newComponent;
    class JSON output;
```

**Lens Used:** Process Flow — The plan centers on runtime dispatch through four algorithm paths sharing a common phase structure, with a new decision point (`--variant`) and new per-path execution branches.

**Color Legend:**
| Color | Category | Description |
|-------|----------|-------------|
| Dark Blue | Terminal | Start and end states |
| Green | New Component | ★ New functions, kernels, flags, instrumentation |
| Orange | Handler | Existing processing steps (unchanged logic) |
| Teal | State | Decision points and routing |
| Dark Teal | Output | JSON profiler output artifact |

---

## Tests

These tests must be written first. They will fail until the variant functions are implemented. All run under `cargo test --features testing`.

### Test data helpers (in `src/metrics.rs` `#[cfg(test)]` module)

Extract repeated data construction into private helper functions:

```rust
fn tw_data_perfect_preservation() -> (Array2<f64>, Array2<f64>, usize) { /* from t_tw_01 */ }
fn tw_data_random_n20_k5(seed: u64) -> (Array2<f64>, Array2<f64>, usize) { /* from t_tw_02 */ }
fn tw_data_hand_check_4pt() -> (Array2<f64>, Array2<f64>, usize) { /* from t_tw_03 */ }
fn tw_data_max_k_n20() -> (Array2<f64>, Array2<f64>, usize) { /* from t_tw_04, k=n/2-1 */ }
fn tw_data_n30_k5(seed: u64) -> (Array2<f64>, Array2<f64>, usize) { /* from t_tw_07 */ }
fn tw_data_n50_k3(seed: u64) -> (Array2<f64>, Array2<f64>, usize) { /* n=50, k=3, from t_tw_08 */ }
fn tw_data_n50_k7(seed: u64) -> (Array2<f64>, Array2<f64>, usize) { /* n=50, k=7, from t_tw_08 */ }
```

### Correctness tests (21 tests: 7 per variant)

```rust
// heap_reuse
#[test] fn t_tw_heap_reuse_01() { let (x,y,k) = tw_data_perfect_preservation();
    assert!((trustworthiness_heap_reuse(x.view(),y.view(),k) - trustworthiness(x.view(),y.view(),k)).abs() < 1e-12); }
#[test] fn t_tw_heap_reuse_02() { /* random_n20_k5 */ }
#[test] fn t_tw_heap_reuse_03() { /* hand_check_4pt */ }
#[test] fn t_tw_heap_reuse_04() { /* max_k_n20 */ }
#[test] fn t_tw_heap_reuse_05() { /* n30_k5 */ }
#[test] fn t_tw_heap_reuse_06() { /* n50_k3 */ }
#[test] fn t_tw_heap_reuse_07() { /* n50_k7 */ }

// flat_partial (same 7 data scenarios, different fn)
#[test] fn t_tw_flat_partial_01() { ... }  // through _07

// flat_simd (same 7 data scenarios, different fn)
#[test] fn t_tw_flat_simd_01() { ... }  // through _07
```

---

## Implementation Steps

### Step 1 — Add `profiling` feature to `Cargo.toml`

In `Cargo.toml` `[features]` section, add one line after `testing`:

```toml
[features]
testing  = ["dep:serde"]
profiling = []
cli      = ["dep:ndarray-npy", "dep:pico-args", "dep:serde_json", "dep:libc"]
```

Verify all three check passes:
```
cargo check
cargo check --features testing
cargo check --features profiling
```

---

### Step 2 — Add step-timing instrumentation to `trustworthiness()` in `src/metrics.rs`

Add timing captures around each of the four steps, gated on `#[cfg(feature = "profiling")]`. The timing is recorded **after** the Rayon parallel section completes (one emit per call, not per thread).

Emit format must match `tw_profiler.rs::parse_step_timing` exactly: `[timing:<step>] <ns>`.

Pattern for each step boundary:

```rust
#[cfg(feature = "profiling")]
let t_start = std::time::Instant::now();

// ... the step's parallel work ...

#[cfg(feature = "profiling")]
eprintln!("[timing:x_dist] {}", t_start.elapsed().as_nanos());
```

Steps to instrument in `trustworthiness()`:
- `x_dist` — after the COMB_DIST_X fill parallel section
- `x_sort` — after the `select_nth_unstable_by` + `knn_x_set` parallel section
- `y_heap` — after the BinaryHeap fill parallel section
- `penalty` — after the penalty accumulation parallel section (before the final normalization)

**Do not change any logic, signatures, or existing behavior.** Only add/remove `#[cfg(feature = "profiling")]` blocks.

---

### Step 3 — Implement `trustworthiness_heap_reuse` in `src/metrics.rs`

Location: immediately after `trustworthiness()` closes.

Key differences from baseline:
- Declare a module-level (inside the variant function body) `thread_local!` for the heap:
  ```rust
  thread_local! {
      static Y_HEAP: RefCell<BinaryHeap<(u64, usize)>> =
          const { RefCell::new(BinaryHeap::new()) };
  }
  ```
- Inside the parallel map closure for the y_heap step:
  ```rust
  Y_HEAP.with(|cell| {
      let mut heap = cell.borrow_mut();
      if heap.capacity() < k + 1 {
          heap.reserve(k + 1 - heap.capacity());
      }
      heap.clear(); // No reallocation — preserves allocation
      for j in 0..n {
          if j == i { continue; }
          let d: f64 = yi.iter().zip(y.row(j).iter())
              .map(|(&a, &b)| (a - b) * (a - b)).sum();
          heap.push((d.to_bits(), j));
          if heap.len() > k { heap.pop(); }
      }
      // Collect neighbors WITHOUT draining (preserves allocation for next row)
      let knn_y_set: HashSet<usize> = heap.iter().map(|&(_, j)| j).collect();
      // ... identical penalty accumulation ...
  })
  ```
- x_dist, x_sort, penalty steps: **identical** to baseline.
- Profiling instrumentation: identical step names, same emit pattern.

Self-exclusion is identical to baseline: `if j == i { continue; }` in the heap fill loop.

---

### Step 4 — Implement `trustworthiness_flat_partial` in `src/metrics.rs`

Location: immediately after `trustworthiness_heap_reuse()` closes.

Declare two thread-locals inside the function body:
```rust
thread_local! {
    static COMB_DIST_Y:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
    static COMB_INDICES_Y: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}
```

Inside the parallel map closure, the y_heap step becomes:
```rust
COMB_DIST_Y.with(|cd| {
    let mut dist_y = cd.borrow_mut();
    dist_y.clear();
    dist_y.resize(n, 0.0);
    let yi = y.row(i);
    for j in 0..n {
        dist_y[j] = yi.iter().zip(y.row(j).iter())
            .map(|(&a, &b)| (a - b) * (a - b)).sum();
    }
    dist_y[i] = f64::INFINITY;  // self-exclusion

    COMB_INDICES_Y.with(|ci| {
        let mut indices_y = ci.borrow_mut();
        indices_y.clear();
        indices_y.extend(0..n);

        // RT-G: comparator must replicate BinaryHeap tie-breaking exactly.
        // Lowest distance wins; on equal distance, lowest index wins.
        // (BinaryHeap pops max: highest (d.to_bits(), j) → on equal d, highest j evicted → lowest j retained)
        indices_y.select_nth_unstable_by(k, |&a, &b| {
            dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b))
        });

        // indices_y[..=k]: k+1 partition. Self (dist=∞) is past position k.
        let knn_y_partition = &indices_y[..=k];
        // ... penalty accumulation using knn_y_partition ...
    });
});
```

x_dist and x_sort steps: **identical** to baseline (use the same COMB_DIST_X / COMB_INDICES thread-locals — but these are declared in `trustworthiness()`; for variants, redeclare identical thread-locals inside the variant function body with the same initialization pattern).

Penalty accumulation: **identical** logic to baseline, iterating over `knn_y_partition` instead of heap entries.

---

### Step 5 — Add `dist_sq_2d_avx2_batch` kernel in `src/metrics.rs`

Add before `trustworthiness_flat_simd` (after `trustworthiness_flat_partial`). Compile-gated:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
unsafe fn dist_sq_2d_avx2_batch(
    yi: &[f64],       // query row: exactly 2 elements
    y_flat: &[f64],   // full Y matrix, row-major, stride 2
    n: usize,
    out: &mut [f64],  // output: n distances
) {
    use std::arch::x86_64::*;
    // Broadcast query point: [yi[0], yi[1], yi[0], yi[1]] in memory order
    // _mm256_set_pd args are (e3,e2,e1,e0): e0=lowest address
    let yi_bc = _mm256_set_pd(yi[1], yi[0], yi[1], yi[0]);

    let mut j = 0usize;
    while j + 1 < n {
        // Load 2 target points: [yj[0], yj[1], yj+1[0], yj+1[1]]
        let yj_pair = _mm256_loadu_pd(y_flat.as_ptr().add(j * 2));
        let diff = _mm256_sub_pd(yi_bc, yj_pair);
        let sq   = _mm256_mul_pd(diff, diff);
        // _mm256_hadd_pd(sq, sq):
        //   lower 128: [sq[0]+sq[1], sq[0]+sq[1]] → dist_j at element 0
        //   upper 128: [sq[2]+sq[3], sq[2]+sq[3]] → dist_{j+1} at element 0 of upper half
        let hadd = _mm256_hadd_pd(sq, sq);
        out[j]     = _mm_cvtsd_f64(_mm256_castpd256_pd128(hadd));
        out[j + 1] = _mm_cvtsd_f64(_mm256_extractf128_pd(hadd, 1));
        j += 2;
    }
    // Scalar tail for odd n
    while j < n {
        out[j] = yi.iter()
            .zip(y_flat[j * 2..j * 2 + 2].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        j += 1;
    }
}
```

---

### Step 6 — Implement `trustworthiness_flat_simd` in `src/metrics.rs`

Identical to `trustworthiness_flat_partial` except the y-distance fill step is:

```rust
let d_y = y.ncols();

#[cfg(target_arch = "x86_64")]
let use_avx2_y = is_x86_feature_detected!("avx2") && d_y == 2 && y.is_standard_layout();
#[cfg(not(target_arch = "x86_64"))]
let use_avx2_y = false;

// ... inside parallel map closure, inside COMB_DIST_Y.with(...):
dist_y.clear();
dist_y.resize(n, 0.0);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
if use_avx2_y {
    let y_flat = y.as_slice().unwrap();         // safe: is_standard_layout checked above
    let yi_slice = &y_flat[i * 2..(i + 1) * 2];
    unsafe { dist_sq_2d_avx2_batch(yi_slice, y_flat, n, &mut dist_y); }
} else {
    let yi = y.row(i);
    for j in 0..n {
        dist_y[j] = yi.iter().zip(y.row(j).iter())
            .map(|(&a, &b)| (a - b) * (a - b)).sum();
    }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
{
    let yi = y.row(i);
    for j in 0..n {
        dist_y[j] = yi.iter().zip(y.row(j).iter())
            .map(|(&a, &b)| (a - b) * (a - b)).sum();
    }
}

dist_y[i] = f64::INFINITY;
```

Everything else (introselect, penalty, profiling) is identical to `trustworthiness_flat_partial`.

---

### Step 7 — Extend `lib.rs` exports

In `src/lib.rs`, the `#[cfg(all(feature = "cli", not(feature = "testing")))]` block currently exports only `trustworthiness`. Extend it to include all three variants so that `--features cli,profiling` (without `testing`) compiles cleanly:

**Current:**
```rust
#[cfg(all(feature = "cli", not(feature = "testing")))]
pub use crate::metrics::trustworthiness;
```

**Replace with:**
```rust
#[cfg(all(feature = "cli", not(feature = "testing")))]
pub use crate::metrics::{
    trustworthiness,
    trustworthiness_heap_reuse,
    trustworthiness_flat_partial,
    trustworthiness_flat_simd,
};
```

The `#[cfg(feature = "testing")]` block already does `pub use crate::metrics::*`, so the variant functions are automatically exported when `testing` is active. No change needed there.

**Also add DELIV-3C-4:** Under the existing `#[cfg(feature = "testing")]` block where specific symbols are re-exported via wrapper fns, the variant functions should be accessible via the `pub use crate::metrics::*` glob. Verify this is sufficient. If the testing block uses explicit re-exports rather than a glob, add explicit lines:

```rust
#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::metrics::trustworthiness_heap_reuse;
#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::metrics::trustworthiness_flat_partial;
#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::metrics::trustworthiness_flat_simd;
```

(If the existing block already does `pub use crate::metrics::*`, these are redundant and should be omitted.)

---

### Step 8 — Add `--variant` flag to `src/bin/tw_profiler.rs`

**Parse the flag** (after existing `warmup` parse, before `stderr_capture`):

```rust
let variant: String = pargs.opt_value_from_str("--variant")?.unwrap_or_else(|| "baseline".to_string());
```

**Replace the warmup loop** to dispatch via the variant:

```rust
macro_rules! call_variant {
    ($fn:expr) => {
        std::hint::black_box($fn(x.view(), y.view(), k))
    };
}
for _ in 0..warmup {
    match variant.as_str() {
        "baseline"     => { call_variant!(spectral_init::trustworthiness); }
        "heap_reuse"   => { call_variant!(spectral_init::trustworthiness_heap_reuse); }
        "flat_partial" => { call_variant!(spectral_init::trustworthiness_flat_partial); }
        "flat_simd"    => { call_variant!(spectral_init::trustworthiness_flat_simd); }
        other => return Err(format!("unknown variant: {other}").into()),
    }
}
```

**Replace the timed loop** with the same match dispatch (without `black_box`).

The `--stderr-capture` redirect already intercepts `eprintln!` on stderr (fd 2) and `parse_step_timing()` already parses `[timing:<step>] <ns>` — no changes needed to either.

Also include `variant` in the JSON output fields (alongside `k`, `iters`, `warmup`) so runs are self-documenting.

---

### Step 9 — Add correctness tests to `src/metrics.rs`

Add to the `#[cfg(test)]` module inside `src/metrics.rs`.

First, add the 7 data-generating helpers (factored from t_tw_01–04, t_tw_07, t_tw_08):

```rust
fn tw_case_01() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* perfect preservation grid, n=20 k=5 */ }
fn tw_case_02() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* random n=20 k=5 seed 99 */ }
fn tw_case_03() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* 4-pt hand check k=1 */ }
fn tw_case_04() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* n=20 k=9 max k */ }
fn tw_case_05() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* n=30 k=5 */ }
fn tw_case_06() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* n=50 k=3 */ }
fn tw_case_07() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) { /* n=50 k=7 */ }
```

Then add 21 tests (3 variants × 7 cases). Example for heap_reuse:

```rust
#[test]
fn t_tw_heap_reuse_01() {
    let (x, y, k) = tw_case_01();
    let delta = (trustworthiness_heap_reuse(x.view(), y.view(), k)
               - trustworthiness(x.view(), y.view(), k)).abs();
    assert!(delta < 1e-12, "heap_reuse 01: |ΔT| = {delta} ≥ 1e-12");
}
// ... _02 through _07 ...

#[test] fn t_tw_flat_partial_01() { ... }  // same pattern
#[test] fn t_tw_flat_simd_01()    { ... }  // same pattern
```

**RT-G guard:** If any correctness test fails with `|ΔT| ≥ 1e-12`, the root cause is the comparator in `select_nth_unstable_by`. The fix: verify that `.total_cmp(&dist_y[b]).then(a.cmp(&b))` (ascending distance, ascending index on ties) exactly replicates the BinaryHeap eviction order.

BinaryHeap reasoning: stores `(d.to_bits(), j)`, pops the **max** tuple when `len > k`. On equal `d.to_bits()`, the pair with higher `j` is larger → higher `j` evicted → lower `j` is retained. The comparator `.then(a.cmp(&b))` orders by ascending index → lower index is "smaller" → lower index is partitioned to `[..=k]` on ties. This replicates baseline exactly.

---

## Verification

1. **Feature check passes:**
   ```bash
   cargo check
   cargo check --features testing
   cargo check --features profiling
   cargo check --features cli
   cargo check --features cli,profiling
   cargo check --features cli,testing,profiling
   ```

2. **All tests pass:**
   ```bash
   cargo test --features testing
   ```
   Expected: all existing t_tw_01–08 pass, all 21 new variant correctness tests pass.

3. **Profiler binary compiles and runs:**
   ```bash
   cargo build --features cli,profiling
   ./target/debug/tw_profiler --x <X.npy> --y <Y.npy> --variant heap_reuse
   ./target/debug/tw_profiler --x <X.npy> --y <Y.npy> --variant flat_partial
   ./target/debug/tw_profiler --x <X.npy> --y <Y.npy> --variant flat_simd
   ```
   With `--stderr-capture tmp.txt`, the captured file must contain lines matching `[timing:x_dist] <ns>`, `[timing:x_sort] <ns>`, `[timing:y_heap] <ns>`, `[timing:penalty] <ns>`.

4. **AVX2 kernel correctness (flat_simd only):**
   Run on a 2D Y matrix with standard layout. Compare output against flat_partial result: should match to `< 1e-12` (guaranteed by the correctness tests that use 2D Y data).

5. **No regression on existing tests:**
   ```bash
   cargo test
   cargo test --features testing
   ```
   Both must pass with zero failures.
