# Scope Report: Validate Trustworthiness Step-Timing on MERFISH Real Data

## Research Question

Does the trustworthiness runtime step-timing breakdown observed on synthetic data (d_x=10, uniform/Gaussian) hold on real UMAP embeddings from MERFISH spatial transcriptomics data? Specifically: is X-space (x_dist + x_sort) still ~61% of total runtime on MERFISH data at practical d_x, validating the conclusion from PR #238 that X-space ANN is the productive next optimization target?

---

## Known / Unknown Matrix

| Category | Known | Unknown |
|----------|-------|---------|
| Current behavior | Step fractions on synthetic n=10K (post-flat_simd): x_dist~32%, x_sort~24%, y_dist~28%, penalty~16% | Step fractions on real MERFISH 10K or 50K data |
| Performance | `tw_profiler` binary exists; MERFISH fixtures confirmed present; builds with `--features cli,profiling` | Whether MERFISH clustering geometry shifts x/y balance; effect of MERFISH PCA d_x on x_dist share |
| Edge cases | y_heap was 70% before flat_simd optimization; flat_simd dropped it to 28% | Whether MERFISH Y-space geometry (strongly clustered 2D) changes cache behavior enough to shift y_dist fraction meaningfully |
| Prior work | PR #238 synthetic-only caveat explicitly documented; 2026-04-06 experiments exist for n=10K Gaussian/uniform | No MERFISH profiling run has ever been executed; MERFISH H5 gate was not run in 2026-04-05 experiment |
| Fixture data | merfish_n10k_x.npy (3.9 MB), merfish_n10k_y.npy (157 KB), n50k variants all confirmed present | Exact d_x of MERFISH PCA reduction embedded in fixtures (expected ~10, unverified) |

---

## Prior Art in Codebase

### `tw_profiler` Binary (`src/bin/tw_profiler.rs`)
Fully functional profiling binary. Required flags: `--x PATH --y PATH --output PATH`. Optional: `--k N` (default 15), `--iters N` (default 5), `--warmup N` (default 2), `--stderr-capture PATH`.

**Critical: `--stderr-capture` is mandatory for step timing.** Without it, no step-level JSON is produced — the `step_timing` key is omitted entirely from output. The binary does NOT include a `--variant` flag (scripts pass it but the binary ignores or rejects it; variant selection is a build-time concern).

Build command for step-timing data:
```
cargo build --release --features cli,profiling --bin tw_profiler
```

Run command:
```
./target/release/tw_profiler \
  --x research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy \
  --y research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy \
  --output temp/merfish_10k_profile.json \
  --k 15 --iters 5 --warmup 2 \
  --stderr-capture temp/merfish_10k_stderr.txt
```

### MERFISH Fixtures
All four fixtures confirmed present:

| File | Size | Notes |
|------|------|-------|
| `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | 3.9 MB | ~10K cells, PCA-reduced X; d_x inferred from size ~= 10 |
| `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | 157 KB | ~10K cells, 2D UMAP embedding |
| `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | 20 MB | ~50K cells |
| `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | 782 KB | ~50K cells |

d_x estimate from file size: 3.9 MB / (10K rows × 8 bytes/f64) ≈ 48 dimensions — **likely higher than 10**. If f32: ≈ 97 dims. If f64: ≈ 48 dims. The expected d_x=10 claim in the issue may be incorrect; actual d_x is unverified without reading the npy header.

### Existing Synthetic-Data Baselines
From `research/2026-04-06-162144/results/profiler_baseline_n10000.json` (post-flat_simd production code, n=10K Gaussian):
- `y_dist` (formerly `y_heap`): **27.6%**
- `x_dist`: **31.9%**
- `x_sort`: **24.2%**
- `penalty`: **16.2%**
- X-space total (x_dist + x_sort): **56.1%**

Pre-flat_simd baseline from `research/2026-04-05-tw-perf-rerun-clean/results/gaussian_n10000_baseline.json`:
- `y_heap`: 70.3%, `x_dist`: 13.0%, `x_sort`: 10.0%, `penalty`: 6.4%, `x_knn_set`: 0.4%

The issue's stated 61% X-space figure appears to be from a slightly different measurement than the 2026-04-06 56% figure; the difference may reflect n=100K vs n=10K. At larger n, the penalty step's O(n²·k) cost could reduce proportionally more from the total, causing x_dist+x_sort to grow slightly — consistent with going from 56% (n=10K) to 61% (n=100K).

### Profiling Step Names (Current Source)
The current `src/metrics.rs` emits `[timing:y_dist]` (renamed from `y_heap` in historical results). The `tw_profiler` JSON key will be `y_dist`. Analysis scripts using `y_heap` as the key will fail on new results.

### Complexity Reference
| Step | Per-row | Total | SIMD path |
|------|---------|-------|-----------|
| x_dist | O(n·d_x) | O(n²·d_x) | AVX2+FMA when d_x ≥ 10 (8 f64/cycle) |
| x_sort | O(n) avg | O(n²) avg | None (introselect) |
| y_dist | O(n·d_y) + O(n) | O(n²·d_y) | Batch 2D AVX2 kernel when d_y=2 (2 points/iteration) |
| penalty | O(k·n) | O(n²·k) | None |

**Note:** AtomicU64 counters accumulate across all Rayon threads; ratios between steps are valid but absolute values are sum-of-threads-time, not wall-clock.

### Cargo Features
- `profiling = []` — gates all profiling instrumentation in `src/metrics.rs`
- `cli = [dep:ndarray-npy, dep:pico-args, dep:serde_json, dep:libc]` — required for `tw_profiler` binary
- Both required simultaneously for step-timing data

---

## External Research

### MERFISH d_x in Practice
Published MERFISH preprocessing pipelines (Scanpy, Giotto, Squidpy) typically use **8–20 PCA components** before UMAP. The Giotto MERFISH hypothalamic tutorial uses `dimensions_to_use = 1:8` (d_x=8). Scanpy tutorials use `n_pcs = 15–20`. Raw panel sizes are 155–500 genes, but PCA reduces to d_x = 8–20 before UMAP.

**Implication**: The file-size-based d_x estimate (~48 dims at f64) is inconsistent with this range. Possible explanations: (1) the fixture uses f32 not f64 (→ ~97 dims, even more inconsistent); (2) d_x is actually ~48, meaning a larger PCA was used for this particular dataset. Actual d_x must be determined by reading the npy header.

### UMAP Y-Space Geometry on Biological Data
MERFISH UMAP embeddings are **strongly clustered** — whole-brain atlases produce 5,322+ distinct cell-type clusters in 2D. This differs from the synthetic Gaussian mixture (8 clusters). Effects on y_dist runtime:
- **Asymptotic cost is unchanged**: d_y=2 pairwise distances are O(n²·2) regardless of clustering
- **Cache effects**: if cells are indexed by cluster, sequential index pairs are more likely to be spatially close → marginally better L1/L2 cache behavior for Y coordinate loads. Estimated effect: a few percent, not a factor-of-2
- **No exploitable sparsity**: full O(n²) sweep is required for trustworthiness (all pairs, no ANN approximation)

### x_dist / y_dist Ratio Scaling
FLOP ratio x_dist:y_dist scales proportionally with d_x:d_y. At d_x=10, d_y=2, the FLOP ratio is 5:1 but the AVX2 x_dist kernel is 8-wide and the batched y_dist kernel processes 2 points per loop — giving an effective throughput ratio closer to (d_x/8) : (2·d_y/4). At higher d_x (15–48), x_dist FLOPs grow linearly, pushing x_dist's share up and y_dist's share down.

**Prediction**: If d_x ≈ 48 (as file size suggests), x_dist should dominate significantly more than at d_x=10.

### ANN Recall on Biological Data
NN-Descent (UMAP's graph construction algorithm) shows degraded recall on biological data with Local Intrinsic Dimensionality (LID) > 20 and high hubness (153–181 for genomic data vs 1.01 for MNIST-Fashion). This caveat applies to any ANN-based trustworthiness speedup validated on synthetic data — but does **not** affect the exact O(n²) `tw_profiler` benchmark being proposed here.

---

## Technical Context

### Algorithm Flow
`trustworthiness()` in `src/metrics.rs:478` uses `rayon`'s parallel iterator over n rows. Each row executes the four steps sequentially using `thread_local!` scratch buffers (no per-row heap allocation). The profiling counters accumulate via `AtomicU64::fetch_add(Ordering::Relaxed)` across all threads.

### Step-Timing Reporting
After the parallel loop, the function emits to **stderr**:
```
[timing:x_dist] <nanoseconds as u64>
[timing:x_sort] <nanoseconds as u64>
[timing:y_dist] <nanoseconds as u64>
[timing:penalty] <nanoseconds as u64>
```
`tw_profiler` captures stderr via Unix `dup2` (when `--stderr-capture` is given), then regex-parses these lines to populate `step_timing` in the JSON output.

### Known Step-Name Inconsistency
Historical results files (2026-04-05, 2026-04-06) use `y_heap` as the JSON key. Current source emits `y_dist`. Any analysis script using `y_heap` must be updated to handle `y_dist` for new runs.

### No Performance Regression Thresholds
There are no wall-time pass/fail gates in CI. The profiler is a measurement tool; step-timing fractions are reported, not asserted. The existing `test_tw_profiler.rs` tests verify JSON structure only, not timing values.

---

## Hypotheses

**H1 (X-dominance holds)**: On MERFISH 10K data, X-space (x_dist + x_sort) remains ≥50% of total trustworthiness runtime. The claim from PR #238 extends to real data because the d_x/d_y ratio is similarly large on real MERFISH (PCA-reduced X vs 2D Y).

**H2 (X-dominance increases)**: If d_x in the MERFISH fixture is higher than 10 (as file size suggests, ~48 dims), x_dist's fraction grows proportionally — potentially 50–65% of total runtime at d_x=48 — making the X-space ANN target even more attractive for real data than synthetic-data profiling implies.

**H3 (Y-space shifts due to clustering geometry)**: Clustered MERFISH Y embeddings exhibit improved cache behavior during y_dist computation, reducing y_dist's fraction by 3–8% relative to synthetic uniform/Gaussian Y. This is a second-order effect and would not change the qualitative conclusion about X-dominance.

**H4 (X-dominance does NOT hold at d_x ≈ 10)**: If the MERFISH fixture was actually PCA-reduced to 8–10 dimensions (inconsistent with file size but possible if stored as f64 with padding or metadata), the x/y fraction would be consistent with synthetic results at d_x=10.

---

## Proposed Investigation Directions

### Direction 1: Direct Benchmark Run (Recommended)
Build `tw_profiler` with `--features cli,profiling`, run it directly on the four MERFISH fixtures (10K and 50K), collect step fractions with ≥3 timed iterations. First verify d_x by reading the npy array shape header before running.

**Steps:**
1. `python3 -c "import numpy as np; x=np.load('...merfish_n10k_x.npy'); print(x.shape)"` to confirm d_x
2. `cargo build --release --features cli,profiling --bin tw_profiler`
3. Run on 10K with `--iters 5 --warmup 2 --stderr-capture temp/merfish_10k_stderr.txt`
4. Optionally run on 50K (will be slower — estimate: ~25x longer than 10K due to O(n²))
5. Parse step fractions; compare against 2026-04-06 baseline

**Trade-offs**: Direct and conclusive. 50K run may take minutes; 10K is sufficient for the primary question.

### Direction 2: Analytical Prediction First
Compute theoretical FLOP ratio (d_x/d_y) from the actual fixture d_x, compare to observed synthetic fractions, predict MERFISH fractions analytically before running. Then run to validate.

**Trade-offs**: Provides a sanity check and a stronger result (prediction vs. actual). Adds ~30 minutes of analysis but strengthens the experiment.

### Direction 3: Multi-k Sweep
Run `tw_profiler` at k=5, 15, 30 on MERFISH 10K to characterize how the penalty fraction scales with k on real data (O(n²·k) for penalty; O(n²) for distances). This is secondary to the main question but useful if penalty optimization becomes relevant.

**Trade-offs**: Multiplies the run count by 3; only necessary if penalty fraction is unexpectedly large.

---

## Success Criteria

1. **d_x confirmed**: Actual shape of `merfish_n10k_x.npy` read and recorded (expected ~10, but may be ~48 based on file size)
2. **Step fractions measured**: At least 3 timed iterations on MERFISH 10K with all four step fractions present in JSON output
3. **Comparison table produced**: MERFISH 10K fractions vs. synthetic 10K fractions (2026-04-06 baseline) side by side
4. **X-dominance verdict**: Binary conclusion — does X-space (x_dist + x_sort) remain ≥50% of total on real MERFISH data?
5. **Optimization path validated or refuted**: Explicit statement on whether X-space ANN remains the productive target for real data

Stretch: Same measurement on MERFISH 50K to confirm the fractions are stable across n.

---

## Metric Context

Trustworthiness profiling falls under the **Performance** quality dimension in this codebase's three-tier framework (Accuracy / Parity / Performance).

| Metric | Dimension | Current Threshold |
|--------|-----------|-------------------|
| `trustworthiness` T(k) | Performance | No hardcoded threshold — score in (0, 1], value is reported |
| `tw_profiler` wall-time | Performance | No pass/fail gate — measured and reported only |
| Step fractions (x_dist, x_sort, y_dist, penalty) | Performance | No thresholds — research artifact, not CI gate |
| `max_eigenpair_residual` | Accuracy | Solver-dependent: 1e-6 (dense EVD), 2e-5 (LOBPCG), 1e-2 (rSVD) |
| `sign_agnostic_max_error` | Parity | 5e-3 |
| `subspace_gram_det` | Parity | 0.95 minimum |

No canonical metric from `src/metrics.rs` directly gates step-timing fractions. The profiling feature is an instrumentation tool for research; results inform future optimization decisions but are not CI-enforced.

**Gap**: There is no existing performance regression test that would catch a regression in trustworthiness wall-time. If an optimization is later added targeting x_dist, there is currently no CI gate to ensure it doesn't slow down other steps.
