# Experiment Plan: y_heap Bottleneck Optimization in Trustworthiness Computation

> Revision 3 — addresses all findings from review-design verdict REVISE (2026-04-06 14:15).
> Required revisions R1–R6 addressed; recommended revisions W1–W7 addressed;
> Red-Team decisions RT-A through RT-J made explicit below.

---

## Motivation

The `trustworthiness()` function in `src/metrics.rs` spends an estimated **70.3% of total
parallel thread-work** in the `y_heap` step — a per-row `BinaryHeap<(u64, usize)>` that scans
all n Y-space points to find the k nearest neighbors in embedding space. This was measured in
`research/2026-04-05-tw-perf-rerun-clean` at n=10K, k=15 over 30 timed iterations
(95% CI: [70.3%, 70.4%]).

**Time-base qualification (R3, RT-H):** The 70.3% figure is a **summed CPU thread-work
fraction**, not a wall-clock fraction. The `AtomicU64` step counters accumulate nanoseconds
across all Rayon worker threads. At T threads, total thread-work can be up to T× wall-clock
elapsed time. The Amdahl upper bound derived from this fraction — `1 / (1 − 0.703) ≈ 3.37×`
— assumes thread-work fraction equals wall-clock fraction, which holds only in single-threaded
execution. **In an 8-thread Rayon context, the true wall-clock speedup ceiling is unknown
without a wall-clock step decomposition.** All Amdahl-derived targets in this document are
labeled as **thread-work fraction upper bounds**, not wall-clock predictions.

Prior optimization work (PR #226/229) applied thread-local buffers and AVX2 to `x_dist`.
The current baseline incorporates those changes; `x_dist` accounts for only ~13% of
thread-work. The `y_heap` step was explicitly flagged as the unresolved dominant cost.

The prior rerun-clean experiment (2026-04-05) applied a thread-local optimization to y_heap
and measured **2× wall-clock slowdown** (thread_local: 0.634s vs baseline: 0.313s at n=10K).
This failure is a critical prior result. **The experiment must establish why it failed before
committing to the same strategy** (Phase 0).

This experiment tests three precisely isolated variants: (a) is BinaryHeap **allocation** the
bottleneck (`heap_reuse`), (b) does replacing the heap with a flat buffer + introselect help or
hurt (`flat_partial`), and (c) does a 2D AVX2 SIMD distance kernel on top of the flat buffer
deliver meaningful throughput gain (`flat_simd`). Results determine which — if any — y_heap
optimization should be shipped and whether to escalate to a KD-tree approach.

---

## Hypothesis

**Null hypothesis (H0):** Optimizing the y_heap step yields no statistically significant
wall-time speedup. Formally: the Criterion 95% CI for the wall-time speedup ratio
(T_baseline / T_flat_simd) at n=10K, k=15 contains 1.0.

**Alternative hypothesis (H1):** The combined variant (`flat_simd`: thread-local flat buffer +
`select_nth_unstable_by` + 2D AVX2 distance kernel) produces a Criterion 95% CI lower bound
strictly greater than 1.0 at n=10K, k=15 — the optimization is reliably faster than baseline.

**Stretch target (exploratory reference only, not a gate):** Point estimate ≥ 1.5×,
corresponding to ~50% y_heap thread-work reduction. This threshold is derived from the 70.3%
thread-work fraction under the single-threaded Amdahl approximation. **It is labeled as a
thread-work-fraction upper bound, not a wall-clock prediction (R3).** A positive primary result
requires only CI LB > 1.0; the stretch target is retained as an ambitious reference point.

---

## Red-Team Decisions

These decisions are made explicitly in response to the review-design revision guidance. Each is
documented as a conscious choice.

### RT-A: Hot-loop benchmark vs cold-production call (Goodhart — accepted)

Criterion's hot-loop throughput is accepted as a proxy for deployment value. The hot-loop
speedup is a necessary but not sufficient condition for production speedup; it is sufficient for
a go/no-go decision given that this codebase calls `trustworthiness()` in evaluation loops
(multiple invocations per pipeline run), not only as single cold calls. **This limitation is
declared in §Threats to Validity (External, R4).** A cold-call validation is out of scope for
this experiment.

### RT-B: Allocator free-list inflation (accepted as lower bound)

The heap_reuse vs baseline delta in a warm Criterion loop may understate production malloc cost
because the allocator free-list absorbs repeated alloc/free cycles after warmup. The measured
heap_reuse speedup is accepted as a **conservative lower bound** on the production allocation
cost reduction. If heap_reuse shows ≥ 10% speedup, the allocation overhead is confirmed as
non-trivial even in the warm-loop case.

### RT-C: d_y=2 specialization — scoped (accepted)

The AVX2 kernel is specialized for d_y=2, which matches both the benchmark parameter and the
expected production UMAP embedding dimensionality. **The result is scoped to d_y=2 only.**
Generalization to d_y ≠ 2 is out of scope; the d_y=2 specialization is the intended production
artifact (UMAP always embeds into 2D for visualization). This scope limit is declared in
§Controlled Variables and §Threats to Validity.

### RT-D: Conflated optimizations in flat_simd (accepted)

The flat_partial → flat_simd step isolates "SIMD distance kernel contribution to the flat
buffer approach" but the flat_partial → baseline delta conflates two simultaneous changes: data
structure (heap → Vec) and selection algorithm (push/evict → introselect). **The decomposition
isolates bundles of changes, not single causes.** If flat_simd shows a positive result driven
primarily by the heap_reuse component, the plan will ship heap_reuse instead of flat_simd
(shipping decision based on which variant has the most reliable positive CI, not on full causal
attribution). This conflation is declared in §Threats to Validity (Internal, W2).

### RT-E: Asymmetric AVX2 investment (intentional)

The baseline never received an equivalent AVX2 investment for y_heap. The measured speedup of
flat_simd therefore reflects "flat buffer + AVX2 combined gain over unoptimized heap." This is
the deployment-relevant comparison; the experiment is not testing whether AVX2 alone helps the
heap implementation. This limit on causal interpretation is declared in §Threats to Validity.

### RT-F: Two-stage escalation threshold (acknowledged risk — partially mitigated, see R1)

The 1.1× escalation trigger was not derived from a power analysis. The Type I error inflation
from the two-stage design is acknowledged; the mitigation is a tightened threshold for the
escalated run (CI LB > 1.05 rather than > 1.0). See §Dependent Variables §Escalation Protocol
for the full justification under R1.

### RT-G: BinaryHeap vs introselect tie-breaking (exact invariance required)

The correctness gate uses `|ΔT| < 1e-12` between baseline and variant trustworthiness scores.
Introselect tie-breaking in `flat_partial` and `flat_simd` must exactly replicate the BinaryHeap
tie-breaking order for the test to pass. The `select_nth_unstable_by` comparator must use
`.total_cmp(&dist_y[b]).then(a.cmp(&b))` — identical to the existing `x_sort` comparator —
to ensure lowest index wins on ties. If the tie-breaking comparator is wrong and the test fails,
it is treated as a **correctness failure**, not an acceptable tolerance difference.

### RT-H: Amdahl bound scope (addressed — thread-work fraction only)

The 3.37× Amdahl upper bound and 1.5× stretch target are derived from thread-work fraction.
The wall-clock speedup ceiling is unknown. See §Motivation for the explicit qualification and
§Dependent Variables for how the profiler metric is labeled. **No wall-clock prediction is
made from these figures.**

### RT-I: Phase 0 completion criterion (explicit gate)

Phase 0 analysis is **sufficient to proceed** when: the root cause is documented as one of the
three candidate explanations (cache pressure, introselect locality, broken implementation),
**with supporting evidence from at least one of:** (a) reading the rerun-clean worktree
src/metrics.rs, or (b) the heap_reuse Criterion result (if heap_reuse is also fast, cache
pressure is ruled out as the sole cause). Insufficient: "root cause unknown, proceeding anyway."
If root cause cannot be determined from available evidence, Phase 0 must be escalated to the
researcher with a written summary before Phase 3 begins.

### RT-J: Hardware generalizability — single-machine scope (declared)

All speedup measurements are specific to the AMD Ryzen 7 9800X3D development machine
(8 physical cores, AVX2+FMA, 96 MiB L3) compiled with `target-cpu=native`. AVX2 instruction
throughput and latency vary across CPU microarchitectures. **The deployment decision is scoped
to "AVX2-capable machines"; the measured magnitude may differ on other AVX2 CPUs.** This is
declared in §Threats to Validity (External).

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Implementation variant | baseline, heap_reuse, flat_partial, flat_simd | Isolates: malloc cost (baseline→heap_reuse), data structure + algorithm (heap_reuse→flat_partial), SIMD distance (flat_partial→flat_simd) |
| Input size n | 1 000, 5 000, 10 000 | Matches existing Criterion bench; n=10K is primary measurement; smaller sizes show scaling |

**Variant definitions:**

- **baseline**: Current production `trustworthiness()` from `src/metrics.rs` — exactly as
  shipped. Thread-local X-space buffers (COMB_DIST_X, COMB_INDICES) and AVX2 for x_dist are
  already present. y_heap uses `BinaryHeap::with_capacity(k+1)` allocated fresh per row.

- **heap_reuse**: Thread-local `BinaryHeap<(u64, usize)>` pre-allocated per thread with
  `with_capacity(k+1)`, then `clear()`-d (not re-allocated) at the start of each row.
  Identical push/pop logic to baseline. Diagnostic variant: isolates malloc cost.

- **flat_partial**: Thread-local `Vec<f64>` (COMB_DIST_Y, size n) and `Vec<usize>`
  (COMB_INDICES_Y, size n). All Y squared distances written to COMB_DIST_Y; COMB_INDICES_Y
  extended with 0..n; then `select_nth_unstable_by(k, ...)`. Self-exclusion: set
  `COMB_DIST_Y[i] = f64::INFINITY`. No SIMD — scalar distance loop.

- **flat_simd**: All of flat_partial, plus a dedicated `dist_sq_2d_avx2_batch` kernel
  filling COMB_DIST_Y using 256-bit AVX2 registers, processing 2 Y-rows per 4-wide lane.
  Specialized for d_y=2. Dispatched conditionally on `use_avx2 && d_y == 2`.

---

## Dependent Variables (Metrics)

### Primary Dependent Variable (Confirmatory)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| Wall-time speedup ratio: T_baseline / T_flat_simd | dimensionless | Criterion bootstrap CI at n=10K, k=15 | NEW — no entry in src/metrics.rs |

H0 is rejected if and only if the 95% CI lower bound strictly exceeds 1.0 for this ratio
at n=10K, k=15, seed=42.

No family-wise error correction is applied — there is a single confirmatory comparison.
α=0.05 is the nominal Type I error rate for this comparison.

**CI coverage limitation (W1 — declared and accepted):** Criterion computes the ratio CI
from independent bootstrap distributions of baseline and variant wall times. The ratio of two
random variables has no guaranteed 95% coverage at the nominal level; the CI lower bound
threshold of 1.0 is used despite imperfect coverage as a practical decision threshold.
This limitation is accepted because: (a) Criterion CIs are directionally reliable for large
speedup ratios, and (b) the cost of a false positive (shipping a marginally faster function)
is low relative to the cost of a false negative (leaving 70.3% thread-work unaddressed).

### Escalation Protocol (R1 — Two-Stage Validity)

The RT-5 escalation rule is modified to address the two-stage Type I error inflation identified
in revision R1:

- **Stage 1** (primary): Run baseline + flat_simd with `sample_size=10`. Evaluate CI LB vs 1.0.
- **Escalation trigger**: If point estimate ≥ 1.1× but CI LB ≤ 1.0 (ambiguous result).
- **Stage 2** (escalated): Re-run baseline vs flat_simd only, with `sample_size=50`.
  Evaluate CI LB vs **1.05** (tightened from 1.0 to partially compensate for two-stage Type I
  error inflation).

**R1 explicit choice: tightened threshold (option c).** The elevated combined Type I error
rate from the two-stage design is not pooled (would require sample counting assumptions not
supported by Criterion's output format). Instead the escalated decision uses CI LB > 1.05
as a practical guard against false positives driven by between-run scheduling variance.
Between-run variance of Criterion at n=10K is declared as an internal threat (§Threats to
Validity).

**Decision outcomes:**
- Stage 1 CI LB > 1.0 → **POSITIVE**: ship flat_simd (or best positive variant).
- Stage 2 CI LB > 1.05 → **WEAK POSITIVE**: ship with caveat.
- Stage 2 CI LB ≤ 1.05 → **INCONCLUSIVE**: escalate to H3 (KD-tree experiment).
- Stage 1 CI LB ≤ 1.0, point estimate < 1.1× → **NEGATIVE**: y_heap flat buffer confirmed
  non-viable; escalate to H3.

### Secondary Dependent Variables (Exploratory, Uncorrected)

| Metric | Unit | Collection Method | Purpose |
|--------|------|-------------------|---------|
| Speedup ratio T_baseline / T_heap_reuse | ratio | Criterion, n=10K | Isolates malloc cost |
| Speedup ratio T_baseline / T_flat_partial | ratio | Criterion, n=10K | Tests heap vs introselect |
| SIMD contribution T_flat_partial / T_flat_simd | ratio | Criterion, n=10K | Isolates SIMD distance gain |
| Speedup ratio at n=5K, n=1K | ratio | Criterion | Scaling signal |
| y_heap thread-work fraction per variant | % | Profiler (profiling feature) | Causal attribution |
| Correctness: \|ΔT\| vs baseline | absolute f64 | Unit test assertion | Exact kNN invariance |

**Step fraction time-base clarification (R3):** The profiler emits `[timing:y_heap]` lines via
stderr; `tw_profiler` parses them per-iteration. Values are **wall-clock nanoseconds per step
per invocation**, not summed thread-time. The fraction `y_heap_ns / total_ns` is a per-call
wall-clock step fraction. The 70.3% profiling result from the rerun-clean experiment was
collected with `AtomicU64` step counters gated on `--features profiling` — that earlier
infrastructure accumulated thread-time (summed across Rayon workers). **The new experiment uses
the existing `eprintln!("[timing:")` stderr mechanism** in metrics.rs variants (compile-time
gated behind `#[cfg(feature = "profiling")]`), which records wall-clock elapsed time per
invocation. Both mechanisms confirm step fractions but have different time bases; this plan uses
the stderr mechanism for simplicity. The step fractions are labeled "per-call wall-clock step
fraction" in all result outputs.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| k | 15 | Existing bench standard; prior research used this value. Results scoped to k=15 (W6). |
| d_y | 2 | SIMD kernel is specialized for d_y=2 (RT-C). |
| d_x | 10 | Existing bench standard; activates x_dist AVX2 path (d_x ≥ 10 gate). |
| Seed | 42 | Project-wide convention (RT-3). Sensitivity to seed choice not tested. |
| Rayon thread count | `RAYON_NUM_THREADS=$(nproc)`, recorded | Controls scheduling jitter. |
| RUSTFLAGS | `-C target-cpu=native` (from `.cargo/config.toml`) | Enables full native ISA including AVX2 (R5). |
| Rust toolchain | nightly-2026-03-26 | Pinned via `rust-toolchain.toml` in experiment directory. |
| Criterion sampling mode | SamplingMode::Flat | Eliminates adaptive sampling variance. |
| Criterion sample_size | 10 (Stage 1), 50 (Stage 2 escalation) | Time-bounded. |
| Criterion measurement_time | 10 s per benchmark | Bounds total Criterion runtime. |
| Criterion warm_up_time | 10 s | Matches existing bench convention. |

**Compilation flag alignment (R5):** The `.cargo/config.toml` already sets
`RUSTFLAGS = ["-C", "target-cpu=native"]`, which is what benchmarks use on the development
machine. CI uses `RUSTFLAGS: "-C target-cpu=x86-64-v3"` (AVX2+FMA+BMI2, but not full native
optimization). Downstream consumers building via `cargo build` without explicit RUSTFLAGS will
compile without any `target-cpu` override (baseline codegen). **The benchmark measures
native-compiled code. The flat_simd variant's AVX2 kernel is guarded by
`is_x86_feature_detected!("avx2")` at runtime; it will activate on any AVX2 machine regardless
of compilation flags. However, surrounding scalar code quality may differ between native and
generic codegen. This is declared as a deployment threat in §Threats to Validity (External).**

---

## Inputs and Data

The experiment uses uniform[0,1] synthetic data generated by a Python script. Real UMAP
embedding data is not used; the deployment claim is scoped accordingly (R2).

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| gaussian_n1000_{x,y}.npy | gen_data.py, seed 42 | n=1000, d_x=10, d_y=2, float64, uniform[0,1] | Small-scale Criterion; warm-up |
| gaussian_n5000_{x,y}.npy | gen_data.py, seed 42 | n=5000, d_x=10, d_y=2, float64, uniform[0,1] | Mid-scale Criterion |
| gaussian_n10000_{x,y}.npy | gen_data.py, seed 42 | n=10000, d_x=10, d_y=2, float64, uniform[0,1] | Primary measurement |

**Data properties required:** Each .npy file must have correct shape, dtype=float64, no
NaN/Inf, and non-degenerate range (max − min > 0.01 per column). Verification is logged to
`results/data_verification.txt`.

**Criterion data generation:** The Criterion bench (`benches/y_heap_variants_bench.rs`) generates
data in-process using `SmallRng::seed_from_u64(42)`. This is statistically equivalent (uniform
[0,1]) but not bitwise identical to the numpy-generated .npy files. The profiler uses the .npy
files. Both draw from the same distribution; they are treated as independent samples for
complementary validation.

---

## Prior Failure Analysis (Phase 0 Prerequisite)

**Critical prior result:** The rerun-clean experiment (2026-04-05) measured 2× slowdown with
a thread-local optimization on y_heap (0.634s vs 0.313s baseline at n=10K).

Before implementing any y_heap variant, the implementer must:

1. Check for surviving rerun-clean worktree/branch:
   ```bash
   git worktree list && git branch -a | grep -i rerun
   ```
2. If accessible: read `src/metrics.rs` from that worktree to identify what the `thread_local`
   and `avx2_kernel` variants actually did.
3. Document root cause hypothesis in `results/prior_failure_analysis.md`. Candidates:
   - **Cache pressure**: COMB_DIST_Y (80KB) + COMB_DIST_X (80KB) per thread exceeds per-core
     L2 (256KB), causing cache thrashing across rows.
   - **Introselect locality**: For the heap (k+1=16 elements, L1-resident), O(n log k) accesses
     a 16-slot working set. Introselect over 80KB has worse temporal locality.
   - **Broken self-exclusion**: A correctness branch or off-by-one in the prior implementation
     caused extra work or a correctness fail that was masked by test tolerance.

**Phase 0 completion gate (RT-I):** Phase 0 is sufficient when root cause is documented with
supporting evidence from (a) reading rerun-clean source, or (b) the heap_reuse result. If
neither is accessible before Phase 3, document "root cause unknown" with the three hypotheses
and proceed — the heap_reuse result will provide diagnostic evidence.

---

## Experiment Directory Layout

```
research/2026-04-06-y-heap-bottleneck-optimization/
├── rust-toolchain.toml               # Pins nightly-2026-03-26
├── environment.yml                   # Minimal Python env (numpy, scipy, matplotlib)
├── scripts/
│   ├── gen_data.py                   # Generates gaussian_n{1K,5K,10K}_{x,y}.npy, seed=42
│   ├── run_criterion.sh              # Criterion benchmark runs (profiling OFF, one variant/group)
│   ├── run_profiler.sh               # Step-timing profiler runs (profiling ON, stderr capture)
│   ├── analyze_results.py            # Loads Criterion + profiler JSON; speedup tables + plots
│   └── dry_run.sh                    # Smoke test with n=1K, sample_size=3
├── data/
│   ├── .gitkeep
│   ├── gaussian_n1000_x.npy          # Generated (gitignored via /data/)
│   ├── gaussian_n1000_y.npy
│   ├── gaussian_n5000_x.npy
│   ├── gaussian_n5000_y.npy
│   ├── gaussian_n10000_x.npy
│   └── gaussian_n10000_y.npy
└── results/
    ├── .gitkeep
    ├── data_verification.txt          # Shape/dtype/finiteness log
    ├── prior_failure_analysis.md      # Phase 0: root cause of rerun-clean slowdown
    ├── hardware_profile.txt           # CPU model, nproc, RAYON_NUM_THREADS, rustc, numpy
    ├── Cargo.lock.snapshot            # Copy of Cargo.lock at run time
    ├── criterion/
    │   ├── .gitkeep
    │   ├── y_heap_baseline_n1000.json
    │   ├── y_heap_baseline_n5000.json
    │   ├── y_heap_baseline_n10000.json
    │   ├── y_heap_heap_reuse_n1000.json
    │   ├── y_heap_heap_reuse_n5000.json
    │   ├── y_heap_heap_reuse_n10000.json
    │   ├── y_heap_flat_partial_n1000.json
    │   ├── y_heap_flat_partial_n5000.json
    │   ├── y_heap_flat_partial_n10000.json
    │   ├── y_heap_flat_simd_n1000.json
    │   ├── y_heap_flat_simd_n5000.json
    │   └── y_heap_flat_simd_n10000.json
    ├── profiler/
    │   ├── .gitkeep
    │   ├── profiler_baseline_n10000.json
    │   ├── profiler_heap_reuse_n10000.json
    │   ├── profiler_flat_partial_n10000.json
    │   └── profiler_flat_simd_n10000.json
    └── analysis/
        ├── .gitkeep
        ├── analysis_report.md         # Speedup table, CI table, causal attribution
        └── speedup_ratios.png         # Bar chart: variants × n
```

**File descriptions:**

- `rust-toolchain.toml`:
  ```toml
  [toolchain]
  channel = "nightly-2026-03-26"
  ```

- `environment.yml`:
  ```yaml
  name: y-heap-bench
  channels:
    - conda-forge
  dependencies:
    - python=3.11.*
    - numpy=2.2.*
    - scipy=1.15.*
    - matplotlib=3.10.*
  ```
  Existing `envs/spectral-test/` prefix may be used directly (numpy 2.2.6, scipy 1.15.2,
  matplotlib 3.10 already present).

- `benches/y_heap_variants_bench.rs`: Criterion bench file in the project root `benches/`
  directory. Four benchmark groups (one per variant), each sweeping n ∈ {1K, 5K, 10K}.
  Separate invocations per group via `--bench y_heap_variants_bench -- <filter>` to prevent
  cache warm-state sharing across variants (W4 partial mitigation). Each invocation has its
  own 10s warmup.

- `scripts/run_criterion.sh`: Invokes `cargo bench` once per variant with 60s thermal gaps
  between invocations. Sets `RAYON_NUM_THREADS=$(nproc)`. **Compiled without `profiling`
  feature.** Appends Criterion JSON to per-variant-per-n output files.

- `scripts/run_profiler.sh`: Builds `tw_profiler --features cli,profiling`. Runs each variant
  (via `--variant` flag) with `--warmup 5 --iters 30` at n=10K only. Captures stderr (step
  timing lines) via `--stderr-capture`. Outputs to `results/profiler/`.

---

## Environment

**No custom Rust environment needed.** The project's existing toolchain (stable Rust, all
dependencies in Cargo.toml) is sufficient for all three variants. The `profiling` feature flag
will be added to `Cargo.toml` as a zero-dependency feature.

**KD-tree (H3) is out of scope** for this experiment and would require adding `kiddo` or
`nabo` as new dependencies. That decision is deferred to a follow-up experiment if this
experiment's primary result is NEGATIVE or INCONCLUSIVE.

**Custom Python environment required for `gen_data.py` and `analyze_results.py`:**

```yaml
name: y-heap-bench
channels:
  - conda-forge
dependencies:
  - python=3.11.*
  - numpy=2.2.*      # .npy generation + verification; pinned for RNG stability
  - scipy=1.15.*     # t.ppf for 95% CI on profiler step fractions
  - matplotlib=3.10.*  # speedup_ratios.png
```

The existing `envs/spectral-test/` conda prefix satisfies all Python dependencies and is
preferred over creating a new environment.

---

## Implementation Phases

### Phase 0: Prior Failure Investigation (Prerequisite)

1. Check for surviving worktree/branch:
   ```bash
   git worktree list
   git branch -a | grep -i "rerun\|tw-perf"
   ```
2. If found: read `src/metrics.rs` from that branch. Identify what the prior thread_local
   y_heap variant did. Check specifically: did it add a `COMB_DIST_Y: RefCell<Vec<f64>>`?
   Did it use `select_nth_unstable_by` or keep the BinaryHeap?
3. Write `results/prior_failure_analysis.md` with: root cause hypothesis, supporting evidence,
   and implication for which variants are worth testing.
4. Apply **Phase 0 gate (RT-I):** Do not proceed to Phase 3 until this document exists with
   at least one candidate root cause and supporting evidence.

### Phase 1: Directory Structure and Tooling

1. Create experiment directory tree:
   ```bash
   mkdir -p research/2026-04-06-y-heap-bottleneck-optimization/{scripts,data,results/{criterion,profiler,analysis}}
   touch research/2026-04-06-y-heap-bottleneck-optimization/data/.gitkeep
   touch research/2026-04-06-y-heap-bottleneck-optimization/results/{.gitkeep,criterion/.gitkeep,profiler/.gitkeep,analysis/.gitkeep}
   ```
2. Create `rust-toolchain.toml` in the experiment directory (content shown above).
3. Create `environment.yml` (content shown above).
4. Verify Python environment:
   ```bash
   envs/spectral-test/bin/python -c "import numpy, scipy, matplotlib; print('OK')"
   ```

### Phase 2: Data Generation

1. Write `scripts/gen_data.py`:
   - Use `numpy.random.default_rng(seed=42)`.
   - Accept `--out-dir` argument (default: `data/`).
   - Generate float64: X shape `(n, 10)`, Y shape `(n, 2)` for n ∈ {1000, 5000, 10000}.
   - Save as `gaussian_n{n}_{x|y}.npy` via `np.save()`.
   - Print verification: filename, shape, dtype, min, max, any NaN/Inf.
2. Run and capture:
   ```bash
   python scripts/gen_data.py --out-dir data/ | tee results/data_verification.txt
   ```
3. Verify log shows correct shapes and no NaN/Inf before proceeding to Phase 3.

### Phase 3: Library Implementation

All Rust changes target `src/metrics.rs` and `Cargo.toml`. No other source files are modified.

**3a. Add `profiling` feature to `Cargo.toml`:**
```toml
profiling = []  # Enables step timing instrumentation in trustworthiness variants
```

**3b. Add step-timing instrumentation (gated on `profiling` feature, R6):**

Add per-step wall-clock instrumentation inside `trustworthiness()` and each variant using
`Instant::now()` + `elapsed().as_nanos()` with conditional compilation. The timing output is
emitted via `eprintln!("[timing:{step}] {ns}")` — consistent with the existing profiler
parse protocol in `tw_profiler.rs::parse_step_timing`.

**R6 memory ordering specification:** The existing codebase uses `eprintln!` stderr output (not
`AtomicU64`) for step timing. This plan retains that mechanism, which has no memory ordering
concerns (stderr writes are serialized by the OS). If `AtomicU64` counters are added for
accumulation across Rayon threads, the required contract is:
- Accumulation: `fetch_add(Ordering::Relaxed)` — threads accumulate independently; no
  cross-thread ordering needed during accumulation.
- Reset before a new measurement window: **`store(0, Ordering::SeqCst)`** on all counters.
  SeqCst ensures that all preceding `fetch_add(Relaxed)` operations from all threads are
  globally visible before the counter is cleared. Using Relaxed or Acquire/Release on the
  zero-stores risks carry-over from a prior measurement window contaminating the next.
- Read after all threads complete: `load(Ordering::SeqCst)` — consistent with SeqCst stores.
This ordering contract must be implemented if AtomicU64 counters are used. For this experiment,
the `eprintln!` mechanism avoids AtomicU64 entirely; the requirement above is stated for future
reference if a thread-accumulating profiler is added.

**3c. Implement three variant functions:**

Add to `src/metrics.rs`:

- `pub fn trustworthiness_heap_reuse(x, y, k) -> f64`: Thread-local
  `RefCell<BinaryHeap<(u64, usize)>>`. At start of each row's y_heap: borrow, call
  `heap.clear()`, proceed with identical push/pop logic. Output to penalty step via
  `heap.iter().map(|(_, j)| *j)` (iterates without draining, preserving allocation).

- `pub fn trustworthiness_flat_partial(x, y, k) -> f64`: Thread-local `COMB_DIST_Y:
  RefCell<Vec<f64>>` and `COMB_INDICES_Y: RefCell<Vec<usize>>`. Per row: clear+resize
  COMB_DIST_Y to n; fill scalar squared distances; set `COMB_DIST_Y[i] = f64::INFINITY`
  for self-exclusion; clear+extend COMB_INDICES_Y with 0..n; call
  `select_nth_unstable_by(k, |&a, &b| COMB_DIST_Y[a].total_cmp(&COMB_DIST_Y[b]).then(a.cmp(&b)))`.
  Output to penalty step: iterate `COMB_INDICES_Y[..=k]` (indices, not distances).

- `pub fn trustworthiness_flat_simd(x, y, k) -> f64`: All of flat_partial, plus conditional
  dispatch to `dist_sq_2d_avx2_batch` when `use_avx2 && d_y == 2 && y.is_standard_layout()`.
  The AVX2 kernel processes 2 rows per 4-wide 256-bit lane:
  ```
  lane = [yi[0], yi[1], y[j,0], y[j,1]]
  ref  = [y[j+1,0], y[j+1,1], y[j+2,0], y[j+2,1]]   (blocked per 2 output rows)
  ```
  Post-fill: set `COMB_DIST_Y[i] = f64::INFINITY` to exclude self.

Expose all three from `src/lib.rs` with `#[cfg(feature = "testing")]` gate (consistent with
other internal-testing exports).

Add variant tests to `tests/integration/test_trustworthiness.rs` asserting
`|trustworthiness_flat_simd(x, y, k) - trustworthiness(x, y, k)| < 1e-12` for all existing
test fixtures (t_tw_01 through t_tw_07).

### Phase 4: Criterion Benchmark

1. Write `benches/y_heap_variants_bench.rs`:
   - Four benchmark groups: `y_heap_baseline`, `y_heap_heap_reuse`, `y_heap_flat_partial`,
     `y_heap_flat_simd`.
   - Each group sweeps n ∈ {1000, 5000, 10000} with `BenchmarkId::new("n", n)`.
   - `SamplingMode::Flat`, `sample_size(10)`, `warm_up_time(Duration::from_secs(10))`,
     `measurement_time(Duration::from_secs(10))`.
   - Uses `make_data(n, 10, 2, 42)` (same signature as existing `trustworthiness_bench.rs`).
   - Add to `Cargo.toml`:
     ```toml
     [[bench]]
     name = "y_heap_variants_bench"
     harness = false
     required-features = ["testing"]
     ```

2. Register all three variant functions with `required-features = ["testing"]` in lib.rs.

3. Write `scripts/run_criterion.sh`:
   - Check: abort if `CARGO_FEATURE_PROFILING` is set (W8 guard).
   - Set `RAYON_NUM_THREADS=$(nproc)`.
   - For each variant group {baseline, heap_reuse, flat_partial, flat_simd}:
     - Run `cargo bench --bench y_heap_variants_bench --features testing -- <variant_filter>`.
     - Redirect JSON output to `results/criterion/y_heap_{variant}_n{n}.json`.
     - Sleep 60s thermal gap between groups.
   - Copy `Cargo.lock` to `results/Cargo.lock.snapshot`.

### Phase 5: Profiler Instrumentation and Run

1. Verify `tw_profiler` binary supports `--variant` flag. If not, add it in Phase 3b:
   the `--variant` flag selects which function to call
   (baseline/heap_reuse/flat_partial/flat_simd).
2. Write `scripts/run_profiler.sh`:
   - Build: `cargo build --release --features cli,profiling`.
   - For each variant: `tw_profiler --x data/gaussian_n10000_x.npy --y data/gaussian_n10000_y.npy
     --k 15 --iters 30 --warmup 5 --variant {variant}
     --stderr-capture results/profiler/stderr_{variant}.txt
     --output results/profiler/profiler_{variant}_n10000.json`.

### Phase 6: Dry Run

1. Write `scripts/dry_run.sh`:
   - Run `gen_data.py --out-dir data/` for n=1K only (or use existing n=1K if present).
   - Run Criterion with `sample_size=3`, `warm_up_time=2s` at n=1K only.
   - Run profiler with `--iters 2 --warmup 1` for baseline only.
   - Verify: JSON files produced, no panics, no NaN in scores, `|variant_score - baseline| < 1e-6`.
2. Confirm dry run passes before running full benchmarks.

### Phase 7: Full Experiment Run

1. Collect hardware profile: `lscpu | tee results/hardware_profile.txt && rustc --version >> results/hardware_profile.txt`.
2. Run `scripts/run_criterion.sh` (Stage 1, sample_size=10). Estimated time: ~4 × 3 × 2 × (10s warmup + 10s measurement) = ~2h with 60s gaps.
3. Evaluate Stage 1 escalation condition from `analyze_results.py --stage1-only`.
4. If escalation triggered: re-run `cargo bench` for baseline vs flat_simd only with `sample_size=50`. Add `--escalated` flag to output file names.
5. Run `scripts/run_profiler.sh` for all four variants. Estimated time: ~35 min.
6. Run `scripts/analyze_results.py` to produce `results/analysis/analysis_report.md` and `speedup_ratios.png`.

---

## Execution Protocol

```
cd research/2026-04-06-y-heap-bottleneck-optimization/
bash scripts/dry_run.sh                          # Smoke test
bash scripts/run_criterion.sh                    # Stage 1 benchmarks (~2h)
python scripts/analyze_results.py --stage1-only  # Check escalation condition
# If escalation triggered:
#   bash scripts/run_criterion_escalated.sh      # Stage 2 (baseline + flat_simd, n=50 samples)
bash scripts/run_profiler.sh                     # Step fractions (~35 min)
python scripts/analyze_results.py               # Full analysis
```

---

## Analysis Plan

`scripts/analyze_results.py` implements the following:

1. **Load Criterion JSON**: For each variant and n, extract mean wall time from Criterion's
   estimate record. Compute speedup ratio as `mean_baseline / mean_variant`. Extract Criterion's
   reported CI bounds if available; if not (W5 FALLBACK), compute bootstrap CI from ±5% bounds.

2. **Primary hypothesis test**: Report Criterion CI for T_baseline / T_flat_simd at n=10K.
   State: CI LB > 1.0 → POSITIVE; CI LB ≤ 1.0, estimate ≥ 1.1× → ESCALATE; otherwise → NEGATIVE.

3. **Secondary speedup table**: Rows = variants × n values. Columns = mean time (ms), speedup
   ratio, CI. Mark statistically positive results (CI LB > 1.0).

4. **Causal decomposition table (W2 — with caveat)**: Step contributions estimated as:
   - Allocation cost: 1/ratio(heap_reuse) − 1 (fraction saved by eliminating per-row malloc)
   - Algorithm cost: ratio(heap_reuse)/ratio(flat_partial) − 1 (flat buffer vs heap ops)
   - SIMD cost: ratio(flat_partial)/ratio(flat_simd) − 1 (pure SIMD contribution)
   Note: Each step conflates a bundle of changes (W2). The table is labeled "bundle attribution,
   not single-cause isolation."

5. **Step fractions**: Load profiler JSON files. Compute `step_ns / total_ns` per iteration.
   Report mean ± std per variant, labeled "per-call wall-clock step fraction." If y_heap
   fraction decreases but wall-time doesn't improve, flag as potential confound for discussion.

6. **Correctness confirmation**: Load `cargo test --features testing` output; confirm all
   t_tw_01..t_tw_07 pass and `|ΔT| < 1e-12` for each variant vs baseline.

7. **Shipping decision logic**: Based on primary result:
   - POSITIVE: recommend shipping `flat_simd` (or `heap_reuse` if heap_reuse matches flat_simd
     within CI overlap — prefer simpler).
   - WEAK POSITIVE (escalated): recommend shipping with caveat; note measurement uncertainty.
   - INCONCLUSIVE / NEGATIVE: recommend H3 (KD-tree experiment).

---

## Success Criteria

- **Conclusive positive:** Stage 1 Criterion 95% CI LB > 1.0 for T_baseline / T_flat_simd at
  n=10K, k=15. All correctness tests pass (`|ΔT| < 1e-12`). Profiler confirms y_heap step
  fraction decreased (supporting causal attribution).

- **Weak positive:** Stage 2 escalated CI LB > 1.05. Tests pass. y_heap fraction decreased.
  Ship with caveat.

- **Conclusive negative:** Stage 1 CI LB ≤ 1.0 and point estimate < 1.1×. heap_reuse result
  confirms whether allocation or algorithm is the bottleneck.

- **Inconclusive:** Stage 2 CI LB ≤ 1.05. Heap_reuse result helps diagnose root cause; escalate
  to H3 with documented root cause hypothesis.

---

## Threats to Validity

### Internal

1. **W4 cache warm-state anomaly across n values**: Within each Criterion process invocation,
   thread-local Vecs grow from n=1K through n=5K to n=10K. At n=5K and n=10K, the buffers are
   pre-grown (amortized allocation), partially absorbing allocation cost that would appear in a
   cold-start run. This biases heap_reuse vs baseline comparisons at n=5K and n=10K toward
   understating the allocation difference. Direction: consistent across variants; within-n
   cross-variant comparisons are minimally affected.

2. **Between-run variance in two-stage escalation (R1)**: The escalation protocol runs Stage 2
   as an independent process invocation. OS scheduling, thermal state, and memory bandwidth
   contention between Stage 1 and Stage 2 are uncontrolled. A favorable Stage 2 run may produce
   CI LB > 1.05 due to scheduling luck rather than true speedup. The tightened threshold (1.05)
   partially compensates; the risk of false positive under escalation is acknowledged as an
   accepted design trade-off.

3. **W3 warm_up_time asymmetry**: Criterion's fixed 10s warm-up means faster variants complete
   more warm-up iterations than slower ones. Faster variants enter measurement with more branch
   predictor warming and instruction cache saturation, biasing in the direction of making them
   appear faster. Effect magnitude is small for large speedup ratios (≥ 1.5×) but may inflate
   apparent speedup for modest ratios (1.1–1.3×).

4. **W2 causal decomposition conflation**: The flat_partial vs heap_reuse comparison
   conflates data structure (BinaryHeap → Vec) and selection algorithm (push/evict →
   introselect). The flat_partial vs flat_simd comparison isolates SIMD distance kernel
   contribution but includes d_y=2 specialization at the architectural level. Causal attribution
   is bundle-level, not single-cause.

5. **Criterion ratio CI coverage**: Criterion's ratio CI is computed from independent bootstrap
   distributions of baseline and variant times. The ratio of two random variables does not
   have guaranteed 95% coverage at the nominal level (W1). CI LB > 1.0 is used as a practical
   decision threshold despite this limitation.

### External

1. **Uniform[0,1] synthetic data vs clustered UMAP embeddings (R2)**: The benchmark Y-data is
   uniform[0,1] random. Real UMAP 2D embeddings have clustered, non-uniform distributions with
   qualitatively different distance histograms. The y_heap step's performance depends on:
   heap eviction frequency (proportional to k/n — similar for uniform and clustered data),
   introselect pivot quality (worse for non-uniform data — can degrade to O(n²) in adversarial
   cases), and SIMD load patterns (unaffected by distribution). **The measured speedup is
   scoped to synthetic uniform[0,1] data at n=10K. Transfer to clustered UMAP embedding
   distributions is unvalidated.**

2. **Hot-loop vs cold-call production context (R4)**: Criterion's 10s warm-up saturates L1/L2
   caches, branch predictors, and allocator free-lists before measurement. In production,
   `trustworthiness()` is called after UMAP computation with Y data freshly allocated. The flat
   buffer's sequential access advantage (enabling SIMD) may be reduced under cold cache
   conditions. **The benchmark measures hot-loop throughput. The cold-call speedup in production
   may differ; the result may overstate the deployment speedup.**

3. **Compilation flags: native vs downstream (R5)**: Benchmark uses `target-cpu=native` from
   `.cargo/config.toml`. CI uses `target-cpu=x86-64-v3`. Downstream `cargo build` without
   explicit RUSTFLAGS uses generic baseline codegen. The AVX2 kernel is runtime-dispatched
   (`is_x86_feature_detected!("avx2")`), so it activates on any AVX2 CPU regardless of
   compilation flags — but surrounding scalar code quality may differ. **Speedup magnitude may
   vary between native-compiled and generic-compiled builds; the benchmark measures
   native-compiled code only.**

4. **k=15 scope limit (W6)**: The BinaryHeap vs introselect crossover depends on k. At k=15,
   log(k) ≈ 4 comparisons per element; at k=50, log(k) ≈ 6. The relative performance of
   BinaryHeap vs introselect may shift at k ≥ 30. **Results are valid for k=15 only; deployment
   decisions at k ∈ {30, 50} require separate validation.**

5. **n=100K out of scope (W7)**: At n=100K the flat buffer per thread is 800KB, exceeding
   per-core L2 cache (typical 256KB–512KB). The flat buffer's sequential-access advantage that
   enables SIMD throughput may reverse under L2 miss pressure. The estimated L2 spill threshold
   for COMB_DIST_Y is n ≈ 32K (256KB / 8 bytes). **The flat buffer approach is validated only
   for n ≤ 10K. Large-scale behavior at n=100K requires a separate experiment.**

6. **Hardware specificity (RT-J)**: AVX2 throughput and pipeline latency vary across CPU
   microarchitectures. The AMD Ryzen 7 9800X3D (Zen 5) has specific AVX2 execution unit
   characteristics. **Speedup magnitudes may differ on Intel, older AMD, or ARM (with AVX2
   emulation) CPUs.**

---

## Estimated Resource Requirements

- **Criterion benchmarks (Stage 1):** ~2h (4 variants × 3 n values × 20s measurement + 60s
  thermal gaps). Stage 2 escalation (if triggered): additional ~30 min.
- **Profiler runs:** ~35 min (4 variants × 30 iterations × warmup at n=10K).
- **Data generation + analysis scripts:** < 5 min.
- **Disk space:** ~50 MB (Criterion JSON, profiler JSON, .npy data files).
- **New Rust dependencies:** None. The experiment uses only `std::collections::BinaryHeap`,
  `std::arch::x86_64`, and the existing thread-local pattern.
- **Rust compilation changes:** Adding `profiling` feature to Cargo.toml (zero deps); adding
  `y_heap_variants_bench.rs` bench; adding three variant functions to `src/metrics.rs`.
