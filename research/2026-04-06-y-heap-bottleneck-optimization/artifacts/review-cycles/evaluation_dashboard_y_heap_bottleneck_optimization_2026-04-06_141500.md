# Evaluation Dashboard — y_heap Bottleneck Optimization
**Review date:** 2026-04-06 · **Verdict: ⚠️ REVISE**

---

## Verdict Banner

```
╔══════════════════════════════════════════════════════════╗
║  VERDICT: REVISE                                         ║
║  7 critical findings · 44 warnings · 10 red-team (warn) ║
║  Experiment type: benchmark                              ║
╚══════════════════════════════════════════════════════════╝
```

The plan is structurally sound in its hypothesis specification, causal decomposition design, and data acquisition. The red-team decisions (RT-1 through RT-5) demonstrate awareness of the primary adversarial concerns. However, **two critical design flaws in the variance protocol** (uncontrolled between-run variance for the escalation path, and non-pooled two-stage sampling) compromise the statistical validity of the primary decision rule. Additionally, **five critical findings across representativeness, ecological validity, and measurement** point to gaps between the benchmark conditions and the deployment context being targeted. All seven critical findings are addressable through plan revision — none are structural STOP triggers.

---

## Classification Summary

| Field | Value | Source |
|---|---|---|
| Experiment type | benchmark | Triage (Rule 1: IVs are method names, DVs are perf metrics, multiple comparators) |
| Active modifiers | +deployment, +multi_metric | Production shipping decision; ≥3 DVs |
| L1 gate result | PASS | No critical L1 findings |
| Stop triggers | 0 | No L1/red-team critical after cap |
| Critical findings | 7 | variance_protocol (2), benchmark_representativeness (2), ecological_validity (2), measurement_alignment (1) |
| Warning findings | 44 | Distributed across 12 non-silent dimensions |
| Red-team findings | 10 | All capped at warning for benchmark type |
| Warning threshold | 65 | 13 active dimensions × 5 budget |

---

## Dimension Scorecard

| Dimension | Level | Weight | Critical | Warning | Info | Status |
|---|---|---|---|---|---|---|
| estimand_clarity | L1 | H | 0 | 1 | 2 | ⚠️ Warning |
| hypothesis_falsifiability | L1 | H | 0 | 2 | 3 | ⚠️ Warning |
| baseline_fairness | L2 | H | 0 | 3 | 2 | ⚠️ Warning |
| unit_interference | L2 | H | 0 | 2 | 5 | ⚠️ Warning |
| error_budget | L3 | M | 0 | 3 | 2 | ⚠️ Warning |
| statistical_corrections | L3 | H* | 0 | 3 | 2 | ⚠️ Warning |
| variance_protocol | L3 | H | 2 | 3 | 2 | 🔴 Critical |
| benchmark_representativeness | L4 | M | 2 | 5 | 1 | 🔴 Critical |
| ecological_validity | L4 | M | 2 | 3 | 0 | 🔴 Critical |
| measurement_alignment | L4 | M | 1 | 4 | 0 | 🔴 Critical |
| reproducibility_spec | L4 | M | 0 | 3 | 2 | ⚠️ Warning |
| data_acquisition | L4 | M | 0 | 1 | 6 | ✅ Clean |
| red_team | — | — | 0 | 10 | 0 | ⚠️ Warning |

*statistical_corrections elevated H due to +multi_metric modifier.

---

## Critical Findings

### variance_protocol

**C1 — Escalation run not pooled with initial run**
*Section:* § Execution Protocol → RT-5

The escalation rule (RT-5) discards the initial 10-sample run's information entirely: if CI LB ≤ 1.0 but the point estimate ≥ 1.1×, a fresh 50-sample run is conducted whose result alone governs the decision. The two runs are statistically independent observations drawn under potentially different OS and cache conditions. The decision threshold applied to the second run is the same 1.0 CI LB used for the first, with no adjustment for the two-stage nature of the test. This creates a path where a true null effect can produce CI LB > 1.0 in the 50-sample run due to favorable run conditions, inflating the effective Type I error rate above the nominal α=0.05.

**C2 — Between-run variance uncontrolled for escalation decision**
*Section:* § Execution Protocol → Step 5

The plan treats separate Criterion invocations as producing comparable CI bounds, but run-to-run variance in wall-clock measurements (OS scheduling, DRAM bandwidth contention, thermal throttling) is not controlled or bounded. The escalation trigger (CI LB ≤ 1.0, estimate ≥ 1.1×) could be satisfied in one run but not in a re-run on the same hardware. The plan does not describe any protocol for verifying that the escalation-triggering run's point estimate is stable across two consecutive unmolested executions before committing to the 50-sample escalation. For a binary production deployment decision, this instability in the escalation path is undeclared and unmitigated.

---

### benchmark_representativeness

**C3 — Uniform[0,1] synthetic Y-space structurally unlike real UMAP embeddings**
*Section:* § Inputs and Data

Real UMAP 2D embeddings exhibit clustered, non-uniform distance distributions: points within clusters are densely packed, inter-cluster distances are large, and the k-nearest-neighbor rank structure reflects this topology. The y_heap step processes pairwise Y-distances; heap eviction frequency, introselect pivot quality, and SIMD load patterns all depend on the distributional shape of those distances. Uniform[0,1] produces broadly spread distances with no cluster structure, which is a qualitatively different input to the y_heap kernel than real deployment data. The speedup measured on uniform data cannot be assumed to bound or predict the speedup on real UMAP outputs. This representativeness gap is **not declared as a threat** despite the plan's +deployment framing — the result is labeled as measuring "deployment value" but is measured on fundamentally non-deployment data.

**C4 — Thread-work fraction used to motivate experiment without wall-clock validation**
*Section:* § Motivation + § Dependent Variables

The experiment's motivation (and stretch target) rests on the 70.3% profiling measurement. That measurement is explicitly characterized as summed CPU thread-time across Rayon workers, not wall-clock elapsed time. The plan correctly notes these are equivalent "only in single-threaded execution" and scopes the profiler metric as causal attribution only. However, no wall-clock step decomposition is presented to validate that y_heap contributes a comparable fraction of wall-clock time. If parallelism efficiency varies across steps (e.g., y_heap is more compute-bound and x_knn_set is more memory-bound with different NUMA behavior), the wall-clock fraction of y_heap could be substantially different from 70.3%. The Amdahl upper bound and the 1.5× stretch target are derived from the thread-time fraction, not from a wall-clock measurement, yet are presented in the motivation as if they predict wall-clock achievable speedup.

---

### ecological_validity

**C5 — Benchmark warm-cache steady-state vs production cold-entry context**
*Section:* § Environment

Criterion runs with 10s warm-up before measurement begins, producing a steady-state measurement where caches (L1/L2/L3), branch predictor tables, and TLB entries are fully saturated by the variant's specific memory access patterns. In production, `trustworthiness()` is called once per evaluation epoch, entering after UMAP graph construction and SGD optimization have evicted its working set from cache. The measured speedup reflects hot-path throughput; the deployment-relevant speedup reflects cold-entry latency. For variants whose advantage is in memory access pattern (the flat buffer's sequential access vs the heap's pointer-chasing), the hot-cache vs cold-cache difference in measured speedup can be qualitatively different in direction, not merely a scalar factor.

**C6 — RUSTFLAGS=-C target-cpu=native benchmark vs non-native production compilation**
*Section:* § Controlled Variables

The benchmark compiles with `-C target-cpu=native`, enabling the full ISA of the test machine including AVX2, FMA, and potentially machine-specific microarchitectural extensions. The flat_simd variant's AVX2 kernel is conditionally compiled and dispatched only when AVX2 is present. The plan notes CI uses `x86-64-v3` as baseline, but does not specify what compilation flags are used for the production `src/metrics.rs` artifact when the crate is consumed downstream. If downstream users compile without `-C target-cpu=native`, the `is_x86_feature_detected!("avx2")` runtime check would still enable the AVX2 path on capable hardware, but if the crate is compiled at `x86-64-v2` or lower, the AVX2 target feature may not be available at compile time, causing the unsafe AVX2 block to either not compile or to compile without the expected intrinsic availability. The benchmark result is therefore comparing conditions that may not match what ships.

---

### measurement_alignment

**C7 — AtomicU64 Relaxed ordering on step_timing counters may compromise reset validity**
*Section:* § Implementation Phases 3b

The plan specifies `fetch_add(Ordering::Relaxed)` for step counter accumulation. In a Rayon multi-threaded context, Relaxed provides atomicity but no happens-before guarantee. The `step_timing::reset()` function (which zeroes all counters) is called at the start of each profiler window. If reset() uses Acquire or SeqCst ordering, it still does not guarantee that all preceding `fetch_add(Relaxed)` operations from other threads have committed to the counters before the zero is written. This creates a race where carry-over thread-time from a prior profiling window can persist into the next measurement, or where the current window's thread-time is partially attributed to the next window. Since the step fractions are the primary causal attribution tool for verifying that y_heap improvement explains the observed wall-clock gain, measurement errors in the step counters could produce concordant (both decrease) or discordant (only one decreases) signals that mislead interpretation of the Criterion results.

---

## Adversarial Findings (Red-Team)

*All red-team findings have `requires_decision: true`. Benchmark type caps red-team severity at warning.*

**RT-A: Goodhart — Microbenchmark hot-loop vs production cold-call** `requires_decision: true`
Criterion measures wall-clock in a tight call loop with warm caches. Flat_simd may win in this condition through instruction-cache residency and SIMD register retention, without delivering comparable speedup when called once per UMAP evaluation epoch from a cold state. The primary metric (CI LB > 1.0) can be satisfied without the optimization being useful in deployment.

**RT-B: Goodhart — Allocator free-list inflation in baseline** `requires_decision: true`
In a tight Criterion loop, the baseline's per-row BinaryHeap allocation/deallocation cycles train the allocator's free-list to immediately satisfy each request. This partially conceals the true allocation overhead compared to production, where the allocator may be operating on a colder state. The measured malloc cost for the baseline (heap_reuse vs baseline delta) is likely understated relative to production.

**RT-C: Data leakage — AVX2 kernel designed for benchmark fixture** `requires_decision: true`
The 2D AVX2 distance kernel is specialized for d_y=2, which is also the evaluation condition. The optimization target and the evaluation condition are the same parameter value. It is not possible to distinguish "fast for the benchmark" from "fast for d_y=2 in general" since only d_y=2 is tested. The plan acknowledges d_y=2 as a specialization limit but does not address that the kernel design was informed by knowing d_y=2 is the evaluation condition.

**RT-D: Asymmetric tuning — Primary metric conflates two optimizations** `requires_decision: true`
The primary metric (T_baseline / T_flat_simd) bundles heap elimination + SIMD into a single measured effect. RT-2 frames this as measuring "deployment value," but the causal decomposition only partially resolves which sub-optimization drives the speedup. If flat_simd's advantage comes entirely from heap elimination and SIMD contributes negligibly, shipping flat_simd requires maintaining SIMD code for no benefit; conversely, if SIMD drives the gain, heap_reuse alone would be the simpler shipping path. The confirmatory test cannot distinguish these cases; the causal decomposition is exploratory and may not achieve statistical confidence.

**RT-E: Asymmetric tuning — Baseline y_heap never received AVX2 investment** `requires_decision: true`
Prior work applied AVX2 to the x_dist step; the y_heap step in the baseline has never been AVX2-optimized. The comparison is between an AVX2-invested flat_simd and an un-invested baseline y_heap. The plan does not assess whether the performance delta reflects "flat buffer enables SIMD" or "any vectorization of y_heap distance computation helps regardless of data structure." An AVX2-accelerated BinaryHeap comparison variant is absent.

**RT-F: Survivorship bias — Escalation threshold (1.1×) not pre-registered vs H0 (1.0)** `requires_decision: true`
The escalation trigger at ≥ 1.1× creates selective additional power only when the first-run estimate is favorable. The 1.1× threshold is not derived from a pre-specified power analysis; it was chosen to capture "near-positive results." A true null effect can produce point estimates of 1.05–1.15× by chance, triggering escalation that uses the larger sample to potentially push CI LB above 1.0. The combined Type I error rate of the two-stage test is uncontrolled.

**RT-G: Evaluation collision — BinaryHeap tie-breaking non-determinism** `requires_decision: true`
The BinaryHeap uses bit-comparison of (u64, usize) pairs. For identical f64 distances (equal Y-distances between rows), ordering depends on the second tuple element (row index), producing a deterministic but arbitrary ordering. The `select_nth_unstable_by` (introselect) uses `partial_cmp` with a `cmp::Ordering::Equal` fallback, which also depends on index ordering but may resolve ties differently. If any benchmark input has tied k-th/k+1-th Y-distances, the correctness tests (|Δ| < 1e-12) may pass on specific seeds but the flat variants may compute a different but equally valid kNN set that yields a different (but correct) trustworthiness score. The 1e-12 tolerance tests algorithmic equivalence, not correctness.

**RT-H: Evaluation collision — Amdahl bound derived from mismatched time base** `requires_decision: true`
The 3.37× Amdahl upper bound and the 1.5× stretch target are derived from the 70.3% thread-work fraction. As acknowledged in the plan, thread-work fraction equals wall-clock fraction only for perfectly parallel, no-synchronization execution. With Rayon and non-trivial inter-thread work stealing, the actual wall-clock fraction of y_heap could be significantly lower than 70.3%. If the true wall-clock y_heap fraction is ~30%, the achievable wall-clock speedup ceiling from total y_heap elimination is ~1.43×, and 1.5× is unachievable by construction. The experiment would then report "stretch target not met" for structural reasons, not implementation quality reasons.

**RT-I: Phase 0 failure analysis is not a formal gate** `requires_decision: true`
Phase 0 requires documenting the root cause of the prior 2× slowdown. The plan does not specify what constitutes a sufficient analysis to proceed — it is a human judgment call. If the root cause is misdiagnosed (e.g., identified as a broken implementation when it was actually cache pressure), the flat_partial and flat_simd variants may reproduce the same regression. The dry run at n=1K is the only early regression check, but n=1K has a much smaller flat buffer (8KB vs 80KB at n=10K) and would not trigger the cache pressure that caused the prior slowdown.

**RT-J: Hardware generalizability not declared as a scope limit** `requires_decision: true`
The AVX2 distance kernel's speedup depends on CPU microarchitecture (Intel vs AMD, Zen 4 vs Zen 5, Alder Lake vs Raptor Lake differ in AVX2 throughput/latency). The benchmark runs on one machine. The deployment decision "ship flat_simd to production" does not scope which hardware configurations will experience the measured speedup, yet the plan does not declare hardware generalizability as a threat or scope the claim to specific CPU families.

---

## Cannot Assess

1. **Phase 0 root cause adequacy** — The failure analysis of the 2× slowdown from the prior rerun-clean experiment depends on whether that git worktree still exists at execution time. The correctness and completeness of the causal diagnosis cannot be evaluated from the plan alone; it is inherently an execution-time judgment call.

2. **AVX2 kernel correctness for tail handling** — The plan specifies that the 2D AVX2 batch kernel processes 4 rows per SIMD iteration with a scalar tail for n % 4 ≠ 0. The correctness of the tail boundary in the SIMD kernel cannot be assessed from the plan's design description; it requires the implementation to exist. The correctness tests at n ∈ {20, 50, 100} (all non-multiples of 4) are the right mitigation, but their adequacy depends on the actual kernel behavior.

3. **Parallelism efficiency of Rayon for each step** — The plan's Amdahl-based reasoning assumes thread-work fraction ≈ wall-clock fraction. The actual wall-clock fraction of y_heap under Rayon's work-stealing scheduler cannot be determined from the plan; it requires execution of the profiler with wall-clock instrumentation per step, which the plan does not include.

4. **Criterion sample_size=10 adequacy for the actual measured variance** — Whether 10 samples provides stable CI bounds depends on the coefficient of variation of the timing measurements, which requires execution to assess. The prior rerun-clean experiment measured ~0.313s baseline at n=10K; the inter-measurement variance of that run is not reported in the plan.

---

## Mechanizable Check Log

| Check | Status | Notes |
|---|---|---|
| Has explicit H0 | ✅ PASS | "CI for speedup ratio contains 1.0" |
| Has explicit H1 | ✅ PASS | "CI LB > 1.0 for T_baseline/T_flat_simd" |
| Decision rule binary (CI LB > 1.0) | ✅ PASS | Single threshold, unambiguous |
| Has escalation rule with terminal condition | ✅ PASS | RT-5 includes H3 escalation |
| All gitignored data paths have generation steps | ✅ PASS | data/*.npy generated by gen_data.py |
| Seed specified | ✅ PASS | seed=42 throughout |
| Rust toolchain pinned | ✅ PASS | nightly-2026-03-26 in rust-toolchain.toml |
| Python env pinned | ⚠️ PARTIAL | environment.yml uses 2.2.* wildcards, not exact patches |
| Cargo.lock snapshot planned | ⚠️ PARTIAL | Plan describes snapshot step; not yet created |
| Correctness tests specified | ✅ PASS | t_tw_09 through t_tw_12 |
| Profiling feature gated from Criterion | ✅ PASS | RT-4 resolution |
| Hardware profile recording specified | ✅ PASS | hardware_profile.txt |
| Dry run defined | ✅ PASS | dry_run.sh with smoke-test protocol |
| Prior failure analysis required | ✅ PASS | Phase 0 prerequisite |
| α explicitly stated | ✅ PASS | α=0.05 |
| Confirmatory/exploratory separation | ✅ PASS | Single confirmatory DV; all others declared exploratory |
| Bonferroni not applied (justified) | ✅ PASS | Single confirmatory comparison |
| Escalation threshold adjustment | ❌ FAIL | No α adjustment for two-stage test |
| Warm-cache vs cold-cache scope declared | ❌ FAIL | Not declared as a threat |
| Production compilation flags matched | ❌ FAIL | RUSTFLAGS=native not confirmed as production baseline |
| Cross-variant warm_up iteration asymmetry addressed | ❌ FAIL | Not declared as a threat |

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 7
warning_count: 44
red_team_count: 10
active_dimensions: 13
warning_threshold: 65
stop_trigger_count: 0
modifiers:
  - +deployment
  - +multi_metric
revision_required_for_go: true
notes: |
  7 critical findings: variance_protocol (2), benchmark_representativeness (2),
  ecological_validity (2), measurement_alignment (1).
  Red-team capped at warning for benchmark type (10 findings, 0 critical after cap).
  No L1 or red-team stop triggers. Verdict driven by non-stop critical findings.
  All critical findings are addressable without changing the core experimental design.
```
