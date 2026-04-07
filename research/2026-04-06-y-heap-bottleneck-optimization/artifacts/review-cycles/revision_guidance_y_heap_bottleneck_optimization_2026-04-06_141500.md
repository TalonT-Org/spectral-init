# Revision Guidance — y_heap Bottleneck Optimization
**Verdict: REVISE** · **Review date:** 2026-04-06

---

## Overview

The plan has a solid core: explicit H0/H1, a causal decomposition design, pre-addressed red-team concerns (RT-1 through RT-5), and complete data acquisition. The revisions below address gaps in the statistical validity of the two-stage escalation path, the representativeness of the benchmark fixture for deployment claims, and the alignment between the profiler's time-base and the claims derived from it.

None of the required revisions demand new implementation work. All are design-level changes to hypothesis framing, threat declarations, and scope boundaries.

---

## Required Revisions (Critical Findings)

### R1 — Address two-stage escalation statistical validity (variance_protocol C1/C2)

**Gap:** The escalation rule (RT-5) triggers a second independent 50-sample run when the first 10-sample run produces estimate ≥ 1.1× but CI LB ≤ 1.0. The second run's result governs the decision without pooling the first run's data, and no threshold adjustment is made for the two-stage nature of the test. Between-run variance (OS scheduling, thermal state, memory bandwidth contention) is uncontrolled and unacknowledged as a threat to the escalation path.

**Risk:** The combined Type I error rate of the two-stage test exceeds the nominal α=0.05 without bound relative to the stated threshold. A true null effect can satisfy CI LB > 1.0 in a favorable second run, yielding a false deployment decision.

**Required change:** The escalation rule requires one of: (a) a declared acknowledgment of the elevated Type I error risk with an explicit decision to accept it as a design trade-off, (b) a protocol for pooling or averaging the two runs, or (c) a tightened decision threshold for the escalated run (e.g., CI LB > 1.05) to partially compensate for the two-stage inflation. The choice among these alternatives is the plan author's; the plan must make the choice explicit. Additionally, the between-run variance of Criterion measurements at n=10K should be declared as an internal threat.

---

### R2 — Declare uniform data vs real UMAP embedding distribution as a threat (benchmark_representativeness C3)

**Gap:** The plan uses uniform[0,1] synthetic Y-coordinates as the benchmark fixture and frames the result as measuring "deployment value." Real UMAP 2D embeddings have clustered, non-uniform distributions with qualitatively different distance histograms. The y_heap step's performance characteristics (heap eviction frequency, introselect pivot quality, SIMD load patterns) are sensitive to the distance distribution shape. This gap is not declared as a threat to external validity despite the plan's +deployment framing.

**Risk:** The measured speedup on uniform data may overestimate or underestimate the speedup on actual UMAP output data. A deployment decision made without this scope boundary misrepresents the claim's generalizability.

**Required change:** Add a declaration in §Threats to Validity (External) that the benchmark data is uniform[0,1] synthetic and that results may not transfer to clustered UMAP embedding distributions. The deployment conclusion should be scoped to "synthetic uniform benchmark at n=10K" rather than stated as a general deployment value.

---

### R3 — Clarify the derivation and scope of the Amdahl-based speedup expectation (benchmark_representativeness C4)

**Gap:** The 70.3% profiling measurement is summed CPU thread-time, not wall-clock elapsed time. The Amdahl upper bound (3.37×) and the 1.5× stretch target are derived from this thread-work fraction. The plan acknowledges the time-base distinction correctly for the profiler metric, but the motivation section presents these numbers as if they predict wall-clock speedup. No wall-clock step decomposition exists to validate that y_heap constitutes a comparable fraction of wall-clock time.

**Risk:** If the true wall-clock y_heap fraction at n=10K is substantially lower than 70.3% (plausible given that other steps have different parallelism characteristics), the achievable speedup ceiling from y_heap optimization is lower than 3.37×, and the 1.5× stretch target may be unachievable by construction rather than by optimization quality. This would produce a misleading "stretch target not met" result.

**Required change:** Qualify the Amdahl bound and stretch target in §Motivation and §Hypothesis with an explicit acknowledgment that these figures are derived from thread-work fractions, not wall-clock fractions, and that the true wall-clock speedup ceiling is unknown without a wall-clock step decomposition experiment. The stretch target may be retained as an exploratory reference but should be labeled "derived from thread-work fraction (upper bound under single-thread assumption)."

---

### R4 — Declare warm-cache microbenchmark vs cold-call production context as a threat (ecological_validity C5)

**Gap:** Criterion's 10s warm-up phase saturates caches and branch predictors before measurement begins. In production, `trustworthiness()` is called once per evaluation epoch after UMAP computation, entering cold. The flat buffer's sequential memory access pattern (which is the primary mechanism expected to enable SIMD gains) may behave qualitatively differently under warm vs cold cache conditions. This is not declared as a threat.

**Risk:** The measured speedup may be specific to Criterion's steady-state hot-loop condition and not transferable to the single-call production pattern. A deployment decision based solely on hot-loop speedup may not materialize in production latency reduction.

**Required change:** Add a declaration in §Threats to Validity (External) that the benchmark measures hot-loop throughput under fully-warm cache conditions, and that the result may overstate or understate the single-call cold-entry speedup in production. The deployment claim should be scoped accordingly.

---

### R5 — Address production compilation flag alignment (ecological_validity C6)

**Gap:** The benchmark uses `RUSTFLAGS=-C target-cpu=native`, which enables the full ISA of the test machine. The plan notes that CI uses `x86-64-v3`. The compilation flags used when this crate is consumed in production (via `cargo build` in downstream projects) are not specified. If the flat_simd variant's AVX2 kernel depends on compile-time feature availability and the downstream build does not use `target-cpu=native` or an equivalent, the runtime `is_x86_feature_detected!("avx2")` check may succeed while the compiled code path differs from what was benchmarked.

**Risk:** The benchmark measures a native-compiled code path; production may run a different compilation artifact with different codegen quality, making the measured speedup not representative of the deployment artifact.

**Required change:** Declare in §Controlled Variables or §Threats to Validity whether `RUSTFLAGS=-C target-cpu=native` is also used in the expected production compilation context. If it is not, declare the compilation flag difference as an external threat and scope the result to "native compilation only."

---

### R6 — Address AtomicU64 ordering for step_timing::reset() (measurement_alignment C7)

**Gap:** The plan specifies `fetch_add(Ordering::Relaxed)` for step counter accumulation, and the `reset()` function description does not specify the memory ordering used for the zeroing operations. In a Rayon multi-threaded context, if `reset()` uses Relaxed or even Acquire ordering on the zero-stores, it does not guarantee that all preceding `fetch_add(Relaxed)` operations from other threads have completed before the counters are cleared. This can cause carry-over thread-time from a prior profiling window to contaminate the next measurement.

**Risk:** The step fractions produced by the profiler may be inaccurate, producing concordant or discordant signals between step fraction change and wall-clock change that mislead causal attribution. A "discordant" result (y_heap fraction decreases but wall-time doesn't improve) could indicate an instrumentation artifact rather than a real phenomenon.

**Required change:** The plan should specify the memory ordering contract for `reset()` explicitly — including which ordering guarantees are required to ensure all prior `fetch_add` operations are visible before the counters are zeroed. The design description in Phase 3b should state the required ordering and the reasoning. This is a design specification gap, not an implementation correctness issue; the fix is to add the ordering requirement to the plan.

---

## Recommended Revisions (Selected High-Value Warnings)

### W1 — Criterion CI ratio coverage is a structural limitation of the decision rule

*Dimensions: estimand_clarity, hypothesis_falsifiability, error_budget, measurement_alignment*

The plan acknowledges that Criterion's ratio CI (derived from independent bootstrap distributions of baseline and variant) does not have guaranteed 95% coverage for the ratio estimand. This limitation is declared but is not reflected in the decision rule (CI LB > 1.0 at nominal 95%). The four separate dimensions all independently identified this as the same core gap.

**Recommended:** Either declare the CI coverage limitation as an accepted risk with a statement of why CI LB > 1.0 is still a useful decision threshold despite imperfect coverage, or widen the decision threshold to CI LB > 1.02 or similar to provide a practical buffer against coverage shortfall.

---

### W2 — Causal decomposition attribution is weaker than presented

*Dimension: baseline_fairness*

The flat_partial → flat_simd step is labeled as isolating "SIMD contribution," but flat_simd also receives d_y=2 specialization that applies at the architectural level. The heap_reuse → flat_partial step conflates two simultaneous algorithmic changes: data structure (heap → flat Vec) and selection algorithm (sequential insert/evict → introselect). These conflations weaken the causal attribution claims in §Analysis Plan.

**Recommended:** Either add a clarifying note that the decomposition steps isolate "bundles of changes" rather than single causes, or add intermediate variants that isolate individual factors. If the budget does not permit additional variants, declare the conflation explicitly in §Threats to Validity (Internal).

---

### W3 — warm_up_time asymmetry across variants

*Dimension: baseline_fairness*

Criterion's fixed warm_up_time=10s means faster variants complete more warm-up iterations than slower ones, creating an asymmetry in pre-measurement CPU microarchitectural state. Faster variants receive more branch predictor warming and instruction cache saturation before measurement.

**Recommended:** Declare this asymmetry in §Threats to Validity (Internal) and note its direction (biases toward showing faster variants as relatively faster than they would appear under uniform warm_up iteration counts).

---

### W4 — Within-process Vec growth across n-values may understate allocation costs

*Dimension: unit_interference*

Within each variant's Criterion process invocation, thread-local Vecs grow from the n=1K run through n=5K to n=10K. Later n-values inherit pre-grown capacity, partially absorbing allocation overhead that would occur in a cold-start run at that n. For baseline_fairness with the heap_reuse variant (which measures malloc cost), this means heap_reuse vs baseline comparisons at n=5K and n=10K may understate the allocation difference.

**Recommended:** Acknowledge this effect in §Threats to Validity (Internal) and note that its direction is consistent across variants (all n-groups benefit), so within-variant comparisons (n=1K vs n=5K vs n=10K) are more affected than cross-variant comparisons at the same n.

---

### W5 — sample_size=10 adequacy unvalidated

*Dimension: error_budget, variance_protocol*

The plan does not demonstrate that 10 independent timing samples produces stable CI bounds for detecting a 1.1× speedup, which is the escalation threshold. Criterion's bootstrap CI width scales with 1/√n; at n=10 the CI is approximately √5 ≈ 2.2× wider than at n=50. The escalation rule's trigger condition (estimate ≥ 1.1×) implies awareness that 10 samples may be insufficient, but the adequacy of 10 samples for the confirmatory comparison at CI LB > 1.0 is not analyzed.

**Recommended:** Either add a brief justification of sample_size=10 (e.g., from prior Criterion runs in this codebase, timing CV at n=10K is <5%, so n=10 is sufficient to detect ≥1.3× speedups) or lower the confirmatory threshold given the CI width expected from n=10.

---

### W6 — k=15 scope limit not declared for deployment generalizability

*Dimension: benchmark_representativeness*

Trustworthiness is routinely evaluated at k ∈ {5, 10, 15, 30, 50}. The relative performance of BinaryHeap vs introselect depends on k (heap per-element cost = O(log k)); a deployment decision at k=15 may not hold at k=50.

**Recommended:** Declare k=15 as a scope limit in §Threats to Validity (External): "results are valid for k=15 only; the crossover between heap and introselect may shift at k≥30."

---

### W7 — n=100K production scale unmeasured

*Dimension: benchmark_representativeness*

At n=100K, the flat buffer per thread is 800KB, exceeding per-core L2 cache. The flat buffer's sequential access advantage (which motivates the SIMD kernel) may reverse under L2 miss pressure. The plan scopes n≤10K but does not declare n=100K as out of scope, even though the profiling motivation references prior work on MERFISH-scale data.

**Recommended:** Explicitly declare n=100K as out of scope with a note that cache hierarchy behavior changes above n≈32K (estimated L2 spill threshold for the flat buffer), and that a separate experiment would be needed to validate the approach at production scale for large n.

---

## Red-Team Decision Points

Each item below requires an explicit plan decision (accept, reject, or mitigate). All items have `requires_decision: true`.

**RT-A: Hot-loop benchmark vs cold-production call (Goodhart)**
The experiment measures hot-loop throughput. The plan should declare whether this is accepted as a proxy for deployment value or whether a cold-call measurement is needed before shipping.

**RT-B: Allocator free-list inflation in microbenchmark (Goodhart)**
The heap_reuse vs baseline delta may understate production malloc cost. The plan should declare whether the measured heap_reuse delta is accepted as a lower bound on production allocation cost.

**RT-C: d_y=2 kernel designed knowing benchmark dimension (Data leakage)**
The AVX2 kernel is specialized for d_y=2, which is also the benchmark parameter. The plan should declare whether the result is scoped to d_y=2 only and whether the d_y=2 specialization is the intended production artifact.

**RT-D: Primary metric (baseline vs flat_simd) conflates two optimizations (Asymmetric tuning)**
The plan should declare whether this conflation is acceptable for the deployment decision, or whether a positive result for flat_simd that is driven entirely by heap_reuse (allocation elimination) would change the shipping path (i.e., ship heap_reuse instead).

**RT-E: Baseline y_heap never received equivalent AVX2 investment (Asymmetric tuning)**
The plan should acknowledge that the measured speedup is the gain from adding both flat buffer AND AVX2 simultaneously, not the isolated contribution of AVX2 to the flat buffer approach.

**RT-F: Two-stage escalation threshold not pre-registered (Survivorship bias)**
The 1.1× escalation trigger was not derived from a power analysis. The plan should either justify the 1.1× threshold analytically, or acknowledge the elevated Type I error risk and declare it acceptable given the low cost of a false positive (shipping a marginally faster function).

**RT-G: BinaryHeap tie-breaking vs introselect tie-breaking (Evaluation collision)**
If tied Y-distances produce different (but equally valid) kNN sets, the |Δ| < 1e-12 tolerance may spuriously fail. The plan should declare whether the correctness test is testing exact numerical equivalence or algorithmic equivalence, and whether equal-validity tie-breaking differences are acceptable.

**RT-H: Amdahl bound derived from thread-work fraction, not wall-clock (Evaluation collision)**
The plan should explicitly scope the 3.37× upper bound and 1.5× stretch target to "thread-work fraction approximation" and declare that the wall-clock speedup ceiling is unknown.

**RT-I: Phase 0 failure analysis has no completion criterion (No formal gate)**
The plan should specify what constitutes a sufficient Phase 0 analysis to proceed. A minimum bar might be: "root cause is documented as one of the three candidates, with evidence from the rerun-clean source code or from the heap_reuse diagnostic result."

**RT-J: Hardware generalizability not scoped (Single machine)**
The plan should declare that the speedup measurement is specific to the test machine (AMD Ryzen 7 9800X3D or equivalent) and that AVX2 performance varies across CPU microarchitectures. The deployment decision should be scoped to "machines with equivalent AVX2 throughput" or "any AVX2-capable machine with the caveat that speedup may vary."
