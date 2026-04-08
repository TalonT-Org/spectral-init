# Revision Guidance — KD-tree Y k-NN Trustworthiness Experiment

**Plan:** `experiment_plan_kdtree_ynn_trustworthiness_50k_2026-04-06_221102.md`
**Verdict:** REVISE  
**Review timestamp:** 2026-04-06_223004

---

## Required Revisions (Critical Findings)

These gaps directly compromise the validity of the reported speedup numbers. They must be addressed before the experiment runs.

---

### R1 — Benchmark isolation: the two algorithm paths must be measured under equivalent compilation conditions
**Dimension:** baseline_fairness | **Section:** Phase 7: Criterion Bench

**Gap:** The plan describes forcing the brute-force and kdtree paths via a compile-time threshold constant, meaning the two benchmark groups are compiled with different code visibility (dead-code elimination, inlining). Any performance difference between variants that derives from compilation artifact rather than algorithmic cost will be indistinguishable from a genuine speedup.

**Risk:** The reported speedup ratio reflects both algorithmic improvement and compilation effects, making it impossible to attribute the measured difference to the KD-tree algorithm alone.

---

### R2 — Speedup metric definition must account for the asymmetric cost structure
**Dimension:** baseline_fairness | **Section:** Phase 5: KD-tree Implementation

**Gap:** The primary speedup metric `flat_simd_time / kdtree_total_time` places O(n log n) tree build cost in the kdtree denominator while the flat_simd numerator has no corresponding build cost. The plan acknowledges build amortization as a separate analysis (H4), but the primary H1 metric does not separate "total wall-time speedup including build" from "query-only speedup excluding build." This distinction is material to the decision of whether the adaptive flag is worth shipping.

**Risk:** A positive H1 result could be driven entirely by efficient queries while the build cost is not amortized, or vice versa. The decision criterion cannot distinguish these cases.

---

### R3 — Profiling atomics must be reset between benchmark groups
**Dimension:** unit_interference | **Section:** Phase 4: Profiling Atomic Addition

**Gap:** The profiling statics (`Y_DIST_NS`, `Y_KDTREE_BUILD_NS`) are process-scoped with no reset mechanism. When Criterion runs the brute-force and kdtree groups sequentially in the same process, the kdtree group's step-timing reads will include nanosecond counts accumulated during the brute-force group.

**Risk:** The step-level timing breakdown — which is used to confirm build amortization and diagnose which code path dominates — will report inflated totals for the kdtree group, making the profiler data uninterpretable.

---

### R4 — Criterion benchmark groups must not share state across groups in the same process
**Dimension:** unit_interference | **Section:** Phase 7: Criterion Bench

**Gap:** The plan proposes reusing `Y_DIST_NS` for both the brute-force measurement and the kdtree query measurement, and running both as groups in the same Criterion binary. Because Criterion does not restart the process between groups, the brute-force group's entire accumulated `Y_DIST_NS` sum will be present when the kdtree group runs, poisoning the kdtree group's step-timing read.

**Risk:** The step-timing comparison between brute-force and kdtree — the primary evidence for the crossover threshold (H2) and build amortization (H4) — will be systematically wrong.

---

### R5 — Multiple comparisons: pre-specify how familywise error is handled across 6 DVs × 5 n values
**Dimension:** statistical_corrections | **Section:** Analysis Plan / Statistical treatment

**Gap:** The plan tests 6 dependent variables across 5 sample sizes without any pre-specified familywise error rate or false discovery rate correction. With up to 30 comparison cells, the probability of at least one spurious positive claim at nominal α is substantially above α.

**Risk:** A "conclusive positive" verdict based on all sub-claims passing simultaneously may partially reflect inflated Type I error rather than genuine algorithmic superiority. Because the conjunctive structure reduces (but does not eliminate) familywise error, a brief statistical rationale for why the conjunction was chosen is needed to distinguish a principled design from a post-hoc one.

---

### R6 — Conjunctive success criterion requires a pre-specified statistical rationale
**Dimension:** statistical_corrections | **Section:** Success Criteria

**Gap:** The "conclusive positive" outcome requires five simultaneous conditions: speedup ≥ 5× at n=50K AND ≥ 10× at n=100K AND correctness passes AND crossover variance ≤ 2× AND build cost ≤ 25% of savings. No rationale explains why these specific thresholds and this specific conjunction were chosen before the data was collected.

**Risk:** Without documentation, the conjunction cannot be distinguished from an ad hoc design selected post-hoc to inflate apparent rigor. This undermines the replicability and credibility of the final conclusion.

---

### R7 — Y_DIST_NS must not be shared as the canonical name for both the brute-force step and the kdtree query step
**Dimension:** measurement_alignment | **Section:** Dependent Variables (Metrics)

**Gap:** The plan proposes reusing `Y_DIST_NS` for both `tw_y_dist_ns` (brute-force path) and `tw_y_kdtree_query_ns` (kdtree query path). In a profiler session where only one path is active (due to the feature flag), the same atomic name produces different physical quantities. Any tool or analysis script that reads `[timing:y_dist]` from profiler output will not know which code path generated the measurement.

**Risk:** The step-timing comparison (brute-force fill+introselect vs kdtree query) is the principal evidence for the query-only speedup estimate. If both paths report under the same label, the comparison requires manual disambiguation of which run used which path, introducing error-prone bookkeeping.

---

## Recommended Revisions (Warning Findings)

Address these before publication to strengthen the evidence quality.

---

### W1 — Align H0 threshold with the "conclusive negative" decision boundary
**Dimension:** hypothesis_falsifiability | **Section:** Success Criteria

**Gap:** H0 states ≤ 1.5× speedup. The "conclusive negative" outcome requires < 2× speedup. A result of 1.6×–2.0× formally rejects H0 under its own definition but still maps to "do not ship." The outcome space between 1.5× and 2.0× has no defined meaning in the success criteria.

---

### W2 — Define a decision for the 2×–5× inconclusive band, or narrow it
**Dimension:** hypothesis_falsifiability | **Section:** Success Criteria

**Gap:** The 2×–5× band encompasses a large range of outcomes that include results well above H0 yet below H1. An inconclusive result in this range produces no action, meaning the experiment cannot definitively answer the shipping question for a substantial portion of plausible outcomes.

---

### W3 — Define primary DV explicitly (total wall time vs Y k-NN sub-step)
**Dimension:** estimand_clarity | **Section:** Hypothesis / Dependent Variables

**Gap:** H1 refers to "wall-time speedup" but the DV table lists both total wall time and Y k-NN sub-step time as separate top-level metrics. The formal contrast (A vs B on *Y* in *Z*) cannot be written unambiguously until the primary outcome variable is declared.

---

### W4 — Add a repeated-run protocol to support the ≤ 2× crossover variance criterion
**Dimension:** variance_protocol | **Section:** Execution Protocol / Success Criteria

**Gap:** The "conclusive positive" criterion requires crossover n to be identified with ≤ 2× variance across repeated runs. The execution protocol specifies only a single Criterion run. There is no mechanism to measure the crossover variance required by the success criterion.

---

### W5 — Add profiler variance reporting (step-level std across iterations)
**Dimension:** variance_protocol | **Section:** Execution Protocol

**Gap:** The profiler step (`run_profiler.sh`) is a single pass of 30 iterations with no variance output. Any OS jitter or thermal event during this run is silently incorporated without detection. The profiler data provides no confidence interval for step-level timings.

---

### W6 — Unify the three informal error thresholds (CV > 10%, std/mean > 0.15, variance > 20%)
**Dimension:** error_budget | **Section:** Analysis Plan / Success Criteria / Threats

**Gap:** Three separate informal thresholds coexist without a unified policy: CV > 10% (flag measurement), std/mean > 0.15 (thermal flag), variance > 20% (declare inconclusive). It is not specified what happens when multiple thresholds trigger simultaneously or which takes precedence.

---

### W7 — Characterize the allocation asymmetry: kiddo heap-per-query vs flat_simd thread-local Vec
**Dimension:** baseline_fairness | **Section:** Phase 5: KD-tree Implementation

**Gap:** The flat_simd path reuses thread-local pre-allocated `Vec` buffers with no per-iteration allocation. kiddo v5 `nearest_n` allocates a heap priority queue per query. This is a structural advantage for flat_simd beyond its algorithmic complexity that should be acknowledged in the analysis as a potential confounder for the speedup measurement.

---

### W8 — Specify the benchmark group order and consider its effect on thermal/cache fairness
**Dimension:** baseline_fairness | **Section:** Phase 7: Criterion Bench

**Gap:** The two Criterion groups are sequential with no order randomization. The second group runs after 20–45 minutes of prior execution and inherits thermal and cache state from the first. The direction of the bias depends on which group runs second.

---

### W9 — Add `source_type` classification and complete verification criteria to data manifest
**Dimension:** data_acquisition | **Section:** Inputs and Data

**Gap:** No data manifest entry is tagged with `source_type` (external / gitignored / generated). The verification step only covers 2 of the 10 generated files. Without source classification and complete verification, a fresh worktree run may silently fail to acquire all required data.

---

### W10 — Specify a response rule for CV > 10% measurements
**Dimension:** statistical_corrections | **Section:** Analysis Plan

**Gap:** The CV > 10% flag currently triggers a narrative note but no defined corrective action. A high-CV measurement that still yields a favorable point estimate could silently contribute to a "conclusive positive" verdict without triggering any invalidation.

---

### W11 — Pre-specify which of the n values is used for the primary H0 test
**Dimension:** statistical_corrections | **Section:** Dependent Variables / Analysis Plan

**Gap:** The speedup ratio `tw_kdtree_speedup` is derived post-hoc from two measured DVs that are also reported as primary metrics, creating correlated DVs. The dependency structure should be acknowledged and, if any corrections are applied, the correlation should be factored in.

---

### W12 — Address the 1e-12 correctness tolerance vs floating-point non-associativity
**Dimension:** measurement_alignment | **Section:** Hypothesis / Phase 6

**Gap:** `trustworthiness()` is a rank-based integer-sum statistic that depends on the aggregation order of parallel Rayon reductions. Non-deterministic floating-point reduction ordering under Rayon can produce differences at the 1e-12 level even when both implementations use identical k-NN sets. The plan does not acknowledge whether the tolerance is set to account for this or whether it assumes deterministic reduction.

---

### W13 — Bound the production call distribution for d_y values
**Dimension:** ecological_validity | **Section:** Controlled Variables / Motivation

**Gap:** The adaptive feature flag will silently fall back to flat_simd for all d_y ≠ 2 calls. The shipping decision is therefore contingent on d_y = 2 being the dominant production case, but no evidence is provided to bound the fraction of production calls that would actually benefit from the flag.

---

## Red-Team Decision Points

All red-team findings require an explicit decision by the plan author.

| ID | Risk | Decision Required |
|---|---|---|
| RT1 | Asymmetric profiling instrumentation: kdtree gets decomposed per-phase profiling; flat_simd does not. Any overhead removable from kdtree is discoverable before the benchmark runs; flat_simd has no equivalent self-analysis opportunity. | Accept (document that flat_simd profiling is not symmetrized) or Address (add equivalent decomposition to flat_simd) |
| RT2 | Leaf-size 32: origin undocumented. If selected via any prior experimentation against n=50K/k=15, the kdtree path has been tuned against its own evaluation. | Accept (assert 32 is the kiddo-recommended default, not tuned) or Address (document selection rationale) |
| RT3 | Uniform random data yields an uninterpretable negative: a speedup < 5× on uniform random cannot distinguish "kdtree is genuinely slow" from "kdtree is fast on real clustered data but not captured here." | Accept (acknowledge in conclusions that a negative result is necessary but not sufficient to reject kdtree) or Address (add at least one structured-data condition) |
| RT4 | Survivorship bias via inconclusive zone: inconclusives can be discarded without a reporting obligation. | Accept (pre-commit that all runs are reported regardless of outcome) or Address (add a protocol requiring inconclusive runs to be logged) |
| RT5 | Evaluation collision: the same warm_up_time parameter differentially benefits kdtree (tree cache warmup) vs flat_simd (smaller warmup footprint). | Accept (document that warm_up_time was not optimized for either path) or Address (specify warm_up_time rationale independently for each group) |
| RT6 | Thread count as confounder: Rayon thread count is not recorded in deliverables. If the two paths have different parallelism profiles at different thread counts, speedup is partially a function of thread count rather than algorithm. | Accept (document thread count in report and scope conclusions to tested thread count) or Address (add RAYON_NUM_THREADS to the list of controlled variables with a fixed value) |
| RT7 | `N_KDTREE_THRESHOLD = 0` may affect shared code paths in both benchmark groups. | Accept (assert threshold affects only the dispatch branch) or Address (verify isolation) |
| RT8 | Post-hoc crossover identification on same n values: the crossover n is identified from the same data used to define the hypothesis, with no held-out validation. | Accept (scope conclusions explicitly as "description of tested n values, not a predictive crossover estimate") or Address (add at least one held-out n value not in the hypothesis) |
