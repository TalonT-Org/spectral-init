# Experiment Design Review — Evaluation Dashboard

**Plan:** `experiment_plan_kdtree_ynn_trustworthiness_50k_2026-04-06_221102.md`
**Review timestamp:** 2026-04-06_223004

---

## Verdict

```
╔══════════════════════════════════════╗
║           VERDICT: REVISE            ║
╚══════════════════════════════════════╝
```

**Classification:** `benchmark`  
**Active dimensions evaluated:** 13  
**Stop triggers:** 0 (no L1 critical findings; red-team severity capped at `warning` for benchmark type)  
**Critical findings:** 7 (from non-stop-trigger dimensions → REVISE, not STOP)  
**Warning findings:** ~54  
**Warning threshold:** 65 (13 dims × 5 budget/dim)

The plan is well-structured and contains a thorough implementation specification. The REVISE verdict is driven by seven critical design gaps across `baseline_fairness`, `unit_interference`, `statistical_corrections`, and `measurement_alignment` that would undermine the trustworthiness of the reported speedup numbers if not addressed.

---

## Dimension Scorecard

| Dimension | Level | Weight | Critical | Warning | Info | Status |
|---|---|---|---|---|---|---|
| estimand_clarity | L1 | H | 0 | 2 | 3 | ⚠ Warnings |
| hypothesis_falsifiability | L1 | H | 0 | 3 | 1 | ⚠ Warnings |
| baseline_fairness | L2 | H | 2 | 4 | 2 | 🔴 Critical |
| unit_interference | L2 | H | 2 | 4 | 0 | 🔴 Critical |
| red_team | L2 | H | 0 | 8 | 0 | ⚠ Warnings |
| error_budget | L3 | H | 0 | 4 | 3 | ⚠ Warnings |
| statistical_corrections | L3 | H | 2 | 4 | 1 | 🔴 Critical |
| variance_protocol | L3 | H | 0 | 6 | 2 | ⚠ Warnings |
| ecological_validity | L4 | M | 0 | 5 | 2 | ⚠ Warnings |
| measurement_alignment | L4 | M | 1 | 4 | 1 | 🔴 Critical |
| reproducibility_spec | L4 | M | 0 | 3 | 3 | ⚠ Warnings |
| benchmark_representativeness | L4 | M | 0 | 5 | 2 | ⚠ Warnings |
| data_acquisition | L4 | M | 0 | 3 | 1 | ⚠ Warnings |

> Dimensions with weight = S (SILENT) are not spawned and not listed: `causal_structure`.

---

## Critical Findings (REVISE Triggers)

### C1 — baseline_fairness: Bench toggle mechanism creates compile-variant asymmetry
**Section:** Phase 7: Criterion Bench  
**Finding:** The two benchmark groups force their respective paths via different compile-time configurations (`N_KDTREE_THRESHOLD = usize::MAX` vs `= 0`). Any difference in inlining, dead-code elimination, or conditional-branch removal between the two compiled variants could inflate or deflate one path's measurement independently of algorithmic cost.

### C2 — baseline_fairness: Speedup ratio denominator includes build cost; numerator does not
**Section:** Phase 5: KD-tree Implementation  
**Finding:** `tw_kdtree_speedup = flat_simd_time / kdtree_total_time` charges the O(n log n) tree construction cost to the kdtree denominator while `flat_simd` has zero build cost. The plan calls this metric a "speedup ratio" without acknowledging that it conflates construction and query costs in the denominator while the numerator has no corresponding setup cost. The build amortization analysis (H4) partially addresses this but is separate from the primary speedup claim.

### C3 — unit_interference: Profiling atomics accumulate across benchmark groups without reset
**Section:** Phase 4: Profiling Atomic Addition  
**Finding:** The profiling statics (`Y_DIST_NS`, `Y_KDTREE_BUILD_NS`) are process-scoped and initialized once at process start. Criterion runs both benchmark groups in the same process without resetting these atomics. The `kdtree` group's reported step timings will include the `brute_force` group's accumulated nanosecond counts, making per-group step timing comparisons invalid.

### C4 — unit_interference: Same-process sequential groups contaminate step-timing reads
**Section:** Phase 7: Criterion Bench  
**Finding:** Because both groups share the same OS process with no reset between them, any step-timing atomic read at the end of the `kdtree` group reflects the cumulative total from both groups. The plan explicitly reuses `Y_DIST_NS` for the kdtree query path, amplifying this problem: the brute-force group's entire accumulated sum is already present in `Y_DIST_NS` when the kdtree group begins accumulating.

### C5 — statistical_corrections: 6 DVs × 5 n values yields 30 comparisons with no FWER pre-specification
**Section:** Analysis Plan / Statistical treatment  
**Finding:** The plan tests six dependent variables simultaneously across five sample sizes (up to 30 comparison cells) with no familywise error rate or false discovery rate correction pre-specified. Claiming simultaneous positive outcomes across multiple DVs without acknowledging inflated Type I error risk is a design gap for a benchmark with H-weight and ≥3 DVs (`+multi_metric` modifier active).

### C6 — statistical_corrections: Conjunctive success criterion bundles 5 sub-claims without statistical rationale
**Section:** Success Criteria  
**Finding:** The "conclusive positive" criterion requires speedup ≥ 5× at n=50K AND ≥ 10× at n=100K AND all correctness tests pass AND crossover variance ≤ 2× AND build cost ≤ 25% of savings — five distinct claims in one AND-conjunction. No pre-specified statistical rationale explains why the conjunctive structure was chosen (it reduces false positives but without documentation this is indistinguishable from an ad hoc post-hoc design).

### C7 — measurement_alignment: Y_DIST_NS reused for both paths makes step-level comparison ambiguous
**Section:** Dependent Variables (Metrics)  
**Finding:** The plan proposes reusing the `Y_DIST_NS` accumulator for both the brute-force path and the KD-tree query path. Both branches feed the same atomic, so a single profiler session where the feature flag selects one path cannot be distinguished in the output from a session that selected the other. The plan never specifies that these are measured in separate runs — it implies they are measurement-comparable outputs of the same instrumentation label.

---

## Warning Highlights (Selected Key Findings)

### Level 1 — Estimand / Falsifiability

- **estimand_clarity (warning):** The primary outcome variable is ambiguous — the plan lists both `trustworthiness()` total wall time and Y k-NN sub-step time as top-level DVs without declaring which is the primary estimand for the H0/H1 contrast.
- **hypothesis_falsifiability (warning):** The "conclusive negative" threshold (< 2× at n=50K) is inconsistent with H0 (≤ 1.5×). A result of 1.6×–2.0× formally rejects H0 yet triggers the "conclusive negative" outcome label, creating a decision zone where H0 is false but the experiment still concludes "do not ship."
- **hypothesis_falsifiability (warning):** The 2×–5× "inconclusive" band covers a wide range with no defined action, making the experiment unable to falsify H1 for that large swath of outcomes.

### Level 2 — Baseline Fairness / Unit Interference

- **baseline_fairness (warning):** kiddo `nearest_n` allocates a heap priority queue per query; `flat_simd` uses pre-allocated thread-local `Vec` buffers. This structural memory allocation asymmetry gives `flat_simd` an advantage beyond its algorithmic complexity.
- **baseline_fairness (warning):** Sequential Criterion groups run for 20–45 minutes total, with no group-order randomization. The second group always inherits thermal and cache history from the first.
- **unit_interference (warning):** Thread-local scratch `Vec` allocators are pre-sized by the first benchmark group, asymmetrically benefiting whichever treatment runs second.
- **unit_interference (warning):** The profiler script accumulates step-timing values across all `n` values in a single process invocation without reset, making cross-size comparisons cumulative rather than per-size.

### Level 2 — Red-Team (all `requires_decision: true`)

See dedicated section below.

### Level 3 — Error Budget / Variance Protocol

- **error_budget (warning):** Three overlapping informal error thresholds exist: CV > 10% (flag), std/mean > 0.15 (thermal flag), and variance > 20% (inconclusive trigger) — without a unified policy for when they conflict.
- **variance_protocol (warning):** The "conclusive positive" criterion requires crossover n identified with ≤ 2× variance across repeated runs, but the execution protocol specifies only a single Criterion run. No repeated-run protocol exists to satisfy this criterion.
- **variance_protocol (warning):** The profiler step has no variance reporting — single-pass 30 iterations with no std output — making step-level timing unreliable under OS/thermal jitter.

### Level 4 — Ecological Validity / Reproducibility

- **ecological_validity (warning):** An adaptive kdtree feature flag presupposes a crossover threshold; the benchmark identifies crossover post-hoc on the same 5 n values used to define the hypothesis. This crossover cannot be validated on held-out n values.
- **ecological_validity (warning):** The benchmark is evaluated on the host ISA (`target-cpu=native`), but the crate is distributed as source. On hardware without AVX2, kiddo's scalar fallback may underperform, making the shipping decision ISA-specific in ways the plan does not address.
- **reproducibility_spec (warning):** The execution protocol specifies a single Criterion run with no mechanism to record the Rayon thread count in the output artifacts — yet thread count is material to speedup ratios for both paths.

### Level 4 — Measurement Alignment / Benchmark Representativeness

- **measurement_alignment (warning):** The 1e-12 correctness tolerance may be violated by non-deterministic Rayon parallel reduction ordering even when both implementations produce identical k-NN sets, because `trustworthiness()` is a rank-based summation and floating-point non-associativity can produce differences at that precision.
- **benchmark_representativeness (warning):** k is fixed at 15; larger k values (30–50) change the kdtree priority queue cost, potentially shifting the crossover threshold. Conclusions at k=15 may not generalize to production calls with different k.
- **benchmark_representativeness (warning):** The expected speedup multipliers (≥5× at n=50K, ≥10× at n=100K) were derived from theoretical asymptotic expectations, not from empirical calibration on this codebase. If prior optimizations to `flat_simd` have already reduced the y_kNN bottleneck fraction, the achievable speedup may differ from the theoretical prediction.

### Level 4 — Data Acquisition

- **data_acquisition (warning):** The data manifest does not classify entries by `source_type` (external / gitignored / generated), making it impossible to confirm all external and gitignored data sources have the required acquisition commands and verification criteria.
- **data_acquisition (warning):** Only 2 of the 10 generated files (`n50000_x.npy` and `n100000_y.npy`) have a stated verification criterion; the remaining eight have none.

---

## Adversarial Findings (Red-Team) — All `requires_decision: true`

| # | Challenge Type | Section | Risk |
|---|---|---|---|
| RT1 | Asymmetric effort | Implementation | kdtree receives dedicated build-time profiling atomic allowing per-invocation optimization discovery; flat_simd baseline has no equivalent decomposition. Any overhead removed from kdtree during implementation will not trigger a symmetric review of flat_simd. |
| RT2 | Goodhart exploitation | Controlled Variables | The leaf-size parameter (32) is baked into the type before benchmarking. If this value was selected through any prior experimentation against the same n/k configuration, the proposed method has been tuned against its own evaluation with no documentation of how 32 was chosen. |
| RT3 | Unfalsifiable in negative direction | Inputs and Data | Uniform random data is acknowledged as conservative (pessimistic) for kdtree speedup. A negative result cannot distinguish between "kdtree is genuinely slow" and "kdtree is fast on real data but this benchmark cannot show it," making H0 unfalsifiable in practice on real-data grounds. |
| RT4 | Survivorship bias | Analysis Plan | The 2×–5× "inconclusive" band allows re-running the experiment (with different leaf size, thread count, or seed) and discarding results as inconclusive rather than negative. No protocol specifies that inconclusives must be reported. |
| RT5 | Evaluation collision | Controlled Variables | Criterion warm_up_time (10s) benefits the kdtree path more than flat_simd because tree construction populates CPU cache and branch predictor state that persists into measured samples; flat_simd's introselect has a smaller warm-up footprint. The same infrastructure parameter affects the two treatments unequally. |
| RT6 | Uncontrolled confounder | Controlled Variables | Rayon thread count is left at host default and not recorded as a controlled variable. If the two paths have different task granularities or parallelism profiles, the speedup ratio is partially a function of thread count rather than algorithm efficiency, and results will not be reproducible on machines with different core counts. |
| RT7 | Goodhart / shared code path | Implementation | `N_KDTREE_THRESHOLD = 0` is a global flag forcing the kdtree path at all n. If this flag also affects any shared code paths used by the flat_simd benchmark group, the measured flat_simd times may reflect the overhead of operating in a non-default configuration. |
| RT8 | Data leakage | Analysis Plan | Crossover n is identified post-hoc from the same five n values used to define the hypothesis. There is no held-out n value to validate the crossover claim; the reported crossover is a description of the training data, not a prediction. |

---

## Cannot Assess

The following dimensions could not be evaluated with confidence due to absent plan content:

1. **Rayon task granularity for flat_simd vs kdtree** — The plan does not characterize whether the two paths have equivalent Rayon task granularity (e.g., chunk sizes, work-stealing patterns). Without this, the unit of parallelism is unclear and the attribution of speedup to algorithmic vs scheduling differences cannot be assessed.

2. **Leaf-size selection provenance** — The KD-tree leaf size (32) is fixed as a compile-time constant but the plan contains no record of how this value was chosen. Whether it was empirically tuned against the evaluation benchmark conditions or selected a priori from library defaults cannot be determined from the plan text.

3. **Thermal baseline characterization** — The plan does not specify the thermal state of the host at benchmark start (idle time, prior workloads). Criterion's warmup addresses short-term cache state but not host thermal envelope. Without a pre-run idle period or thermal probe, it is not possible to assess whether the warm_up_time is sufficient for the host's thermal recovery curve.

4. **Production d_y distribution** — The plan restricts the kdtree path to d_y = 2 and asserts this "covers dominant UMAP use case" without providing evidence about the actual distribution of d_y values in production calls via umap-rs. The fraction of calls that would benefit from the adaptive flag is uncharacterized.

---

## Mechanizable Check Log

The following checks could be automated in a future design-review pipeline:

| Check | Binary? | Signal |
|---|---|---|
| `sample_size` ≥ 30 (Criterion default) | Yes | `sample_size(10)` fails |
| All DVs have a unique canonical name | Yes | Pass |
| Success criteria have exactly one primary DV | No | Ambiguous (wall time vs sub-step) |
| Profiling atomics have reset mechanism | Yes | Fail (no reset anywhere in plan) |
| Two benchmark groups in same process | Yes | Fail (same Criterion binary) |
| FWER correction name present in statistical plan | Yes | Fail (absent) |
| Repeated-run count ≥ 3 for variance criterion | Yes | Fail (single run) |
| Data manifest has `source_type` for all entries | Yes | Fail (absent) |
| All generated files have verification criteria | Yes | Fail (2/10 covered) |
| Seed documented for all RNG-dependent steps | Yes | Pass (seed 42 for both) |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 7
warning_count: 54
red_team_count: 8
active_dimensions: 13
warning_threshold: 65
```
