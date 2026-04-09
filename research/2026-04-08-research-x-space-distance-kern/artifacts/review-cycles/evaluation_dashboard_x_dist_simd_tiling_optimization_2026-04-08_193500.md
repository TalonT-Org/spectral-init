# Evaluation Dashboard: X-space Distance Kernel SIMD Optimization

**Plan:** `experiment_plan_x_dist_simd_tiling_optimization_2026-04-08_192500.md`  
**Review timestamp:** 2026-04-08 19:35:00  
**Reviewer:** review-design skill (automated multi-level analysis)

---

## Verdict

```
╔══════════════════════════════════════╗
║           VERDICT:  G O              ║
║  0 critical · 50 warnings · 18 info  ║
║  threshold: 70 warnings (14 dims×5)  ║
╚══════════════════════════════════════╝
```

The plan is clear in intent, well-scoped, and has a falsifiable primary hypothesis. Fifty warnings
were raised across 14 dimensions — well within the proportional budget of 70. The most actionable
concerns cluster in three areas the implementer should be aware of before execution:

1. **X_DIST_NS thread-time vs. wall-clock confusion** — the step-timing profiler accumulates
   summed thread-time across all Rayon workers, not wall-clock elapsed time for the step. This
   affects Amdahl model validity and the H1 "≥4× x_dist speedup" claim operationalization.

2. **Asymmetric evaluation design** — the baseline is intentionally "broken" while experimental
   variants receive iterative implementation effort, creating structural asymmetry that six
   adversarial (red-team) findings flag as threats.

3. **Multiple comparisons not acknowledged** — four hypotheses (H1–H4) evaluated against shared
   measurement data with no FWER discussion.

---

## Triage Classification

| Field | Value | Source |
|-------|-------|--------|
| experiment_type | benchmark | frontmatter extraction (Rule 1: IVs=method names, DVs=performance metrics, multiple comparators) |
| hypothesis_h0 | Looped SIMD kernel does NOT improve total trustworthiness time by >1.5× at n=10K, d_x=50 | ## Hypothesis |
| hypothesis_h1 | Looped SIMD reduces x_dist by ≥4× and total trustworthiness by ≥1.5× | ## Hypothesis |
| estimand | Treatment=SIMD kernel variant, Outcome=total trustworthiness wall-clock, Population=n=10K d_x=50 synthetic on 9800X3D, Contrast=ratio of Criterion medians | extracted |
| metrics | 4 DVs: total wall-clock, x_dist step time, TW parity, dist_sq_* microbench | ## Dependent Variables |
| baselines | current (2-fixed-load AVX2), avx2_looped, avx512_looped, tiled variants | ## Independent Variables |
| statistical_plan | Criterion median + 95% CI, speedup ratios, Amdahl projection, CI overlap check | ## Analysis Plan |
| success_criteria | ≥1.5× with non-overlapping CIs, correctness |Δ|<1e-6, H2/H3/H4 verdicts | ## Success Criteria |

**Secondary modifiers active:**
- `+causal` — Amdahl mechanism claim in H1 → causal_structure escalated S→L (spawned)
- `+multi_metric` — 4 DVs (≥3) → statistical_corrections escalated M→H
- `+deployment` — production ship decision → ecological_validity floor = M

---

## Dimension Scorecard

| Dimension | Level | Weight | Findings | Severity Summary |
|-----------|-------|--------|----------|-----------------|
| estimand_clarity | L1 | H | 8 | 4W 4I |
| hypothesis_falsifiability | L1 | H | 5 | 3W 2I |
| baseline_fairness | L2 | H | 6 | 4W 2I |
| causal_structure | L2 | L (+causal) | 3 | 2W 1I |
| unit_interference | L2 | H | 6 | 6W |
| red_team | concurrent | — | 6 | 6W (all requires_decision) |
| error_budget | L3 | M | 4 | 3W 1I |
| statistical_corrections | L3 | H (+multi_metric) | 4 | 4W |
| variance_protocol | L3 | H | 4 | 4W |
| benchmark_representativeness | L4 | M | 4 | 3W 1I |
| ecological_validity | L4 | M | 4 | 3W 1I |
| measurement_alignment | L4 | M | 4 | 3W 1I |
| reproducibility_spec | L4 | M | 4 | 3W 1I |
| data_acquisition | L4 | M | 4 | 2W 2I |

**Totals:** 0 critical · 50 warnings · 18 info across 14 active dimensions  
**Warning threshold:** 70 (14 × 5)  
**causal_structure (S in base matrix):** activated by `+causal` modifier; passed foothold validation (Amdahl mechanistic content present)

---

## Level 1 Findings — Estimand Clarity

> Severity calibration: `benchmark` → absent formal estimand = `warning` (not critical)

**[W] H1 double-estimand** `## Hypothesis`
The primary estimand conflates two distinct outcomes in the same hypothesis statement. H1 simultaneously claims ≥4× speedup on the x_dist step AND ≥1.5× speedup on total trustworthiness wall-clock time. These are separable contrasts (A vs B on Y1, A vs B on Y2), each with its own threshold and measurement method. The success criteria section treats them as separate verdicts, but the hypothesis itself does not identify which outcome is the primary estimand against which H0/H1 is adjudicated.

**[W] H4 not a contrast** `## Hypothesis`
H4 ("Block-triangular symmetry exploitation is infeasible") has no independent variable being manipulated, no measurement protocol, and no analysis step. As written it is a design assumption rather than a testable contrast — it cannot be written as A vs B on Y in Z.

**[I] Baseline compilation ambiguity** `## Independent Variables`
It is unclear whether the baseline is the existing production kernel or a freshly compiled version under the same `target-cpu=native` flag. If the baseline has never been compiled with AVX-512 availability, compiler auto-vectorization may produce a different scalar tail than expected.

**[W] All metrics are NEW; instrumentation unvalidated** `## Dependent Variables`
All four metrics are marked "NEW" — the measurement instruments do not yet exist at plan time. The x_dist step time relies on `tw_profiler` and a specific JSON field (`step_timing.x_dist`) that have not been defined or validated against the measurement boundary they are intended to isolate.

**[I] Correctness gate gating condition ambiguous** `## Dependent Variables`
The plan does not specify which kernel variants must pass the correctness gate, nor whether it applies to the baseline. If the baseline itself fails the gate, it is unclear whether the experiment proceeds.

**[W] Baseline-shifting across H1, H2, H3** `## Analysis Plan`
The speedup ratio baseline shifts meaning across hypotheses: "current" for H1, "avx2_looped" for H2, presumably best-SIMD for H3. This baseline-shifting means the three secondary contrasts are not commensurable under a single estimand definition.

**[I] Amdahl projection has no pass/fail criterion** `## Analysis Plan`
The Amdahl projection check is listed as an analysis step but has no defined outcome. It is unclear whether deviation between observed and predicted speedup affects the H1 verdict or is merely diagnostic.

**[I] Auto-vectorization not controlled in estimand** `## Threats to Validity`
The threat of compiler auto-vectorization of the scalar tail is acknowledged but not controlled for. The actual SIMD coverage of the baseline is unspecified, making the estimand ("84% fall to scalar arithmetic") potentially incorrect.

---

## Level 1 Findings — Hypothesis Falsifiability

> Severity calibration: `benchmark` → comparison goal without formal H0 = `warning` (not critical)

**[W] H4 asserted without measurement** `## Hypothesis`
H4 ("symmetry exploitation infeasible, memory cost ≥ 800 MB") rests on an inline calculation rather than any empirical observation scheduled in the analysis plan. No section in the Analysis Plan measures or verifies this figure. H4 cannot be falsified or supported by the experiment as designed.

**[W] Amdahl fractions hardcoded, unverified** `## Analysis Plan`
The Amdahl projection uses hardcoded fractions (0.411, 0.589) with no measurement step to verify these fractions on the current machine and build configuration before the benchmark runs. If actual fractions differ, the projection comparison is uninterpretable.

**[I] H3 verdict gap between 5–10%** `## Success Criteria`
Tiling gains between 5% and 10% have no prescribed outcome ("< 5% → skip", "> 10% → add"), leaving an ambiguous region where no decision rule applies.

**[W] Inconclusive escape clause not bounded** `## Success Criteria`
The CV > 15% inconclusive condition has no maximum retry count or terminal fallback decision rule. If high variance persists, the experiment has no terminal path — neither H0 nor H1 is ever accepted.

**[I] H3 machine-specificity not acknowledged** `## Threats to Validity`
H3's claim is explicitly conditioned on the 96 MB V-Cache machine. The plan does not acknowledge that the H3 verdict is machine-local and cannot be generalized.

---

## Level 2 Findings — Baseline Fairness

**[W] Revert mechanism unspecified** `## Execution Protocol`
The protocol "reverts src/metrics.rs (or uses git stash/branch)" but does not specify which mechanism will be used, nor includes a verification step confirming the revert was clean before each variant's measurement run. The two mechanisms differ in stale build artifact behavior.

**[W] Fixed execution ordering** `## Implementation Phases`
The baseline always runs first (Phase 3 before Phases 4–6). The baseline is measured cold while experimental variants run with a warmer OS page cache, branch predictor state, and Rayon thread pool initialization. No counterbalanced ordering is specified.

**[W] Correctness gate applied asymmetrically** `## Dependent Variables`
The plan does not explicitly state that the sklearn parity test is run against the baseline variant. If the baseline is assumed correct and only experimental variants are gated, the check is applied asymmetrically.

**[W] Phase 7 tiling symmetry unspecified** `## Implementation Phases — Phase 7`
Phase 7 is conditional on SIMD kernels failing. It may not receive the same n×d_x sweep, Criterion sample count, or correctness gate as Phases 4 and 5. Symmetric resource allocation is uncertain.

**[I] Controlled variables not verified per run** `## Controlled Variables`
The plan relies on ambient environment state for thread count and compiler flags rather than explicitly verifying these values per variant run.

**[I] Auto-vectorization of baseline scalar tail unverified** `## Threats to Validity`
The threat is acknowledged but no mitigation or characterization is specified, leaving the baseline's effective SIMD coverage (and therefore the speedup denominator) as unknown.

---

## Level 2 Findings — Causal Structure

> Activated by `+causal` modifier (Amdahl mechanism claim in H1)

**[W] Amdahl model misspecification: thread-time ≠ wall-clock** `## Hypothesis`
The 58.9% x_dist fraction is derived from profiler atomics that accumulate *summed thread-time* across all 8 Rayon workers, not wall-clock elapsed time for the step. The wall-clock fraction is approximately `58.9% / T` if other steps parallelize equally. Using the summed thread-time ratio directly in Amdahl's formula overestimates predicted total speedup. This is a systematic misspecification of the mechanistic model.

**[W] Auto-vectorization effect unquantified in model** `## Hypothesis`
If the compiler auto-vectorizes the baseline's scalar tail under `target-cpu=native`, the measured x_dist speedup S is smaller than the theoretical maximum. The model does not quantify how much smaller S becomes, leaving the projected ≥4× x_dist speedup falsifiable in direction but not in magnitude.

**[I] Penalty step O(n) scan shifts Amdahl fractions at large n** `## Analysis Plan`
The penalty step contains an O(n) scan per violating neighbor that grows faster than x_dist (O(n·d_x)) as n increases. The model's fixed fractions (0.411, 0.589) do not account for this n-dependent shift, making the ≥1.5× total speedup threshold more stringent at large n.

---

## Level 2 Findings — Unit Interference

**[W] CPU thermal state spillover** `## Threats to Validity`
AMD Zen 5 boosts aggressively. The first variant executes partly during the CPU's ramp-up transient while subsequent variants run at sustained boost frequency. Thermal state at Criterion measurement start differs systematically across variants, with no randomized or counterbalanced ordering.

**[W] Cache state spillover across variants** `## Threats to Validity`
The 96 MB 3D V-Cache can hold the entire working set for n=1K and n=5K, and possibly n=10K. After the first variant warms the cache, subsequent variants see a pre-warmed cache not representative of production cold-start. The fixed execution order means this benefit is not distributed evenly.

**[W] Microarchitectural state contamination** `## Threats to Validity`
The branch predictor, instruction TLB, and µop cache trained on one variant's access patterns persist into the next variant's Criterion warmup phase. For SIMD variants that differ in inner-loop branching structure (looped vs. scalar tail), this can bias warmup samples if warmup duration is insufficient.

**[W] Incremental compilation artifact leakage** `## Execution Protocol`
When switching kernel variants via revert-and-recompile, Rust's incremental compilation may reuse cached object files from the previous build. Shared codegen units compiled together with the changed kernel may not be fully recompiled, potentially leaving traces of a prior variant's binary in the artifact.

**[W] Background Criterion activity competing for CPU** `## Execution Protocol`
Criterion performs CPU-bound post-benchmark analysis and HTML report generation. If a prior variant's report generation overlaps with the next variant's measurement window, effective Rayon thread capacity varies between variants in a non-deterministic, ordering-dependent way.

**[W] correctness.json attribution risk** `## Execution Protocol`
The protocol appends each variant's correctness results to a shared `correctness.json`. If a silent kernel-switch failure (stash conflict, compilation error swallowed) occurs, a subsequent variant's append may follow the prior entry without clear demarcation, misattributing results.

---

## Adversarial Findings (Red-Team)

> All findings `requires_decision: true`. Severity capped at `warning` for `benchmark` type.

**[W] Goodhart exploitation via profiling feature asymmetry** `## Analysis Plan`
`requires_decision: true`  
The speedup ratio is `baseline_median / variant_median`. The `profiling` Cargo feature adds `Instant::now()` and `AtomicU64::fetch_add` probes inside the hot parallel loop. If the baseline binary is compiled with `--features profiling` (to obtain step_timing breakdowns) while variants are measured without it, the baseline wall-clock time is inflated by instrumentation overhead, artificially inflating the speedup ratio. The plan does not specify that feature sets must be identical across all timing runs used in the final verdict.

**[W] Data leakage via evaluation-dimension foreknowledge** `## Controlled Variables / Dependent Variables`
`requires_decision: true`  
The benchmark locks d_x to {10, 50} before kernel design is complete. If the kernel is iteratively implemented and tested against these exact dimensions, the loop stride and unroll factor will converge on what is optimal for d_x=50 specifically (50/8 = 6 remainder 2), not for the general case. Fixing evaluation dimensions before finalizing the kernel design creates a leakage path from the evaluation set into the implementation choices.

**[W] Asymmetric tuning: baseline deliberately broken** `## Phase 3 vs. Phases 4–5`
`requires_decision: true`  
The baseline is explicitly described as "broken" (covering only 8 of 50 dimensions with SIMD) and receives zero engineering effort. The experimental variants undergo dedicated implementation phases (4 and 5) with iterative improvement. A fair comparison would require the baseline to represent the best achievable performance of the same algorithmic approach (fixed 2-load AVX2), not the current unoptimized state.

**[W] Survivorship bias via asymmetric stopping** `## Phase 6 — Conditional Evaluation`
`requires_decision: true`  
Phase 6 states: "if looped AVX2 already satisfies ≥1.5×, AVX-512 is not required for a pass verdict." This creates asymmetric stopping: the experiment terminates with "pass" when results look good, but continues into Phase 7 (tiling, additional optimisation) when they do not. Each additional phase applies further optimisation effort exclusively to experimental variants. An experiment that would narrowly miss ≥1.5× at AVX2-only may cross the threshold after tiling is added.

**[W] Evaluation collision: profiler inside measured code** `## tw_profiler / step_timing infrastructure`
`requires_decision: true`  
The `tw_profiler` and x_dist step timing execute `Instant::now()` and `AtomicU64::fetch_add` inside the Rayon parallel map on every row. On an 8-thread run at n=50K, this is ~100K atomic operations in the hot path per call. The act of measuring x_dist inflates x_dist wall-clock time. If the baseline is profiled (to obtain the 58.9% breakdown) but variants are benchmarked via Criterion without the `profiling` feature, the reported step-time reduction will overstate actual speedup.

**[W] Asymmetric effort: iterative improvement feedback loop** `## Phase 4–5 vs. Phase 3`
`requires_decision: true`  
Phases 4 and 5 involve creating NEW optimized kernels through multiple implementation passes; the experimenter sees the speedup ratio after each revision and can stop iterating when the ratio is satisfactory. The baseline is frozen. The final experimental variant is the one that produces the best observed ratio against this specific baseline — equivalent iterative effort applied to the baseline would narrow the gap.

---

## Cannot Assess

The following dimensions could not be fully evaluated from the plan document alone:

1. **tw_profiler boundary definition** — The `step_timing.x_dist` field's exact timing boundary is
   not defined in the plan. Whether it includes data movement, Rayon task dispatch overhead, or
   only kernel arithmetic cannot be assessed. The plan states the binary "requires extending" in
   Phase 3, meaning the boundary has not been defined yet.

2. **Phase 7 tiling equivalence** — Phase 7 is conditional on SIMD kernels failing the ≥1.5×
   criterion. Whether its measurement protocol will be fully symmetric with Phases 4–5 cannot be
   assessed because it is not defined — its execution depends on an outcome that may not occur.

3. **Incremental compilation scope** — Whether the Rust incremental compilation cache's codegen
   unit boundaries overlap the changed kernel code cannot be assessed without knowing the
   crate's LTO configuration and codegen-units setting in Cargo.toml.

4. **Criterion auto-configuration adequacy** — Whether Criterion's default warm-up and sample
   count settings are sufficient for sub-microsecond `dist_sq_*` microbenchmarks on this specific
   hardware cannot be assessed without knowing the expected kernel latency in nanoseconds and the
   resulting sample count auto-selected by Criterion's sampling routine.

---

## Mechanizable Check Log

Binary checks that could be automated in future CI or pre-flight validation:

| Check | Automatable | Finding Reference |
|-------|-------------|-------------------|
| All metrics marked NEW before any script exists | Yes — grep plan for "NEW" | L1 estimand |
| RNG seed declared in controlled variables | Yes — parse controlled variables table | L3 variance |
| Profiling feature flag consistency across variant runs | Yes — parse script invocations for `--features` | Red-team |
| Correctness gate applied to baseline variant in execution protocol | Yes — check if baseline appears in parity test invocation | L2 baseline |
| H4 appears in success criteria without corresponding data manifest entry | Yes — cross-reference hypotheses against analysis steps | L4 data_acquisition |
| Criterion version pinned to patch level | Yes — check Cargo.toml criterion version string | L4 reproducibility |
| Feature branch retention plan stated | Yes — check for SHA or archive reference in plan | L4 reproducibility |
| H3 verdict covers full [0%, ∞%) range | Yes — check that lower + upper bounds form a partition | L1 falsifiability |

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: GO
experiment_type: benchmark
critical_count: 0
warning_count: 50
red_team_count: 6
active_dimensions: 14
warning_threshold: 70
secondary_modifiers:
  - +causal
  - +multi_metric
  - +deployment
notable_requires_decision:
  - measurement_alignment: X_DIST_NS thread-time vs wall-clock misoperationalization
  - reproducibility_spec: feature branch deletion risk
  - reproducibility_spec: kernel switching protocol ambiguity
  - red_team: profiling feature asymmetry
  - red_team: asymmetric stopping
  - red_team: evaluation collision
  - red_team: asymmetric tuning
  - red_team: asymmetric effort
  - red_team: data leakage via dimension foreknowledge
```
