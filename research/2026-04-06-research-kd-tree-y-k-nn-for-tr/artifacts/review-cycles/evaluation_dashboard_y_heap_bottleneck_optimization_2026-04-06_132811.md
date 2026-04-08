# Evaluation Dashboard: y_heap Bottleneck Optimization

**Verdict: REVISE**
**Experiment type:** benchmark
**Plan file:** `.autoskillit/temp/plan-experiment/experiment_plan_y_heap_bottleneck_optimization_2026-04-06_131330.md`
**Reviewed:** 2026-04-06

---

## Classification Summary

| Field | Value | Source |
|---|---|---|
| experiment_type | benchmark | extracted (Rule 1: IVs = method names, DVs = perf metrics, multiple comparators) |
| secondary_modifiers | +multi_metric (4 DVs ≥ 3) → statistical_corrections M→H | extracted |
| hypothesis_h0 | speedup < 1.5× at n=10K, k=15, 95% CI overlaps 1.0× | extracted |
| hypothesis_h1 | ≥ 1.5× speedup CI LB > 1.0×; y_heap fraction ≤ 40% | extracted |
| estimand | variant (baseline/h1/h2) vs baseline on trustworthiness wall-time at n=10K, k=15 | extracted |
| primary_metric | y_heap_speedup_ratio (Criterion mean ratio, n=10K) | extracted |
| baselines | baseline (BinaryHeap), h1_introselect, h2_simd | extracted |
| statistical_plan | 95% CI from Criterion; 10 samples SamplingMode::Flat; CI LB > 1.0 threshold | extracted |

---

## Verdict Banner

```
┌─────────────────────────────────────────────────────────┐
│  VERDICT: REVISE                                        │
│                                                         │
│  22 critical findings across 7 dimensions require      │
│  design revision before execution.                      │
│  No STOP-class structural defects (L1/red-team clear). │
└─────────────────────────────────────────────────────────┘
```

The plan is well-motivated, operationally detailed, and the hypothesis is falsifiable. Revision is required primarily around: (1) baseline description accuracy vs existing codebase, (2) measurement incommensurability (CPU time vs wall-time for step fraction), (3) absent power analysis and multiple-comparisons structure, (4) under-specified reproducibility environment, and (5) thread-local buffer state interference protocol.

---

## Dimension Scorecard

| Dimension | Weight | Level | Critical | Warning | Info | Status |
|---|---|---|---|---|---|---|
| estimand_clarity | H | L1 | 0 | 2 | 3 | ⚠ WARN |
| hypothesis_falsifiability | H | L1 | 0 | 2 | 2 | ⚠ WARN |
| baseline_fairness | H | L2 | 2 | 5 | 1 | 🔴 CRITICAL |
| unit_interference | M | L2 | 2 | 3 | 1 | 🔴 CRITICAL |
| red_team | — | L2 | 0 | 5 | 3 | ⚠ WARN (capped at warning for benchmark) |
| error_budget | M | L3 | 3 | 4 | 1 | 🔴 CRITICAL |
| statistical_corrections | H | L3 | 3 | 4 | 1 | 🔴 CRITICAL |
| variance_protocol | H | L3 | 2 | 5 | 1 | 🔴 CRITICAL |
| benchmark_representativeness | M | L4 | 1 | 4 | 3 | 🔴 CRITICAL |
| ecological_validity | M | L4 | 0 | 5 | 3 | ⚠ WARN |
| measurement_alignment | M | L4 | 3 | 2 | 1 | 🔴 CRITICAL |
| reproducibility_spec | M | L4 | 5 | 6 | 2 | 🔴 CRITICAL |
| data_acquisition | M | L4 | 1 | 2 | 2 | 🔴 CRITICAL |
| resource_proportionality | L | L4 | — | — | — | ✅ Not assessed (see Cannot Assess) |
| causal_structure | S | — | — | — | — | Silent (not spawned) |

**Active dimensions:** 13 | **Warning threshold:** 65 | **Warnings counted:** ~51 | **Criticals counted:** 22

---

## Critical Findings

### L2 — Baseline Fairness

**BF-1** (section: Independent Variables)
The baseline variant is described as "current BinaryHeap" but the existing codebase already employs thread-local flat buffers and `select_nth_unstable_by` for X-space KNN. Only the Y-space step retains a BinaryHeap. The plan's baseline label misrepresents the current state of production code, conflating the full-function baseline with a pure-BinaryHeap-everywhere strawman.

**BF-2** (section: Independent Variables / Implementation Phases)
The plan does not specify whether "baseline" is the current production function or a purpose-built regression stub. If the former, H1 is already partially present in the baseline (allocation elimination is established for X-space); if the latter, the plan provides no construction specification. Either way, the baseline's compile-time optimization profile, Rayon configuration, and code quality relative to the variants are undefined.

### L2 — Unit Interference

**UI-1** (section: Implementation Design)
Thread-local buffers (`COMB_DIST_Y`, `COMB_INDICES_Y`) persist across Criterion benchmark iterations. Capacity growths from larger-n runs carrying into smaller-n runs on the same Rayon thread constitute cross-iteration state leakage. The plan acknowledges reuse as intentional by design but does not address cross-(variant × n) contamination.

**UI-2** (section: Implementation Design)
All variants use `make_data()` inputs with the same seed. Inputs generated for earlier variants may already reside in L2/L3 cache when later variants are measured, providing systematic warm-cache advantages to later-ordered variants. This cross-variant spillover is not addressed in the threats section.

### L3 — Error Budget

**EB-1** (section: Analysis Plan)
No power analysis is present. The 10-sample budget is asserted without any justification of expected effect size, within-sample variance, or the resulting Type II error rate at the 1.5× threshold.

**EB-2** (section: Analysis Plan)
Type I error rate is not acknowledged. The 95% CI criterion is not stated as applying family-wise across multiple DVs and multiple n values (minimum 9 comparisons), leaving effective alpha undefined and inflated above 0.05.

**EB-3** (section: Execution Protocol)
The 10-sample budget is never connected to the CI width required to distinguish 1.0× from 1.5× speedup. If within-variant measurement noise is large relative to the 0.5× gap, 10 samples may produce CIs too wide to reach a conclusive outcome under either the positive or negative criteria.

### L3 — Statistical Corrections

**SC-1** (section: Analysis Plan)
No multiple comparisons correction procedure is pre-specified despite a minimum of 9 pairwise comparisons (3 variant pairs × 3 n values) across 4 DVs. Family-wise error rate and FDR are entirely unaddressed.

**SC-2** (section: Analysis Plan)
The significance criterion (95% CI LB > 1.0) is stated only for speedup at n=10K, without correction for the multiple variant comparisons (baseline vs h1, baseline vs h2, h1 vs h2).

**SC-3** (section: Dependent Variables)
With 4 DVs and the +multi_metric modifier active (H-weight), no correction structure (Bonferroni, Holm, Benjamini-Hochberg) is declared to govern inference across the DV family.

### L3 — Variance Protocol

**VP-1** (section: Controlled Variables)
Rayon thread count is "System default" with no fixation mechanism. For a parallel workload, this means scheduling jitter, cache contention, and NUMA effects are uncontrolled across benchmark runs, making cross-run comparisons unreliable.

**VP-2** (section: Execution Protocol)
The plan states `SamplingMode::Flat` but this setting applies only to the new bench being created; other benches in the project use Criterion's default `SamplingMode::Auto`. The uniform "10 samples" claim is therefore inconsistent with the broader benchmark suite's measurement regime.

### L4 — Benchmark Representativeness

**BR-1** (section: Inputs and Data)
Synthetic Y drawn from uniform [0,1) does not reflect the clustered, manifold-structured distributions that UMAP embeddings produce in practice. Trustworthiness nearest-neighbor rank distributions and heap access patterns are sensitive to data locality properties; the speedup ratios observed under uniform Y may differ substantially on real embedding outputs.

### L4 — Measurement Alignment

**MA-1** (section: Hypothesis)
H1_alt requires both ≥ 1.5× wall-time speedup AND y_heap fraction ≤ 40%, but the plan does not formally define whether these are conjoined (both required), disjoined (either sufficient), or ordered gates (fraction only evaluated if speedup passes). The compound criterion is ambiguous.

**MA-2** (section: Dependent Variables)
The y_heap step fraction metric uses an `AtomicU64` accumulator that sums CPU time across parallel threads, not wall-time. In a parallel execution context, summed thread-time can exceed wall-time substantially, making this metric incommensurable with the wall-time speedup ratio used for the primary hypothesis. The two metrics cannot be directly compared on a fractional basis.

**MA-3** (section: Analysis Plan)
The step fraction metric is stated to "validate that observed speedup comes from y_heap reduction," but a reduction in y_heap fraction is consistent with both genuine speedup of that step and slowdown of other steps. The plan lacks a mechanism to distinguish these cases, so the step-fraction metric cannot support the causal attribution it is assigned to validate.

### L4 — Reproducibility Spec

**RS-1** (section: Environment)
No Rust toolchain version is specified. The project uses a nightly toolchain; without a pinned version, a reproducer using a different nightly date compiles with a different compiler, invalidating benchmark comparisons. Prior experiments in this repo explicitly document and verify the toolchain version.

**RS-2** (section: Environment)
`Cargo.lock` is not tracked in git. Without a committed or archived `Cargo.lock`, a reproducer cannot guarantee identical dependency trees across runs, and semver-compatible patch updates can change codegen or behavior.

**RS-3** (section: Environment)
No `environment.yml` or equivalent Python dependency specification is provided for `gen_data.py`. Prior experiments in this repo specify `python=3.11`, `numpy=2.2.6`, etc. An unspecified numpy version affects uniform random output layout and dtype behavior.

**RS-4** (section: Controlled Variables)
Rayon thread count is "System default" with no mechanism to fix, document, or record it. A reproducer on a different-core-count machine obtains structurally incomparable throughput numbers.

**RS-5** (section: Controlled Variables / Environment)
`RUSTFLAGS=-C target-cpu=native` resolves to a machine-specific ISA. The concrete CPU and enabled instruction sets are not documented. The project's CI history (commit 3c08f61) records a prior SIGILL failure from this exact ISA ambiguity.

### L4 — Data Acquisition

**DA-1** (section: Inputs and Data / Experiment Directory Layout)
The `data/` directory is tracked only via `.gitkeep` (consistent with all experiments in this repo), so `.npy` files are absent in a fresh worktree. `gen_data.py` as Step 1 of the execution protocol provides implicit coverage, but the data dependency is not formally declared, and no verification criterion is specified to confirm the generated files are well-formed before downstream steps consume them.

---

## Adversarial Findings (Red-Team) — All `requires_decision: true`

| # | Section | Severity | Finding |
|---|---|---|---|
| RT-1 | Hypothesis | warning | The 1.5× threshold appears set post-profiling, making it achievable without generalizing beyond the specific n=10K, k=15 configuration. No evidence of pre-registration before profiling. Goodhart exploitation risk. |
| RT-2 | Implementation Design | warning | H2 receives a bespoke low-level AVX2 kernel; baseline receives no analogous tuning effort. Any speedup conflates "better algorithm" with "more implementation effort applied to variant." Asymmetric effort not acknowledged. |
| RT-3 | Controlled Variables | warning | Single seed (42) across all runs; no documentation of how seed was selected. If the seed was chosen after informal exploration, data layout may be atypically favorable for SIMD-aligned access patterns in H2. |
| RT-4 | Dependent Variables / Implementation Design | warning | AtomicU64 instrumentation is embedded in both the treatment implementations and the measurement layer. Atomic overhead interacts differently with H1's flat buffer layout vs H2's SIMD kernel, potentially distorting the step-fraction attribution used to validate H1_alt. |
| RT-5 | Analysis Plan | warning | 10 samples is Criterion's stated minimum, widening CIs and increasing the probability that the CI LB clears 1.0× by chance under Rayon scheduling noise. The plan does not justify this reduction relative to the CI precision required by the decision boundaries. |
| RT-6 | Independent Variables / Controlled Variables | info | k=15 is fixed across all n values, but the graph density ratio (k/n) changes meaningfully from n=1K to n=10K, potentially causing non-monotonic speedup behavior attributed to algorithmic properties that are actually density-scaling artifacts. |
| RT-7 | Threats to Validity | info | AtomicU64 overhead (~10µs) is not specified as subtracted from the step-fraction denominator. The y_heap fraction metric could understate the true fraction in a non-uniform way across variants. |
| RT-8 | Success Criteria | info | The 1e-12 correctness gate may be tighter than f64 floating-point reproducibility across SIMD vs scalar code paths with FMA enabled. A correct H2 implementation could be classified as failing parity. |

---

## Cannot Assess

1. **resource_proportionality**: The plan includes an "Estimated Resource Requirements" section (~30 min wall time, <10MB disk), but no subagent was spawned for this L-weight dimension. The stated resource estimates appear plausible given 3 variants × 3 n values × 10 samples × 10s warmup, but conformance to the 3-hour budget constraint from the scope report was not independently verified.

2. **H2 fallback path correctness under non-AVX2 conditions**: The plan acknowledges H2 silently falls back to scalar on non-AVX2 hardware, but no evaluation protocol is specified for validating that the fallback path produces results equivalent to the non-SIMD baseline. Whether this affects correctness gate coverage cannot be assessed without knowing the CI runner's CPU capabilities at experiment time.

3. **tw_profiler `--variant` flag output schema**: The plan introduces a new CLI flag and its output JSON fields are referenced by `analyze_results.py`, but the schema contract between `tw_profiler` and the analysis script is not described in the plan. Whether `step_timing.y_heap` JSON path is stable across variants and whether the profiler output format is compatible with the analysis script cannot be assessed.

4. **`SamplingMode::Flat` interaction with Criterion's bootstrap CI**: The CI width under 10-sample flat sampling depends on within-variant timing variance, which is not characterized for this workload at any n. Whether 10 samples provide sufficient bootstrap support for the stated 95% CI nominal coverage cannot be assessed without pilot timing data.

---

## Mechanizable Check Log

| Check | Status | Description |
|---|---|---|
| Plan file parseable | ✅ PASS | File exists and is readable |
| Frontmatter present | ❌ ABSENT | No YAML frontmatter — fields extracted from prose |
| Hypothesis H0/H1 present | ✅ PASS | Both stated with quantitative thresholds |
| Success criteria present | ✅ PASS | Positive, negative, and inconclusive zones defined |
| Threats to validity section present | ✅ PASS | Internal and external threats documented |
| Controlled variables table present | ✅ PASS | 6 controlled variables listed |
| Random seed declared | ✅ PASS | seed=42 in controlled variables |
| Cargo.lock tracked | ❌ FAIL | `.gitignore` excludes `Cargo.lock` |
| Rust toolchain version pinned | ❌ FAIL | No `rust-toolchain.toml` or version pin in plan |
| environment.yml present | ❌ FAIL | Plan explicitly states "No environment.yml will be created" |
| Power analysis present | ❌ FAIL | No sample-size justification |
| Multiple comparisons correction declared | ❌ FAIL | No correction procedure named |
| Primary DV declared | ⚠ PARTIAL | Speedup ratio implied as primary but not formally designated |
| Step fraction time basis stated | ❌ FAIL | CPU-time vs wall-time incommensurability not acknowledged |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 22
warning_count: 51
red_team_count: 8
active_dimensions: 13
warning_threshold: 65
```
