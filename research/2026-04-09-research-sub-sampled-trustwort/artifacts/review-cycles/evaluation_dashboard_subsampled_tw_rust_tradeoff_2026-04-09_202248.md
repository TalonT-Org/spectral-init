# Design Review: Evaluation Dashboard
## Sub-Sampled Trustworthiness — Rust Error/Speed Trade-off Validation

**Verdict: REVISE**
**Experiment Type:** configuration_study
**Review Timestamp:** 2026-04-09 20:22:48

---

## Verdict Banner

> **REVISE** — 13 critical findings across 8 dimensions. The plan has a sound overall structure and clearly articulated hypotheses, but contains unaddressed measurement design flaws (SIMD bias claim untestable, timing variance protocol insufficient, statistical framework absent) and practical blockers (gitignored data fixtures with no acquisition steps, visual-inspection-gated execution protocol). The plan cannot be implemented and executed by an agent in its current form, and several measurement claims cannot be verified by the proposed metrics.

---

## Dimension Scorecard

| Dimension | Level | Weight | Critical | Warning | Info | Status |
|---|---|---|---|---|---|---|
| estimand_clarity | L1 | H | 0 | 3 | 1 | ⚠ warnings |
| hypothesis_falsifiability | L1 | H | 0 | 3 | 1 | ⚠ warnings |
| baseline_fairness | L2 | H | 1 | 3 | 2 | 🔴 critical |
| causal_structure | L2 | L | 1 | 0 | 0 | 🔴 critical |
| unit_interference | L2 | — | 0 | 4 | 1 | ⚠ warnings |
| error_budget | L3 | H | 0 | 4 | 0 | ⚠ warnings |
| statistical_corrections | L3 | H | 3 | 1 | 0 | 🔴 critical |
| variance_protocol | L3 | H | 3 | 1 | 0 | 🔴 critical |
| ecological_validity | L4 | M | 1 | 3 | 1 | 🔴 critical |
| measurement_alignment | L4 | M | 1 | 4 | 0 | 🔴 critical |
| data_acquisition | L4 | M | 2 | 3 | 0 | 🔴 critical |
| agent_implementability | L4 | H | 2 | 4 | 0 | 🔴 critical |
| red_team | — | — | 0 | 6 | 0 | ⚠ warnings (cap applied) |

**Active dimensions:** 13 | **Warning threshold:** 65 | **Observed warnings:** 39

---

## Critical Findings

### [BLOCKING] Consolidated: SIMD Systematic Bias Claim — Neither Isolable Nor Measurable
**Dimensions:** causal_structure (L2) + measurement_alignment (L4)

The plan's 4th motivating claim — "SIMD floating-point ordering does not introduce systematic bias vs the Python scalar path" — cannot be established by this experiment's design.

- *causal_structure:* The dependent variable `|T_sub - T_exact|` conflates sub-sampling variance (stochastic, decreasing with m) and SIMD rounding error (deterministic, independent of m). No condition in the design holds m constant while varying only the scalar vs SIMD execution path, so the two error sources are never isolated. The claim of "no systematic bias" requires a design that can separate a deterministic path-dependent offset from stochastic sampling noise.

- *measurement_alignment:* Claim 3 asserts "no systematic bias vs the Python scalar path," but no metric compares Rust SIMD output to a Rust scalar or Python scalar reference. Both T_sub and T_exact are computed via the same SIMD path; any systematic SIMD rounding error present in both cancels out entirely. The claim is untestable by any metric in this experiment.

---

### [BLOCKING] baseline_fairness — H4 Cross-Language Speedup Ratio Not Anchored
**Section:** H4 Cross-Language Parity

The plan's H4 pass criterion compares "Rust speedup ratio at same (n,m) matches Python within 2x." However, the Rust experiment uses introselect O(n) partial sort, AVX2+FMA SIMD, and Rayon parallelism, while Python sklearn trustworthiness uses a different algorithmic path. The speedup ratio (exact_T / sub_T) is a function of the absolute complexity of each implementation, not just the sampling ratio n/m. Comparing these ratios across languages without normalizing for implementation complexity differences means the 2x tolerance is not anchored to any justified equivalence criterion — it may accept or reject results for implementation reasons rather than conceptual ones.

---

### [BLOCKING] statistical_corrections — No Family-Wise Error Rate Control
**Section:** Overall Verdicts / Hypothesis Evaluation

6 hypotheses are evaluated simultaneously with no FWER or FDR correction pre-specified. The verdict logic treats each threshold independently, but simultaneous evaluation of 6 binary outcomes inflates the probability of at least one false PASS under noise. No Bonferroni, Holm, Benjamini-Hochberg, or equivalent correction is mentioned.

---

### [BLOCKING] statistical_corrections — H1/H5 Lack a Pre-Specified Test Statistic
**Section:** H1/H5 Accuracy threshold hypotheses

H1 and H5 use mean|ΔT| compared against a fixed threshold (0.01) with no pre-specified test statistic or sampling distribution. A point-estimate mean provides no inferential guarantee: the threshold could be satisfied by the sample mean while the true mean exceeds it. No t-test, one-sided confidence interval, or bootstrap procedure is pre-specified to bound the probability of incorrectly concluding PASS.

---

### [BLOCKING] statistical_corrections — No Alpha Level Pre-Specified
**Section:** Overall Verdicts

No alpha level is pre-specified anywhere in the plan. Without a stated Type I error rate, threshold-based verdicts cannot be interpreted as statistical decisions with known error guarantees, making the entire inference framework non-falsifiable in a formal sense.

---

### [BLOCKING] variance_protocol — Single Timed Run Per Trial
**Section:** Wall-clock timing

Single timed run per (m, seed) trial with no repeated measurement or median means each data point is a single sample with no within-cell noise estimate. A single outlier timing (OS jitter, cache miss, scheduler preemption) is indistinguishable from signal. The variance metric std(T_sub) across 10 seeds conflates sub-sampling variance with timing noise, making it impossible to decompose algorithmic variance from measurement variance.

---

### [BLOCKING] variance_protocol — Rayon Non-Determinism Invalidates Seed Reproducibility
**Section:** Rayon threads

Rayon's work-stealing scheduler produces non-deterministic floating-point summation order across trials even when the same StdRng seed is used. This means two runs with identical seeds at the same (n, m) cell can produce different numeric outputs and different execution paths, invalidating seed-as-reproducibility-control. The 10-seed variance metric therefore includes an uncontrolled non-deterministic component that cannot be attributed to sub-sampling alone.

---

### [BLOCKING] variance_protocol — Warmup Insufficient for n=50K Thermal Profile
**Section:** CPU thermal threat

A single 1-iteration warmup is insufficient mitigation for thermal throttling across a continuous process running trials that include n=50K cells estimated at ~17 minutes. Frequency scaling during later trials (particularly late m-values or late seeds within n=50K) will inflate wall-clock times in a position-dependent manner correlated with trial order, biasing std(T_sub) upward for high-m cells and confounding the H3 variance-decay slope test.

---

### [BLOCKING] ecological_validity — Hardware Generalizability Not Established
**Section:** hardware_generalizability

The deployment decision (b) — whether the Rust speedup justifies the same m=2000 recommendation as Python — is hardware-contingent, but the experiment runs on a single unnamed machine. A non-SIMD path or a 2-core CI/CD environment could invalidate the speed half of the justification while leaving the error half intact. Either the recommendation must be decoupled from speed, or the experiment must establish a worst-case hardware bound.

---

### [BLOCKING] data_acquisition — All Fixtures in Gitignored Paths, No Acquisition Steps
**Section:** Inputs and Data

All six data files (merfish_n10k_x.npy, merfish_n10k_y.npy, merfish_n50k_x.npy, merfish_n50k_y.npy, exact_merfish_10000.json, sub_A_merfish_10000_m*_s*.json) reside under gitignored research/ paths and will be absent in a fresh worktree. The plan contains no acquisition, generation, or fallback step to reconstitute them, leaving the experiment unrunnable from a clean implementation worktree.

---

### [BLOCKING] data_acquisition — H4 Python Reference Data Has No Acquisition Step
**Section:** H4 Cross-Language Parity

H4 cross-validation depends on pre-computed Python reference results (exact_merfish_10000.json and sub_A_merfish_10000_m{m}_s{seed}.json) that are described as "existing" but have no acquisition, generation, or verification step. There is no specified command to produce or confirm these files before H4 is evaluated.

---

### [BLOCKING] agent_implementability — Execution Protocol Requires Visual Inspection
**Section:** Execution Protocol step 6

Step 6 explicitly requires "visual inspection of PNG plots" and a human reading summary.md for hypothesis verdicts. Neither action is automatable by a code-generating agent. No machine-readable pass/fail criterion is defined for the analysis outputs that would allow an agent to confirm experiment success.

---

### [BLOCKING] agent_implementability — H4 Data Dependency Unresolvable by Agent
**Section:** Phase 3 / analyze_results.py

analyze_results.py is specified to read from research/2026-04-09-subsampled-tw-tradeoff/results/raw/ for H4 cross-validation. This path is gitignored and no acquisition step is defined. An agent cannot produce the cross_validation.png output or satisfy H4 analysis without this data.

---

## Notable Warning Findings

### unit_interference — Ascending m-Order Aliases System Effects onto m Dimension
The ascending m ordering (250→7500) means trial order is perfectly correlated with m magnitude. Any time-varying system effect (thermal throttling, OS memory reclamation, allocator compaction) will be systematically aliased onto the m dimension rather than absorbed as noise. There is no randomization or interleaving of m values across seeds.

### unit_interference — Normalization Sanity Check Contaminates First Timed Trials
The normalization sanity check (m=n) immediately before the timed trials fully warms CPU caches and thread-local buffers with an n-scale workload, making the first timed trial (m=250, seed=0) non-representative of a cold-start m=250 measurement.

### baseline_fairness — Python Prior Experiment Conditions Not Documented
Python results used in H4 were produced under conditions not documented in this plan (Python version, sklearn version, hardware, threading). Speedup ratios are sensitive to these factors, making the cross-language comparison potentially asymmetric in an unverifiable way.

### error_budget — 0.01 Threshold Not Statistically Grounded
The 0.01 threshold for H1/H5 is inherited from prior Python research without statistical grounding. The prior observed value is 0.00165, making the threshold ~6x the signal. No tolerance interval, confidence bound, or domain justification links this value to downstream embedding quality.

### ecological_validity — Single Dataset Insufficient for Production Default
Single-domain MERFISH (scRNA-seq, d=50) is insufficient to set a general production default. trustworthiness_subsampled() will be called on image, text, tabular, and other omics datasets with different intrinsic dimensionalities and local structure densities.

### ecological_validity — Fixed k=15 Does Not Cover Production Range
k values of 5–200 are common in production. Sub-sampling approximation error is a function of k; smaller k amplifies sampling noise at a given m. The m=2000 recommendation should be validated across boundary k values before shipping as a default.

### ecological_validity — Known Limitations Not Constraining the Deployment Decision
The plan enumerates dataset- and hardware-specific threats but simultaneously claims to determine a general-purpose default m. The deployment decision should be explicitly scoped to "scRNA-seq-class datasets on AVX2 x86 hardware" or the scope broadened to match the recommendation's intended generality.

---

## Adversarial Findings (Red-Team) — All Require Decision

All red-team findings are capped at `warning` per configuration_study severity ceiling.

| Finding | Risk |
|---|---|
| Threshold leakage via same-dataset calibration | Every quantitative threshold was derived from a Python experiment on the same MERFISH dataset. This is not held-out validation — it is fitting thresholds to the answer key and grading against it. |
| Goodhart exploitation via threshold slack | The 6x margin on H1 (0.00165 observed vs 0.01 threshold) creates room for a pathological implementation (e.g., cached subsample) to pass without producing useful sub-sampling behavior. |
| Survivorship bias via fixed sequential seeds | Seeds 0-9 are not drawn from an adversarial distribution. High-variance seeds are averaged away. Per-seed worst-case is not reported. |
| Asymmetric tuning: m=2000 chosen after seeing accuracy curve | m=2000 was selected from a prior Python accuracy sweep on MERFISH before the Rust experiment was designed, making the Rust experiment a confirmation exercise rather than an independent validation. |
| Evaluation collision: H6 is a tautological self-consistency check | If both T_sub and T_exact use the same Rust code path (m=n simply means no rows skipped), any systematic implementation bug would cause both to be wrong identically. H6 does not constitute independent validation. |
| H5 threshold extrapolation without empirical basis | The 0.01 threshold at n=50K is extrapolated from n=10K with no empirical evidence that sub-sampling error does not scale differently at 5x n. |

---

## Cannot Assess

The following dimensions could not be evaluated due to absent plan content:

1. **Machine-level reproducibility environment** — The plan does not identify the target machine (CPU model, core count, cache topology, RAM) beyond "same machine, same run." It is impossible to assess whether results would be reproducible across machines or whether the hardware profile artifact captures sufficient information for reproduction.

2. **Python environment for H4 baseline** — The Python environment producing the reference results for H4 cross-validation is described as "optional" with no version pinning. It is impossible to assess whether H4 cross-validation is replicable by a third party.

3. **Thread-local buffer independence across trials** — The plan claims "buffers are always n-length regardless of m" but does not describe the buffer lifecycle (allocation per-call vs. per-thread-lifetime). It is impossible to assess the actual scope of inter-trial state contamination from buffer reuse.

---

## Mechanizable Check Log

### Fixed Checks

| Check | Status | Notes |
|---|---|---|
| All implementation phases have runnable verification criteria | ❌ FAIL | Phase 3 (infrastructure) has no verification step |
| All file paths in the implementation plan resolve to valid locations | ⚠ CANNOT CONFIRM | MERFISH fixture paths are gitignored; symlink targets may not exist in fresh worktree |

### Ad-Hoc Checks (contributed by dimension subagents)

| Check | Status | Notes |
|---|---|---|
| All data_manifest entries have acquisition steps for gitignored paths | ❌ FAIL | Six data files have no acquisition/generation steps |
| Seeds are fixed and documented for all stochastic steps | ✅ PASS | Seeds 0-9 documented and deterministic (StdRng::seed_from_u64) |
| No human-only action required in execution protocol | ❌ FAIL | Step 6 requires visual inspection of PNG plots |
| Hypothesis thresholds are derived from held-out data | ❌ FAIL | All thresholds calibrated on same MERFISH dataset used for evaluation |
| Template file existence verified before implementation | ⚠ UNVERIFIED | tw_profiler.rs listed as template; plan does not verify its existence |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: configuration_study
critical_count: 13
warning_count: 39
blocking_count: 13
required_count: 20
advisory_count: 6
red_team_count: 6
active_dimensions: 13
warning_threshold: 65
```
