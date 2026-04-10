# Design Review: Revision Guidance
## Sub-Sampled Trustworthiness — Rust Error/Speed Trade-off Validation

**Verdict: REVISE**
**Review Timestamp:** 2026-04-09 20:22:48

This document describes WHAT needs to change in the experimental design. It does not prescribe implementation instructions.

---

## Required Revisions (Critical Findings — Address Before Execution)

### 1. Data Acquisition Strategy — Gitignored Fixtures
**Risk:** The experiment cannot be run from a clean worktree. All MERFISH fixture files reside under gitignored paths and will be absent after cloning or in a fresh implementation worktree. The plan states "All data already exists" without providing any acquisition path for when it doesn't exist.

**What the plan must address:** A data acquisition section specifying, for each gitignored data file: (a) where it originates, (b) how to obtain or reconstitute it in a fresh worktree, and (c) a pre-flight verification criterion that confirms the file is present and correct before execution begins. This applies to both the MERFISH .npy fixtures and the Python reference results required by H4.

---

### 2. H4 Python Reference Data Acquisition
**Risk:** H4 cross-validation depends on Python experiment results (exact_merfish_10000.json, sub_A_merfish_10000_m*_s*.json) from a prior research directory. These files have no acquisition step. If absent, H4 silently fails or the analysis script errors out. An agent implementing this plan has no path to resolve this dependency.

**What the plan must address:** Either (a) define H4 as conditional on the presence of Python reference data, with a clear fallback verdict if absent, or (b) include an explicit step to generate or locate the Python reference files before the analysis phase begins.

---

### 3. Machine-Readable Analysis Outputs — Remove Visual Inspection Gate
**Risk:** Execution Protocol step 6 gates the experiment conclusion on visual inspection of PNG plots, which is not automatable. An agent cannot confirm experiment success or read hypothesis verdicts from plots.

**What the plan must address:** The hypothesis verdict evaluation must produce a machine-readable outcome (e.g., a structured file with per-hypothesis PASS/FAIL/INCONCLUSIVE flags and observed metric values) that can be consumed without visual inspection. PNG plots are appropriate as supplementary output, but cannot be the primary verdict mechanism.

---

### 4. SIMD Systematic Bias Claim — Redesign or Descope
**Risk:** The claim "SIMD floating-point ordering does not introduce systematic bias vs the Python scalar path" cannot be supported by the proposed design. mean|ΔT| conflates sub-sampling variance (stochastic) with SIMD rounding (deterministic and independent of m). The two error sources are not separable by any metric in the plan. The claim is untestable as written.

**What the plan must address:** Either (a) remove the SIMD systematic bias claim from the motivation and success criteria, scoping the experiment purely to sub-sampling accuracy and speedup, or (b) restructure the design to include a condition that isolates SIMD rounding — such as a scalar (non-SIMD) reference path or a Python-reference comparison of T_exact values — that can distinguish the two error sources.

---

### 5. Statistical Framework — Alpha Level and Inference Procedure
**Risk:** No alpha level is pre-specified. No test statistic or sampling distribution is defined for H1 and H5 (the primary accuracy claims). Without these, pass/fail decisions on H1–H6 are informal point-estimate comparisons with no known error rate. The experiment cannot produce statistically defensible conclusions.

**What the plan must address:** For the primary accuracy hypotheses (H1, H5): specify the test statistic (e.g., one-sample t-test on |ΔT| values, or equivalently a confidence interval on mean|ΔT|), the pre-specified alpha level, and the decision rule. For regression-based hypotheses (H2, H3): specify the minimum number of design points for the fit and whether R²/slope thresholds are treated as point estimates or require confidence bounds. An alpha level (e.g., 0.05) must be stated even if formal testing is not used, to establish a basis for the "PASS" interpretation.

---

### 6. Multiple Testing — Correction or Acknowledged Absence
**Risk:** 6 hypotheses are tested simultaneously with no family-wise error rate control. At a nominal 5% per-test rate, the familywise Type I error exceeds 26%. The plan makes a joint shipping decision ("all H1-H6 PASS") that treats all 6 as simultaneously true, compounding the risk.

**What the plan must address:** Either (a) pre-specify a correction procedure (e.g., Bonferroni, Holm) for the 6 simultaneous tests, or (b) explicitly acknowledge that the experiment is confirmatory (expected to pass based on prior evidence) and state the per-test alpha with a note that FWER correction was intentionally omitted and why. The absence must be a deliberate choice, not an oversight.

---

### 7. Timing Variance Protocol — Repeated Measurements
**Risk:** Single timed run per (m, seed) trial means each speedup data point is a single-sample measurement indistinguishable from an outlier (OS jitter, cache miss). Variance attributed to sub-sampling (std of T_sub) is conflated with timing noise. The H2 and H3 conclusions — which depend on the quality of the speedup and variance measurements — cannot be trusted from single-sample timing.

**What the plan must address:** The timing measurement protocol must specify multiple timing repetitions per trial (e.g., 3–5 timed calls per (m, seed) cell with median or mean reported). The number of repetitions and the aggregation method must be specified. For timing-dependent hypotheses (H2, H3), the protocol must distinguish timing variance from sub-sampling variance.

---

### 8. Rayon Non-Determinism and Seed Reproducibility
**Risk:** Rayon's work-stealing scheduler produces non-deterministic floating-point summation order across trials, meaning the same StdRng seed does not guarantee the same numeric output across runs. The claim that "10 seeds provides a reasonable variance estimate" is undermined by the non-deterministic execution path. Results are not reproducible across re-runs even with fixed seeds.

**What the plan must address:** The plan must acknowledge the Rayon non-determinism and either (a) establish that the variance introduced by Rayon's scheduling is negligible compared to sub-sampling variance (this requires a dedicated measurement, e.g., repeated runs with seed=0 at fixed m to measure Rayon-induced variance), or (b) frame the 10-seed variance estimate as a combined measure of sub-sampling + Rayon variance and adjust the analysis plan accordingly.

---

### 9. Trial Order — Thermal and Cache Aliasing
**Risk:** The ascending m-order execution (250→7500) perfectly correlates trial position with m magnitude. Thermal throttling and CPU cache warm-up effects accumulate monotonically with trial position, creating a confound that is indistinguishable from m-dependent signal. This directly threatens the validity of H2 (speedup linearity) and H3 (variance decay).

**What the plan must address:** The trial order must be randomized across m values (and ideally across seeds), or the plan must include a mitigation that separates trial-position effects from m-dependent effects (e.g., running m values in non-sequential order, or including a position covariate in the analysis model).

---

### 10. Hardware Scope for Deployment Decision
**Risk:** The deployment decision — whether the Rust speedup "justifies the same m=2000 recommendation as Python" — is hardware-contingent. A single unnamed machine cannot establish a generalizable claim about Rust's speed advantage. Users on non-SIMD or low-core-count hardware may not experience the claimed speedup.

**What the plan must address:** The deployment decision must be explicitly scoped to the hardware class tested (AVX2 x86 with at least N cores) and the m=2000 recommendation must be justified by the accuracy half of the argument alone (error < 0.01), independent of any speed claim that requires specific hardware.

---

### 11. Phase 3 Verification Step Missing
**Risk:** Phase 3 (experiment infrastructure: symlinks, scripts) has no verification step. An agent cannot confirm that the infrastructure was created correctly before proceeding to Phase 4 (dry run). A silent Phase 3 failure would only manifest as a cryptic runtime error during execution.

**What the plan must address:** Phase 3 must include a concrete acceptance criterion that can be evaluated without human judgment — e.g., confirming that the experiment directory structure exists and symlink targets are valid.

---

## Recommended Revisions (Warning Findings — Address to Strengthen Conclusions)

### Dataset Scope for Production Default
The m=2000 recommendation is validated on a single scRNA-seq dataset (MERFISH, d=50). The "Threats to Validity" section correctly identifies this but does not constrain the deployment decision. The recommendation should be explicitly scoped to "scRNA-seq-class high-dimensional data on AVX2 x86 hardware" rather than stated as a general default.

### k Sensitivity
All experiments use k=15. The error/speed trade-off at m=2000 may differ at k=5 (smaller neighborhood, more sensitivity to sample omission) or k=50. If shipping a general recommendation, at minimum acknowledge that k=15 is the only validated configuration.

### H4 Comparability Criteria — Pre-Specify "Non-Comparable" Conditions
H4 currently has an escape hatch: "FAIL due to non-comparable measurement conditions" can be invoked after seeing results without pre-specifying what makes conditions comparable. The criteria for judging H4 as comparable vs. non-comparable should be stated before data collection.

### Threshold Justification
The 0.01 threshold for H1/H5 is 6x the observed Python value (0.00165). While a conservative threshold is appropriate for a validation study, the choice should be justified in terms of downstream impact: "what embedding quality loss corresponds to a |ΔT| of 0.01?" This would make the threshold defensible beyond "it's what Python used."

### Per-Seed Worst-Case Reporting
With 10 seeds, the mean|ΔT| can conceal a high-variance seed that produces |ΔT| > 0.01. The analysis plan should include a worst-case seed report alongside the mean, so users know the tail risk of a single sub-sampling draw.

### H6 Normalization — Independence from Exact Baseline
H6 validates T_sub(m=n) == T_exact within 1e-10. If both use the same Rust code path, this is a self-consistency check rather than an independent validation. Adding a comparison to the Python-computed T_exact (0.5362038060873342, available in the prior experiment results) would provide independent normalization validation.

---

## Red-Team Decision Points (All require explicit decisions by the plan author)

### 1. Threshold Leakage — Acknowledge or Add Held-Out Validation
All quantitative thresholds (0.01, R²≥0.95, slope≤-0.3, 2x) were derived from a Python experiment on the same MERFISH dataset. This experiment confirms thresholds on the dataset they were calibrated on. **Decision required:** (a) Accept that this is a confirmatory study (not a held-out validation) and scope the conclusions accordingly, or (b) introduce a second dataset to provide an out-of-sample validation of the thresholds.

### 2. Goodhart Exploitation — Tighten or Justify the 6x Threshold Slack
The 6x margin between observed value (0.00165) and threshold (0.01) is large enough that an implementation with significant flaws could still pass. **Decision required:** Either tighten the acceptance threshold (e.g., 2× the Python observed value = 0.003) or explicitly justify that 0.01 is the appropriate production tolerance and the 6x margin is intentional safety slack.

### 3. m=2000 Asymmetric Tuning — Frame as Confirmation, Not Discovery
m=2000 was chosen from the prior Python accuracy sweep on MERFISH. The Rust experiment evaluates accuracy at this pre-selected m. This is a confirmation study, not an independent selection. **Decision required:** Frame H1 and H5 explicitly as "confirming that the Python-recommended m=2000 works in Rust," not as "determining the optimal m for Rust." This framing change does not weaken the study — it accurately represents what it can establish.

### 4. H6 Tautological Self-Consistency — Clarify What Is Being Validated
If T_sub(m=n) and T_exact both use the same Rust code path, H6 may be a mathematical identity rather than an empirical validation of normalization correctness. **Decision required:** Clarify what H6 is testing. If it is purely a mathematical identity (normalization denominator simplifies to 1 when m=n), state that. If it is intended as an implementation correctness check, specify what independent reference it should be checked against.

### 5. H5 Threshold Extrapolation — Explicitly Acknowledge or Validate
The 0.01 threshold at n=50K is extrapolated from n=10K observations. Sub-sampling error could scale differently at larger n due to graph density changes. **Decision required:** Either (a) acknowledge that H5 is a novel, exploratory test without an established threshold, and treat a stricter criterion for "inconclusive" at n=50K, or (b) provide a theoretical or empirical argument for why the 0.01 threshold holds at n=50K.
