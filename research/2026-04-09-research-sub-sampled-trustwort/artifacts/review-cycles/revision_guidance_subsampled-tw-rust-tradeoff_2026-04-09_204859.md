# Revision Guidance — Sub-Sampled Trustworthiness Error/Speed Trade-off (Rust Validation)

**Plan:** `experiment_plan_subsampled_tw_rust_tradeoff_2026-04-10_032816.md`
**Verdict:** REVISE
**Timestamp:** 2026-04-09_204859

---

## Required Revisions (Critical — BLOCKING)

### R1 — Data Acquisition: Automated Path for MERFISH Fixtures
**Dimensions:** `agent_implementability`, `data_acquisition`
**Risk:** H1, H5, and H6 depend on MERFISH `.npy` files that are absent in a fresh worktree. The only stated reconstitution path requires a manual dataset download with no URL, no checksum, and no phase placement. An implementing agent cannot satisfy the Phase 1 preflight acceptance criterion without this data.

**What is lacking:**
- A machine-executable acquisition step or a documented pre-condition confirming the data is present in a known location
- A verification criterion (shape check exists; checksum or byte-size criterion is absent) to detect partial or corrupted downloads
- Phase placement for the manual download pre-condition in the ordered acquisition steps
- Clarity on whether the experiment assumes data is pre-positioned (developer-workstation model) vs requires fresh acquisition (CI/agent model)

---

## Required Revisions (Warnings)

### R2 — Circular Validation: Scope Claims Must Reflect Same-Dataset Confirmation
**Dimensions:** `red_team`, `ecological_validity`, `measurement_alignment`
**Risk:** The success criteria state that passing H1–H6 confirms "the Rust sub-sampled trustworthiness is ready to ship with m=2000 default (for MERFISH-class data, k=15, AVX2 x86)." This claim is not supported by independent validation — m=2000 and the 0.01 threshold were selected on the same MERFISH data. The Rust confirmatory study reproduces the conditions of the selection experiment.

**What is lacking:**
- An explicit acknowledgment in the success criteria and conclusions that this constitutes same-dataset confirmation, not out-of-sample validation
- Qualification of the shipping recommendation as "confirmed on the same dataset used for selection; generalization to other scRNA-seq datasets is unvalidated"
- The plan's "threshold leakage" external validity threat does not propagate its implications to the shipping recommendation

---

### R3 — Multiple Testing: Substantiate or Qualify the FWER Omission
**Dimensions:** `statistical_corrections`, `error_budget`
**Risk:** The plan asserts FWER correction is intentionally omitted, but the justification relies on three unsubstantiated claims: (1) confirmatory status without a prior study citation; (2) "qualitatively distinct hypotheses" that misclassifies H1 and H5 as distinct when they test the same construct; (3) ~15% FWER assuming independence, which is violated by the correlation structure.

**What is lacking:**
- A citation to PR #260 as the pre-registered prior evidence establishing the expected effect sizes
- An explicit acknowledgment that H1 and H5 test the same construct, and a stated rationale for why repeated testing without correction is acceptable in this specific confirmatory context
- A pre-declared FWER ceiling (a numeric bound committed to before data collection, not derived from the per-test alphas post-hoc)
- Clarification of H5's family membership: if H5 is exploratory, it should not be counted in the FWER computation; if it is confirmatory, it warrants a correction

---

### R4 — Power Analysis: Characterize H2, H3, and H5
**Dimension:** `error_budget`
**Risk:** The plan provides a power justification only for H1 (10 seeds, ~90% power, σ≈0.002). H2 (OLS on 7 m-values), H3 (log-log OLS on 7 m-values), and H5 (same structure as H1) have no Type II error characterization. If H2 or H3 fail, the study cannot distinguish "sub-sampling genuinely lacks linear scaling" from "7-point OLS was underpowered to detect the pattern."

**What is lacking:**
- A power assessment for H2 at the stated R² threshold (0.90) given n=7 OLS points
- A power assessment for H3 at the slope threshold (β = -0.3) given n=7 log-log points and an uncharacterized residual variance
- A justification for why 10 seeds is adequate for H5 (same structure as H1 but the σ may differ at n=50K)
- Validation or citation for the σ≈0.002 assumption in H1's power calculation

---

### R5 — Hypothesis Specification: Compound Null, Regression Stratification, Partial Outcomes
**Dimensions:** `estimand_clarity`, `hypothesis_falsifiability`
**Risk:** Reproducibility of verdict logic depends on unambiguous hypothesis specifications. Three gaps reduce reproducibility:

**What is lacking:**
- **H1 composite null:** The top-level H0 combines H1's accuracy claim with H2's linearity claim in a single OR null. This prevents a clean single-question rejection. The individual hypotheses are well-specified; the top-level composite is not.
- **H2 regression stratification:** The plan does not state whether the OLS for H2 pools both n=10K and n=50K into one fit or fits per-stratum. Different implementations will produce different R² values depending on this choice.
- **H2 FAIL condition:** The PASS rule (R² CI lower bound > 0.90) is stated, but the FAIL condition is implicit. A reader must infer that ≤0.90 constitutes FAIL.
- **H2 linearity vs goodness-of-fit:** R² > 0.90 does not distinguish linear from monotone non-linear fits. If the purpose is to confirm linearity (not just high correlation), an additional test for functional form is needed.
- **H3 partial-outcome verdict:** No verdict is defined when β ≤ -0.3 but p ≥ 0.05, or β > -0.3 but p < 0.05.

---

### R6 — Variance Protocol: Warmup Ambiguity, Core Count, Non-determinism Decision Rule
**Dimensions:** `variance_protocol`
**Risk:** Three gaps in the variance protocol introduce uncontrolled sources of timing variance or leave the experiment without a defined response to a flagged validity threat.

**What is lacking:**
- **Warmup:** Whether the first of the 5 timed repetitions is excluded from the median is not specified in the Execution Protocol commands. The Threats to Validity section mentions "warmup discard" in passing but the protocol does not implement it explicitly.
- **Core count:** The Rust toolchain version is recorded as a controlled variable; the core count and CPU model are not. Without recording these, speedup ratios are not reproducible on different hardware even with the same binary.
- **Rayon non-determinism decision rule:** The plan states that max|T_run1 - T_run2| > 1e-6 will be "flagged in results" but does not define the protocol response: retrial? data exclusion? report-and-proceed? This requires a decision before execution.

---

### R7 — Asymmetric Optimization: Characterize T_exact Path
**Dimension:** `red_team`
**Risk:** The speedup ratio (exact_median_ms / sub_median_ms) is the primary metric for H2. If the exact trustworthiness path in the binary uses a different (less optimized) code path than the sub-sampled path, the measured speedup reflects implementation asymmetry rather than sub-sampling efficiency.

**What is lacking:**
- A statement of whether `--mode exact` in `tw_subsample_experiment` uses the same AVX2/Rayon optimized path as `--mode subsample` or delegates to a different code path
- If the exact mode uses `spectral_init::trustworthiness(x, y, k)` from the library (not the copied inner loop), the plan should acknowledge this asymmetry and state why it is acceptable or expected

---

### R8 — Implementability: Dry-Run Coherence and Interface Specifications
**Dimension:** `agent_implementability`
**Risk:** Several under-specifications prevent an implementing agent from satisfying acceptance criteria without human disambiguation.

**What is lacking:**
- **Dry-run H6 equivalence:** The dry-run includes trial `(n=10K, m=10000, seed=0)`. The plan should state explicitly that `m=10000 = n=10000` satisfies the H6 sanity check requirement (`--mode sanity` with m=n), since this equivalence is not stated and is non-obvious to an agent reading the trial list.
- **Dry-run analysis on 3-trial subset:** The analysis script must produce `verdicts.json` with all 6 hypothesis keys from only 3 trials. The plan should specify what value each hypothesis key holds when insufficient data is available (e.g., `"INSUFFICIENT_DATA"`, `null`, or a partial verdict structure).
- **Sanity check JSON output:** The acceptance criterion for `--mode sanity` checks `|T_sub - T_exact| < 1e-10` but does not specify which JSON fields in the binary's output contain T_sub and T_exact for this comparison.
- **`cli` feature existence:** The plan should confirm that the `cli` feature already exists in `Cargo.toml` (or note that it must be added) and enumerate what dependencies or code paths it gates.

---

## Red-Team Findings — Decision Points Required

The following six adversarial findings each require an explicit decision before the plan is finalized. Partial responses are insufficient; each decision point should be answered with a stated position that is incorporated into the plan text.

### RT1 — Goodhart Exploitation
**Risk:** The sub-sampled implementation could be tuned (SIMD rounding, Rayon chunk size, introselect tie-breaking) such that the MERFISH error distribution stays below 0.01 without the bound generalizing to any other dataset or distance regime.
**Decision required:** Does the plan intend to claim generalization, or only same-dataset reproduction? If reproduction only, the success criteria shipping recommendation should be scoped accordingly and labeled as "Rust implementation parity with Python study" rather than "validated default."

### RT2 — Data Leakage
**Risk:** m=2000 was selected and validated on MERFISH in PR #260. This study validates the same parameter on the same dataset. There is no independent holdout.
**Decision required:** Is the absence of an independent holdout explicitly acknowledged in the study's conclusions? If so, this finding is informational only. If the study implies independent confirmation, the conclusions require qualification.

### RT3 — Asymmetric Tuning
**Risk:** The sub-sampled path is described with explicit SIMD and parallelism optimizations. The plan does not describe the exact trustworthiness path's optimization level.
**Decision required:** Does `--mode exact` use the same optimized code path as `--mode subsample`? (See also R7 above.)

### RT4 — Survivorship Bias
**Risk:** No outlier exclusion criteria are pre-registered for seeds 0–9. H5's failure is pre-designated "inconclusive" rather than falsifying.
**Decision required:** (a) Is the seed set {0–9} exhaustive with no post-hoc exclusion permitted? If a seed produces anomalous results, what is the protocol? (b) Is H5 included in the confirmatory family or is it a separately reported exploratory result?

### RT5 — Evaluation Collision
**Risk:** The same Rust process measures both the treatment (trustworthiness computation) and the timing. First-trial cold-start effects (thread pool initialization, page faults) affect both T and timing inseparably.
**Decision required:** Is warmup handled before timed measurement (see R6)? If yes, is the warmup documented in the Execution Protocol commands so an implementing agent applies it consistently?

### RT6 — Overfitting to Held-Out Set
**Risk:** m=2000 is the "recommended default" being confirmed, and the 0.01 threshold was set knowing that m=2000 produces ~0.00165 on MERFISH — a 6x margin. Confirming a pre-set recommendation on its own calibration data does not constitute validation.
**Decision required:** Should the study conclusions be re-framed as "Rust implementation reproduces Python study results" rather than "validates m=2000 as a cross-language default"? If the stronger claim is intended, it requires evidence from a dataset not used in threshold selection.

---

## Advisory Revisions (Recommended)

The following gaps are non-blocking but would strengthen the study. They are not required to proceed.

- **H4 operational semantics:** The plan states Python reference files are "explicitly flagged as absent" but does not describe what "flagged" means in `verdicts.json`. Define the schema for the conditional H4 case.
- **H5 framing consistency:** H5 is labeled "Exploratory" but uses the same formal test and alpha as confirmatory H1. Consider either (a) labeling H5 as a secondary confirmatory test with the same correction applied, or (b) using a descriptive/graphical treatment without a formal pass/fail verdict.
- **Resource proportionality — Rayon non-determinism:** 6 trials (3 configs × 2 reps) is a thin characterization of the floating-point non-determinism distribution, particularly at n=50K. Consider whether a wider sweep (more (m,seed) pairs at n=50K) is warranted given the threat's stated prominence in the internal validity section.
- **Secondary threshold (0.003) justification:** The "2x the Python observed value" rationale is stated but the plan does not clarify whether the Python study was conducted under comparable conditions (same k, n, m). If conditions differ, the referent for the secondary threshold is undefined.
- **Composite H0 scope:** The top-level H0/H1 covers only H1 and H2. Consider replacing it with a structured success definition that maps to all six individual hypotheses, or removing the composite null and relying exclusively on the individual hypothesis verdicts.
