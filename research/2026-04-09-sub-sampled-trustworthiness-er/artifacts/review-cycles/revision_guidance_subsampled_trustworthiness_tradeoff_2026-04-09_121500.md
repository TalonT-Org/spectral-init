# Revision Guidance: Sub-sampled Trustworthiness Error/Speed Trade-off

**Verdict:** REVISE
**Experiment Type:** benchmark
**Dashboard:** evaluation_dashboard_subsampled_trustworthiness_tradeoff_2026-04-09_121500.md

---

## Required Revisions (Critical Findings)

These gaps must be addressed before the plan is ready for execution.

---

### R1 — Speedup Metric Status Must Be Resolved (measurement_alignment)

**Gap:** The dependent variables table marks the speedup ratio as "No threshold — purely descriptive," but the success criteria list speedup >= 3x as a co-equal acceptance condition alongside |ΔT| and std.

**Risk:** If speedup is confirmatory (with threshold), the experiment must collect timing data with sufficient precision to assert >= 3x. If it is descriptive, the success criteria section is overstated. The contradiction means the experiment cannot unambiguously answer the research question's third clause regardless of the measured values.

**What needs to change in the design:** The plan must declare one consistent role for the speedup metric — either a pre-specified acceptance threshold that is operationally defined (measured timing methodology, units, conditions) or an explicitly descriptive outcome that is removed from the success criteria's acceptance logic.

---

### R2 — Multiple Comparisons: No FWER/FDR Control Pre-specified (statistical_corrections)

**Gap:** The plan evaluates H1–H6 and threshold-tests across 40 (approach, dataset, n, m) cells simultaneously, with no pre-specified multiple-comparisons correction procedure and no acknowledgment of family-wise error rate inflation.

**Risk:** Declining formal tests does not eliminate the multiple comparisons problem. With enough cells, at least one is likely to satisfy all three thresholds by chance. Reporting the best-passing cell as the headline result without correction is Goodhart exploitation.

**What needs to change in the design:** The plan must either (a) pre-specify how threshold tests across cells are aggregated (e.g., require all m values in a contiguous range to pass, not just any one), (b) acknowledge the multiplicity risk explicitly and constrain the reporting to pre-chosen target cells rather than the best-observed, or (c) apply a correction procedure to the threshold-based success determination.

---

### R3 — Approach A Denominator Must Be Validated (data integrity)

**Gap:** The custom denominator `m*k*(2*n-3*k-1)` in Approach A's formula is stated as an "unbiased estimator" with no cited derivation source. The prior experiment (H5) failed due to a normalization bug.

**Risk:** If the denominator is incorrect, |ΔT| values for Approach A will be systematically biased in a direction that cannot be detected by the dry-run check (which only guards against |ΔT| ≈ 0.47). An incorrect denominator could make A appear more accurate than B, which is the experiment's key comparative claim.

**What needs to change in the design:** The plan must cite or derive the normalization constant for the row-subsampled trustworthiness formula. The validation protocol must include a check that Approach A's T_sub at large m (e.g., m = n) converges to T_exact, not just that |ΔT| ≠ 0.47 at m=1000.

---

### R4 — MERFISH Fixture Provenance and Verification Undocumented (reproducibility_spec, data_acquisition)

**Gap:** MERFISH fixtures are referenced by local relative paths with no checksum, hash, or data provenance (what source, what UMAP configuration, what preprocessing). The data manifest lacks source_type labels and post-acquisition verification criteria for any fixture.

**Risk:** An independent party (or a future re-run on a different machine) cannot verify they are using byte-identical data. Without knowing the UMAP parameters that produced Y, the embedding quality context for the trustworthiness measurements cannot be interpreted.

**What needs to change in the design:** Each data manifest entry must include a source_type (gitignored/external/generated) and a verification criterion (file hash, byte size, or shape/dtype assertion). The MERFISH embedding parameters must be documented (or a reference to where they are documented must be provided).

---

### R5 — Execution Protocol Missing Symlink Creation Step (data_acquisition)

**Gap:** Phase 1 of the implementation plan specifies creating symlinks in `data/merfish/` for MERFISH fixtures, but this step is absent from the Execution Protocol's command block. The protocol jumps from environment creation to Gaussian generation without materializing the MERFISH data.

**Risk:** A sequential executor following only the Execution Protocol block will attempt to run `compute_exact.py` with missing MERFISH data.

**What needs to change in the design:** The Execution Protocol must include an explicit MERFISH data acquisition step (symlink creation or equivalent) between environment setup and exact baseline computation.

---

### R6 — Primary Use Case Scale Absent: n ≥ 100K (benchmark_representativeness, ecological_validity)

**Gap:** The plan tests only n=10K and n=50K, sizes where exact computation is already feasible. The primary use case for `trustworthiness_subsampled` (n ≥ 100K) is absent from the test bed. Python speedup measurements also do not validate Rust performance.

**Risk:** Any default subsample size recommendation derived from n ≤ 50K data requires extrapolation beyond the measured range for the actual operating regime. The error/speed trade-off characterization for n=50K may not predict behavior at n=100K–1M.

**What needs to change in the design:** The plan must either (a) bound the scope of the recommendation explicitly ("recommendation is valid for n ≤ 50K; n > 50K requires further study"), (b) include a larger fixture or extrapolation validation strategy with stated uncertainty, or (c) add a note distinguishing the algorithmic scaling characterization (valid across scales if CLT holds) from the absolute subsample size recommendation (scale-dependent).

---

### R7 — Variance Source Ambiguity in std(T_sub) (variance_protocol, measurement_alignment)

**Gap:** `std(T_sub)` across 10 seeds is used as the variance acceptance criterion, but the plan does not document that Y (the embedding) is fixed across seeds — i.e., that the only variance source is subsample selection. The Gaussian Y is generated by the experiment itself; if any stochastic element in the generation or trustworthiness pipeline is not fully seeded, `std(T_sub)` conflates estimation variance with run-to-run noise.

**Risk:** If `std(T_sub)` reflects non-subsample variance, the variance criterion (std < 0.005) measures something other than sub-sampling reliability, and passing or failing the criterion cannot be attributed to the sub-sampling approach alone.

**What needs to change in the design:** The plan must explicitly document which components of the pipeline are fixed (X, Y for MERFISH; X, Y from fixed seed for Gaussian) and which are varied across seeds (subsample indices only). The scope of the 10-seed variance must be stated: "seeds govern only subsample index selection."

---

## Recommended Revisions (Warning Findings)

These gaps reduce confidence in the results but are not blocking.

---

### W1 — Asymmetric Warm-up Protocol (baseline_fairness)

The exact baseline discards 1 warm-up call and takes the median of 3 timed runs; Approach A and B timing relies on 10 random seeds with no explicit warm-up per trial. Speedup ratios may be systematically favorable to the exact baseline due to this asymmetry.

**Design gap:** The warm-up and measurement strategy for the speedup denominator (sub-sampled time) is underspecified relative to the numerator (exact time).

---

### W2 — k-Clamping Asymmetry Makes A and B Non-Comparable at Small m (baseline_fairness)

At small m (e.g., m=250), Approach B clamps n_neighbors to min(k, m-1), solving a structurally different problem (reduced-k trustworthiness on a subset). Approach A uses full k against the full n. At these m values, |ΔT| comparisons between A and B measure different quantities.

**Design gap:** The plan acknowledges this clamping but does not segregate the clamped-m results from the unclamped-m results in the primary analysis or success criteria. Results at clamped m should be reported separately or excluded from the joint comparison.

---

### W3 — Approach B Structural Bias Not Analyzed (red_team)

Approach B's k-NN is computed in the lower-cardinality subset space (m points only), not the full n-point space. This structurally inflates apparent accuracy compared to exact trustworthiness, which uses all n points. A positive B result at small m may reflect reduced problem difficulty rather than sub-sampling accuracy.

**Design gap:** The plan does not include an analysis of how B's effective k-NN shrinkage affects |ΔT| as a function of m, or a comparison metric that controls for this structural difference.

---

### W4 — Threshold Rationale Not Pre-registered (red_team)

The thresholds |ΔT| < 0.01 and std < 0.005 are stated without a pre-registered rationale tied to downstream decision quality (e.g., perceptual significance for the visual evaluation pipeline, or practitioner tolerance). No baseline variance for exact trustworthiness across seeds is reported, making it impossible to assess whether these thresholds are tight or loose relative to measurement noise.

**Design gap:** The plan references "visual eval pipeline tolerance per scope report §Metric Context" for |ΔT| < 0.01 but does not cite the std < 0.005 threshold's origin. Both thresholds should be traceable to a stated tolerance requirement.

---

### W5 — Single Real Dataset Limits Generalizability (benchmark_representativeness)

MERFISH (gene expression, n=10K–50K, d=50) is the sole real-manifold dataset. The recommendation to add `trustworthiness_subsampled` to `src/metrics.rs` will be made based on one data modality.

**Design gap:** Results should be scoped in the plan text to "MERFISH-like data" rather than "sub-sampled trustworthiness in general," and the success criteria should reflect this scoping.

---

### W6 — Execution Protocol Timing Methodology Not Symmetric (measurement_alignment)

The exact baseline uses median of 3 timed runs. The timing methodology for Approach A and B sub-sampled runs is not specified in the execution protocol (single call? warm-up? median?). This asymmetry means the speedup ratio's numerator and denominator are measured with different precision.

**Design gap:** The timing methodology for sub-sampled runs must be stated explicitly and at a comparable precision level to the exact baseline, or the asymmetry must be acknowledged as a limitation on speedup interpretation.

---

### W7 — Transitive Dependency Versions Not Pinned (reproducibility_spec)

Environment.yml pins only direct dependencies to minor-version wildcards. Transitive dependencies (joblib, threadpoolctl, BLAS/LAPACK backend) that affect multithreaded floating-point behavior in sklearn are not pinned.

**Design gap:** The environment specification is insufficient for byte-level reproducibility of timing results across machines or time. A lock file or tighter pin specification is needed for scientific reproducibility of the speedup measurements.

---

### W8 — CLT Slope Deviation Has No Defined Impact on H1 (error_budget, red_team)

The CLT slope check (expect ≈ -0.5) is described as a validation step, but no criterion is stated for when a deviation from -0.5 would alter the H1 verdict or be treated as an inconclusive outcome.

**Design gap:** If the empirical slope is -0.3 (much weaker variance reduction), H1 may still technically pass at large m while extrapolation to smaller m is misleading. The plan needs a stated treatment for slope deviations.

---

### W9 — mean(|ΔT|) Does Not Guard Against Per-Seed Exceedances (measurement_alignment)

The success criteria use mean(|ΔT|) < 0.01, but a practitioner performing a single run experiences |ΔT| from one seed, not the mean. The variance criterion (std < 0.005) is separate and does not bound the worst-case per-seed error.

**Design gap:** The relationship between the mean-based acceptance criterion and per-run practitioner risk is not stated. A worst-case or high-percentile metric may better represent the reliability claim.

---

## Red-Team Decision Points

The following require explicit author decisions before execution. Each has `requires_decision: true`.

| ID | Risk | Decision Needed |
|----|------|-----------------|
| RT1 | Approach B structural bias inflates accuracy at small m | Decide whether B is being evaluated as an estimator of full-n trustworthiness or as trustworthiness of the subset. If the latter, the estimand and research question must be updated to match. |
| RT2 | Thresholds lack pre-registration and baseline noise context | Decide whether to document threshold provenance or run a pilot to establish measurement noise floor before the full sweep. |
| RT3 | H6 extrapolation is out-of-distribution | Decide whether to present H6 results as bounded extrapolation with explicit uncertainty or remove H6 from the hypothesis list. |
| RT5 | Single real dataset limits scope of recommendation | Decide the scope of the API decision: MERFISH-specific validation only, or a broader claim requiring additional datasets. |
| RT6 | Approach A denominator unvalidated | Decide whether to add a known-ground-truth validation to the design (e.g., A at m=n should equal exact T exactly). |
| RT8 | 40 cells, no multiple-comparisons control | Decide whether to pre-select the comparison cells of interest (e.g., fix m=2000 for A and m=5000 for B as the primary test), treating remaining cells as exploratory. |
| RT9 | Approach A asymmetric development effort | Decide whether to document that A and B are not equally mature implementations and scope the conclusions accordingly, or add equivalent correctness safeguards to B. |
