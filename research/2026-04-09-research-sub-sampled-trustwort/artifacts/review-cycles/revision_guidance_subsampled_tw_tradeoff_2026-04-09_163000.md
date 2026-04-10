# Revision Guidance: Sub-sampled Trustworthiness Error/Speed Trade-off

**Verdict:** REVISE
**Plan:** `experiment_plan_subsampled_trustworthiness_tradeoff_2026-04-09_160000.md`
**Review timestamp:** 2026-04-09T16:30:00

Findings describe WHAT is lacking or at risk in the experimental design. The fix is the plan author's responsibility.

---

## Required Revisions (Critical Findings)

### R1 — Estimand Mismatch Between Approaches A and B

**Gap:** Approach B is defined as an estimator of "Subset-T" (trustworthiness of the m-point subgraph), not of full-n trustworthiness T_exact. Yet |ΔT_B| = abs(T_exact - T_sub_B) and the success threshold mean < 0.01 are applied to both approaches identically. This metric cannot separate random sub-sampling variance from the structural gap between two different estimands. Approach B cannot reduce |ΔT_B| to zero even with infinite sub-samples, because the quantities it estimates and what it is measured against are definitionally different.

**Risk:** H1_B's success or failure depends on whether MERFISH neighborhood structure causes Subset-T to approximate full-n T, not on whether Approach B is a good estimator. A "B passes" verdict would be dataset-specific coincidence, not a demonstration of approximation quality. The shipping decision based on this verdict may be invalid.

**Dimensions:** measurement_alignment (critical), estimand_clarity (warning), baseline_fairness (warning), red_team (warning)

---

### R2 — Asymmetric Primary Evaluation Cells (m=2000 vs m=5000)

**Gap:** H1_A is evaluated at m=2000, H1_B at m=5000 — a 2.5× difference. Larger m produces lower variance and lower estimation error for both approaches. The four-cell outcome table and shipping decisions rest on verdicts drawn from structurally unequal operating points, with Approach B tested under materially more favorable conditions.

**Risk:** If Approach B passes at m=5000 but would fail at m=2000, the asymmetric evaluation inflates B's apparent quality relative to A. The comparative conclusion in the outcome table — particularly the "B passes only" and "both pass" cells — is not a fair cross-approach comparison.

**Dimensions:** baseline_fairness (critical), red_team (warning)

---

### R3 — Asymmetric Implementation Correctness Exposure

**Gap:** Approach A uses a custom Python implementation with a known bug history (normalization failure in H5), while Approach B delegates entirely to sklearn. The sanity check (|T_A(m=n) - T_exact| < 1e-10) applies only to Approach A. There is no equivalent correctness gate for Approach B's use of sklearn on a subset.

**Risk:** If Approach A underperforms, the design cannot distinguish a genuine accuracy limitation from a residual implementation defect. The two interpretations produce opposite shipping decisions. Without symmetric correctness validation, a negative result for A is uninterpretable.

**Dimensions:** baseline_fairness (critical)

---

### R4 — Statistical Error Budget Uncharacterized

**Gap:** The plan specifies no Type I or Type II error rates for the threshold-based quality gate. The CLT justification for 10 seeds is circular (expresses CI width as 0.62σ without providing σ). The max(|ΔT|) threshold comparand is a positively biased, high-variance statistic at n=10: the sample maximum underestimates the true worst-case tail and is dominated by whichever single seed produces the largest deviation. No confidence intervals are reported around any of the three decision statistics.

**Risk:** The quality gate for a shipping decision operates at an unknown false acceptance / false rejection rate. A "pass" verdict could reflect adequate estimator quality or an underpowered evaluation that fails to detect unacceptable worst-case behavior. The max(|ΔT|) threshold can trigger H0 (failure) from a single atypical seed without this being distinguished from a genuine high worst-case.

**Dimensions:** error_budget (critical ×3), variance_protocol (warning)

---

### R5 — Multiple Comparisons Framework Absent

**Gap:** Three distinct multiplicity problems are unaddressed: (a) within each primary hypothesis, three conditions are tested jointly (AND conjunction) with no within-hypothesis correction; (b) two parallel primary hypotheses are evaluated with no family-wise error rate defined; (c) the full family of 9 dependent variables and 5 secondary analyses (H2–H6) has no pre-specified alpha level, correction procedure, or confirmatory/exploratory classification.

**Risk:** The probability of at least one erroneous borderline judgment across three within-hypothesis conditions is uncontrolled. The family-wise Type I error rate across the two primary hypotheses is undefined. The secondary analyses provide no statistical guarantees, leaving their evidentiary weight indeterminate and creating HARKing vulnerability.

**Dimensions:** statistical_corrections (critical ×4)

---

### R6 — Timing Protocol Underspecified

**Gap:** The plan discards 1 warmup run but does not specify how many timed runs are recorded per seed trial. If a single timed run per seed is taken, OS scheduling jitter, CPU frequency variation, and memory allocation timing are not isolated. The plan does not state whether the warmup protocol is applied per (approach, m, dataset) cell or only for exact computation. Library-level warmup asymmetry (sklearn NearestNeighbors, BLAS thread pool initialization) is also unaddressed across the 560-trial loop.

**Risk:** Wall time measurements conflate sub-sampling variance with system noise. If H4 (speed scaling) or the default-m recommendation incorporates timing evidence, that evidence may not reflect stable estimates of computational cost. Differences in warmup state between Approach A and B trials within the same loop could introduce systematic timing bias.

**Dimensions:** variance_protocol (critical), unit_interference (warning)

---

### R7 — Python Speedup Has No Ecological Validity for Rust Deployment

**Gap:** The plan measures speedup as wall_exact_s / wall_sub_s in Python, yet the deployment target is a Rust function in `src/metrics.rs`. Python overhead (GIL, NumPy dispatch, interpreter startup) is absent in Rust. The speedup metric is explicitly acknowledged as non-transferable, yet it is used to justify the recommended default subsample size.

**Risk:** The cost side of the accuracy-vs-speed tradeoff — the central reason for sub-sampling — cannot be characterized for the actual deployment environment from Python measurements. A default-m recommendation derived partly from Python speedup may be poorly calibrated for Rust production workloads.

**Dimensions:** ecological_validity (critical), benchmark_representativeness (critical), measurement_alignment (warning)

---

### R8 — Two Datasets Insufficient for a Shipping Decision

**Gap:** Only two datasets at a single dimensionality (d=50) and two scales (10K, 50K) are evaluated. One dataset (Gaussian) is explicitly acknowledged as producing incommensurable evidence (T ≈ 0.5, no manifold structure). Both share the same input dimensionality, the same output dimension (2D), and overlapping size ranges.

**Risk:** The shipping decision and default-m recommendation generalize to arbitrary UMAP embeddings at arbitrary n, k, and dimensionality. Two datasets are insufficient to distinguish whether the observed error-vs-m curves are properties of the estimator or artifacts of MERFISH-specific cluster density. The evidence base is too narrow to support the decision the plan intends.

**Dimensions:** benchmark_representativeness (critical ×2)

---

### R9 — Memory Release Between Trials Not Addressed

**Gap:** Approach A allocates an (m, n) pairwise distance matrix per trial (up to ~10 GB at m=25K, n=50K). Python's memory allocator does not guarantee physical page return to the OS before the next trial. The plan specifies no memory management between trials, no iteration order justification, and no parallelization guard.

**Risk:** Approach B trials that follow large Approach A allocations within the same process may execute under artificially elevated memory pressure, contaminating timing measurements in a way that depends on trial execution order. If trials are parallelized post-plan to reduce the 2–4 hour wall time, concurrent 10 GB allocations could trigger system-wide swapping across all trials.

**Dimensions:** unit_interference (critical), unit_interference (warning)

---

## Recommended Revisions (Warning Findings)

### W1 — "B Passes Only" Outcome Lacks Falsification Criteria
**Gap:** The "B passes only" outcome maps to "evaluate whether subset-T is acceptable product metric" — an open-ended consequence with no stated condition under which the evaluation concludes false. This is the one cell in the four-cell table without a pre-specified decision rule.
**Risk:** The experiment cannot conclusively falsify the product-worthiness claim for Approach B when it is the sole passer; the verdict defers to post-hoc judgment.

### W2 — Inconclusive Zone Operationally Incomplete
**Gap:** The inconclusive zone [0.008, 0.012] applies only to mean(|ΔT|), has no pre-specified operational consequence, and leaves max(|ΔT|) and std(T_sub) without equivalent zones. "Report as near threshold" is observational, not decisional.
**Risk:** Results landing in this zone allow post-hoc avoidance of a definitive falsification conclusion. The asymmetric treatment across the three conditions can produce mixed signals with no pre-committed resolution rule.

### W3 — Threshold Justification Absent
**Gap:** The thresholds (0.01, 0.02, 0.005) have no stated derivation from product requirements, perceptual significance, prior empirical distributions of T, or prior sub-sampling literature.
**Risk:** Without documented basis, there is no way to assess whether a "pass" verdict represents a meaningful guarantee of accuracy. Thresholds set without rationale can be unconsciously calibrated to a preferred outcome.

### W4 — Single-Cell Primary Evaluation Without Contradiction Rule
**Gap:** H1_A and H1_B are each evaluated at a single (m, n, k) cell. The analysis plan reports results for many additional (m, n) combinations, but provides no rule for how to handle contradictions between the primary cell verdict and secondary conditions.
**Risk:** If results at n=50K or adjacent m values contradict the primary cell verdict, post-hoc selection of favorable conditions is possible without violating any pre-specified rule.

### W5 — H6 Extrapolation Unfalsifiable Within Scope
**Gap:** H6 extrapolates the fitted error curve to n=100K and reports a predicted |ΔT| at m=10K, but specifies no criterion under which the extrapolation would be considered invalid. No holdout check at an intermediate n is included.
**Risk:** The extrapolation cannot be contradicted by the data collected in this experiment, making it unfalsifiable within the experiment's scope. Claims derived from it carry no empirical support.

### W6 — std Estimate Uncertainty Not Propagated Into Decision
**Gap:** The std threshold (std < 0.005) is evaluated against a sample std computed from 10 seeds. The 95% CI on σ from 10 samples spans roughly [0.69σ, 1.83σ], meaning the true std could plausibly exceed or fall under 0.005 purely from estimation error.
**Risk:** A "pass" on the std condition may reflect estimation noise rather than genuine low sub-sampling variance. The decision rule treats the sample std as if it were the true σ.

### W7 — Regression Fits Have No Pre-specified Inferential Criteria
**Gap:** H2 (slope ≈ −0.5) and H4 (slope ≈ 1 for A, ≈ 2 for B) use log-log regression on a small grid of m values, but specify no confidence interval width, R² threshold, slope tolerance, or correction for running both regressions.
**Risk:** The slope estimates have undetermined precision, and the "≈ −0.5" standard is informal. There is no pre-committed criterion for what range of estimated slopes constitutes adequate agreement.

### W8 — Warmup Protocol Not Confirmed Symmetric Across Approaches
**Gap:** 1 warmup run is described at the experiment level, but it is not confirmed that this warmup applies per (approach, m, dataset) cell or only to the exact computation. Approach A and B have fundamentally different memory and compute profiles.
**Risk:** A shared warmup for the exact baseline may differentially benefit or disadvantage one approach depending on cache reuse, creating systematic timing bias between the approaches.

### W9 — MERFISH Access via Symlinks Without Version Control
**Gap:** `data/merfish/` uses symlinks to `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`. No git commit SHA or file checksums are recorded for the fixture files.
**Risk:** If those fixtures are regenerated with different parameters in a future commit, this experiment's results would silently change. Independent reproduction requires out-of-band knowledge of the correct fixture version.

### W10 — Root .gitignore Covers Generated Gaussian Data
**Gap:** The root `.gitignore` unconditionally ignores `/data/`. Gaussian data written to `data/gaussian/` inside the experiment directory is covered by this rule and would be silently untracked.
**Risk:** In a fresh worktree, shape verification via git status is impossible for generated files. Downstream acquisition steps cannot confirm data readiness through git state.

### W11 — gen_synthetic.py Output Directory Mismatch
**Gap:** The referenced `gen_synthetic.py` from the prior experiment directory writes output to a hardcoded path in its own directory tree. The plan does not specify the exact invocation with the correct output directory for the new experiment.
**Risk:** An implementor following the plan without supplying an explicit output-dir override would produce Gaussian files in the wrong location, causing downstream scripts to fail without a clear diagnostic.

### W12 — Gaussian Y Has No UMAP Structure (H3 Comparability Limited)
**Gap:** Gaussian Y arrays are random 2D normal projections with no neighborhood preservation. MERFISH Y is an actual UMAP embedding. The H3 comparison (std_MERFISH / std_Gaussian) compares a realistic embedding against a degenerate baseline rather than a range of embedding qualities.
**Risk:** H3 cannot characterize how sub-sampling variance behaves across a representative range of embedding quality — only between a well-structured embedding and a structureless one.

### W13 — k=15 Specificity Not Addressed in Shipping Scope
**Gap:** All results are at k=15 only. The sub-sampling error/speed curve is k-dependent, yet the recommended default m will be used by callers at arbitrary k values.
**Risk:** The default m derived at k=15 may be miscalibrated for users operating at k=5 or k=50, which are plausible deployment values.

### W14 — Transitive Dependencies and micromamba Version Not Pinned
**Gap:** Only 5 direct packages are pinned in `environment.yml`. Transitive dependencies (threadpoolctl, joblib, etc.) and the micromamba tool itself are not pinned to versions.
**Risk:** A later conda-forge solve may resolve different transitive versions, potentially altering numerical results or triggering incompatibilities.

---

## Red-Team Findings (Adversarial — All `requires_decision: true`)

**[RT-1] Asymmetric m-parameter selection**
Approaches A and B are evaluated at different m values. If B would fail at m=2000 or A would pass at m=5000, the parameter selection is masking a performance gap. **Decision required:** Is the asymmetric m choice justified by a principled rationale (e.g., operational cost constraints), or should both approaches be evaluated at matched m values for the primary verdicts?

**[RT-2] Wrong benchmark metric for Approach B**
Approach B estimates subset trustworthiness but is compared against full-n T_exact. Agreement is coincidental. **Decision required:** Is the plan intended to evaluate whether Subset-T approximates full-n T on MERFISH (a dataset-specific claim), or whether Approach B is a useful estimator of its own estimand? These require different metrics and success criteria.

**[RT-3] Threshold derivation risk (Goodhart)**
Thresholds (0.01, 0.02, 0.005) have no stated derivation. **Decision required:** Should thresholds be derived from a product requirement specification, a prior empirical distribution of T on representative datasets, or a literature precedent? Without pre-registered rationale, post-hoc threshold adjustment remains possible.

**[RT-4] Asymmetric tuning exposure**
Approach A is a custom implementation subject to researcher iteration; B uses sklearn with fixed defaults. **Decision required:** Is there a protocol for documenting which implementation decisions were made before vs. after seeing preliminary error magnitudes? Without such documentation, the evaluation cannot guard against implicit tuning against the evaluation metric.

**[RT-5] Static Y arrays as hidden covariate**
All trials reuse the same pre-computed Y arrays. **Decision required:** Is the experiment intended to characterize sub-sampling variance for a specific embedding (valid for the fixture-based use case) or to generalize to freshly computed embeddings? If the latter, the design needs to incorporate embedding variance.

**[RT-6] No pre-specified seed exclusion rule**
max(|ΔT|) across 10 seeds can be triggered by a single outlier. **Decision required:** Is there a pre-specified criterion for whether any seed is considered a valid trial? If not, the plan should state that no post-hoc exclusions are permitted, and max(|ΔT|) reflects the true realized worst case including any degenerate draws.

**[RT-7] Trustworthiness concentration near 1.0**
High-quality embeddings produce T_exact near 1.0, compressing the signal range and making any estimator pass an absolute error threshold easily. **Decision required:** Should the experiment validate that the estimator performs adequately at moderate T_exact values (e.g., 0.7–0.9)? If not, the scope of the "acceptable for production" claim should be bounded to high-quality embedding regimes.

**[RT-8] Evaluation collision — sklearn in treatment and measurement**
Both Approach A's sanity check and Approach B use sklearn. A systematic sklearn error would cancel in A's verification and remain undetected. **Decision required:** Should the experiment include a reference implementation comparison that does not use sklearn (e.g., a Python implementation of the full-n trustworthiness formula from first principles) to detect potential sklearn artifacts?
