# Review-Design Evaluation Dashboard

## Verdict: REVISE

**Plan:** `experiment_plan_subsampled_trustworthiness_tradeoff_2026-04-09_160000.md`
**Experiment type:** `benchmark`
**Secondary modifiers:** `+multi_metric` (9 DVs ≥ 3 → `statistical_corrections` elevated M → H)
**Review timestamp:** 2026-04-09T16:30:00

---

## Classification Summary

| Field | Value | Source |
|-------|-------|--------|
| experiment_type | benchmark | extracted (Rule 1: IVs are method names, DVs are performance metrics, multiple comparators) |
| hypothesis_h0_A | mean(ΔT_A) ≥ 0.01 OR max ≥ 0.02 OR std ≥ 0.005 at m=2000, MERFISH n=10K | extracted |
| hypothesis_h1_A | mean(ΔT_A) < 0.01 AND max < 0.02 AND std < 0.005 at m=2000, MERFISH n=10K | extracted |
| hypothesis_h0_B | mean(ΔT_B) ≥ 0.01 OR max ≥ 0.02 OR std ≥ 0.005 at m=5000, MERFISH n=10K | extracted |
| hypothesis_h1_B | mean(ΔT_B) < 0.01 AND max < 0.02 AND std < 0.005 at m=5000, MERFISH n=10K | extracted |
| estimand | Contrast: sub-sampled T vs exact full-n T on MERFISH n=10K, k=15, varying m and approach | extracted |
| baselines | T_exact via sklearn on full dataset; Approach A vs B compared | extracted |
| statistical_plan | Threshold-based; 10 seeds per cell; no formal alpha or correction | extracted |
| success_criteria | Three threshold conditions met per hypothesis; inconclusive zone [0.008, 0.012] for mean | extracted |

---

## Dimension Scorecard

| Dimension | Level | Weight | Spawned | Findings | Critical | Warning | Info |
|-----------|-------|--------|---------|----------|----------|---------|------|
| estimand_clarity | L1 | H | ✓ | 5 | 0 | 3 | 2 |
| hypothesis_falsifiability | L1 | H | ✓ | 9 | 0 | 6 | 3 |
| baseline_fairness | L2 | H | ✓ | 7 | 2 | 4 | 1 |
| unit_interference | L2 | H | ✓ | 9 | 1 | 4 | 4 |
| causal_structure | L2 | S | ✗ | — | — | — | — |
| error_budget | L3 | H | ✓ | 7 | 3 | 3 | 1 |
| statistical_corrections | L3 | H | ✓ | 8 | 4 | 4 | 0 |
| variance_protocol | L3 | H | ✓ | 7 | 1 | 4 | 2 |
| benchmark_representativeness | L4 | M | ✓ | 6 | 2 | 4 | 0 |
| ecological_validity | L4 | M | ✓ | 6 | 1 | 4 | 1 |
| measurement_alignment | L4 | M | ✓ | 6 | 3 | 3 | 0 |
| reproducibility_spec | L4 | M | ✓ | 10 | 0 | 6 | 4 |
| data_acquisition | L4 | M | ✓ | 8 | 1 | 6 | 1 |
| red_team | L3/concurrent | H | ✓ | 8 | 0 | 8 | 0 |

**Summary:** 17 critical · 59 warnings · 19 info across 13 active dimensions.

---

## Critical Findings (REVISE triggers)

> No STOP triggers: no critical findings from `estimand_clarity`, `hypothesis_falsifiability`, or `red_team`.

### Baseline Fairness

**[BF-1] Asymmetric primary evaluation cells** `critical`
H1_A is evaluated at m=2000 while H1_B is evaluated at m=5000 — a 2.5× difference in subsample size. Because larger m produces lower variance and lower estimation error, Approach B is tested under materially more favorable conditions than Approach A. The four-cell outcome table's shipping decisions rest on verdicts drawn from non-matching operating points, making cross-approach comparisons structurally unequal.

**[BF-2] Asymmetric implementation correctness exposure** `critical`
Approach A uses a custom Python implementation while Approach B delegates entirely to sklearn's validated code path. The prior experiment failure (H5) was caused by a normalization bug in a custom implementation of this same function. If Approach A underperforms in this experiment, the design provides no way to distinguish a genuine accuracy limitation from a residual implementation defect — the two interpretations produce opposite shipping decisions.

### Unit Interference

**[UI-1] Memory allocation not guaranteed released between trials** `critical`
Approach A allocates an (m, n) pairwise distance matrix per trial — up to ~10 GB at m=25K, n=50K. Python's memory allocator does not guarantee physical page return to the OS before the next trial begins. Under sequential execution, subsequent Approach B trials (far smaller memory footprint) may execute under elevated memory pressure from prior Approach A allocations, contaminating wall_s measurements for B in a way that depends on trial execution order.

### Error Budget

**[EB-1] Error rates of the quality gate uncharacterized** `critical`
No Type I or Type II error rates are specified for the threshold-based decision rule. The experiment produces a binary PASS/FAIL quality gate for shipping, but the probability that an acceptable implementation is incorrectly rejected (Type II) or an unacceptable one is incorrectly accepted (Type I) is nowhere acknowledged or bounded.

**[EB-2] Power justification circular** `critical`
The CLT-motivated rationale for 10 seeds — "±1.96·σ/√10 ≈ 0.62·σ for 95% CI width" — expresses CI width as a multiple of σ without providing any prior estimate of σ. Without a known or bounded σ, 10 seeds cannot be shown adequate to resolve the threshold comparisons at the required precision.

**[EB-3] max(|ΔT|) is a biased, high-variance threshold comparand** `critical`
The sample maximum across 10 seeds is used as a direct comparand against the 0.02 threshold. The sample maximum is positively biased relative to the true population maximum, has high sampling variance at n=10, and its sampling distribution is not addressed. A single extreme seed can trigger H0 (failure), but the plan provides no mechanism to distinguish a genuinely high worst-case from a statistical artifact of a small sample.

### Statistical Corrections

**[SC-1] Within-hypothesis multiplicity uncorrected** `critical`
H1_A and H1_B each require three conditions to hold jointly (mean AND max AND std), but no correction for the three simultaneous threshold comparisons within each hypothesis is pre-specified. A false favorable judgment on any single borderline condition propagates into a false overall pass, and the probability of at least one erroneous borderline result across three conditions is uncontrolled.

**[SC-2] Family-wise error across two primary hypotheses undefined** `critical`
Two parallel primary hypotheses (H1_A and H1_B) are evaluated jointly with no pre-specified family-wise error correction. The plan does not state whether they constitute a family requiring joint error control or fully independent claims.

**[SC-3] 9-DV family uncorrected** `critical`
The analysis encompasses 9 dependent variables with no pre-specified multiple comparisons correction covering this full family. With ≥3 DVs, the `+multi_metric` modifier elevates `statistical_corrections` to H-weight, making this absence a high-priority gap.

**[SC-4] Secondary analyses uncategorized and uncorrected** `critical`
H2–H6 are evaluated without a declared alpha level, p-value criteria, or correction procedure, and without specifying whether each is confirmatory or exploratory. Five parallel secondary analyses with undefined evidentiary status leave the false discovery rate uncontrolled.

### Variance Protocol

**[VP-1] Timing based on single run per seed** `critical`
The plan specifies 1 warmup run discarded but does not specify how many timed runs are recorded per seed trial. If wall_s is taken from a single timed run, OS scheduling jitter and memory allocation variability are unmitigated. A single-run timing measurement per seed conflates sub-sampling variance with system noise and provides no basis for separating these sources.

### Measurement Alignment

**[MA-1] |ΔT_B| is not a valid metric for Approach B's deployment fitness** `critical`
|ΔT_B| = abs(T_exact - T_sub_B) mixes two structurally distinct sources of discrepancy: random sub-sampling variance and a permanent estimand gap (Approach B estimates Subset-T, not full-n T). Unlike Approach A, Approach B cannot reduce |ΔT_B| to zero even with infinite sub-samples, because the two quantities differ by construction. A threshold on |ΔT_B| cannot determine whether Approach B is an accurate estimator of its own estimand, only whether Subset-T happens to approximate full-n T on this particular dataset.

**[MA-2] Shared threshold conflates incommensurable error types** `critical`
The same threshold (mean < 0.01) is applied to both approaches despite the fact that for Approach A the deviation reflects random estimation error around a shared estimand, while for Approach B it includes a potentially systematic structural offset. A threshold that is calibrated as a tolerance for random error (A) has no principled interpretation as a tolerance for estimand mismatch (B).

**[MA-3] Gaussian and MERFISH conditions test incommensurable quantities** `critical`
For Gaussian data, the plan acknowledges "comparison is about variance, not absolute error" (T ≈ 0.5). For MERFISH data, the comparison is between sub-sampled and exact estimates of a meaningful quality score. The |ΔT| metric produces structurally different evidence under each condition: one tests variance around a chance-level floor, the other tests proximity to a meaningful ground truth. The two cannot be combined under the same framework without a transformation that the plan does not specify.

### Ecological Validity

**[EV-1] Python speedup has no ecological validity for the Rust deployment target** `critical`
The plan uses speedup (wall_exact_s / wall_sub_s, measured in Python) as evidence for the recommended default subsample size, yet the stated motivation is to ship a Rust function in `src/metrics.rs`. Python speedup reflects GIL overhead, NumPy dispatch costs, and interpreter behavior that are absent in the Rust production path. The speedup-based component of the default-m recommendation is derived from an environment that does not represent the deployment context.

### Benchmark Representativeness

**[BR-1] Two datasets insufficient for a generalizable shipping decision** `critical`
Only two datasets (MERFISH and Gaussian) at a single dimensionality (d=50) and two scales (10K, 50K) are evaluated. The shipping decision and default-m recommendation extrapolate to an unbounded space of dataset types, dimensionalities, embedding qualities, and sizes. Two datasets are insufficient to distinguish whether the observed error-vs-m curves are properties of the estimator or artifacts of MERFISH-specific neighborhood structure.

**[BR-2] Speed cost model is wrong environment for the shipping decision** `critical`
The plan explicitly acknowledges that Python wall-clock speedups do not predict Rust speedups, yet the default subsample size recommendation is partly justified by computational cost. The cost side of the accuracy-vs-speed tradeoff — the core justification for sub-sampling — cannot be quantified for the actual production environment from Python measurements.

---

## Warning Highlights (selected)

> Full warning list in revision guidance. Selected high-priority warnings below.

**[EC-W1] `estimand_clarity`** — Approach A and B are evaluated against identical numerical thresholds despite targeting fundamentally different estimands. The plan does not establish whether the same error tolerance is meaningful or appropriate for both. `requires_decision: true`

**[EC-W2] `estimand_clarity`** — The outcome table treats H1_A and H1_B verdicts as jointly informing a single shipping decision, but the two approaches estimate non-comparable quantities. A pass on H1_B does not imply B approximates the same quantity as A. `requires_decision: true`

**[HF-W1] `hypothesis_falsifiability`** — When H0_A is not rejected but H1_B is supported ("B passes only"), the operational consequence is "evaluate whether subset-T is acceptable" — an open-ended outcome with no stated falsification criteria for product-worthiness. `requires_decision: true`

**[HF-W2] `hypothesis_falsifiability`** — The inconclusive zone [0.008, 0.012] has no pre-specified operational consequence. The experiment design provides no pre-committed rule for what happens when a result lands in this zone.

**[HF-W3] `hypothesis_falsifiability`** — Threshold values (0.01, 0.02, 0.005) are stated without justification from product requirements, prior empirical distributions, or perceptual significance.

**[VP-W1] `variance_protocol`** — The std threshold (std < 0.005) is itself evaluated as a DV from only 10 seeds. The 95% CI on σ estimated from 10 samples spans roughly [0.69σ, 1.83σ], meaning the sample std could plausibly exceed or fall under 0.005 purely due to estimation error.

**[RP-W1] `reproducibility_spec`** — MERFISH data/merfish/ is accessed via symlinks to `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`. The plan does not record git commit SHA or checksums of the fixture files.

**[DA-W1] `data_acquisition`** — Root `.gitignore` unconditionally ignores `/data/`. Generated Gaussian files written to `data/gaussian/` inside the experiment directory would be silently git-ignored in a fresh worktree, making them invisible to tracking and verification.

**[DA-W2] `data_acquisition`** — The plan references reusing `gen_synthetic.py` from the prior experiment directory, but that script writes output to a hardcoded path in its own directory tree. The plan does not specify the exact invocation with the correct `--output-dir` for the new experiment.

---

## Adversarial Findings (Red Team)

> All red-team findings carry `requires_decision: true`. Severity capped at `warning` for benchmark type.

**[RT-1] Asymmetric m-parameter selection** `warning · requires_decision`
Approaches A and B are evaluated at different m values (2000 vs 5000). If B would fail at m=2000 or A would pass at m=5000, the asymmetric parameter selection could be masking a meaningful performance gap between approaches, producing a misleading "both pass" outcome that flatters B or conceals A's advantage.

**[RT-2] Wrong benchmark metric for Approach B** `warning · requires_decision`
Approach B estimates subset trustworthiness but is compared against full-n T_exact. Any agreement between B's estimate and T_exact is coincidental or dataset-dependent, not structurally guaranteed. The benchmark is measuring cross-estimand distance, not estimation quality.

**[RT-3] Threshold derivation risk (Goodhart)** `warning · requires_decision`
The pass/fail thresholds (0.01, 0.02, 0.005) have no stated derivation from downstream use-case requirements, prior literature, or pilot data. Thresholds set without pre-registration rationale can be unconsciously set to ensure a preferred outcome.

**[RT-4] Asymmetric tuning exposure** `warning · requires_decision`
Approach A is a custom implementation subject to researcher optimization (algorithm choice, batching, distance defaults), while Approach B delegates entirely to sklearn with fixed defaults. Any implementation debugging or adjustment applied to Approach A during development constitutes asymmetric effort against the evaluation, with no controls documenting which decisions were made before vs. after seeing error magnitudes.

**[RT-5] Static Y arrays — hidden covariate** `warning · requires_decision`
All trials reuse the same pre-computed Y arrays for MERFISH data. If the embedding Y was produced or selected with awareness of which parameters perform well on this dataset, repeated evaluation against a fixed Y conflates estimation variance with dataset-specific embedding artifacts. Results may not generalize to independently computed embeddings.

**[RT-6] Survivorship bias — no pre-specified seed exclusion rule** `warning · requires_decision`
With 10 seeds and max(|ΔT|) evaluated across them, a single outlier seed can cause failure. The experiment does not pre-specify whether seeds with anomalous behavior may be excluded, leaving open the possibility of post-hoc seed-level exclusions that selectively deflate the max statistic.

**[RT-7] Goodhart: trustworthiness concentrates near 1.0** `warning · requires_decision`
If the MERFISH embedding is high-quality, T_exact is close to 1.0 and any estimator can achieve mean |ΔT| < 0.01 simply because the signal range is compressed. Passing the threshold on this dataset does not demonstrate the estimator is useful in regimes where T_exact is moderate (0.7–0.9), which is where accurate estimation matters most for real deployment decisions.

**[RT-8] Evaluation collision — sklearn in treatment and measurement** `warning · requires_decision`
Approach A's correctness is verified against sklearn's full-n trustworthiness, and Approach B also calls sklearn internally. If sklearn has any quirks or version-dependent behavior, both treatment and measurement instrument share the same code path. A systematic sklearn error would cancel in A's comparison and remain undetected, making the benchmark unable to catch a class of systematic errors.

---

## Cannot Assess

The following dimensions could not be fully evaluated due to absent plan content:

1. **Iteration order and memory release sequencing** — The plan does not specify the loop order over (approach, dataset, n, m, seed) in `run_subsampling.py`. Cannot assess whether large Approach A allocations systematically precede Approach B trials within each iteration, or whether any process-level memory management is applied between trials.

2. **Effect size and σ for power characterization** — No prior estimate of sub-sampling variance σ (in units of |ΔT|) is provided. Cannot assess whether 10 seeds achieves adequate classification accuracy for plausible effect sizes near the 0.01 threshold — the power analysis is entirely absent.

3. **Hardware specification** — The plan states "same machine for all runs" without specifying CPU model, core count, RAM, or OS. Cannot assess whether timing results are interpretable or comparable to any reference, or whether they would reproduce on a different machine.

4. **Approach A implementation correctness (pre-measurement)** — The implementation specified in the plan has not yet been executed. The sanity check (|T_A(m=n) − T_exact| < 1e-10) is described as the first dry-run test, but the design cannot be evaluated for correctness at review time. Cannot assess whether the denominator formula handles edge cases (m ≈ n, near-degenerate k-NN neighborhoods) correctly.

---

## Mechanizable Check Log

| Check | Result | Notes |
|-------|--------|-------|
| YAML frontmatter present | FAIL | No `---` delimiters in plan; all fields extracted from prose |
| Seeds pre-specified | PASS | Sub-sampling: 0–9; Gaussian generation: 42 |
| Environment dependencies pinned | PARTIAL | 5 direct packages pinned; transitive dependencies not pinned |
| Success criteria operationalized | PASS | Three threshold conditions per hypothesis with explicit values |
| Dry-run flag specified | PASS | `--dry-run` flag in `run_subsampling.py` |
| Analysis script described | PASS | `analyze_results.py` with all analyses enumerated |
| Sanity check for Approach A | PASS | `|T_A(m=n) - T_exact| < 1e-10` specified |
| Data acquisition for primary hypotheses | PASS | MERFISH fixtures confirmed on disk |
| Data acquisition for secondary hypotheses (H3) | PARTIAL | Gaussian generation described but output-dir unclear |
| Multiple comparisons correction pre-specified | FAIL | No correction procedure for any analysis |
| Type I/II error rates acknowledged | FAIL | No error rate characterization anywhere in plan |
| Hardware spec documented | FAIL | "Same machine" only; no identifying detail |
| Gitignore handling for generated data | FAIL | Root `/data/` gitignore covers `data/gaussian/` |
| Fixture file checksums recorded | FAIL | No SHA or checksum for MERFISH .npy files |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 17
warning_count: 59
red_team_count: 8
active_dimensions: 13
warning_threshold: 65
```
