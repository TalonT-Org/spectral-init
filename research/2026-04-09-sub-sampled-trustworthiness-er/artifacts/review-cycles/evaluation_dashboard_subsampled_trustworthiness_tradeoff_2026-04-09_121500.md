# Evaluation Dashboard: Sub-sampled Trustworthiness Error/Speed Trade-off

**Verdict: REVISE**
**Experiment Type:** benchmark
**Plan File:** experiment_plan_subsampled_trustworthiness_tradeoff_2026-04-09_120000.md
**Review Date:** 2026-04-09

---

## Verdict Banner

> ⚠ **REVISE** — The plan has a sound research question and a well-scoped execution
> strategy, but critical design gaps in statistical validity, ecological correspondence,
> measurement definition, data provenance, and benchmark representativeness require
> resolution before execution. No L1 structural defects were found; all issues are
> addressable without abandoning the experimental approach.

---

## Classification Summary

| Field | Value | Source |
|-------|-------|--------|
| experiment_type | benchmark | triage (rule 1: IVs include method names Approach A/B, DVs are performance metrics, multiple comparators) |
| secondary_modifiers | +multi_metric (6 DVs ≥ 3) | statistical_corrections M → H |
| hypothesis_h0 | Sub-sampling does not achieve \|ΔT\| < 0.01 at any m < n/2, or std > 0.005 | extracted |
| hypothesis_h1 | Exists m << n with \|ΔT\| < 0.01, std < 0.005 across 10 seeds, speedup >= 3x | extracted |
| estimand | Approach A/B vs. exact computation on \|ΔT\|/speedup in MERFISH data at varying m | extracted |
| baselines | Exact sklearn trustworthiness (ground truth); Approach A; Approach B | extracted |
| statistical_plan | Threshold comparisons with 95% CI; no formal tests; 10 seeds | extracted |
| success_criteria | Three-outcome scheme: conclusive positive, conclusive negative, inconclusive | extracted |

---

## Dimension Scorecard

| Dimension | Weight | Level | Findings | Severity Summary |
|-----------|--------|-------|----------|-----------------|
| estimand_clarity | H (L1) | L1 | 4 findings | 2 warning, 2 info |
| hypothesis_falsifiability | H (L1) | L1 | 1 finding | 1 info |
| baseline_fairness | M (L2) | L2 | 5 findings | 3 warning, 2 info |
| unit_interference | M (L2) | L2 | 6 findings | **1 critical**, 4 warning, 1 info |
| causal_structure | S | — | Not spawned (SILENT for benchmark) | — |
| red_team | H | L3 | 10 findings | 7 warning, 3 info |
| error_budget | M (L3) | L3 | 3 findings | 1 warning, 2 info |
| statistical_corrections | H (L3) | L3 | 5 findings | **2 critical**, 3 warning |
| variance_protocol | H (L3) | L3 | 5 findings | **1 critical**, 3 warning, 1 info |
| benchmark_representativeness | L (L4) | L4 | 8 findings | **2 critical**, 5 warning, 1 info |
| ecological_validity | M (L4) | L4 | 5 findings | **2 critical**, 2 warning, 1 info |
| measurement_alignment | M (L4) | L4 | 6 findings | **1 critical**, 4 warning, 1 info |
| reproducibility_spec | L (L4) | L4 | 11 findings | **1 critical**, 6 warning, 4 info |
| data_acquisition | M (L4) | L4 | 6 findings | **2 critical**, 2 warning, 2 info |

**Active dimensions:** 13 (causal_structure SILENT for benchmark)
**Warning threshold:** 65 (13 × 5)
**Critical findings (non-L1 stop triggers):** 12
**Warning findings:** ~42
**Stop triggers:** 0

---

## Critical Findings (Non-Stop)

These critical findings do not trigger STOP (they are not from estimand_clarity, hypothesis_falsifiability, or red_team dimensions), but they must be resolved before the plan is viable.

### C1 — unit_interference: RNG State Isolation Not Guaranteed
**Section:** Implementation Details — Approach B
Approach B uses `np.random.default_rng(seed)` correctly, but no design constraint prevents sklearn's internal trustworthiness implementation or other sweep-loop calls from consuming global NumPy legacy RNG state (`np.random.*`). If legacy RNG state is mutated, later trials in the same process run will have their effective random draws altered by earlier trial ordering.

### C2 — statistical_corrections: FWER Uncontrolled (No Pre-specified Correction)
**Section:** Hypothesis Testing
With 6 DVs, 2 approaches, 5–7 subsample sizes, and 2 datasets, the family-wise error rate across simultaneous threshold comparisons is uncontrolled. No correction procedure (Bonferroni, Holm, BH) is pre-specified, and no justification is given for why threshold-comparison framing is statistically equivalent to FWER/FDR control.

### C3 — statistical_corrections: Formal Test Dismissal Does Not Eliminate Multiple Comparisons
**Section:** Hypothesis Testing
The claim "no formal statistical tests are needed" does not eliminate the multiple comparisons problem. With H1–H6 evaluated jointly and 40 (approach, dataset, n, m) cells, the probability of spurious threshold satisfaction is non-trivial and left unaddressed.

### C4 — variance_protocol: std(T_sub) Conflates Distinct Variance Sources
**Section:** Analysis Plan
`std(T_sub)` across seeds is reported as the variance measure, but it conflates subsample selection variance with any embedding non-determinism. The plan does not document whether these sources are separable or state that the 10-seed spread captures only subsample variance.

### C5 — benchmark_representativeness: Only Two Structural Extremes Tested
**Section:** Inputs and Data
Only two datasets are used: MERFISH (real, clustered manifold) and Gaussian (synthetic, isotropic). No intermediate cases (image embeddings, text embeddings, overlapping clusters) are included. A recommended default subsample size derived from two structural poles may not hold for the majority of UMAP use cases.

### C6 — benchmark_representativeness: Primary Use Case (n ≥ 100K) Absent
**Section:** Threats to Validity (External)
The plan's own threat #4 acknowledges that n=10K and n=50K are sizes where exact computation is already feasible. The regime where `trustworthiness_subsampled` would actually be invoked in production (n ≥ 100K) is entirely absent from the test bed, requiring extrapolation beyond the measured range for the core recommendation.

### C7 — ecological_validity: Python Benchmark Does Not Validate Rust Deployment
**Section:** Environment
The experiment is conducted entirely in Python using sklearn, but the intended deployment is `src/metrics.rs` (Rust). Speedup ratios and error characteristics measured in Python do not directly validate the behavior of the Rust implementation.

### C8 — ecological_validity: Python Speedups Structurally Disconnected from Rust Performance
**Section:** Threats to Validity (External)
The plan acknowledges that Python wall-clock speedups may not transfer to Rust (SIMD, Rayon, scratch-buffer optimizations), but this gap is structural: the experiment produces no Rust-environment evidence. Decisions about adding `trustworthiness_subsampled` to `src/metrics.rs` rest entirely on out-of-environment measurements.

### C9 — measurement_alignment: Speedup Metric Status Contradicted Between Table and Success Criteria
**Section:** Dependent Variables / Success Criteria
The dependent variables table marks the speedup ratio as "No threshold — purely descriptive," yet the success criteria lists speedup >= 3x as a co-equal acceptance condition alongside |ΔT| and std. The metric's confirmatory vs. descriptive status is undefined, undermining whether the experiment can answer the research question's third clause.

### C10 — reproducibility_spec: MERFISH Fixture Provenance Undocumented
**Section:** Inputs and Data
MERFISH fixtures are referenced by local relative paths with no checksum, file hash, or data provenance record. An independent party cannot verify they are using byte-identical data, and the origin of the fixtures (how processed, from what source, what UMAP parameters produced Y) is undocumented.

### C11 — data_acquisition: No Source-Type Labels or Verification Criteria for MERFISH Fixtures
**Section:** Inputs and Data / data_manifest
The data manifest does not assign source_type labels (external, gitignored, generated) to entries, and no post-acquisition verification criteria (shape, dtype, file hash, byte size) are specified for either MERFISH fixture after the symlink operation. MERFISH data under a symlinked research path is effectively gitignored and requires explicit acquisition documentation.

### C12 — data_acquisition: Execution Protocol Omits Symlink Creation Step
**Section:** Execution Protocol
Phase 1 specifies creating symlinks in `data/merfish/` for MERFISH fixtures, but this step is absent from the Execution Protocol's numbered command block. The protocol begins with environment creation and Gaussian generation, leaving MERFISH data acquisition invisible to sequential execution.

---

## Adversarial Findings (Red-Team) — All Require Decision

All findings below have `requires_decision: true`.

| # | Severity | Finding Summary |
|---|----------|-----------------|
| RT1 | warning | **Approach B structural bias:** B's k-NN is computed in a lower-cardinality space (subset only), inflating apparent accuracy. A positive B result does not answer whether sub-sampling estimates full-n trustworthiness. |
| RT2 | warning | **Threshold pre-registration absent:** |ΔT| < 0.01 and std < 0.005 thresholds are set without a rationale tied to downstream decision quality. No baseline variance for exact T across seeds is reported. Thresholds could be chosen post-hoc to match whichever cell passes. |
| RT3 | warning | **H6 extrapolation beyond measured range:** The CLT fit is only validated at n=10K and n=50K. Applying it to n=100K is an out-of-distribution prediction. |
| RT4 | info | **T_sub timing methodology unspecified:** Speedup measurement methodology differs between exact (median-of-3) and sub-sampled (unspecified). If sub-sampled uses a single cold run, the speedup ratio is artificially inflated. |
| RT5 | warning | **Single real dataset:** MERFISH is the only real-manifold dataset and the one that motivated the experiment. Gaussian provides no meaningful adversarial signal. Positive results are specific to MERFISH manifold geometry. |
| RT6 | warning | **Approach A denominator unvalidated:** The custom denominator `m*k*(2*n-3*k-1)` is stated as "unbiased estimator" but cites no derivation source. An incorrect denominator could make A appear more accurate than B in a direction that supports the key comparative claim. |
| RT7 | info | **Dry-run guard too weak:** Only checks |ΔT| ≠ 0.47 (prior bug signature). A different normalization bug producing |ΔT| ~ 0.03 would pass the guard while producing a false positive. |
| RT8 | warning | **Multiple comparisons / Goodhart exploitation:** 40 (approach, dataset, n, m) cells with no multiple-comparisons correction. The probability that at least one cell satisfies all thresholds by chance is non-trivial. Reporting the best-passing cell as the headline result is classical Goodhart exploitation. |
| RT9 | warning | **Asymmetric development effort:** Approach A receives a custom self-distance fix and custom denominator; Approach B uses sklearn defaults. Accuracy differences between approaches may reflect implementation polish rather than algorithmic superiority. |
| RT10 | info | **CLT slope check lacks failure criterion:** Deviation from the expected -0.5 slope has no stated impact on the H1 test outcome, leaving the variance-scaling validation without a defined failure path. |

---

## Cannot Assess

The following aspects of the experimental design could not be evaluated due to absent plan content:

1. **Hardware and OS specifications not recorded** — The plan does not specify CPU count, BLAS thread count, or memory bandwidth of the target machine. Wall-clock speedup ratios and the 2 GB peak memory claim cannot be assessed for reproducibility across different execution environments.

2. **MERFISH embedding parameters not documented** — The plan does not specify what UMAP parameters (n_neighbors, min_dist, random_state) produced the pre-computed 2D UMAP embeddings stored in the MERFISH fixtures. Whether the embedding quality (T_exact ≈ 0.95–0.99, as expected by the plan) is typical of UMAP outputs in general, or specific to a particular configuration, cannot be assessed.

3. **Approach A denominator derivation** — The custom trustworthiness denominator `m*k*(2*n-3*k-1)` is stated without citation. Whether this is the correct normalization constant for the row-subsampled trustworthiness estimator cannot be verified from the plan text alone.

4. **Inter-trial isolation mechanism** — The plan does not specify whether each (seed, m, approach, dataset) trial runs in a subprocess or shares a Python process with other trials. Whether unit interference risks (findings C1, L2-UI-2) materialize in practice cannot be assessed without knowing the process isolation model.

---

## Mechanizable Check Log

Binary checks that could be automated in future review tooling:

| Check | Automatable | Result |
|-------|-------------|--------|
| Seeds 0–N specified in plan | Yes | PASS (seeds 0–9 stated) |
| Environment.yml present with pinned deps | Yes | PASS (environment.yml specified) |
| MERFISH fixture paths stated | Yes | PASS (4 paths listed with sizes) |
| Gaussian fixture generation seeded | Yes | PASS (seed=42 in gen_gaussian_50d.py) |
| Dry run phase defined | Yes | PASS (Phase 6 with specific minimal config) |
| Execution protocol present | Yes | PASS (numbered bash commands) |
| Symlink/acquisition step in execution protocol | Yes | FAIL (missing from protocol command block) |
| Checksum/hash for external fixtures | Yes | FAIL (not specified) |
| Source-type labels on all data manifest entries | Yes | FAIL (no source_type field) |
| Multiple-comparisons correction pre-specified | Yes | FAIL (explicitly declined) |
| Speedup metric threshold stated | Yes | FAIL (contradicted between table and success criteria) |
| Approach A denominator cited | Yes | FAIL (no citation) |

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 12
warning_count: 42
red_team_count: 10
active_dimensions: 13
warning_threshold: 65
```
