# Evaluation Dashboard — Sub-Sampled Trustworthiness Error/Speed Trade-off (Rust Validation)

**Slug:** `subsampled-tw-rust-tradeoff`
**Timestamp:** 2026-04-09_204859
**Experiment Type:** `configuration_study`
**Active Dimensions:** 13

---

## Verdict

```
╔══════════════════════════════════════════════════════════════╗
║                        REVISE                                ║
║  1 critical · ~35 warnings · 1 BLOCKING · ~20 REQUIRED       ║
╚══════════════════════════════════════════════════════════════╝
```

**Reason:** One BLOCKING critical finding (data access infrastructure absent in fresh worktrees with no automated reconstitution), combined with substantive warning clusters across statistical methodology, hypothesis specification, and variance protocol. Warning count (~35) is well below the proportional threshold of 65, so REVISE is driven by the critical finding, not warning volume.

---

## Dimension Scorecard

| Dimension | Weight | Level | Findings | Severity Summary |
|-----------|--------|-------|----------|-----------------|
| estimand_clarity | H | L1 | 4 warn, 2 info | No criticals |
| hypothesis_falsifiability | H | L1 | 4 warn, 1 info | No criticals |
| baseline_fairness | — | L2 | 1 warn | Code duplication asymmetry |
| unit_interference | — | L2 | 1 warn | Exact-first cache confound |
| red_team | — | Adv | 6 warn | All `requires_decision: true` |
| error_budget | H | L3 | 5 warn, 2 info | Power gaps for H2/H3/H5 |
| statistical_corrections | H | L3 | 4 warn, 3 info | FWER justification gaps |
| variance_protocol | H | L3 | 4 warn, 1 info | Non-det scope; warmup; core count |
| measurement_alignment | M | L4 | 2 warn, 2 info | abs_delta_T ambiguity; timing |
| agent_implementability | H | L4 | 2 crit, 7 warn, 2 info | **BLOCKING**: data absent in fresh worktree |
| data_acquisition | M | L4 | 1 crit, 3 warn, 1 info | **BLOCKING**: manual download, no URL |
| ecological_validity | L | L4 | 2 warn, 2 info | Hardware generalizability |
| resource_proportionality | L | L4 | 1 warn, 1 info | Rayon non-det budget thin |

> **Note on L2 findings:** `baseline_fairness` and `unit_interference` subagents produced findings that referenced Python scripts, `utils.py`, `M_VALUES_10K`, and `Approach A/B` not present in the plan text — these appear to be hallucinated findings from filesystem context bleed. Only findings grounded in the plan's stated design are retained above. This is flagged for awareness; it does not affect the verdict.

---

## Critical Findings (BLOCKING)

### C1 — MERFISH Data Absent in Fresh Worktree, No Automated Reconstitution
**Dimensions:** `agent_implementability` + `data_acquisition` (consolidated)
**Priority:** BLOCKING

The plan acknowledges that MERFISH `.npy` fixture files are not committed to git and will be absent in a fresh worktree. The only reconstitution path requires manually downloading the Allen Brain Cell Atlas MERFISH dataset (Yao et al. 2023). The plan states explicitly: *"The download is manual; no automated script exists."* No URL, no checksum, no verification criterion, and no phase placement in the ordered acquisition steps are provided for this manual step.

**Gap:** H1, H5, and H6 all depend on MERFISH data that cannot be acquired without human intervention in a fresh environment. The preflight acceptance criterion (`python3 scripts/preflight_check.py` exits 0) cannot be satisfied by an agent when symlink targets are absent and no automated reconstitution path exists.

**Risk:** Full experiment is unexecutable in a fresh worktree without undocumented manual setup. This is not a design flaw in the hypotheses or statistical methodology — it is an infrastructure gap that blocks execution.

---

## Warning Findings Summary (REQUIRED priority, consolidated by cluster)

### W1 — Circular Validation: m=2000 Confirmed on Its Fitting Data
**Dimensions:** `red_team` (data_leakage + overfitting_to_held_out_set + goodhart_exploitation) + `ecological_validity` + `measurement_alignment`
**Priority:** REQUIRED
**Requires decision:** Yes (red-team)

The parameter m=2000 was selected and the 0.01 accuracy threshold was calibrated in PR #260 using the same MERFISH dataset (Yao et al. 2023) used in this study. The plan acknowledges threshold leakage as an external validity threat but classifies it as "acceptable for a confirmatory design." This creates a circular validation structure: the confirmatory study tests a parameter on the exact data used to select it, with a threshold tuned to that data's known performance (mean|ΔT| = 0.00165 → 6x margin to threshold).

**Gap:** No held-out dataset, no held-out parameter value, no independently derived threshold. H1 passing is nearly guaranteed by construction and does not constitute independent confirmation.

**Decision point (red-team):** Is the circular nature of this validation acknowledged explicitly in the study's stated conclusions and scope claims? If yes, the finding is informational. If the study claims "independent Rust confirmation," the claim requires qualification.

---

### W2 — Multiple Testing Correction: Inadequate Justification and Unresolved Family
**Dimensions:** `statistical_corrections` + `error_budget`
**Priority:** REQUIRED

The FWER correction is omitted based on four claims: (1) confirmatory replication, (2) qualitatively distinct hypotheses, (3) H6 deterministic, (4) nominal FWER ~15% acceptable.

**Gaps:**
- The "confirmatory replication" claim is asserted without citing PR #260 or providing documented prior effect sizes; the confirmatory framing is unverifiable without this citation.
- H1 (n=10K) and H5 (n=50K) test the identical construct (mean|ΔT| < 0.01) with the same test and the same null hypothesis. They are not qualitatively distinct. The FWER contribution from this repeated testing is unaddressed.
- The ~15% FWER bound assumes independence across H1/H2/H3/H5; H1 and H5 are positively correlated (same dataset class) and H2/H3 both operate over the same m-sweep data. The bound is unreliable.
- No a priori FWER ceiling is declared. Stating that 15% is "acceptable" after observing the value is not a pre-specification.
- The family size is indeterminate at design time (H4 is conditional on external data).

---

### W3 — Power Analysis Present Only for H1
**Dimension:** `error_budget`
**Priority:** REQUIRED

A power justification exists for H1 (10 seeds, ~90% power at σ≈0.002, α=0.01). No power analysis is provided for H2 (OLS + bootstrap on 7 m-values), H3 (log-log OLS on 7 m-values), or H5 (same structure as H1).

**Gaps:**
- H2 uses bootstrap R² on n=7 points; the probability of detecting non-linearity (Type II error for R² threshold) is uncharacterized.
- H3 uses log-log OLS on n=7; power depends on residual variance of log(std(T)) across seeds, which is uncharacterized.
- The σ≈0.002 assumption for H1's power justification is asserted without empirical basis or pilot-study citation. If true σ is larger, the 90% claim does not hold.
- H5 is structurally identical to H1 but has no power statement.

---

### W4 — Hypothesis Specification Gaps (Compound Null, Regression Stratification, Partial Outcomes)
**Dimensions:** `estimand_clarity` + `hypothesis_falsifiability`
**Priority:** REQUIRED

Several hypotheses lack precise specification:

**H1 compound null:** The top-level H0 reads "mean|ΔT| >= 0.01 OR speedup does not scale linearly with n/m" — combining H1 and H2 into a single rejection region. This conflates an accuracy threshold test with a regression property. The composite null is narrower than the full set of sub-hypotheses (H3–H6 have no corresponding clause in it).

**H2 regression stratification:** The estimand for H2 is under-specified — the plan does not state whether the OLS fit pools both n=10K and n=50K together or fits per-stratum. The rejection region is also implicit: the PASS rule is stated but the FAIL condition is not written out, leaving falsifiability dependent on reader inference.

**H2 linearity operationalization:** R² > 0.90 does not distinguish linear from monotone non-linear relationships (e.g., power-law curves over a narrow range can achieve R² > 0.90). The hypothesis claims linearity but the test does not verify functional form.

**H3 partial outcome:** The dual decision rule (β ≤ -0.3 AND p < 0.05) has no stated verdict for partial outcomes (β ≤ -0.3 but p ≥ 0.05, or vice versa).

---

### W5 — Variance Protocol Gaps (Warmup, Core Count, Non-determinism Scope, Missing Decision Rule)
**Dimensions:** `variance_protocol` + `measurement_alignment`
**Priority:** REQUIRED

**Gaps:**
- **Warmup:** The plan states "5 timed reps, median reported" and mentions "warmup discard" in the thermal/cache threat mitigation, but the Execution Protocol commands do not include an explicit warmup invocation before the timed repetitions. It is unspecified whether rep #1 is discarded or included in the median.
- **Core count:** Rayon thread pool is `default (num_cpus)` but the number of cores is not recorded as a controlled variable and not pinned. Variation in core availability (shared machines, CPU frequency scaling) introduces uncontrolled timing variance.
- **Non-determinism scope:** The Rayon non-determinism check covers only 3 (m, seed) combinations run twice each. This is insufficient to bound floating-point summation variance across the full 148-trial design, particularly for large n and small m where work-stealing chunk sizes differ.
- **Missing decision rule:** The plan flags Rayon delta > 1e-6 as a warning trigger but does not specify the protocol response — whether trials are re-run, excluded, or merely noted. Requires decision.

**Requires decision (variance_protocol):** What is the protocol if max|T_run1 - T_run2| > 1e-6 across the Rayon non-determinism trials?

---

### W6 — Asymmetric Measurement Substrate (Code Duplication)
**Dimensions:** `baseline_fairness` + `measurement_alignment`
**Priority:** REQUIRED

The Rust binary copies the inner loop from `src/metrics.rs` rather than calling the library function directly. This means:
- Wall-clock timing for sub-sampled trustworthiness reflects a copied code path, not the production library function
- If the copy diverges from the library in any SIMD path, branch, or memory layout, speedup ratios measure a different function than what would be shipped
- The plan mitigates this with H6 (T_sub(m=n) = T_exact within 1e-10), which validates numerical identity but does not validate performance equivalence of the copied path

**Gap:** The measurement substrate for the key metric (speedup_ratio) is not the production code path. The plan acknowledges code duplication but does not state how code divergence in non-numerical characteristics (e.g., memory allocation patterns, SIMD dispatch) would be detected.

---

### W7 — Asymmetric Optimization Effort Between T_exact and T_sub
**Dimension:** `red_team` (asymmetric_tuning)
**Priority:** REQUIRED
**Requires decision:** Yes

The plan describes the sub-sampled path as using AVX2+FMA SIMD and Rayon work-stealing parallelism (explicitly optimized in the motivation section). The plan does not characterize whether the exact trustworthiness (`--mode exact`) invocation through the same binary uses an equivalent implementation path, or whether the exact computation is the unoptimized library baseline. If T_exact is slower due to implementation differences rather than algorithmic work differences, speedup ratios (H2) reflect implementation asymmetry rather than sub-sampling efficiency.

**Decision point:** Does the exact trustworthiness mode in `tw_subsample_experiment` use the same AVX2/Rayon optimization path as the sub-sampled mode, or a different path?

---

### W8 — Implementability Gaps (Symlink Dependencies, Binary Interface, Dry Run Coherence)
**Dimension:** `agent_implementability`
**Priority:** REQUIRED

Beyond the BLOCKING data acquisition gap, several implementability concerns remain:

- **Symlink targets unverifiable:** The Python reference symlink `python_ref/ -> ../../2026-04-09-subsampled-tw-tradeoff/results/raw/` points to a sibling experiment that may not have been run. The plan states the behavior when absent, but `analyze_results.py`'s handling of this case is not fully specified.
- **`cli` feature not confirmed:** The plan adds the binary with `required-features = ["cli"]` but does not confirm this feature exists in `Cargo.toml` or enumerate dependencies it gates.
- **Dry-run trial set vs H6 requirement:** The three dry-run trials are `(n=10K, m=2000, seed=0)`, `(n=10K, m=10000, seed=0)`, and `(n=10K, m=500, seed=0)`. The H6 sanity check requires `m=n`. For n=10K, m=10000 equals n=10000, which satisfies H6 — but the plan does not explicitly state this equivalence, creating interpretive ambiguity for an implementing agent.
- **Dry-run analysis on 3-trial subset:** The analysis script must produce `verdicts.json` with "all 6 hypothesis keys" from only 3 data points. The plan does not specify whether hypotheses without sufficient data should produce `null`, `INSUFFICIENT_DATA`, or partial verdicts.
- **Sanity check JSON output schema:** The acceptance criterion for `--mode sanity` requires `|T_sub - T_exact| < 1e-10`, but the expected JSON field names for the sanity output are not specified in the plan.

---

### W9 — Ecological Validity: Hardware and Dimensionality Scope
**Dimension:** `ecological_validity`
**Priority:** REQUIRED

- The speedup curve is measured on one hardware configuration (unstated core count, AVX2 x86). The m=2000 default recommendation will be shipped to users on machines with different core counts, potentially different SIMD support levels (ARM, AVX-512, scalar fallback), and different memory bandwidth characteristics. The speedup claims in H2 are hardware-contingent and not labeled as such in the success criteria shipping recommendation.
- The plan correctly scopes to "scRNA-seq-class high-dimensional data on AVX2 x86 hardware with k=15" but the introduction and success criteria do not repeat this scoping caveat.

---

## Adversarial Findings (Red-Team — all `requires_decision: true`)

All six universal + type-specific adversarial challenges were evaluated. Findings are summarized here; full descriptions appear in the warning cluster sections above.

| # | Challenge | Severity | Risk Summary |
|---|-----------|----------|--------------|
| RT1 | Goodhart exploitation | warning | Threshold and dataset jointly selected with m=2000; confirmation nearly guaranteed by construction |
| RT2 | Data leakage | warning | m=2000 validated on its selection dataset — no independent holdout |
| RT3 | Asymmetric tuning | warning | T_exact optimization level not characterized vs T_sub SIMD/Rayon path |
| RT4 | Survivorship bias | warning | No outlier exclusion policy for seeds; H5 failure pre-designated "inconclusive" |
| RT5 | Evaluation collision | warning | Same Rust process is both treatment executor and timer; first-trial vs subsequent-trial conditions may differ |
| RT6 | Overfitting to held-out set | warning | m=2000 confirmed on fitting data; no independent parameter validation |

> RT2 and RT6 describe the same underlying structural concern (circular validation) — grouped under W1 above. RT3 corresponds to W7.

---

## Cannot Assess

The following aspects could not be evaluated from the plan text alone:

1. **Test machine hardware specification** — The plan does not state the CPU model, core count, clock speed, cache sizes, or memory bandwidth of the machine where the experiment will run. The claimed ~90-120 minute runtime and speedup ratios cannot be independently assessed, and the ecological validity of hardware-class claims cannot be verified.

2. **`src/metrics.rs` implementation completeness and `cli` feature definition** — The plan references copying "the inner loop" from `src/metrics.rs` and uses `required-features = ["cli"]`, but the plan does not describe the module structure, public function signatures, or what the `cli` feature gates. Whether the implementing agent can successfully copy the correct loop without examining the source file is unverifiable from the plan alone.

3. **Reproducibility of prior Python reference data (H4)** — The plan conditionally depends on JSON files from `research/2026-04-09-subsampled-tw-tradeoff/results/raw/`. Whether these files were produced under comparable conditions (same n, k, m grid, hardware class) cannot be assessed from this plan.

4. **σ of |ΔT| across seeds** — The power justification for H1 assumes σ ≈ 0.002. This value is not derived from pilot data visible in the plan. If σ is substantially larger, the 90% power claim and the 10-seed sample size decision are both affected.

---

## Mechanizable Check Log

**Fixed checks:**

| Check | Status | Notes |
|-------|--------|-------|
| All implementation phases have runnable verification criteria | PARTIAL | Phases 1–4 have acceptance criteria. Phase 4 dry-run H6 criterion ambiguous (m=n equivalence not stated). Analysis acceptance criterion does not verify trial completeness. |
| All file paths in the implementation plan resolve to valid locations | UNKNOWN | Symlink targets reference sibling research directories whose existence cannot be verified from the plan. The `src/bin/tw_subsample_experiment.rs` path is valid. |

**Ad-hoc checks (contributed by subagents):**

| Check | Status | Notes |
|-------|--------|-------|
| Rayon non-determinism decision rule pre-specified | FAIL | No response protocol if max delta > 1e-6 |
| H2 FAIL condition explicitly stated | FAIL | PASS rule stated; FAIL condition implicit |
| H3 partial-outcome verdict defined | FAIL | Dual decision rule with no mixed-outcome verdict |
| MERFISH acquisition step automated | FAIL | Manual-only, no URL, no checksum |
| Power analysis present for all stochastic hypotheses | FAIL | Present only for H1 |
| `cli` feature existence confirmed | UNKNOWN | Not described in plan |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: configuration_study
critical_count: 1
warning_count: 35
blocking_count: 1
required_count: 20
advisory_count: 14
red_team_count: 6
active_dimensions: 13
warning_threshold: 65
```
