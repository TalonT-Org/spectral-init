# Evaluation Dashboard — Sub-sampled Trustworthiness Error/Speed Trade-off

## Verdict: STOP

**Classification:** `configuration_study`
**Fail-fast gate:** TRIGGERED — critical findings in `hypothesis_falsifiability` halted analysis at Level 1.
**Analysis depth:** Level 1 only (L2, L3, L4, red-team NOT run).

> STOP is issued because the hypothesis structure contains fundamental falsifiability defects that must be resolved before the experiment produces interpretable results. Proceeding would risk an experiment whose primary outcome cannot be unambiguously evaluated regardless of the data collected.

---

## Classification Summary

| Field | Value | Source |
|---|---|---|
| Experiment type | configuration_study | triage subagent |
| Primary IV | Subsample size m (9 values swept) | extracted |
| Secondary IVs | Approach (A/B), Dataset, Seed | extracted |
| Hypothesis H0 | mean(|ΔT|) ≥ 0.01 OR max(|ΔT|) ≥ 0.02 at primary m values | extracted |
| Hypothesis H1 | mean(|ΔT|) < 0.01 AND max(|ΔT|) < 0.02 AND std < 0.005 at primary m | extracted |
| Estimand | Sub-sampling accuracy on MERFISH n=10K with k=15 | extracted |
| Success criteria | Conclusive positive, negative, inconclusive | extracted |

**Secondary modifiers active:**
- `+deployment` — motivation references production crate (shipping `trustworthiness_subsampled`) → `ecological_validity` floor raised to M
- `+multi_metric` — 9 DVs defined → `statistical_corrections` weight +1 tier from H (stays H, already maximum)

**Effective dimension weights:**

| Dimension | Weight |
|---|---|
| causal_structure | S (silent) |
| variance_protocol | H |
| statistical_corrections | H |
| ecological_validity | M |
| measurement_alignment | M |
| resource_proportionality | L |
| data_acquisition | M |

---

## Dimension Scorecard

Only Level 1 dimensions were evaluated (fail-fast gate triggered).

| Dimension | Weight | Level | Findings | Severity Summary |
|---|---|---|---|---|
| estimand_clarity | H | 1 | 4 | 2 warning, 2 info |
| hypothesis_falsifiability | H | 1 | 8 | **2 critical (STOP triggers)**, 4 warning, 2 info |
| variance_protocol | H | — | NOT RUN | L1 gate triggered |
| statistical_corrections | H | — | NOT RUN | L1 gate triggered |
| ecological_validity | M | — | NOT RUN | L1 gate triggered |
| measurement_alignment | M | — | NOT RUN | L1 gate triggered |
| resource_proportionality | L | — | NOT RUN | L1 gate triggered |
| data_acquisition | M | — | NOT RUN | L1 gate triggered |

---

## STOP Triggers (Critical Findings)

### STOP-1 — Hypothesis Unfalsifiable in Practice

**Dimension:** `hypothesis_falsifiability`
**Section:** `## Success Criteria`
**Severity:** critical

The "at least one primary cell passes" rule for H1 acceptance makes H0 unfalsifiable in practice: if Approach A passes and Approach B fails, H1 is declared supported — yet the plan also states "the passing approach is recommended" as a partial-support outcome. H0 requires BOTH primary cells to fail ALL criteria simultaneously. Because the two approaches use different estimands (full-set trustworthiness vs. subset trustworthiness), Approach A and Approach B are not interchangeable, and a result where only one passes does not cleanly reject H0 for the failing approach. The asymmetry between the disjunctive H1 acceptance rule and the conjunctive H0 non-rejection rule means there is no single outcome that unambiguously rejects H1 while simultaneously validating sub-sampling as a concept.

---

### STOP-2 — "Partial Support" Branch Has No Pre-Specified Decision Rule

**Dimension:** `hypothesis_falsifiability`
**Section:** `## Analysis Plan`
**Severity:** critical

The "one passes and one fails" outcome is labeled "partial support — the passing approach is recommended," but no pre-specified decision rule governs which approach is recommended or what the operational consequence is. This creates a post-hoc rationalization opportunity: any single-arm pass becomes a positive finding regardless of which arm fails. Without a pre-registered preference ordering between Approach A and Approach B, the partial-support branch cannot be distinguished from a favorable reinterpretation of a mixed negative result.

---

## Level 1 Findings — All Dimensions

### estimand_clarity

**Finding EC-1** | severity: warning | section: `## Hypothesis`

H0/H1 bundles two distinct estimands (Approach A's estimate of full-n trustworthiness and Approach B's trustworthiness of the subset embedding) into a single null/alternative pair with different m values per arm. The hypothesis cannot be written as a single formal contrast (A vs B on Y in Z) because the outcome Y is not the same quantity for both arms. This creates ambiguity about what the hypothesis is actually testing: a claim about sub-sampling accuracy in general, or two separate claims about two incommensurable quantities.

**Finding EC-2** | severity: warning | section: `## Analysis Plan`

The "at least one" success criterion conflates two structurally different estimand questions into a single composite endpoint. Because the two approaches target different estimands (full-n T estimate vs. subset-T estimate), a pass by either arm does not constitute evidence for a unified claim. The composite success criterion is not derivable from a single formal contrast, and it is unclear which research question is actually being answered when only one arm passes.

**Finding EC-3** | severity: info | section: `## Hypothesis`

The population is specified in the hypothesis body (MERFISH n=10K, k=15) but not embedded in the formal statement of H0 and H1 themselves. The estimand's population condition is implicit rather than stated, which makes the formal contrast incomplete as written.

**Finding EC-4** | severity: info | section: `## Hypothesis`

H1 asserts that sub-sampling "provides sufficient accuracy for embedding quality monitoring on MERFISH-like data at n ≤ 50K," but the confirmatory design tests only n=10K. The scope of the estimand in the conclusion (n ≤ 50K) does not match the scope of the confirmatory population (n=10K), creating a gap between the estimand as tested and the estimand as claimed.

---

### hypothesis_falsifiability

**Finding HF-1** | severity: critical | **STOP trigger** | section: `## Success Criteria`

(See STOP-1 above.)

**Finding HF-2** | severity: critical | **STOP trigger** | section: `## Analysis Plan`

(See STOP-2 above.)

**Finding HF-3** | severity: warning | section: `## Hypothesis`

H2 (variance scaling) specifies an acceptable log-log slope range of [-0.7, -0.3], but no formal H0 is stated for H2. The plan states only that "if slope ∉ [-0.7, -0.3], variance scaling model is rejected," without specifying what conclusion is drawn about sub-sampling viability or how a slope outside this range interacts with the primary decision. H2 is a comparison goal with a named rejection zone but no formal null hypothesis structure.

**Finding HF-4** | severity: warning | section: `## Hypothesis`

H3 ("MERFISH error profile differs from synthetic Gaussian") has no specified metric, threshold, or decision criterion — any observed difference or similarity between datasets can be described as consistent with H3. Without a falsifiable criterion for what "differs" means quantitatively, H3 cannot be rejected by any experimental outcome.

**Finding HF-5** | severity: warning | section: `## Hypothesis`

H5 ("accuracy relationship at n=10K holds qualitatively at n=50K") uses the term "qualitatively" without defining it. No threshold or operational criterion is provided for what constitutes failure of H5. Any n=50K result can be interpreted as qualitatively consistent or inconsistent with the n=10K result post-hoc.

**Finding HF-6** | severity: warning | section: `## Hypothesis`

H6 ("error/speed curves can be extrapolated to n=100K with stated uncertainty") does not define what "stated uncertainty" means, what bound on extrapolation error constitutes failure, or how extrapolation quality would be assessed empirically. Because n=100K data is not collected, H6 cannot be empirically falsified within the experiment.

**Finding HF-7** | severity: info | section: `## Success Criteria`

The "inconclusive" category is defined by a near-threshold range for mean(|ΔT|) ∈ [0.008, 0.012] or CLT slope outside [-0.7, -0.3]. The near-threshold range [0.008, 0.012] overlaps with both the H0 threshold (≥ 0.01) and the H1 threshold (< 0.01), meaning a mean(|ΔT|) of exactly 0.009 could be classified as either H1-supporting or inconclusive depending on the analyst's interpretation. The boundary between conclusive positive and inconclusive is not fully pre-specified.

**Finding HF-8** | severity: info | section: `## Hypothesis`

H4 ("Approach A achieves lower |ΔT| than Approach B at matched m") compares approaches at matched m values, but the primary cells use different m values (m=2000 for A, m=5000 for B). No pre-specified matched-m comparison point is identified for the primary analysis, leaving the scope of H4 testing ambiguous.

---

## Adversarial Findings

**Red-team analysis was NOT run** (fail-fast gate triggered before Level 2). No adversarial findings.

---

## Cannot Assess

The following dimensions could not be evaluated because the fail-fast gate halted analysis at Level 1. These are noted for completeness and should be reviewed once the STOP findings are resolved:

1. **Variance protocol** (H-weight): The plan includes a variance protocol section documenting fixed/varied components and 10-seed replication. Whether seeds are fixed appropriately and run-to-run variance is adequately addressed cannot be assessed without L3 analysis.

2. **Statistical corrections** (H-weight): The plan uses threshold comparison rather than inferential statistics. Whether multiple comparisons (9 primary metrics across 2 arms) are addressed, and whether the lack of formal p-values is appropriate for the experiment type, cannot be assessed without L3 analysis.

3. **Ecological validity** (M-weight, elevated via +deployment): The plan targets a production shipping decision. Whether the MERFISH n=10K test conditions match the intended deployment context (sub-sampling in the production `spectral-init` crate) cannot be assessed without L4 analysis.

4. **Measurement alignment** (M-weight): Whether the metrics (mean(|ΔT|), max(|ΔT|), std) actually measure what the research question claims cannot be assessed without L4 analysis.

5. **Data acquisition** (M-weight): The plan includes a data manifest with SHA-256 checksums and symlink steps. Whether the data acquisition strategy is complete and acquisition steps exist for all gitignored fixtures cannot be assessed without L4 analysis.

6. **Benchmark representativeness / reproducibility**: Not evaluated.

7. **Red-team adversarial challenges**: Not run. Known candidate risks not assessed: asymmetric tuning (Approach A has custom implementation while B uses sklearn), Goodhart exploitation (passing primary cell by cherry-picking seeds or m values in exploratory zone), and HARKing vulnerability in secondary exploratory hypotheses.

---

## Mechanizable Check Log

| Check | Result | Notes |
|---|---|---|
| Primary cells explicitly pre-selected (bold m values in IV table) | PASS | m=2000 (A), m=5000 (B) marked bold |
| H0/H1 explicitly labeled in plan | PASS | Both present with threshold values |
| Threshold provenance cited | PASS | W4 cites scope report §Metric Context §2.2 |
| Inconclusive range defined | PASS | [0.008, 0.012] stated |
| Data manifest with checksums present | PASS | SHA-256 and shapes in Data Manifest table |
| Symlink step in Execution Protocol | PASS | Step 2 added (R5) |
| m=n convergence check in dry-run checklist | PASS | Dry-run checklist includes this check |
| Seed range explicitly specified | PASS | Seeds 0–9 documented |
| Approach asymmetry in implementation documented | PASS | RT9 disposition recorded |
| "at least one" success rule formalized | FLAG | Creates falsifiability defect (STOP-1) |
| Partial-support branch pre-specified | FLAG | No decision rule for single-arm pass (STOP-2) |
| Secondary hypotheses H2–H6 have formal H0 | FLAG | Only H2 has a rejection zone; H3, H5 unfalsifiable |

---

## Summary

The plan is a well-documented second-revision effort that has resolved many prior design review findings. The variance protocol, data manifest, threshold provenance, and implementation asymmetry documentation represent meaningful improvements. The STOP verdict is issued on specific structural issues in the hypothesis/success criteria logic — not on the overall experimental setup quality.

The two STOP findings share a common root: the plan tests two approaches with different estimands under a single composite hypothesis structure, creating asymmetric falsifiability. This is a design issue that can be resolved by either (a) separating the approaches into two independent hypotheses with independent decision rules, or (b) defining a pre-registered priority ordering for the single-arm pass case.

---

```yaml
# --- review-design machine summary ---
verdict: STOP
experiment_type: configuration_study
critical_count: 2
warning_count: 6
red_team_count: 0
active_dimensions: 2
warning_threshold: 10
```
