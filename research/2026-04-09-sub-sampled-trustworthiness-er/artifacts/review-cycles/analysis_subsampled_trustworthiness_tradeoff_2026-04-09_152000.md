# Triage Analysis — Sub-sampled Trustworthiness Error/Speed Trade-off

**Triage complete (BEFORE any guidance written)**
ADDRESSABLE: 2 | STRUCTURAL: 0 | DISCUSS: 0

---

## Input Files

- Dashboard: `evaluation_dashboard_subsampled_trustworthiness_tradeoff_2026-04-09_151651.md`
- Plan: `experiment_plan_subsampled_trustworthiness_tradeoff_2026-04-09_123000.md`
- Prior guidance: `revision_guidance_subsampled_trustworthiness_tradeoff_2026-04-09_121500.md`

---

## Stop Triggers Parsed

| ID | Dimension | Severity | Classification |
|----|-----------|----------|----------------|
| STOP-1 (HF-1) | hypothesis_falsifiability | critical | ADDRESSABLE |
| STOP-2 (HF-2) | hypothesis_falsifiability | critical | ADDRESSABLE |

---

## Feasibility Validation Results (Parallel Subagents)

### STOP-1 — Hypothesis Unfalsifiable in Practice

**Verdict:** ADDRESSABLE

**Evidence:** The composite H1 acceptance rule ("At least one pre-selected primary cell satisfies ALL of...") combined with the conjunctive H0 non-rejection rule ("Both pre-selected primary cells fail at least one criterion") creates an asymmetry. The "partial support" branch in the Analysis Plan ("If one passes and one fails: partial support — the passing approach is recommended") has no corresponding entry in the Success Criteria section, leaving the partial-support outcome structurally undefined relative to H0/H1. The different estimands (full-set trustworthiness for Approach A vs. subset trustworthiness for Approach B) mean the two arms are not testing the same quantity.

**Fix sketch:** Decompose into two independent hypothesis pairs: H1_A (Approach A row sub-sampling achieves mean(|ΔT|) < 0.01, max(|ΔT|) < 0.02, std(T_sub) < 0.005 at m=2000) and H1_B (Approach B subset embedding achieves the same thresholds at m=5000), each with its own null. Remove the composite "at least one" rule entirely. Define four exhaustive, mutually exclusive outcome cells: (A-pass, B-pass), (A-pass, B-fail), (A-fail, B-pass), (A-fail, B-fail), with pre-registered implementation recommendations for each cell. This eliminates the disjunctive/conjunctive asymmetry and makes every outcome independently falsifiable.

---

### STOP-2 — "Partial Support" Branch Has No Pre-Specified Decision Rule

**Verdict:** ADDRESSABLE

**Evidence:** "If one passes and one fails: partial support — the passing approach is recommended." No preference ordering between Approach A and Approach B is stated; either arm passing triggers the same "recommended" label despite the two approaches estimating different quantities (full-n trustworthiness vs. subset embedding trustworthiness).

**Fix sketch:** Separate the composite hypothesis into two independent hypotheses (H1_A and H1_B), each with its own null, alternative, and decision rule. Remove the "partial support" branch entirely — each arm reports pass/fail independently. Optionally, pre-register a priority ordering (e.g., Approach A is primary because it estimates the full-n quantity; Approach B is secondary/exploratory) so that if only B passes, the operational consequence is explicitly specified.

---

## Goalposts-Moving Detection (Step 1.5)

Prior revision guidance was provided (`revision_guidance_subsampled_trustworthiness_tradeoff_2026-04-09_121500.md`). Each ADDRESSABLE finding was checked against prior guidance themes.

| Finding | Goalposts-Moving | Analysis |
|---------|-----------------|---------|
| STOP-1 | No | Genuinely new issue. The composite "at least one" hypothesis structure emerged as a consequence of correctly resolving prior RT1 (Approach B now has a different estimand) and R2/RT8 (primary cells pre-selected). The prior guidance did not address composite hypothesis falsifiability — it addressed multiple-comparisons across 40 cells (R2) and pre-selection of primary cells (RT8), which are distinct concerns. The falsifiability defect is a new structural consequence of the revision, not a theme escalation. |
| STOP-2 | No | Genuinely new issue. The partial-support branch ("passing approach is recommended") was not present in the prior plan. It appeared in the revised plan as a consequence of the two-arm composite structure. No prior guidance entry addresses pre-specification of single-arm-pass decision rules. |

**No reclassifications due to goalposts-moving.**

---

## Resolution

```
resolution = revised
```

At least one finding (both findings) are ADDRESSABLE → revision path.

---

## Root Cause Summary

Both STOP findings share a single root: the plan evaluates two approaches with incommensurable estimands under a single composite hypothesis, creating asymmetric falsifiability. This root emerged from the correct resolution of prior findings RT1/W3 (distinguishing Approach B's estimand) combined with R2/RT8 (pre-selecting primary cells). The fix for both findings is the same mechanical change: decompose the composite hypothesis into two independent per-approach hypotheses, each with its own H0/H1 and decision rule.
