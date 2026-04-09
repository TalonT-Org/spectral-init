# Revision Guidance — Sub-sampled Trustworthiness Error/Speed Trade-off

**Round:** 2nd revision  
**Source dashboard:** `evaluation_dashboard_subsampled_trustworthiness_tradeoff_2026-04-09_151651.md`  
**Resolution:** revised  
**Stop triggers addressed:** 2 (STOP-1, STOP-2) — both ADDRESSABLE  

---

## Required Fixes

Both STOP findings share a single root: the composite "at least one primary cell passes" hypothesis structure tests two approaches with incommensurable estimands under a single H0/H1 pair, creating asymmetric falsifiability. The fix for both is the same mechanical change.

---

### FIX-1 — Decompose the Composite Hypothesis into Two Independent Hypotheses

**Addresses:** STOP-1 (HF-1) and STOP-2 (HF-2)

**What must change:**

Replace the single composite H0/H1 with two independent, per-approach hypothesis pairs:

**H1_A (Approach A — row sub-sampling, unbiased estimator of full-n T):**
- H0_A: mean(|ΔT_A|) ≥ 0.01 OR max(|ΔT_A|) ≥ 0.02 OR std(T_sub_A) ≥ 0.005 at m=2000 on MERFISH n=10K, k=15
- H1_A: mean(|ΔT_A|) < 0.01 AND max(|ΔT_A|) < 0.02 AND std(T_sub_A) < 0.005 at m=2000 on MERFISH n=10K, k=15

**H1_B (Approach B — subset embedding, estimator of subset-T):**
- H0_B: mean(|ΔT_B|) ≥ 0.01 OR max(|ΔT_B|) ≥ 0.02 OR std(T_sub_B) ≥ 0.005 at m=5000 on MERFISH n=10K, k=15
- H1_B: mean(|ΔT_B|) < 0.01 AND max(|ΔT_B|) < 0.02 AND std(T_sub_B) < 0.005 at m=5000 on MERFISH n=10K, k=15

Each hypothesis is evaluated independently. Each produces an independent pass/fail verdict.

**Replace the Analysis Plan decision tree** with a four-cell pre-specified outcome table:

| A verdict | B verdict | Outcome | Operational consequence |
|-----------|-----------|---------|------------------------|
| H1_A supported | H1_B supported | Both pass | Ship `trustworthiness_subsampled` with Approach A (preferred, unbiased estimator) as default; Approach B supported as an alternative for subset-T use cases |
| H1_A supported | H0_B not rejected | A passes only | Ship `trustworthiness_subsampled` with Approach A only; Approach B not recommended at m=5000 for accuracy parity |
| H0_A not rejected | H1_B supported | B passes only | Note: Approach B estimates a different quantity (subset-T, not full-n T); this outcome means neither approach estimates full-n T accurately at these m values; evaluate whether subset-T is an acceptable product metric before shipping |
| H0_A not rejected | H0_B not rejected | Both fail | Sub-sampling does not provide acceptable accuracy on MERFISH at the tested m values; larger m or alternative approaches required |

**Replace the Success Criteria section** to match the four-cell table above. Remove the "Conclusive positive (at least one)" rule. Remove the "partial support — the passing approach is recommended" branch.

**Why this is the correct fix:** The two approaches measure incommensurable quantities (full-n T vs. subset-T). A composite H1 that accepts if either passes conflates two different research questions. Independent hypotheses make each outcome independently falsifiable: H1_A is accepted or rejected by Approach A data alone; H1_B by Approach B data alone. The four-cell table pre-registers all combinations, eliminating post-hoc rationalization.

**What does NOT need to change:**
- The primary cell m values (m=2000 for A, m=5000 for B) — these remain as pre-specified
- The threshold values (mean < 0.01, max < 0.02, std < 0.005) — these remain unchanged
- The secondary hypotheses H2–H6 — these are exploratory and unaffected
- The data collection, analysis scripts, and execution protocol — no changes required

---

## Design Questions for Human Review

None. Both STOP findings have mechanical fixes. No DISCUSS findings were identified.

---

## Structural Findings (for context)

None. No STRUCTURAL findings were identified. Both STOP triggers are fully addressable by the hypothesis decomposition above.

---

## Context: Secondary Hypotheses (Not Stop Triggers)

The following warning-level findings from the dashboard are noted for awareness but do not block execution. They were not re-evaluated in this round (fail-fast gate halted L2+ analysis):

- **HF-3 (warning)**: H2 variance scaling has a rejection zone but no formal H0 structure. Acceptable for an exploratory hypothesis — ensure the analysis report labels it exploratory.
- **HF-4 (warning)**: H3 "MERFISH differs from Gaussian" has no quantitative criterion. Label as exploratory/descriptive in the analysis report.
- **HF-5 (warning)**: H5 "qualitatively holds at n=50K" uses undefined "qualitatively." Define a concrete criterion (e.g., "crossover m/n ratio within 2×") or label as purely descriptive.
- **HF-6 (warning)**: H6 extrapolation cannot be empirically falsified. Already downgraded to exploratory (R6); ensure the analysis report clearly marks it as an out-of-distribution projection.
- **HF-7 (info)**: Inconclusive range [0.008, 0.012] overlaps with thresholds. Consider whether the inconclusive category applies per-hypothesis (H1_A inconclusive, H1_B conclusive) or to the overall experiment.
- **HF-8 (info)**: H4 matched-m comparison has no pre-specified matched m point. Specify the comparison m value (e.g., m=2000 or m=5000) before analysis.

These will be evaluated in the next design review pass once the STOP findings are resolved.
