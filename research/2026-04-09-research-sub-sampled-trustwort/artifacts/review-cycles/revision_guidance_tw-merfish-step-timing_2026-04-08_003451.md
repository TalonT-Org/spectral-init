# Revision Guidance: Trustworthiness Step-Timing Validation on MERFISH Real Data

**Verdict:** REVISE
**Experiment type:** exploratory
**Date:** 2026-04-08

---

## Required Revisions (Critical Findings)

### R1: Scope generalization language to match single-dataset design
**Dimension:** benchmark_representativeness | **Severity:** critical

The Motivation section states this experiment validates whether the PR #238 conclusion "holds on real biological data." This generalization language is broader than what a single MERFISH dataset from one tissue type, one assay, and one PCA reduction can support. The generalization boundary section (correctly limiting claims) contradicts the broader framing in the Motivation.

**Gap:** The plan's motivating claim and its actual design reach are misaligned. Readers following the Motivation framing may interpret results as evidence about biological data generally, while the design only supports claims about this specific MERFISH dataset.

**Risk:** Overclaiming in the final report. If the Motivation framing is carried into the report's conclusions, the single-dataset limitation becomes a credibility vulnerability when peer-reviewed.

### R2: Document MERFISH fixture distribution mechanism
**Dimension:** reproducibility_spec | **Severity:** critical

The MERFISH fixture files are accessed via relative symlinks but the plan does not state whether these files are committed to git, stored in git-lfs, or exist only on the experimenter's local disk. An independent party cannot assess whether they can obtain the data.

**Gap:** The reproducibility chain is broken at the data access point. Environment, toolchain, and analysis scripts are documented, but the primary input data's availability is unspecified.

**Risk:** The experiment may be non-reproducible by design if the .npy files are local-only. If they are git-tracked (as the data_acquisition agent confirmed on the current machine), stating this explicitly costs nothing and closes the gap.

---

## Recommended Revisions (Warning Findings)

### Baseline Design Asymmetries (W1, W2, W3, W4, W22)

The Gaussian baseline differs from MERFISH in d_x (10 vs ~48), scale coverage (n=10K only vs n=10K+50K), SIMD code paths, and interpretive threshold anchoring. These are acknowledged as confounds but treated as acceptable for exploratory work. Recommend:

- Acknowledging in the Analysis Plan that the primary comparison conflates d_x and geometry effects, and that a matched-d_x Gaussian control would be needed to isolate either
- Stating explicitly that the success criteria thresholds are post-hoc anchored to the Gaussian baseline and should not be treated as pre-registered decision boundaries in the report

### Unit Interference Controls (W5, W6, W7)

Sequential execution ordering, absent cache flushing, and thermal accumulation create systematic directional biases. For an exploratory experiment, these are acceptable if acknowledged. Recommend:

- Documenting the planned execution order (Gaussian first → MERFISH 10K → MERFISH 50K) as a controlled variable with acknowledged bias direction
- Noting that replicate 1 of each dataset has different I/O characteristics than replicates 2-3 due to page cache warming

### Measurement Proxy Limitations (W8, W9, W10, W11)

The thread-ns metric systematically overweights high-SIMD-throughput steps. The magnitude and direction of this bias are dataset-dependent and unquantified. Recommend:

- Adding a brief theoretical bound on the proxy gap (e.g., stating the ratio of SIMD utilization between x_dist and y_dist kernels at representative d_x values)
- Noting in the Success Criteria that "X-space dominance confirmed" is a statement about compute-share, with an explicit caveat about the proxy-to-wall-clock gap

### Reproducibility Gaps (W12, W13, W14, W15)

MERFISH fixture provenance is absent, the Gaussian generating script is not referenced, no environment pinning exists, and RAYON_NUM_THREADS is not explicitly set. Recommend:

- Adding a brief provenance note for MERFISH fixtures (referencing the prepare_merfish.py script in the source research directory)
- Referencing the Gaussian gen_synthetic.py script
- Setting RAYON_NUM_THREADS=16 explicitly as a controlled variable

### Statistical Awareness (W24, W25, W27)

Post-observation threshold selection, absent DV priority ordering, and unjustified replicate count reduce interpretive rigor. Recommend:

- Declaring x_space_pct as the primary DV and the four individual step fractions as secondary DVs with no formal verdict weight
- Stating the expected CI half-width at R=3, df=2 for a representative between-run std (e.g., "if between-run std = 3pp, 95% CI half-width = 3 × 4.30 / sqrt(3) = 7.4pp")
- Adding a sentence noting that the 8-DV × 2-comparison inspection burden means individual CI non-overlaps should be interpreted jointly, not as 8 independent findings

---

## Red-Team Findings (Decision Points)

All red-team findings are capped at `info` severity for this exploratory experiment type. Each is presented as a decision point for the plan author.

### RT-1: Goodhart exploitation
Success criteria are structural (files exist, table produced) rather than epistemic. Consider adding a criterion requiring the uncertainty bounds to be informative (e.g., "95% CI half-width for x_space_pct < 15pp on at least one configuration").

### RT-2: Data leakage via fixture reuse
The "fresh" Gaussian baseline uses the same seed, d_x, and n as PR #238. This is by design (same-conditions re-run), not a flaw, but the interpretive framework is anchored to a known result. The plan acknowledges this. No revision needed if the report maintains the same transparency.

### RT-3: Asymmetric tuning of warmup/iters
warmup=2, iters=5 were not validated against MERFISH variance. If the dry run (Phase 2) shows high within-run CV for MERFISH, consider increasing iters for all configurations symmetrically before Phase 4.

### RT-4: Survivorship bias via inconclusive handling
No maximum replicate count is specified. Consider pre-specifying: "If the initial R=3 result is inconclusive, R will not be extended in this experiment. A follow-up experiment with R=10 would be designed separately."

### RT-5: Evaluation collision
The profiler is both subject and instrument. This is inherent to the measurement approach and cannot be eliminated without external instrumentation (e.g., hardware performance counters). Accept as a known limitation.

### RT-6: HARKing vulnerability
The directional expected outcome and post-hoc guideposts create HARKing risk. The plan's explicit acknowledgment of post-observation threshold selection is the primary mitigation. Ensure the report carries forward this transparency without softening it.
