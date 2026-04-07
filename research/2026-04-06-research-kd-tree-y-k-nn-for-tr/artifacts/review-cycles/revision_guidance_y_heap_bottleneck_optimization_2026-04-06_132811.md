# Revision Guidance: y_heap Bottleneck Optimization

**Verdict:** REVISE
**Experiment type:** benchmark
**Review date:** 2026-04-06

This document describes what the experimental design is lacking or at risk. It does not prescribe implementation steps — the plan author determines how to address each gap.

---

## Required Revisions (Critical Findings)

### 1. Baseline Description and Construction Specification

**Gap:** The plan describes "baseline (current BinaryHeap)" as if the existing `trustworthiness()` function is a pure-BinaryHeap implementation everywhere. The actual codebase already employs thread-local flat buffers and introselect for the X-space KNN step; only the Y-space step retains a BinaryHeap. The baseline's actual relationship to H1 and H2 is therefore ambiguous.

**Risk:** If the baseline already contains part of what H1 claims to test, the speedup ratio will be artificially compressed — the measured improvement is only over the Y-space portion, not a full before/after comparison of the proposed optimization. The experimental conclusion cannot be generalized beyond the specific sub-step that was actually changed.

**Gap:** The plan does not specify whether the "baseline" variant is the current production function or a purpose-built regression stub. This leaves the baseline's compilation flags, Rayon parallelism configuration, and code quality relative to the variants undefined, making baseline fairness unverifiable.

**Risk:** An unspecified baseline construction creates systematic asymmetry that cannot be detected post-hoc.

---

### 2. Measurement Incommensurability: Step Fraction Metric

**Gap:** The `y_heap_step_fraction` metric is defined as the output of an `AtomicU64` accumulator that sums nanosecond durations across all parallel Rayon threads. This measures summed CPU time, not wall-clock time. In a parallel execution context, the sum of per-thread times will substantially exceed wall-clock elapsed time (by up to the thread-count factor). The plan then uses this fraction to validate the H1_alt condition (≤ 40%) alongside a wall-time-based speedup ratio — these two quantities are on incompatible time bases.

**Risk:** The step-fraction component of H1_alt is either unmeasurable as stated, or will systematically overstate y_heap's fraction of total time (summed CPU time denominator inflates numerator relative to total). The fraction metric as defined cannot be used to validate a ≤ 40% reduction target without establishing the correct denominator.

---

### 3. H1_alt Compound Criterion Ambiguity

**Gap:** H1_alt requires both "≥ 1.5× wall-time speedup" AND "y_heap fraction ≤ 40%" but the plan does not specify how these conditions combine. Is it a conjunction (both must be satisfied)? A disjunction (either is sufficient)? An ordered gate (fraction only evaluated if speedup passes first)?

**Risk:** Without a formal definition of the compound criterion, the experiment cannot produce a determinate verdict. The author and reader may reach different conclusions from the same data depending on implicit assumptions about the joint condition.

---

### 4. Step Fraction as Causal Attribution

**Gap:** The plan states that a reduction in y_heap fraction "validates that the observed speedup comes from y_heap reduction." A reduction in y_heap fraction is consistent with two scenarios: (a) the y_heap step ran faster; or (b) other steps ran slower. The plan has no mechanism to distinguish these cases.

**Risk:** If H2's SIMD kernel introduces overhead elsewhere (e.g., via cache eviction of X-space data), total wall-time speedup could be observed alongside an artifactual y_heap fraction reduction that does not represent genuine y_heap improvement. The attribution claim in the analysis plan would be false positive.

---

### 5. Power Analysis Absent

**Gap:** The 10-sample Criterion budget is stated as a protocol parameter but is never connected to the effect size being tested (1.5× speedup, with inconclusive zone down to 1.1×), the expected within-variant measurement noise, or the resulting Type II error rate.

**Risk:** The experiment may be systematically underpowered for its own decision boundaries. The probability of landing in the inconclusive zone (1.1×–1.5× with CI overlapping 1.0×) under the true data-generating process is unknown. Consuming 15–20 minutes of Criterion runtime without a decision-capable result is a plausible outcome that is not accounted for.

---

### 6. Multiple Comparisons Structure Absent

**Gap:** The plan contains a minimum of 9 primary comparisons (3 variant pairs × 3 n values) across 4 DVs, plus attribution and scaling sub-analyses. No correction procedure (Bonferroni, Holm, Benjamini-Hochberg, or family-wise error rate acknowledgment) is pre-specified for any of these.

**Risk:** With uncorrected multiple comparisons, the probability of at least one spurious significant result (CI LB > 1.0) grows substantially above 5%. The plan does not designate a single primary DV to anchor a correction hierarchy, leaving all comparisons at nominally equal but inflated alpha.

---

### 7. Type I Error Rate and CI Interpretation

**Gap:** The significance criterion "CI lower bound strictly > 1.0" applies a 95% CI threshold, but the nominal alpha level (0.05) is not declared, and no acknowledgment is made that this criterion is being evaluated across multiple (variant × n) combinations simultaneously.

**Risk:** The effective Type I error rate is elevated above 0.05 in proportion to the number of simultaneous CI evaluations. The plan's stated criterion is a necessary but not sufficient condition for significance under the analysis structure actually used.

---

### 8. Rayon Thread Count Uncontrolled

**Gap:** Rayon thread count is "System default" with no fixation, documentation, or recording mechanism. The thread count directly affects parallel scheduling jitter, allocation contention, and per-thread cache behavior — all of which differ between H1 (thread-local reuse) and the baseline (per-row allocation).

**Risk:** Two runs on machines with different core counts will produce structurally incomparable speedup ratios. The experiment cannot be reproduced without knowing the thread count at execution time, and the measured speedup is conditional on an undocumented variable.

---

### 9. Reproducibility Environment Under-Specified

**Gap (Rust toolchain):** No Rust toolchain version is pinned or documented. The project uses a nightly toolchain; codegen and optimization differ across nightly dates. Prior experiments in this repository explicitly document and verify the toolchain version.

**Risk:** A reproducer on a different date's nightly obtains a different binary with potentially different performance characteristics, making benchmark comparisons invalid across time.

**Gap (Cargo.lock):** `Cargo.lock` is not tracked in git. Semver-compatible dependency updates can change codegen.

**Gap (Python environment):** No numpy version is specified for `gen_data.py`. Numpy's random output layout and dtype defaults have changed across major versions.

**Gap (Hardware documentation):** `target-cpu=native` expands to a machine-specific ISA that is not documented in the plan. The project's CI history records a prior SIGILL failure from this ambiguity (commit 3c08f61).

**Risk (combined):** Four independent reproducibility gaps in the environment specification means an independent reproducer cannot reconstruct the experimental conditions. Any performance delta observed is conditional on undocumented software and hardware state.

---

### 10. Data Acquisition Dependency Not Formally Declared

**Gap:** The `data/` directory contains only `.gitkeep` in the repository (consistent with all experiments in this repo). The `.npy` files required by `run_profiler.sh` are absent in a fresh worktree. The execution protocol lists `gen_data.py` as Step 1, providing implicit coverage, but the file dependency is not formally declared and no verification criterion is specified for the generated files.

**Risk:** If `run_profiler.sh` is run before `gen_data.py`, or if `gen_data.py` produces malformed output, the profiler step will silently fail or produce garbage results. Without a verification step, this failure mode is undetected until analysis.

---

## Recommended Revisions (Warning Findings)

### Estimand Clarity

- The plan contains multiple DVs but does not formally designate one as the primary estimand for H0 rejection. The analysis section implies speedup ratio at n=10K is primary, but this should be declared explicitly to anchor the correction hierarchy and prevent post-hoc primary DV selection.
- The H1-alone sub-contrast (baseline vs h1_introselect) participates in the attribution analysis but has no declared estimand, population, or outcome specification. Its findings will be descriptive rather than inferentially bounded; this scope limitation should be stated.

### Hypothesis Falsifiability

- The 1.1×–1.5× inconclusive zone is not mapped to H0 acceptance or rejection. A result in this zone is neither H0 nor H1_alt; the plan should specify what conclusion is drawn and what action follows (e.g., increase sample_size, escalate to H3).
- The correctness gates (parity < 1e-6, delta < 1e-12) are not integrated into the H0/H1_alt falsification logic. A variant that passes speed threshold but fails correctness has no defined hypothesis outcome.

### Baseline Fairness

- The plan states W4 mitigation requires "separate Criterion invocations with cold starts," but the execution protocol shows a single `cargo bench --bench y_heap_variants_bench` invocation. The contradiction between mitigation strategy and execution command should be resolved.
- The H2 variant is described as "h1 + 2D AVX2 batch kernel," meaning H2 includes H1's optimization. The attribution analysis assumes H2 − H1 isolates SIMD contribution. If H2 is not tested in isolation (without H1's flat buffer), this decomposition is unverifiable.
- The AtomicU64 step timing instrumentation placement should be described as equivalent across all three variants to ensure the overhead is symmetric.

### Unit Interference

- The plan should specify whether (variant × n) combinations are run in a fixed order or randomized within a Criterion invocation. Fixed order means the warm state from earlier runs systematically benefits later runs within the same variant.
- The plan should specify whether the AtomicU64 accumulator is reset between Criterion iterations (not just between benchmark groups).

### Error Budget

- The inconclusive zone probability under the true data-generating process should be acknowledged as unknown. The plan should describe what action is taken if the experiment lands in this zone (e.g., increase sample_size to 30–100 and re-run).
- The attribution analysis and scaling analysis each imply inferential claims but carry no stated significance criterion. Their exploratory vs. confirmatory status should be declared.
- Criterion's CI is constructed over wall-time measurements, not over the derived speedup ratio (a ratio of two random variables). The ratio CI's nominal coverage may differ from 95%.

### Statistical Corrections

- The primary DV (speedup ratio at n=10K) should be formally declared as the sole confirmatory DV for correction-hierarchy purposes, with remaining DVs designated exploratory.
- The step fraction metric's role as a confirmatory gate (it appears in H1_alt) should be explicitly assigned an alpha level and declared within the correction family.

### Variance Protocol

- `measurement_time` should be specified in the benchmark configuration to bound the CI collection window and enable prediction of total benchmark runtime.
- The seed scope (data generation vs solver initialization) should be clarified. If `SmallRng` is used for data and a different RNG is used for profiler-side data, both should be documented.

### Benchmark Representativeness / Ecological Validity

- The plan correctly scopes results to k=15, n≤10K, and uniform Y, but does not acknowledge d_y>2 as an additional generalizability gap. The AVX2 kernel is specialized for d_y=2; behavior at d_y=3+ is undocumented.
- The n≤10K scope limitation is most consequential for the H1+H2 vs H3 decision. The plan recommends escalating to H3 (KD-tree) if results are inconclusive; the scale at which H3 becomes beneficial should be stated as an open question.

### Measurement Alignment

- The relationship between y_heap step fraction and total wall-time speedup should be formally quantified: given that x_dist already represents 13% of CPU time (from PR #229 profiling), an upper bound on achievable total wall-time speedup from y_heap optimization alone should be stated and compared to the 1.5× threshold.

### Reproducibility Spec

- The `scripts/gen_data.py` and Rust `make_data()` use independent RNG implementations with the same seed. Whether they produce statistically equivalent data (same distribution parameters, not bitwise identical) should be documented. The cross-validation relationship between the two data sources should be stated.
- CPU thermal state and boost configuration during benchmark execution should be documented, or the plan should acknowledge this as an uncontrolled variable and report it in results.
- The `--variant` flag's output schema for `tw_profiler` should be specified in the plan (at minimum: the JSON field names that `analyze_results.py` depends on).

### Data Acquisition

- The `data/` generation should have a verification step confirming the generated files are well-formed (correct shape, value range, file size) before the profiler step consumes them.
- The dual-sourcing note ("in-process bench + npy profiler") should explicitly state whether the two data sources use identical distribution parameters and whether cross-validation is planned or out of scope.

---

## Red-Team Decision Points — All `requires_decision: true`

Each of the following requires an explicit author decision before the experiment proceeds. The decision does not need to be "resolve the risk" — it may be "accept the risk and document it" — but the decision must be made consciously.

### RT-1: Post-Profiling Threshold (Goodhart Risk)
**Risk:** The 1.5× threshold may have been selected after observing the 70.3% profiling result, making the bar calibrated to expected gains rather than to answering whether the optimization is meaningful. If the threshold was post-hoc, the experiment tests "can we hit the target we set after seeing the data" rather than "does the optimization produce a meaningful improvement."
**Decision required:** Was the 1.5× threshold pre-specified before the profiling result was available, or is it post-hoc? If post-hoc, the threshold should be documented as exploratory, and the decision criterion for H3 escalation adjusted accordingly.

### RT-2: Asymmetric Implementation Effort
**Risk:** H2 receives a custom low-level SIMD kernel while the baseline receives no analogous tuning. The measured speedup reflects both algorithmic superiority and differential engineering effort. If the baseline were similarly optimized (e.g., with a compiler-auto-vectorized equivalent), the speedup ratio would likely be smaller.
**Decision required:** Is the asymmetric comparison intentional (realistic production deployment scenario: "what can we gain by adding this optimization to existing code") or is it a fairness gap that should be acknowledged as a threat to the conclusion that H2 is a better algorithm?

### RT-3: Single Seed Selection
**Risk:** Seed 42 may produce data layouts that are atypically favorable for SIMD-aligned Y access patterns in H2. Without documentation of how the seed was selected, survivorship bias from seed choice cannot be ruled out.
**Decision required:** Was seed 42 selected for reasons unrelated to performance (e.g., convention, prior bench matching), or was it explored? If the latter, should additional seeds be included in the benchmark design?

### RT-4: Evaluation Collision (AtomicU64 in Treatment and Measurement)
**Risk:** The AtomicU64 instrumentation is embedded inside the same parallel iteration that is being benchmarked. The atomic overhead interacts differently with H1's flat-buffer layout (more predictable memory access) vs H2's SIMD kernel (wider register state). This means the measurement infrastructure is not neutral across treatments.
**Decision required:** Should the step-fraction profiling be separated from the wall-time Criterion benchmark (profiler runs separately, Criterion runs without AtomicU64 overhead), or is the overhead considered acceptable and documented as part of the measurement uncertainty?

### RT-5: 10-Sample Budget vs CI Precision
**Risk:** Criterion's minimum sample count is 10. With only 10 samples, the bootstrap CI width may be insufficient to reliably distinguish the 1.1×–1.5× inconclusive zone from the ≥1.5× positive zone, particularly under Rayon scheduling noise.
**Decision required:** Is 10 samples accepted as a deliberate trade-off (time-bounded experiment) with the understanding that an inconclusive result may simply reflect insufficient power, or should a pre-specified sample escalation rule be included (e.g., if CI overlaps 1.0× at 10 samples, re-run with 50 samples)?
