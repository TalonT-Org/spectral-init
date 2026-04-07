# Review-Design Evaluation Dashboard
## Experiment: y_heap Bottleneck Optimization in Trustworthiness Computation (Revision 3)

---

## Verdict

```
╔══════════════════════════════════════╗
║           VERDICT:  GO               ║
╚══════════════════════════════════════╝
```

**Experiment type:** benchmark  
**Active dimensions:** 14  
**Warning threshold:** 70 (14 × 5)  
**Critical findings:** 0  
**Warning findings:** 18  
**Red-team findings:** 6 (all capped at warning per benchmark type)  
**Stop triggers:** 0  

---

## Classification Summary

| Field | Value | Source |
|---|---|---|
| experiment_type | benchmark | frontmatter/triage |
| hypothesis_h0 | CI for T_baseline/T_flat_simd at n=10K contains 1.0 | frontmatter |
| hypothesis_h1 | CI lower bound strictly > 1.0 at n=10K, k=15 | frontmatter |
| estimand | flat_simd vs baseline on wall-time speedup ratio at n=10K, k=15 | frontmatter |
| primary_metric | Criterion 95% CI speedup ratio | frontmatter |
| baselines | baseline (current production `trustworthiness()`) | frontmatter |
| statistical_plan | CI LB > 1.0 (Stage 1); CI LB > 1.05 (Stage 2 escalation); α=0.05 nominal | frontmatter |
| success_criteria | Three criteria (positive/negative/inconclusive) | frontmatter |

**Active secondary modifiers:**
- `+causal`: causal decomposition table present → `causal_structure` S→L
- `+multi_metric`: 7 DVs (≥3) → `statistical_corrections` M→H
- `+deployment`: production motivation → `ecological_validity` floor = M (already M)

---

## Dimension Scorecard

| Dimension | Weight | Findings | Highest Severity | Notes |
|---|---|---|---|---|
| estimand_clarity | H (L1) | 0 | info | Formal contrast clear: T_baseline/T_flat_simd at n=10K, k=15 |
| hypothesis_falsifiability | H (L1) | 0 | info | H0 rejection criterion explicit and falsifiable |
| baseline_fairness | M (L2) | 1w, 3i | warning | Fixed run order introduces thermal sequencing asymmetry |
| causal_structure | L (L2) | 2w, 1i | warning | Bundle attribution declared; shipping logic depends on it |
| unit_interference | M (L2) | 1w, 4i | warning | W4 n-sweep warm-state declared; cross-variant isolation adequate |
| error_budget | H (L3) | 3w, 1i | warning | sample_size=10 unjustified; Type II risk undeclared |
| statistical_corrections | H (L3) | 2w, 2i | warning | heap_reuse shipping path lacks pre-specified equivalence criterion |
| variance_protocol | H (L3) | 1w, 1i | warning | Stage 2 false-negative risk undeclared |
| ecological_validity | M (L4) | 1w, 3i | warning | n=10K evidence vs potential scale gap; other threats well-declared |
| measurement_alignment | M (L4) | 2w, 2i | warning | hot-loop vs deployment (declared RT-A); heap_reuse selection gap |
| reproducibility_spec | M (L4) | 3w, 4i | warning | environment.yml mix; RAYON_NUM_THREADS not pinned; prior_failure_analysis.md manual |
| data_acquisition | M (L4) | 2w, 4i | warning | Criterion in-process data not independently verified; Phase 0 scope narrow |
| benchmark_representativeness | M (L4) | 0w, 2i | info | Scope limits well-declared (W6, W7, R2, RT-C, RT-J) |
| red_team | — | 6w | warning | All capped at warning per benchmark type; all requires_decision: true |

---

## Level 1 Findings (Fail-Fast Gate)

### estimand_clarity
**Result: CLEAN — no warnings or critical findings.**
The claim is expressible as a formal contrast: flat_simd vs baseline on wall-time speedup ratio at n=10K, k=15, seed=42. Treatment, comparator, outcome variable, and population are all named.

### hypothesis_falsifiability
**Result: CLEAN — no warnings or critical findings.**
H0 rejection criterion is precisely stated (CI LB > 1.0). The condition under which H0 is retained (CI LB ≤ 1.0, point estimate < 1.1×) is spelled out in the escalation protocol and success criteria. The experiment is falsifiable.

**L1 GATE: PASSED — no critical findings. Proceeding to full analysis.**

---

## Level 2 Findings

### baseline_fairness (M)

**[W-BF-1] WARNING — Variant group execution ordering fixed, not counterbalanced**
Section: `## Controlled Variables`
The plan specifies 60-second thermal gaps between variant groups and separate process invocations, but the variant execution order is fixed (baseline first, flat_simd last per the script description). If earlier variants benefit from lower CPU temperatures or higher boost clock states, and later variants run at thermally throttled frequency, the ordering introduces an asymmetric resource condition not controlled or declared as a threat. Shorter execution windows benefit from boost clocks; longer ones may not.
`requires_decision: false`

**[I-BF-2] INFO — AVX2 asymmetry declaration location**
The intentional AVX2 asymmetry (baseline has no SIMD for y_heap, flat_simd does) is declared in §Threats to Validity but the plan does not specify whether this declaration appears in raw benchmark output headers or only in the design document. Readers of raw Criterion results may not encounter the scope limitation.

**[I-BF-3] INFO — heap_reuse pre-allocation size not specified**
The plan says heap_reuse uses `with_capacity(k+1)` pre-allocated per thread. Whether this over-allocates, under-allocates, or matches the baseline exactly per row is unspecified.

**[I-BF-4] INFO — Thread affinity and NUMA binding uncontrolled**
`RAYON_NUM_THREADS=$(nproc)` is set identically for all variants, but thread affinity is not declared. On heterogeneous-core or multi-socket systems this could introduce uncontrolled resource asymmetry.

### causal_structure (L)

**[W-CS-1] WARNING — AVX2 asymmetry creates alternative explanation for speedup magnitude**
Section: `## RT-E` / `## Threats to Validity`
The framing "flat buffer + AVX2 combined gain over unoptimized heap" is appropriately qualified, but if the primary result is POSITIVE and leads to shipping flat_simd, the magnitude of the speedup will be attributed to the flat buffer + SIMD approach rather than to the AVX2 investment asymmetry. The plan does not specify how this alternative explanation is communicated in the final analysis report, creating a risk that the speedup magnitude is interpreted as evidence for architectural superiority rather than a combined "optimized vs unoptimized" comparison.
`requires_decision: false`

**[W-CS-2] WARNING — Shipping decision between flat_simd and heap_reuse depends on causal attribution the design cannot fully support**
Section: `## Analysis Plan`
The "prefer simpler" fallback (ship heap_reuse if it matches flat_simd within CI overlap) implicitly treats the flat_partial → flat_simd SIMD gap as a causal decomposition step. RT-D declares this conflation. If the gap between heap_reuse and flat_simd is largely attributable to the algorithm change (introselect vs push/evict, bundled into flat_partial) rather than to SIMD, the "prefer simpler" decision would be selecting a variant for the wrong reason. The design cannot isolate which bundle causes the gap; the decision depends on a causal inference the experiment is not powered to make.
`requires_decision: false`

**[I-CS-3] INFO — Bundle attribution table correctly labeled**
The causal decomposition formulas are labeled "bundle attribution, not single-cause isolation" throughout. No unqualified causal claims are present.

### unit_interference (M)

**[W-UI-1] WARNING — W4 cache warm-state within n-sweep is declared but not mitigated**
Section: `## Controlled Variables`
Within each process invocation, thread-local Vecs grow from n=1K through n=5K to n=10K. At n=5K and n=10K, allocation cost is absorbed by pre-grown buffers. The direction (consistently biases allocation savings toward understating heap_reuse advantage) is declared as W4. However, no mitigation is implemented — the n-sweep ordering is fixed within each invocation, meaning the n=10K primary measurement always follows the n=1K and n=5K warm-up sweeps. An intra-invocation thermal carry-over from the n=5K to n=10K sweep within the same process is also not analyzed.
`requires_decision: false`

---

## Level 3 Findings

### error_budget (H)

**[W-EB-1] WARNING — sample_size=10 not justified by any error analysis**
Section: `## Dependent Variables`
Stage 1 uses `sample_size=10` with no power analysis, coefficient of variation estimate, or minimum detectable effect calculation. The CI width from 10 Criterion samples at n=10K is unknown relative to the 1.0 threshold. If the true speedup is in the 1.1–1.3× range (plausible given the prior 2× slowdown failure and the exploratory nature), 10 samples may produce a CI that is too wide to reject H0, triggering a false escalation to Stage 2 when a larger initial sample would have been sufficient. The sample size appears to be chosen by convention, not by the expected measurement distribution.
`requires_decision: false`

**[W-EB-2] WARNING — Type II error risk (false negative for modest speedups) not acknowledged**
Section: `## Dependent Variables`
The plan declares Type I error inflation (RT-F, W1) but does not acknowledge Type II error (failing to detect a real speedup). The escalation trigger of 1.1× point estimate implicitly creates a detection floor, but no false-negative rate is stated or bounded for speedups in the 1.05×–1.1× range where Stage 1 would neither confirm nor escalate. A real speedup in this region would be silently classified as NEGATIVE without any declared probability of missing it. Given the prior 2× slowdown failure, the system's behavior near the detection floor is highly relevant.
`requires_decision: false`

**[W-EB-3] WARNING — Effective Type I error rate for the overall experiment is unquantified and likely above nominal α=0.05**
Section: `## Dependent Variables`
Two compounding factors make the nominal α=0.05 assignment inaccurate: (a) W1 declares the ratio CI has no guaranteed 95% coverage because it derives from two independent bootstrap distributions — the direction and magnitude of coverage distortion are not bounded; (b) RT-F acknowledges two-stage Type I inflation from the sequential escalation design but does not compute the inflated familywise α. The combination of an uncorrected bootstrap coverage issue and an unquantified two-stage inflation means the effective experiment-level Type I error rate is unknown and possibly substantially above 5%.
`requires_decision: false`

**[I-EB-4] INFO — Two-stage threshold tightening is a practical guard, not a formal correction**
The CI LB > 1.05 threshold for Stage 2 is a qualitative compensation declared in RT-F. This is acceptable for a benchmark experiment at this development stage.

### statistical_corrections (H)

**[W-SC-1] WARNING — heap_reuse shipping decision path lacks pre-specified equivalence criterion**
Section: `## Dependent Variables` / `## Analysis Plan`
The POSITIVE shipping branch includes "or heap_reuse if heap_reuse matches flat_simd within CI overlap — prefer simpler." This decision is contingent on an exploratory secondary DV (T_baseline/T_heap_reuse) via CI overlap comparison — a decision pathway with no pre-specified equivalence margin, no alpha, and no hypothesis. If flat_simd clears the confirmatory threshold but heap_reuse does not independently clear it, applying informal CI overlap to select heap_reuse over flat_simd introduces an uncorrected inferential step. The shipped artifact is determined by a secondary metric not subject to the confirmatory H0 test.
`requires_decision: true`

**[W-SC-2] WARNING — Two-stage escalation Type I inflation acknowledged but not formally bounded**
Section: `## Dependent Variables`
RT-F acknowledges that the two-stage design inflates familywise Type I error but does not compute or bound the inflated α. At H weight (due to +multi_metric), with a binary shipping recommendation downstream, the gap between nominal α=0.05 and actual experiment-level error rate is a material undisclosed risk. The tightened threshold (1.05) is an unprincipled partial compensation.
`requires_decision: true`

**[I-SC-3] INFO — Single confirmatory comparison well-separated from exploratory secondaries**
The primary DV is clearly pre-specified and labeled. Secondary DVs are explicitly labeled "exploratory, uncorrected." This separation is adequate for H-weight context.

**[I-SC-4] INFO — Scale-sensitivity DVs correctly scoped**
n=5K and n=1K results are labeled exploratory and do not route into any shipping branch.

### variance_protocol (H)

**[W-VP-1] WARNING — Stage 2 false-negative risk not declared**
Section: `## Escalation Protocol`
R1 declares the Stage 2 false-positive risk (escalation trigger may fire due to scheduling luck) and partially compensates with a tightened threshold. The symmetric false-negative risk — Stage 2 suffers from scheduling degradation and produces CI LB < 1.05 despite a true speedup above 1.05× — is not addressed. Under this scenario, the experiment would classify a real optimization as INCONCLUSIVE and escalate to H3 (KD-tree), losing the benefit of a genuine improvement. At H weight, undeclared false-negative risk in the decision protocol is a design gap.
`requires_decision: false`

**[I-VP-2] INFO — Seeds and flat sampling mode are adequate**
Criterion bench: `SmallRng::seed_from_u64(42)`. .npy files: `numpy.random.default_rng(seed=42)`. `SamplingMode::Flat` eliminates adaptive sampling variance. These are adequate variance controls for a benchmark experiment.

---

## Level 4 Findings

### ecological_validity (M)

**[W-EV-1] WARNING — W7 scopes n=10K result but does not flag the gap between evidence and implied deployment scale**
Section: `## Threats to Validity §External`
W7 correctly declares the flat buffer approach is validated only at n≤10K. However, the plan's motivation (RT-A) explicitly targets a deployment context where `trustworthiness()` is called in evaluation pipelines. If production use cases operate at n≥32K (approaching the L2 spill threshold of ~32K elements for COMB_DIST_Y), the flat buffer's sequential access advantage may reverse under L2 miss pressure. W7 says "a separate experiment is required" but does not declare that the shipping recommendation from this experiment should be explicitly bounded to n≤10K contexts. The scope gap between evidence and shipping decision is not propagated into the recommendation framing.
`requires_decision: false`

**[I-EV-2] INFO — R2, R4, R5, W6, RT-J are adequately declared**
The hot-loop vs cold-call gap (R4), synthetic data vs UMAP distributions (R2), compilation flag variation (R5), k=15 scope limit (W6), and hardware specificity (RT-J) are all declared with appropriate specificity for M-weight benchmark evaluation.

**[I-EV-3] INFO — d_y=2 specialization scope is appropriate**
RT-C correctly scopes the SIMD kernel to d_y=2, which matches the intended UMAP 2D visualization use case. No undeclared scope restriction here.

**[I-EV-4] INFO — sample_size=10 Criterion configuration not declared as a validity threat**
SamplingMode::Flat with sample_size=10 produces bootstrap CIs from a small sample. Whether 10 samples adequately characterizes CI width for the decision threshold is addressed by the escalation protocol (Stage 2 with n=50) but is not declared as an independent ecological validity risk for Stage 1 results.

### measurement_alignment (M)

**[W-MA-1] WARNING — Primary metric measures hot-loop throughput, not deployment latency**
Section: `## Dependent Variables`
H1 claims "the optimization is reliably faster than baseline" without specifying that this is isolated hot-loop wall-time under Criterion's warm-cache conditions, not end-to-end or cold-call deployment latency. RT-A declares this limitation. The risk is that a POSITIVE verdict (CI LB > 1.0 in hot-loop) is presented as evidence for deployment improvement when the deployment mechanism (warm cache after UMAP computation) differs from Criterion's warm-up regime. This is a declared limitation, not an undiscovered one, but the alignment gap should be reflected in the analysis report framing.
`requires_decision: false`

**[W-MA-2] WARNING — heap_reuse shipping selection has no pre-specified equivalence margin**
Section: `## Analysis Plan` / `## Shipping Decision Logic`
(Same finding as W-SC-1, measurement alignment perspective.) The "prefer simpler if within CI overlap" decision for heap_reuse involves no pre-specified threshold for what constitutes "within CI overlap" — whether 95% CIs share any overlap, whether point estimates are within 5%, etc. This makes the decision gate informal and non-reproducible across analysts.
`requires_decision: true`

**[I-MA-3] INFO — Time-base change from prior experiment declared**
R3 explicitly documents the change from `AtomicU64` summed thread-time (prior experiment) to `eprintln!` per-call wall-clock fractions (new experiment). These are not directly comparable; the plan correctly treats them as separate measurements.

**[I-MA-4] INFO — Correctness gate tolerance not calibrated to numerical regime**
The `|ΔT| < 1e-12` gate is applied without establishing whether 1e-12 is tight or loose relative to the expected floating-point rounding error for O(n×k) ≈ 150K accumulated terms at n=10K, k=15. The gate may not be discriminating at the regime of interest. This is unlikely to be a material issue given the test fixtures are small (t_tw_01 through t_tw_07), but the calibration gap is undeclared.

### reproducibility_spec (M)

**[W-RS-1] WARNING — environment.yml mixes conda-forge channel and pip; resolver not specified**
Section: `## Experiment Directory Layout`
The environment.yml specifies conda-forge channel with pinned minor versions (numpy=2.2.*, scipy=1.15.*, matplotlib=3.10.*) but the plan notes the existing `envs/spectral-test/` conda prefix may be used directly. An independent party cannot deterministically reconstruct the Python environment without knowing which resolver was used, its version, and whether the conda or pip path was taken.
`requires_decision: false`

**[W-RS-2] WARNING — RAYON_NUM_THREADS=$(nproc) not pinned; results depend on core count**
Section: `## Controlled Variables`
`nproc` returns the number of online logical cores, which varies by machine configuration. Criterion results for parallel Rayon code depend on thread count. An independent party on a different machine cannot know what core count is required or acceptable for the results to be considered comparable.
`requires_decision: false`

**[W-RS-3] WARNING — prior_failure_analysis.md is a manually authored prerequisite with no independent reproduction specification**
Section: `## Phase 0`
This Phase 0 prerequisite is a manually authored analysis document. An independent party reproducing the experiment from scratch has no specification for what this file must contain, how it is produced, or what evidence it must cite to satisfy the Phase 0 gate. If the gate requires reading the rerun-clean worktree and that worktree is not available in a fresh clone, the prerequisite cannot be satisfied.
`requires_decision: false`

**[I-RS-4] INFO — Cargo.lock.snapshot is post-hoc**
The Cargo.lock snapshot is recorded at run time rather than committed prior to execution. This documents dependency state but does not guarantee reproducibility if crates.io resolves differently in a future reproduction attempt.

**[I-RS-5] INFO — 60-second thermal gap adequacy not empirically validated**
Thermal state equivalence between variant invocations is not captured in any artifact, making it impossible to verify inter-run thermal variance.

**[I-RS-6] INFO — analyze_results.py command invocation fully specified in Execution Protocol**
The execution protocol gives exact commands, including `--stage1-only` and conditional `--escalated` flags. This is adequate for reproducibility at M weight.

**[I-RS-7] INFO — No convergence criterion for independent reproduction matching**
No tolerance is stated for what constitutes a "matching" reproduction result (e.g., point estimate within 5%, or CI LB on same side of 1.0).

### data_acquisition (M)

**[W-DA-1] WARNING — Criterion in-process data (primary metric source) has no independent verification step**
Section: `## Inputs and Data`
The primary hypothesis test runs on data generated in-process by `SmallRng::seed_from_u64(42)`, not on the verified .npy files. The data verification in Phase 2 applies only to the .npy files used by the profiler. The primary Criterion data's distribution properties (uniform[0,1], correct shapes, no numerical anomalies) are assumed from the `make_data(n, 10, 2, 42)` function without an equivalent verification step.
`requires_decision: false`

**[W-DA-2] WARNING — Phase 0 acquisition may be insufficient if root cause requires timing data, not just source code**
Section: `## Phase 0`
The Phase 0 acquisition step specifies reading `src/metrics.rs` from the rerun-clean worktree. The 2× slowdown root cause may require reviewing the prior experiment's step-timing artifacts (profiler outputs) in addition to source code. The acquisition step is scoped to source reading only, which may be insufficient for a data-supported root cause determination.
`requires_decision: false`

**[I-DA-3] INFO — Hypothesis coverage complete**
Every success criterion (positive, weak positive, negative, inconclusive) maps to a named data source (Criterion CI, profiler step fractions, correctness tests).

**[I-DA-4] INFO — Gitignored .npy files have explicit acquisition steps**
Phase 2 specifies `gen_data.py --out-dir data/` with full parameter detail. Verification is logged to `data_verification.txt`.

**[I-DA-5] INFO — Dependency ordering is acyclic and correctly sequenced**
Phase 0 gates Phase 3; Phase 1-2 precede Phase 3; dry run gates full run; analysis runs last.

**[I-DA-6] INFO — Directive compliance**
Hardware profile, Cargo.lock snapshot, and environment.yml artifacts all have explicit generation or commit steps.

### benchmark_representativeness (M)

**[I-BR-1] INFO — Scope restrictions are comprehensive and well-declared**
W6 (k=15), W7 (n≤10K), R2 (synthetic data), RT-C (d_y=2), RT-J (hardware specificity) together cover the primary generalizability restrictions. No major undeclared scope restriction was found that is not addressed elsewhere in the plan.

**[I-BR-2] INFO — W4 reversed-order check not included for new variants**
The plan declares W4 (cache warm-state within n-sweep) but does not include a reversed n-order check to empirically bound the magnitude of the warm-state bias for the new variants. This is accepted as a limitation (W4 is declared as an accepted design trade-off in the plan).

---

## Adversarial Findings (Red-Team)

All red-team findings are capped at `warning` per benchmark type. All `requires_decision: true`.

**[RT-1] WARNING — Goodhart exploitation: hot-loop exploits warm-cache SIMD throughput advantage**
Section: `## Metric Design`
The cheapest way to score well on T_baseline/T_flat_simd is to write an implementation that exploits Criterion's warm-cache hot-loop regime — exactly what sequential flat-array access + SIMD does. The measured ratio may reflect "cache-warm SIMD throughput advantage" rather than "deployment latency advantage." RT-A acknowledges this but the risk that a cold-call or interleaved-invocation deployment shows a smaller or reversed speedup is not fully resolved. The prior thread_local 2× slowdown failure demonstrates that warm-loop results can contradict cold-call behavior in this exact codebase.
`requires_decision: true`

**[RT-2] WARNING — Data leakage: single fixed input at (n=10K, k=15) may have been construction-time tuned**
Section: `## Data / Input Construction`
The benchmark uses a single fixed synthetic input (seed=42, n=10K, k=15) known at implementation time. The flat_simd kernel's AVX2 lane widths (2 rows per 4-wide 256-bit lane) and the introselect implementation could have been unconsciously tuned to perform optimally at this specific shape. No validation at alternative (n, k) pairs is included as a guard. The correctness gate (`|ΔT| < 1e-12`) does not protect against this.
`requires_decision: true`

**[RT-3] WARNING — Asymmetric tuning: baseline y_heap received no AVX2 investment**
Section: `## Baseline Configuration`
RT-E declares this intentional and deployment-relevant. Residual risk: the measured speedup reflects "fully optimized vs fully unoptimized" for y_heap, not "algorithmic approach A vs B under equal effort." The experiment cannot distinguish "flat buffer is inherently better" from "flat buffer was optimized and baseline was not." The shipping recommendation is made on evidence that conflates architectural advantage with optimization investment asymmetry.
`requires_decision: true`

**[RT-4] WARNING — Survivorship bias: asymmetric two-stage stopping rule inflates effective Type I rate**
Section: `## Escalation Protocol`
The plan allows the experiment to stop at Stage 1 on a positive result (CI LB > 1.0) but extends to Stage 2 on an ambiguous result (point estimate ≥ 1.1×, CI LB ≤ 1.0). A result that marginally clears Stage 1 is published as success; a result that marginally fails Stage 1 gets a second chance. This asymmetric stopping rule inflates effective Type I error above α=0.05. No Bonferroni or alpha-spending correction is applied. RT-F acknowledges this; the tightened 1.05 threshold is a partial but unprincipled compensation.
`requires_decision: true`

**[RT-5] WARNING — Evaluation collision: AMD Ryzen 9800X3D 3D V-Cache introduces non-uniform core access**
Section: `## Infrastructure Independence`
The 9800X3D has a 3D V-Cache stacked on specific cores, creating non-uniform L3 access latency across the processor's physical cores. Criterion does not pin measurements to specific cores. If different measurement iterations land on V-Cache vs non-V-Cache cores, within-run variance may be inflated by hardware topology noise rather than algorithmic variation. This is not declared in §Threats to Validity.
`requires_decision: true`

**[RT-6] WARNING — Asymmetric effort: no "heap + AVX2" control variant isolates algorithmic contribution**
Section: `## Optimization Effort`
The four variants represent monotonically increasing optimization effort applied exclusively to the flat buffer path. No control variant applies equivalent AVX2 investment to the original heap approach. The experiment is structured to validate a pre-chosen direction, not to compare approaches under equal investment. The ratio T_baseline/T_flat_simd measures "optimized vs unoptimized" rather than "architecture A vs architecture B," overstating the case for shipping flat_simd as categorically superior.
`requires_decision: true`

---

## Cannot Assess

The following design aspects could not be evaluated from the plan text alone and were outside the scope of design review:

1. **Introselect tie-breaking exact invariance**: The plan specifies a `.total_cmp(&dist_y[b]).then(a.cmp(&b))` comparator (RT-G) to match BinaryHeap tie-breaking order. Whether this comparator faithfully replicates the production BinaryHeap's eviction behavior for all k+1 capacity states cannot be verified from the design document alone — it requires implementation inspection.

2. **3D V-Cache core assignment distribution**: Whether Criterion's thread placement interacts with the AMD Ryzen 9800X3D's non-uniform V-Cache topology (RT-5 above) in a way that materially affects variance cannot be assessed without hardware characterization data.

3. **Criterion bootstrap ratio CI coverage**: The actual coverage of the ratio CI for the specific timing distributions in this benchmark (W1) cannot be assessed without pilot sample data. The direction and magnitude of coverage distortion from the ratio-of-distributions bootstrap are distribution-dependent.

4. **SmallRng distribution properties**: Whether `SmallRng::seed_from_u64(42)` produces a uniform[0,1] distribution whose tail behavior matches `numpy.random.default_rng(seed=42)` closely enough that the Criterion and profiler inputs can be treated as independent samples from the same population cannot be assessed from design review alone.

5. **n=10K to n=100K performance extrapolation**: Whether the flat buffer's sequential access advantage reverses at n≥32K (the estimated L2 spill threshold for COMB_DIST_Y at 256KB/8 bytes) cannot be assessed without running the larger-n benchmark — which W7 correctly declares out of scope.

---

## Mechanizable Check Log

Binary checks that could be automated in CI or as pre-run validation:
- [ ] `CARGO_FEATURE_PROFILING` not set when running `run_criterion.sh` (plan specifies this guard; W8)
- [ ] `data_verification.txt` contains no NaN/Inf lines (automated from gen_data.py output)
- [ ] Four Criterion JSON files per variant present before `analyze_results.py` runs
- [ ] `prior_failure_analysis.md` exists and is non-empty before Phase 3 begins (Phase 0 gate)
- [ ] Rust toolchain matches `rust-toolchain.toml` (nightly-2026-03-26)
- [ ] `RAYON_NUM_THREADS` value recorded in `hardware_profile.txt`
- [ ] All t_tw_01..t_tw_07 pass with `cargo test --features testing` before benchmarks proceed

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: GO
experiment_type: benchmark
critical_count: 0
warning_count: 18
red_team_count: 6
active_dimensions: 14
warning_threshold: 70
stop_triggers: 0
revision: 3
```
