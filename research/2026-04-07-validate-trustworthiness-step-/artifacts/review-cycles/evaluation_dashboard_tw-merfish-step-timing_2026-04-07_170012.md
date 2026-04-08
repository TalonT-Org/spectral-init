# Design Review: Trustworthiness Step-Timing on MERFISH Real Data

**Verdict: REVISE**
**Experiment Type:** robustness_audit
**Plan:** `experiment_plan_tw_merfish_step_timing_2026-04-07_165019.md`
**Review timestamp:** 2026-04-07 17:00:12

---

## Verdict Banner

```
┌─────────────────────────────────────────────────────────┐
│  VERDICT: REVISE                                        │
│  The plan has critical design gaps that must be         │
│  addressed before execution produces interpretable      │
│  results. The core measurement metric (thread-aggregate │
│  ns fraction) is not aligned with the optimization ROI  │
│  inference, and the baseline comparison is structurally │
│  asymmetric. Multiple critical issues across 5+         │
│  dimensions warrant revision before compute is spent.   │
└─────────────────────────────────────────────────────────┘
```

**Classification summary:**
- Experiment type: `robustness_audit` (tests generalization of PR #238's synthetic-data conclusion to real MERFISH data)
- Secondary modifiers active: `+deployment` (production engineering investment decision), `+multi_metric` (7 DVs → statistical_corrections L→L from S)
- Active dimensions: 14
- Warning budget: 14 × 5 = 70
- Critical findings: 22
- Warning findings: 38
- Red-team findings: 8 (all capped at warning per robustness_audit severity ceiling)
- Stop triggers (L1 critical or red-team critical after cap): 0

---

## Dimension Scorecard

| Dimension | Weight | Level | Findings | Worst Severity |
|---|---|---|---|---|
| estimand_clarity | H | L1 | 4 (2W, 2I) | warning |
| hypothesis_falsifiability | H | L1 | 4 (3W, 1I) | warning |
| baseline_fairness | — | L2 | 5 (2C, 2W, 1I) | **critical** |
| causal_structure | M | L2 | 5 (2C, 3W) | **critical** |
| unit_interference | — | L2 | 4 (1C, 2W, 1I) | **critical** |
| error_budget | — | L3 | 5 (3C, 2W) | **critical** |
| statistical_corrections | L | L3 | 3 (1W, 2I) | warning |
| variance_protocol | M | L3 | 5 (1C, 3W, 1I) | **critical** |
| ecological_validity | H | L4 | 6 (2C, 3W, 1I) | **critical** |
| measurement_alignment | H | L4 | 5 (3C, 2W) | **critical** |
| data_acquisition | H | L4 | 4 (1C, 2W, 1I) | **critical** |
| reproducibility_spec | — | L4 | 9 (5C, 3W, 1I) | **critical** |
| benchmark_representativeness | — | L4 | 6 (2C, 4W) | **critical** |
| red_team | — | RT | 8 (0C, 7W, 1I) | warning (capped) |

W = warning, C = critical, I = info. Red-team severity capped at "warning" for robustness_audit.

---

## Level 1 Findings (Fail-Fast)

*No critical findings — L1 gate passed.*

### estimand_clarity (warning)

- **[W]** `## Hypothesis` — H1 contains an embedded secondary assertion (`x_space ≥ 50% regardless of direction`) that functions as a separate estimand from the primary ±5pp contrast. The plan conflates two distinct claims without specifying how they relate if one is confirmed and the other is not.
- **[W]** `## Independent Variables` — MERFISH n=50K is listed as an IV with no stated estimand, hypothesis, threshold, or comparison baseline. It is unclear whether n=50K tests the same contrast or a distinct scale-stability claim.
- **[I]** `## Independent Variables` — Stretch-goal k values (5 and 30) have no associated contrast or expected direction.
- **[I]** `## Analysis Plan` — The ±5pp threshold is applied to a point estimate from a historical run rather than a confidence interval; the operationalization of the contrast is ambiguous.

### hypothesis_falsifiability (warning)

- **[W]** `## Hypothesis` — H1 has a compound structure creating an unfalsifiable region: a result where x_space_pct falls between ~45–51.1% satisfies the >5pp divergence criterion but is ambiguous on the ≥50% assertion.
- **[W]** `## Success Criteria` — The success criteria collapse compound H1 into a single threshold check (x_space_pct ≥ 50%), discarding the ±5pp component entirely. H0 could be rejected (>5pp divergence) while success criteria simultaneously declare H0 confirmed (x_space_pct < 50%). The mapping is logically inconsistent.
- **[W]** `## Hypothesis` — H0 is operationalized with two different thresholds (51.1% from ±5pp band, and 50% cutoff from success criteria). Which governs the conclusion is unspecified.
- **[I]** `## Hypothesis` — The "regardless of direction" clause makes H0 non-confirmable for any result where x_space_pct ≥ 50%.

---

## Level 2 Findings

### baseline_fairness (**critical**)

- **[C]** `## Inputs and Data` — The synthetic Gaussian baseline was collected in a prior session with unknown machine state, thermal conditions, and background load. Environmental differences between the historical collection and the MERFISH run are confounded with the dataset difference, making it impossible to attribute step-fraction divergence solely to data characteristics.
- **[C]** `## Inputs and Data` — The synthetic baseline uses a fixed pre-existing JSON (from `2026-04-06-y-heap-bottleneck-optimization`) rather than being re-run under the controlled conditions specified in this plan. The historical baseline's warmup count, iteration count, and profiler binary version are not verified to match the current controlled-variable table.
- **[W]** `## Controlled Variables` — The synthetic baseline's build profile, commit hash, and feature flags are not verified to match HEAD. An older binary or different feature flag set would make the step-timing measurement methodology non-equivalent.
- **[W]** `## Analysis Plan` — The comparison pairs a single historical result (no CI, no run variance) with a fresh 5-iteration run. The sample provenance asymmetry means observed differences cannot be attributed to data characteristics alone.
- **[I]** `## Threats to Validity` — The y_dist/y_heap key mismatch indicates the archived synthetic JSON may have schema differences from the current profiler output; imperfect field mapping would make the step fractions non-comparable.

### causal_structure (**critical**)

- **[C]** `## Analysis Plan — d_x Effect` — The plan attributes x_dist_pct to d_x magnitude via a FLOP-count model (`O(n² × d_x)`), but FLOP count and thread-aggregate ns are not equivalent. Memory bandwidth, cache pressure, and SIMD throughput are independent mediators. The theoretical prediction (`d_x / (d_x + d_y + constant) × 100`) is an unsupported causal proxy.
- **[C]** `## Analysis Plan — Clustering Geometry Effect` — The plan attributes lower y_dist_pct to "cache effects from clustered 2D layout," but this is confounded by: (1) simultaneous change in d_x altering the ratio, (2) MERFISH differing from synthetic along many axes beyond clustering, and (3) access patterns for y_dist depending on neighbor index ordering, not raw spatial clustering. These confounders are not identified or controlled.
- **[W]** `## Threats to Validity` — Thread-aggregate ns ≠ wall-clock is acknowledged but not identified as a confounder for the mechanism claims. If x_dist and y_dist parallelise at different efficiencies, the thread-ns ratio is not a valid proxy for computational cost ratio.
- **[W]** `## Hypothesis` — H1's two causal mechanisms (d_x magnitude and clustering geometry) cannot be disentangled since both change simultaneously between synthetic and MERFISH data. The design lacks any isolation of these factors.
- **[W]** `## Threats to Validity` — x_dist and y_dist use different SIMD kernels (AVX2+FMA vs. batched 2-point). The observed ns ratio is partially a function of hardware instruction throughput differences, not solely data geometry.

### unit_interference (**critical**)

- **[C]** `## Implementation — Phase 2` — Warmup iterations emit `[timing:...]` lines to the stderr capture file under the profiling feature flag. `parse_step_timing()` reads the entire capture file and accumulates every line indiscriminately, so warmup-call timing values are summed into the reported `step_timing` alongside timed-iteration values. The profiler JSON therefore contains 7 samples (2 warmup + 5 timed) mixed together, not the 5 timed iterations described in the analysis plan.
- **[W]** `## Implementation — Phase 2` — Thread-local Vec buffers (`COMB_DIST_X`, etc.) persist across all `trustworthiness()` calls within the same Rayon thread. Timed iterations inherit pre-allocated buffer capacity established during warmup, making them structurally non-equivalent to warmup iterations in allocation cost — a difference that is not documented in the variance model.
- **[W]** `## Threats to Validity` — n=10K and n=50K run sequentially with no cooldown or thermal monitoring. CPU thermal throttling, DVFS frequency stepping, and L3 cache state from the 10K run are not isolated from the 50K run's early iterations.
- **[I]** `## Implementation — Phase 2` — Rayon thread pool scheduling state and OS scheduler history from warmup iterations persist into timed iterations. The rationale for choosing 2 warmup iterations to stabilize thread-pool behavior is not documented.

---

## Level 3 Findings

### error_budget (**critical**)

- **[C]** `## Success Criteria` — No power analysis or sample size justification is present. No assessment of whether n=5 timed iterations is sufficient to detect a ±5pp difference against measurement noise is made.
- **[C]** `## Analysis Plan` — Type I and Type II error rates are entirely unacknowledged. Binary verdicts and threshold comparisons are made across 7 metrics without any stated alpha level or false positive/negative risk discussion.
- **[C]** `## Success Criteria` — The historical baseline is a single point estimate with no distributional characterization. Measurement uncertainty in the baseline is unacknowledged; it is impossible to determine whether observed differences fall within baseline noise or represent genuine signal.
- **[W]** `## Hypothesis` — The ±5pp and ≥50% thresholds are stated without statistical or domain justification. No acknowledgment that these are heuristic choices.
- **[W]** `## Analysis Plan` — n=5 variance estimates are highly unstable, inflating Type II error risk for true differences near the ±5pp decision boundary.

### statistical_corrections (warning)

- **[W]** `## Analysis Plan` — No multiple comparisons correction is pre-specified despite 7 DVs and at least 4 named comparison claims. Family-wise error rate is uncontrolled across simultaneous comparisons on correlated metrics (x_dist_pct, y_dist_pct, x_space_pct share geometry).
- **[I]** `## Analysis Plan` — Primary vs. secondary comparison hierarchy is labeled but the confirmatory/exploratory boundary is not formalized.
- **[I]** `## Success Criteria` — The binary verdict threshold (x_space_pct ≥ 50%) is a single-comparison rule but overlaps inferentially with the primary comparison; conflict resolution is not specified.

### variance_protocol (**critical**)

- **[C]** `## Execution Protocol` — No independent process re-runs are planned. Five within-run iterations share process state (thread pools, allocator arenas, TLB, branch predictor) and are not statistically independent. A single outlier event (OS jitter, thermal throttle) cannot be distinguished from signal without replicated independent runs.
- **[W]** `## Controlled Variables` — No random seed is specified. Although `trustworthiness()` is deterministic, the plan does not document this fact, and the absence of a documented seed rationale leaves readers unable to distinguish "seeds not needed" from "seeds overlooked."
- **[W]** `## Analysis Plan` — Per-step timing variance across the 5 iterations is not tracked or reported. Only mean step fractions are computed. Without per-iteration step-fraction values or standard deviations, the stability of the bottleneck ranking cannot be assessed.
- **[W]** `## Threats to Validity` — With n=5 samples, the 95% CI on the mean is approximately ±1.13 × std, wide enough that a meaningful timing difference could fall within noise at typical microbenchmark coefficients of variation.
- **[I]** `## Controlled Variables` — Rayon thread pool size is not listed as a controlled variable. Thread-aggregate ns counters are directly proportional to pool size; uncontrolled pool size would invalidate cross-run comparisons.

---

## Level 4 Findings

### ecological_validity (**critical**)

- **[C]** `## Inputs and Data` — The plan itself acknowledges n=10K may understate X-dominance at production scale (PR #238's 61% figure may be from n=100K; penalty cost grows relative to distance cost as n increases). Testing at n=10K may structurally suppress X-dominance fractions relative to the deployment context where the optimization decision applies.
- **[C]** `## Inputs and Data` — d_x is listed as "TBD (~48 or ~8-20)" — a 3-6× ambiguity range. Since X-space cost scales with d_x, the ecological validity of the X fixture is indeterminate at plan authorship time.
- **[W]** `## Inputs and Data` — A single biological dataset (mouse hypothalamus MERFISH) is used to represent the full class of real-world deployment targets. MERFISH has atypically high spatial structure relative to scRNA-seq atlas data or other spatial transcriptomics modalities.
- **[W]** `## Inputs and Data` — The Y-space fixture is a pre-computed UMAP embedding from a fixed random seed. Different embeddings of the same data produce different 2D geometry and potentially different y_dist cache effects.
- **[W]** `## Controlled Variables` — The experiment is run on a single local machine with AVX2+FMA support. Production environments (cloud VMs, CI runners, ARM systems) may have different SIMD profiles that alter step-fraction ratios.
- **[I]** `## Inputs and Data` — n=50K data is listed as available and its scale-stability check is identified as a threats-mitigation measure, but the n=50K run is labeled "stretch — comment out if time-constrained," leaving the most directly addressable ecological validity gap optionally unexecuted.

### measurement_alignment (**critical**)

- **[C]** `## Dependent Variables` — x_space_pct is computed from thread-aggregate nanoseconds, which conflate parallelism degree with compute cost. A step running on 8 threads accumulates 8× the ns per wall-clock second versus a single-threaded step of identical wall duration. The bottleneck conclusion drawn from this metric (x_dist + x_sort ≥ 50%) does not measure wall-clock share.
- **[C]** `## Threats to Validity` — The plan acknowledges "thread-aggregate ns ≠ wall-clock" but then asserts "ratios between steps are valid for bottleneck analysis" without qualification. This assertion holds only if all steps have identical parallelization efficiency — a condition that is not verified and is likely violated given different SIMD kernels.
- **[C]** `## Analysis Plan` — The binary verdict "Is x_space_pct ≥ 50%?" uses thread-ns fraction as a proxy for wall-clock optimization ROI. Optimization ROI is determined by wall-clock share (Amdahl's law), not thread-ns share. A step with high thread-ns fraction but high parallelism efficiency may yield less wall-clock speedup than a step with lower thread-ns fraction and poor parallelism.
- **[W]** `## Dependent Variables` — Wall-clock mean_s is a single aggregate scalar for the entire call; it cannot decompose wall-clock time by step and therefore cannot independently validate the per-step thread-ns fractions.
- **[W]** `## Dependent Variables` — d_x is listed as a dependent variable, but it is a covariate or independent variable of the design, not an outcome metric. Its misclassification as a DV may create confusion about the measurement structure.

### data_acquisition (**critical**)

- **[C]** `## Inputs and Data` — The plan does not specify a verification step confirming the synthetic baseline JSON exists and is intact before the analysis script depends on it. If the file is absent from a fresh worktree, no generation command is provided.
- **[W]** `## Inputs and Data` — d_x is listed as "TBD" at plan authorship time. The pre-flight check resolves it, but downstream analysis steps that depend on it (theoretical prediction, sanity check) are written against an unknown value. If the pre-flight is skipped, no fallback value is documented.
- **[W]** `## Inputs and Data` — The stretch n=50K run output (`merfish_n50k_k15.json`) is listed in the results layout and in `run_profiler.sh`, but labeled "comment out if time-constrained" without a criterion for when to proceed. If skipped, the scale-stability success criterion cannot be evaluated.
- **[I]** `## Inputs and Data` — File validity is asserted via file-size ratios rather than checksums or shape assertions. Binary `.npy` files can have size-consistent corruption (endianness, dtype mismatch) not detectable by size alone.

### reproducibility_spec (**critical**)

- **[C]** `## Controlled Variables` — Software version is specified as "tw_profiler at HEAD" with no commit hash. HEAD is a moving reference; any later re-run silently executes different code with no divergence detection.
- **[C]** `## Controlled Variables` — Rust toolchain is described as "at HEAD" with no channel, version string, or date anchor. Toolchain version affects codegen, SIMD autovectorization, and optimization behavior — all material to timing results.
- **[C]** `## Controlled Variables` — Machine is described only as "Local development machine." CPU model, core count, cache sizes, SIMD extension level, and memory bandwidth are absent. Timing results cannot be interpreted or reproduced without this information.
- **[C]** `## Inputs and Data` — MERFISH fixture files are referenced by path only, with no checksums. An independent party cannot verify byte-identical inputs.
- **[C]** `## Inputs and Data` — The synthetic baseline JSON is referenced by path with no checksum and no indication of the commit or profiler version that produced it. Its provenance is unverifiable.
- **[W]** `## Environment` — Python environment specifies only minor versions (numpy=2.2, scipy=1.15). Without pinned patch versions or a lockfile, the environment is not fully reproducible.
- **[W]** `## Execution Protocol` — The four-step protocol lacks concrete commands, flags, and input file paths. An independent party cannot execute it unambiguously from the plan alone.
- **[W]** `## Execution Protocol` — No intermediate artifact validation steps are defined; no expected value ranges or sanity checks allow silent failure detection.
- **[I]** `## Controlled Variables` — OS identity, kernel version, and background process state are not recorded.

### benchmark_representativeness (**critical**)

- **[C]** `## Inputs and Data` — A single MERFISH dataset from one tissue type (mouse hypothalamus) does not support the claimed generalization to "real biological data" as a category. MERFISH is an atypically structured assay; the conclusion cannot be extended to scRNA-seq, ATAC-seq, or other spatial transcriptomics modalities without additional evidence.
- **[C]** `## Inputs and Data` — n=10K is explicitly acknowledged as potentially understating X-dominance at production scale. The success criterion is evaluated at a scale that may structurally suppress the X-space fraction, making a negative result inconclusive rather than falsifying.
- **[W]** `## Inputs and Data` — A single k=15 is not representative of the full range of production k values. y_heap cost scales as O(n·k) while x_dist is O(n²); at larger k, y_heap fraction grows relative to x_dist, affecting the bottleneck balance.
- **[W]** `## Inputs and Data` — d_x ambiguity (8-20 vs ~48) leaves the actual operating point in the fraction space undefined. If d_x is at the low end (~8-10), the experiment may not represent the high-d_x deployment scenario claimed.
- **[W]** `## Threats to Validity` — The generalization boundary is implicit. The plan claims the result "informs whether to proceed with X-space ANN approximation" without scoping the inference to specific dataset types, n values, or d_x ranges. This leaves the conclusion open to over-application.
- **[W]** `## Success Criteria` — The prior MERFISH experiments (2026-04-05-tw-perf-rerun-clean) used the same dataset, so this experiment does not broaden the generalization boundary — it re-tests the same specific fixture under a different profiling lens.

---

## Adversarial Findings (Red-Team)

*All capped at "warning" per robustness_audit severity ceiling. All require a decision.*

| # | Challenge | Section | Finding |
|---|---|---|---|
| 1 | Goodhart exploitation | `## Analysis Plan` | x_space_pct ≥ 50% can be satisfied by the measurement methodology itself — the metric is fixed by d_x and n (per-row work ratio), not by data geometry. MERFISH could produce identical x_space_pct to Gaussian because the ratio is constructed by d_x, not because the bottleneck is the same. |
| 2 | Data leakage / post-hoc thresholds | `## Data` | The ±5pp null band and ≥50% H1 cutoff were derived after observing the 56.1% synthetic baseline. The thresholds were not pre-registered before seeing baseline data. A ±5pp band centered on a known result is very likely to accommodate expected MERFISH variation, making H0 rejection improbable by design. |
| 3 | Asymmetric tuning | `## Controlled Variables` | The historical baseline was collected with a potentially different warmup count (profiler JSON shows warmup: 5; this plan specifies warmup: 2). At least one prior baseline JSON (`gaussian_n10000_combined.json`) shows all counter values at zero, suggesting the baseline profiling infrastructure may not have been active during that collection. Fresh MERFISH run uses controlled, active profiling while the comparison baseline may have been measured under different or inoperative instrumentation. |
| 4 | Survivorship bias | `## Data` | The MERFISH n=10K fixture is the same fixture used in prior profiling experiments (research/2026-04-05-tw-perf-rerun-clean/). There is no evidence the dataset was chosen blind. If preliminary runs informed the choice to target MERFISH for real-data validation, the selection was not independent of the expected result. |
| 5 | Evaluation collision | `## Controlled Variables` | The profiling instrumentation (`AtomicU64` `fetch_add` per row per step) adds symmetric overhead in proportion to the number of `fetch_add` calls. For shorter per-element kernels (e.g., the flat_simd y_dist batch), the Instant::now()+fetch_add overhead represents a larger fraction of measured step time than for longer per-element kernels (x_dist at high d_x). The instrumentation overhead is asymmetric, inflating x_space_pct relative to uninstrumented wall-clock ratios. |
| 6 | Unfalsifiable mechanism claim | `## Analysis Plan` | The d_x effect analysis ("compare observed x_dist/y_dist ratio to theoretical prediction") has no quantitative expected ratio or tolerance defined. Without a pre-specified match criterion, the d_x analysis is interpretive rather than falsifiable. |
| 7 | Counter accumulation / step timing validity | `## Threats to Validity` | AtomicU64 counters are reset at the top of each `trustworthiness()` call per commit #242 (`fix(profiling): reset AtomicU64 counters at top of trustworthiness()`). The stderr capture file receives one emission per call (after each iteration). If `parse_step_timing` accumulates all emitted lines including warmup, the step_timing JSON contains 7 sample sets (2 warmup + 5 timed), not 5 timed iterations. The mean computed from all 7 is biased by the warmup samples, which may not have reached steady-state timing. |
| 8 | Unrealistic threat distribution (type-specific) | `## Independent Variables` | MERFISH and Gaussian n=10K differ on at least three confounding dimensions simultaneously: (1) data distribution (clustered vs. uniform), (2) intrinsic dimensionality d_x (MERFISH gene expression is high-d; Gaussian is typically low-d), and (3) neighborhood structure. Changing all three at once prevents attribution of any step-fraction difference to any single factor. The "robustness" being tested covers a conflated bundle of changes, not an isolated real-data distribution shift. |

---

## Cannot Assess

The following design properties could not be evaluated from the plan text:

1. **Parallelisation efficiency per step** — The plan does not specify the Rayon thread count, nor does it characterise how work is distributed across threads for each step (x_dist, y_dist, penalty). Without this, it is impossible to assess whether thread-aggregate ns fractions are biased upward or downward relative to wall-clock fractions for each specific step.

2. **Instrumentation overhead per step** — The plan does not quantify the overhead of `Instant::now()` + `AtomicU64::fetch_add()` relative to the per-element compute cost for each step. For steps with short per-element kernels (y_dist flat_simd), this overhead may be material. Without a no-instrumentation baseline, the net effect on step fractions cannot be estimated.

3. **Baseline iteration protocol provenance** — The exact warmup count, iteration count, and profiler feature flags used to produce `profiler_baseline_n10000.json` are not recorded in that file's context visible in the plan. Whether it matches the current plan's controlled variables is unverifiable from the plan alone.

4. **MERFISH fixture provenance and data pipeline** — The plan states d_x is "TBD" and infers it from file size. The actual PCA pipeline that produced the fixture (number of components, preprocessing steps) is not described. Without this, the ecological validity of the d_x value is unassessable at plan-writing time.

---

## Mechanisable Check Log

Binary checks that could be automated in future CI gate:

- [ ] `experiment_type` field present in YAML frontmatter
- [ ] `commit_hash` field present and non-empty in controlled variables
- [ ] `machine_spec` field includes CPU model + core count
- [ ] All input files have SHA-256 checksums recorded
- [ ] `hypothesis_h0` and `hypothesis_h1` both present and non-empty
- [ ] `success_criteria` operationalization is consistent with H0/H1 thresholds
- [ ] `environment.yml` specifies patch-level versions or includes lockfile reference
- [ ] `n_independent_runs` ≥ 3 for timing experiments

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: robustness_audit
critical_count: 22
warning_count: 38
red_team_count: 8
active_dimensions: 14
warning_threshold: 70
```
