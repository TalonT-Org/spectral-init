# Evaluation Dashboard: Trustworthiness Step-Timing Validation on MERFISH Real Data

## Verdict: REVISE

**Experiment type:** exploratory
**Classification confidence:** High — plan explicitly self-identifies as exploratory/descriptive
**Secondary modifiers:** +multi_metric (8 DVs)

**Summary:** The experiment plan is well-structured with extensive revision history addressing 12 prior findings. The exploratory framing is appropriate and honestly applied. However, two critical gaps require revision: (1) the motivation's generalization language exceeds what a single-dataset design can support, and (2) the MERFISH fixture files' distribution mechanism is not documented, creating a reproducibility barrier. 27 warning-level findings identify design awareness gaps that should be acknowledged, primarily around the thread-ns proxy limitation, baseline asymmetries, and minimal replicate count.

---

## Dimension Scorecard

| Dimension | Weight | Level | Findings | Critical | Warning | Info |
|---|---|---|---|---|---|---|
| estimand_clarity | H | L1 | 3 | 0 | 0 | 3 |
| hypothesis_falsifiability | H | L1 | 5 | 0 | 1 | 4 |
| baseline_fairness | M | L2 | 7 | 0 | 4 | 3 |
| causal_structure | L | L2 | 5 | 0 | 1 | 4 |
| unit_interference | M | L2 | 6 | 0 | 3 | 3 |
| red_team | — | RT | 6 | 0 | 0 | 6 |
| error_budget | M | L3 | 5 | 0 | 1 | 4 |
| statistical_corrections | L | L3 | 5 | 0 | 1 | 4 |
| variance_protocol | L | L3 | 5 | 0 | 1 | 4 |
| ecological_validity | M | L4 | 3 | 0 | 2 | 1 |
| benchmark_representativeness | M | L4 | 5 | 1 | 2 | 2 |
| measurement_alignment | M | L4 | 7 | 0 | 4 | 3 |
| reproducibility_spec | M | L4 | 9 | 1 | 4 | 4 |
| data_acquisition | M | L4 | 10 | 0 | 3 | 7 |
| **Total** | | | **81** | **2** | **27** | **52** |

---

## Critical Findings (Required Revisions)

### C1: Generalization language exceeds single-dataset design reach
**Dimension:** benchmark_representativeness | **Section:** ## Inputs and Data | **Level:** L4

The experiment uses exactly one real dataset covering one tissue type (hypothalamus), one spatial transcriptomics assay (MERFISH), and one implicit PCA reduction. The stated generalization boundary correctly limits claims to this dataset, but the framing in the Motivation section asserts that this validates whether the PR #238 conclusion "holds on real biological data" — a broader claim than one dataset supports. A single MERFISH dataset cannot confirm or deny that X-space dominance is a general property of biological data; it can only confirm or deny it for this specific dataset.

### C2: MERFISH fixture distribution mechanism not documented
**Dimension:** reproducibility_spec | **Section:** ## Inputs and Data | **Level:** L4

The MERFISH fixture files are accessed via relative symlinks within the repository but it is not stated whether they are committed to git, stored in git-lfs, or exist only on the original experimenter's local disk. If the files are not committed or otherwise distributed, an independent party has no way to obtain the actual data, making the experiment non-reproducible even with all scripts and environment details captured.

---

## Warning Findings by Dimension

### baseline_fairness (4 warnings)

- **W1:** The Gaussian baseline has only one scale point (n=10K), while MERFISH has two (n=10K and n=50K). Scale-stability analysis can only be performed for MERFISH — any observed scale-related shift cannot be compared against an equivalent Gaussian scale gradient.
- **W2:** The Gaussian baseline fixture has a known, fixed d_x of 10, while MERFISH d_x is unknown and likely much higher (~48). The primary comparison variable (d_x) is confounded with dataset geometry.
- **W3:** The success criteria thresholds (45% and 50%) were calibrated against the observed synthetic Gaussian baseline (~56.1%), creating an asymmetric interpretive framework anchored to the Gaussian reference point.
- **W4:** SIMD kernel differences between Gaussian (d_x=10, AVX2 8-wide) and MERFISH (d_x unknown, possibly different code path) mean thread-ns overcount bias differs between datasets, making x_space_pct comparison not directly symmetric.

### unit_interference (3 warnings)

- **W5:** Sequential run ordering (Gaussian → MERFISH 10K → MERFISH 50K) with no inter-run pause or ordering randomization creates thermal-state bias: Gaussian baseline runs cold while MERFISH runs after thermal loading.
- **W6:** No OS file-cache flushing between replicate invocations. After first invocation loads .npy files, replicates 2-3 read from page cache, making replicate 1 not exchangeable with replicates 2-3 for absolute wall-clock measurements.
- **W7:** Cross-dataset CPU L3 cache contamination between configurations is not acknowledged — the 96 MB V-Cache holds entire n=10K working sets that persist across process boundaries.

### measurement_alignment (4 warnings)

- **W8:** The primary metric (thread-aggregate ns fraction) conflates compute density with wall-clock optimization ROI. Steps with higher SIMD throughput accumulate more thread-ns per wall-clock second.
- **W9:** x_dist uses 8-wide AVX2+FMA while y_dist uses a batched 2-point kernel, causing systematic overweighting of high-SIMD-throughput steps. The magnitude of this bias is unquantified.
- **W10:** The expected outcome is based on FLOP throughput scaling with d_x, but thread-ns captures throughput, not FLOP count — cache reuse and SIMD utilization effects are conflated.
- **W11:** The success criterion ("X-space dominance confirmed" at x_space_pct >= 50%) is tied to the optimization decision but calibrated to thread-ns share, not wall-clock share. The decision criterion is not calibrated to the proxy gap.

### reproducibility_spec (4 warnings)

- **W12:** MERFISH fixture files have no documented provenance — no information about the original biological dataset, preprocessing pipeline, or generation script.
- **W13:** Gaussian fixture generating script is not part of this experiment's artifact set and is not referenced.
- **W14:** No environment pinning mechanism — environment.json records observed versions but does not enforce or restore them for replicators.
- **W15:** RAYON_NUM_THREADS left at system default (16) without specifying that it should be explicitly set for reproducibility across different hardware.

### data_acquisition (3 warnings)

- **W16:** No pre-execution integrity checksums for input fixtures. SHA256 is recorded after runs start, not as a pre-execution gate. A corrupt .npy file would not be detected before profiling.
- **W17:** MERFISH fixture generation pipeline not documented — if git-tracked files were lost, there is no documented path to regenerate them.
- **W18:** The experiment directory does not yet exist. Phase 1 treats symlink creation as implementation, not a hard acquisition gate. If Phase 1 fails silently, profiler scripts fail with unclear errors.

### ecological_validity (2 warnings)

- **W19:** The MERFISH dataset was selected for convenience (only available fixture), not representativeness of typical deployment workloads.
- **W20:** Thread-count sensitivity is not assessed. The compute-share fractions measured at 16 threads may not reflect fractions at different thread counts in deployment.

### benchmark_representativeness (2 warnings)

- **W21:** The null expectation's generalizability is overstated relative to the design's reach — observing no difference with three confounders uncontrolled would only confirm the specific combination, not similarity in general.
- **W22:** Synthetic baseline uses only one parameterization (d_x=10). Varying d_x to match MERFISH dimensionality would isolate dimensionality from geometry effects.

### hypothesis_falsifiability (1 warning)

- **W23:** Post-hoc thresholds (45%, 50%) were chosen after observing the 56.1% baseline, reducing discriminative sharpness. A MERFISH result of 48% would satisfy "not confirmed" by the stated rule yet remain ambiguous.

### error_budget (1 warning)

- **W24:** Post-observation threshold selection inflates effective Type I error. The practice is acknowledged honestly but the report should carry explicit caveats to avoid overclaiming.

### statistical_corrections (1 warning)

- **W25:** No priority ordering among the 8 DVs for verdict determination. Post-hoc emphasis on whichever metric produces the clearest result is a latent researcher-degrees-of-freedom risk.

### causal_structure (1 warning)

- **W26:** Secondary analysis attributes n-scaling shifts in step fractions to specific mechanisms (cache effects, scheduling overhead) without controlled manipulation — the design supports detecting shifts but not attributing them.

### variance_protocol (1 warning)

- **W27:** R=3 is a minimal replicate count. With t*=4.30 at df=2, 95% CIs will be wide. Expected CI width and minimum detectable difference are not quantified.

---

## Adversarial Findings (Red-Team)

All red-team findings are capped at `info` severity for exploratory experiment type. Each requires an explicit decision.

| # | Challenge | Finding | requires_decision |
|---|---|---|---|
| RT-1 | Goodhart exploitation | Success criteria are structural (files produced, CIs computed) rather than epistemic. An experimenter can satisfy all criteria regardless of whether step-timing measurements are meaningful proxies for optimization ROI. No criterion requires uncertainty bounds to be small enough to distinguish outcomes. | true |
| RT-2 | Data leakage | The "fresh" Gaussian baseline re-instantiates the same distribution (seed=2026, d_x=10, n=10K) from PR #238, which produced the 56.1% x_space_pct that interpretive guideposts derive from. The baseline is not independent of the prior result. | true |
| RT-3 | Asymmetric tuning | warmup=2 and iters=5 were not validated against MERFISH's variance profile. The historical baseline used warmup=5, iters=30. If MERFISH exhibits higher variance, R=3 x iters=5 yields wider CIs for MERFISH, making Gaussian look more decisive. | true |
| RT-4 | Survivorship bias | The plan provides no pre-specified decision rule for the inconclusive case. An experimenter could run additional replicates selectively for MERFISH until CIs narrow to cross the desired guidepost. Maximum replicate count is not capped. | true |
| RT-5 | Evaluation collision | The profiler binary is both subject and instrument. If MERFISH d_x is substantially higher, the x_dist SIMD loop grows in instruction count relative to y_dist, making cache pressure asymmetry non-proportional and biasing step fractions. | true |
| RT-6 | HARKing vulnerability | Directional expected outcome stated before data collection, but d_x is unknown at plan time. Interpretive guideposts (45%/50%) were selected post-observation of the 56.1% synthetic baseline. The exploratory label does not prevent presenting findings as "consistent with expectation" when the expectation was formed after observing the most relevant prior data. | true |

---

## Cannot Assess

1. **Resource proportionality:** No per-step resource budgets stated — cannot assess whether the compute investment (20-80 minutes) is proportional to the expected information gain from each dataset configuration.
2. **Run ordering randomization:** No randomization mechanism described for the execution order of dataset configurations and replicates — cannot assess whether sequential ordering introduces systematic bias beyond what between-run variance captures.
3. **MERFISH PCA reduction parameters:** The PCA reduction that produced the MERFISH fixtures is not characterized — cannot assess whether the dimensionality reduction itself introduces artifacts relevant to step-timing fractions.

---

## Mechanizable Check Log

| Check | Status | Notes |
|---|---|---|
| Plan has hypothesis section | PASS | Research question clearly stated |
| Plan has success criteria | PASS | Three outcome categories defined |
| Plan has controlled variables table | PASS | 8 variables controlled |
| Plan has threats to validity | PASS | Internal (5) and external (4) threats listed |
| Plan has execution protocol | PASS | Full protocol with commands |
| All input files referenced with paths | PASS | 6 fixtures with full paths |
| All DVs have collection method | PASS | 8 DVs with methods specified |
| Warmup/iter counts specified | PASS | warmup=2, iters=5, R=3 |
| Experiment type self-identified | PASS | "exploratory/descriptive" |
| Post-hoc thresholds acknowledged | PASS | Explicit note about 45%/50% being post-observation |
| Fixture checksums pre-specified | FAIL | Checksums recorded post-execution only |
| Data distribution mechanism stated | FAIL | Git-tracked vs LFS vs local-only not stated |
| Multiplicity awareness documented | FAIL | 8 DVs with no acknowledgment of family-wise error |
| Replicate count justified | FAIL | R=3 chosen without sensitivity analysis |

---

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: exploratory
critical_count: 2
warning_count: 27
red_team_count: 6
active_dimensions: 14
warning_threshold: 70
```
