# Evaluation Dashboard: Trustworthiness Step-Timing Validation on MERFISH Data

## Verdict: REVISE

| Field | Value |
|-------|-------|
| Experiment type | benchmark |
| Classification confidence | High (Rule 1: IVs are dataset names, DVs are performance metrics, multiple comparators) |
| Secondary modifiers | +multi_metric (8 DVs) |
| Verdict trigger | 3 critical findings in baseline_fairness and unit_interference dimensions |

## Dimension Scorecard

| Dimension | Level | Weight | Findings | Severity Summary |
|-----------|-------|--------|----------|------------------|
| estimand_clarity | L1 | H | 2 | 2 warning |
| hypothesis_falsifiability | L1 | H | 2 | 2 warning |
| baseline_fairness | L2 | H | 3 | 2 critical, 1 warning |
| unit_interference | L2 | H | 3 | 1 critical, 2 warning |
| error_budget | L3 | H | 2 | 2 warning |
| statistical_corrections | L3 | H | 3 | 3 info |
| variance_protocol | L3 | H | 3 | 3 warning |
| benchmark_representativeness | L4 | M | 2 | 2 info |
| ecological_validity | L4 | M | 2 | 2 warning |
| measurement_alignment | L4 | M | 2 | 2 warning |
| reproducibility_spec | L4 | M | 2 | 2 warning |
| data_acquisition | L4 | M | 1 | 1 info |
| red_team | Adv | — | 5 | 5 warning |

**SILENT dimensions (not spawned):** causal_structure (S for benchmark)

## Level 1: Estimand & Falsifiability

**estimand_clarity** (2 warnings):

1. **[warning]** The primary estimand conflates two distinct contrasts without separating them into formal comparisons. H0/H1 test MERFISH 10K against a 50% threshold (a one-sample contrast), but the Analysis Plan also specifies a two-sample contrast (MERFISH 10K vs Gaussian 10K on x_space_pct). These are different estimands with different inferential structures; only the threshold contrast is formally stated as a hypothesis.

2. **[warning]** The comparison between MERFISH and Gaussian conditions does not constitute a clean formal contrast because the two datasets differ on both dimensionality (d_x=50 vs d_x=10) and data geometry simultaneously. The contrast is ambiguous about which independent variable is the claimed cause of any observed difference.

**hypothesis_falsifiability** (2 warnings):

3. **[warning]** The conclusive-negative band (x_space_pct < 50% with CI UB < 55%) and the inconclusive band (CI spans 50%) overlap with the post-hoc threshold acknowledgment. A result in the narrow corridor near 50% could be read as either conclusive-negative or inconclusive depending on rounding and interpretation, leaving H0 acceptance ambiguous. *(requires_decision)*

4. **[warning]** The 50% threshold was chosen after observing the historical flat_simd result of 56.2%, creating no falsification standard independent of the data that motivated the experiment. If the result falls near 50%, the author retains discretion to reframe what constitutes H0 support. *(requires_decision)*

## Level 2: Baseline Fairness & Unit Interference

**baseline_fairness** (2 critical, 1 warning):

5. **[CRITICAL]** The Gaussian baseline is generated in-session from a fixed seed while MERFISH data is pre-existing git-committed fixture data. These two provenance paths are asymmetric: the Gaussian dataset is subject to any in-session environmental effects (RNG library version, generation overhead, memory state) that the MERFISH dataset is not exposed to.

6. **[CRITICAL]** The two primary compared systems differ on two confounded dimensions simultaneously: input dimensionality (d_x=10 vs d_x=50) and data geometry (synthetic Gaussian vs real biological MERFISH distribution). Step-timing fraction comparisons cannot be attributed to either factor independently, making the comparison structurally unfair as a basis for conclusions about either variable.

7. **[warning]** The historical flat_simd Gaussian reference originates from a prior experiment on a potentially different hardware session. Comparing current measurements against this reference introduces an asymmetry where the reference system had different resource availability than the current systems.

**unit_interference** (1 critical, 2 warnings):

8. **[CRITICAL]** The three dataset runs share OS page-cache state across their boundaries. The warm-cache state produced by earlier datasets alters the I/O latency profile of later datasets. The interference is asymmetric and non-recoverable within a single sequential pass.

9. **[warning]** Thermal accumulation across the sequential run order means MERFISH 50K executes under a different sustained CPU/DRAM thermal envelope than Gaussian 10K, potentially triggering frequency throttling.

10. **[warning]** Replicate 1 of each dataset carries state from the preceding dataset's cache-warming while simultaneously serving as first contact with the new dataset's working set, creating a hybrid measurement not representative of either cold or fully warm state.

## Level 3: Statistical Rigor

**error_budget** (2 warnings):

11. **[warning]** No formal power analysis is present. With R=5 and df=4, the study's ability to detect true differences in x_space_pct and the four step-fraction DVs is unquantified.

12. **[warning]** Type I and Type II error rates are not acknowledged. With 8 DVs evaluated and no multiplicity correction, the family-wise Type I error rate is inflated beyond the nominal per-comparison level.

**statistical_corrections** (3 info):

13. **[info]** No formal multiple comparisons correction is pre-specified despite 8 DVs. The joint-pattern language mitigates overclaiming in the narrative, but the formal significance markers carry an unquantified inflated error rate.

14. **[info]** DV priority ordering is defined, but specific thresholds at which secondary DVs would independently trigger a conclusion are not stated before data collection.

15. **[info]** Conservative ratio bounds are not formal 95% CIs but are presented using the same visual convention.

**variance_protocol** (3 warnings):

16. **[warning]** The profiler binary has no documented inter-run seed state. Although trustworthiness() is deterministic given fixed inputs, Rayon's thread-pool scheduling and implicit OS-level PRNG state are not confirmed as seeded or reset between R=5 runs.

17. **[warning]** Execution order is fixed and never randomized across replicates. Any systematic position effect (cache warm, CPU frequency scaling, memory pressure) is fully confounded with dataset identity across all runs.

18. **[warning]** With df=4 (R=5), the t-interval is wide and the design has limited power to detect inconclusive variance patterns. The hard cap (no extension) removes the ability to reduce uncertainty post-hoc.

## Level 4: External Validity & Reproducibility

**benchmark_representativeness** (2 info):

19. **[info]** The synthetic baseline (one Gaussian, d_x=10) does not cover structured or clustered synthetic geometries that would stress-test timing variance across graph topologies.

20. **[info]** Single k=15 means step-timing fractions may not hold at k values that shift the graph density regime.

**ecological_validity** (2 warnings):

21. **[warning]** Step timings are thread-summed CPU nanoseconds, not wall-clock time. In production, the optimization decision is whether wall-clock latency improves. Thread-summed CPU time can diverge substantially from wall-clock at high parallelism.

22. **[warning]** Speedup conclusions at n=10K may not apply at production-scale biological datasets (n=100K-500K), which are an order of magnitude larger.

**measurement_alignment** (2 warnings):

23. **[warning]** x_space_pct is a share-of-thread-nanoseconds metric, but the research question asks whether X-space dominates runtime (wall-clock). If x_dist and x_sort are more parallelized than y_dist and penalty, x_space_pct will overstate or understate the wall-clock share by a factor proportional to the parallelism ratio — independent of the SIMD asymmetry the plan acknowledges.

24. **[warning]** At d_x=50 with AVX2 processing 8 f64/cycle vs y_dist at ~4 f64/cycle, x_dist accumulates ~2x as many thread-ns per unit wall-time. x_space_pct=70% may correspond to a materially lower actual wall-clock share, making the metric a biased proxy for the quantity that drives the investment decision.

**reproducibility_spec** (2 warnings):

25. **[warning]** The hardware is described only as a "16-core system" with no CPU model, microarchitecture, or ISA feature set. The binary uses `target-cpu=native` and runtime AVX2 dispatch, making step-timing results dependent on specific CPU capabilities that are undocumented.

26. **[warning]** No environment.yml or installable environment specification was created. Python dependencies are documented as prose version pins only with no lockfile or channel specification.

**data_acquisition** (1 info):

27. **[info]** The plan names the Gaussian generation script as `gen_gaussian_baseline.py`, but the committed script in the prior experiment directory is `gen_synthetic.py`. The manifest entry references a non-existent artifact, though the Gaussian files themselves are present.

## Adversarial Findings (Red-Team)

All red-team findings set `requires_decision: true`. Severity capped at warning for benchmark type.

28. **[warning] Goodhart/Evaluation Collision** *(requires_decision)*: The AtomicU64 thread-summed ns proxy conflates wall-clock time with FLOPs-per-ns in a way that systematically compresses x_dist's reported share relative to y_dist on AVX2 hardware. At d_x=50, the proxy is structurally incapable of cleanly answering whether x-space ops are truly >=50% of computational burden. The 50% threshold test operates on a measurement biased in the direction of rejecting H1.

29. **[warning] Asymmetric Tuning / Confound** *(requires_decision)*: The two datasets differ on three simultaneous axes: dimensionality (50 vs 10), neighborhood geometry, and k-NN graph sparsity pattern. Prior research found MERFISH runs 2x slower with 3x wider CIs than Gaussian at n=50K. The 2x wall-time difference is unexplained geometry or sparsity, not dimensionality alone, yet the analysis attributes step fraction differences to dimensionality.

30. **[warning] Data Leakage / Historical Contamination** *(requires_decision)*: The 56.2% historical baseline was measured pre-PR#242 (which reset AtomicU64 counters). Any divergence from 56.2% could be explained by the counter-reset code change rather than a true change in step distribution. The historical comparison is not interpretable as a replication.

31. **[warning] Evaluation Collision** *(requires_decision)*: The profiling instrumentation (AtomicU64 fetch_add inside the Rayon parallel loop) adds millions of atomic operations during the measurement window. Probe-induced perturbation is larger as a fraction of step time for shorter steps (x_sort, penalty) and smaller for the dominant step, systematically compressing fast-step fractions and inflating slow-step fractions.

32. **[warning] Survivorship / Sample Size** *(requires_decision)*: With only 5 timed iterations (df=4), the sample size is calibrated for Gaussian variance but MERFISH produces ~3x wider CIs. At n=50K the design will almost certainly produce an "inconclusive" result on MERFISH regardless of the true step distribution, because the sample size is inadequate for the MERFISH variance profile.

## Cannot Assess

1. **CPU frequency scaling and boost behavior** — No documentation of whether turbo boost, frequency governors, or power management are controlled. Cannot assess whether thermal throttling artifacts are present in step timing measurements.

2. **Inter-session measurement stability** — No protocol for running the experiment across separate sessions. Cannot assess whether the single-session results are stable across reboots, OS updates, or background process variation.

3. **Rayon work-stealing scheduling determinism** — No documentation of whether Rayon's work-stealing scheduler produces deterministic thread-to-core mapping. Cannot assess whether thread scheduling variance contributes to step-timing variance.

## Mechanizable Check Log

| Check | Automatable | Status |
|-------|-------------|--------|
| YAML frontmatter present | Yes | FAIL — no frontmatter |
| Hypothesis H0/H1 present | Yes | PASS |
| Success criteria defined | Yes | PASS |
| Controlled variables table present | Yes | PASS |
| Resource estimates present | Yes | PASS |
| Threats to validity section present | Yes | PASS |
| Data paths resolve to existing files | Yes | PASS (git-committed) |
| Execution order documented | Yes | PASS |
| Python dependency versions stated | Yes | PASS |
| Environment lockfile present | Yes | FAIL — prose only |
| Hardware specification complete | Partial | FAIL — "16-core system" only |
| Multiple comparisons correction stated | Yes | FAIL — none specified |
| Power analysis present | Yes | FAIL — none present |

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 3
warning_count: 23
red_team_count: 5
active_dimensions: 13
warning_threshold: 65
```
