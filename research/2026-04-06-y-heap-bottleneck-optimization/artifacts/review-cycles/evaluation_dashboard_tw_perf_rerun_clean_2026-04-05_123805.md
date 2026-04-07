# Experiment Design Review: `tw_perf_rerun_clean`

**Plan file:** `.autoskillit/temp/plan-experiment/experiment_plan_tw_perf_rerun_clean_2026-04-05_122948.md`
**Review timestamp:** 2026-04-05 12:38:05
**Reviewer:** review-design skill (multi-level parallel analysis)

---

## ⚠️ VERDICT: REVISE

**Experiment type:** `benchmark`  
**Active secondary modifiers:** `+multi_metric` (5 DVs → statistical_corrections weight H), `+deployment` (production-scale claim)

The plan contains **20 critical findings** and **26 warnings** across 10 dimensions. No L1 stop triggers were raised (estimand_clarity and hypothesis_falsifiability findings are capped at `warning` for benchmark type). However, the volume and depth of structural defects — including an IV-to-script variant mismatch, unimplemented benchmark isolation, an undefined correction family, and three measurement validity failures — require resolution before execution.

---

## Classification Summary

| Field | Value | Source |
|-------|-------|--------|
| experiment_type | benchmark | L1 triage (Rule 1: IVs are method names, DVs are performance metrics, multiple comparators) |
| hypothesis_h0/h1 | Present for all 4 hypotheses | Prose |
| estimand | Partially formal — 3/4 hypotheses have A-vs-B structure; H0/H1-clean is a one-sample threshold test | L1 agent |
| metrics | 5 DVs: wall_clock_speedup, delta_tw, criterion_speedup_100k, per_step_fraction, partial_rank_ci_half_width | Prose |
| baselines | baseline variant; Gaussian CI at n=10K as H-partial-MERFISH comparator | Prose |
| statistical_plan | Holm-Bonferroni, α=0.05, m=4 (disputed — see Dimension 5); power analysis documented at sample_size=100 | Prose |
| success_criteria | PASS/FAIL/INCONCLUSIVE table for all 4 hypotheses | Prose |

---

## Dimension Scorecard

| Dimension | Weight | Findings | Severity Summary |
|-----------|--------|----------|-----------------|
| estimand_clarity | H (L1) | 4 | 2 warning, 2 info |
| hypothesis_falsifiability | H (L1) | 4 | 4 warning (1 downgraded from critical per calibration rubric) |
| baseline_fairness | H (L2) | 4 | 2 critical, 1 warning, 1 info |
| unit_interference | H (L2) | 5 | 2 critical, 2 warning, 1 info |
| red_team | — | 6 | 6 warning (2 downgraded from critical per benchmark cap) |
| error_budget | H (L3) | 7 | 2 critical, 4 warning, 1 info |
| statistical_corrections | H (L3, +multi_metric) | 6 | 3 critical, 2 warning, 1 info |
| variance_protocol | H (L3) | 7 | 2 critical, 3 warning, 2 info |
| data_acquisition | M (L4) | 6 | 3 critical, 2 warning, 1 info |
| reproducibility_spec | M (L4) | 7 | 3 critical, 3 warning, 1 info |
| ecological_validity | M (L4, +deployment) | 4 | 1 critical, 2 warning, 1 info |
| measurement_alignment | M (L4) | 4 | 3 critical, 1 warning |
| benchmark_representativeness | M (L4) | 5 | 2 critical, 2 warning, 1 info |
| causal_structure | **S** | — | Not spawned (benchmark type) |

---

## Level 1 Findings (Fail-Fast Gate) — PASSED

### estimand_clarity

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 1 | warning | ## Hypotheses | H-100K has an inverted null/alternative framing: H0 encodes the desired-outcome condition (CI entirely above 1.5×) and H1 encodes failure. A formal contrast is recoverable, but the direction is stated backwards from convention, risking misclassification of a conclusive positive as null retention. |
| 2 | warning | ## Hypotheses | H0/H1-clean lacks a treatment comparator. The estimand tuple is incomplete: outcome (tw_x_dist fraction) is clear, but there is no A-vs-B structure — it is a one-sample threshold test. If the intent is baseline-only profiling, that must be stated explicitly. |
| 3 | info | ## Hypotheses | H5 has a complete formal contrast: (approx[m=5000] vs exact, outcomes=speedup ratio and \|delta\|, population=MERFISH n=10K, conjunctive threshold). |
| 4 | info | ## Hypotheses | H-partial-MERFISH has a clean formal contrast: (partial_rank on MERFISH PCA-50 vs Gaussian, outcome=CI half-width, population=n=10K). |

### hypothesis_falsifiability

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 5 | warning | ## Hypotheses | H5: H0 requires BOTH conditions to fail ("neither ≥5× NOR \|delta\| < 0.001"), but the success criteria table treats failure of EITHER as conclusive negative. The observable result that retains H0 is ambiguous between the two formulations. |
| 6 | warning | ## Hypotheses | H-100K: H0 is the desired/positive outcome (speedup confirmed), inverting the standard skeptical null. The author would celebrate H0 retention, not treat it as a null supported against the alternative. Recommend inverting H0/H1 to restore conventional falsifiability framing. *[Calibration: was critical from agent; capped to warning for benchmark type.]* |
| 7 | warning | ## Hypotheses | H0/H1-clean: the 40%–50% zone is inconclusive per success criteria but formally retains H0 (<50%). The boundary between null retention and inconclusiveness is undefined in the hypothesis itself. |
| 8 | warning | ## Hypotheses | H-partial-MERFISH: p > 0.05 does not confirm H0 (absence of evidence ≠ evidence of absence). No positive criterion for H0 confirmation is operationalized. |

**L1 gate: PASSED** — No critical findings after calibration. Proceeding to L2–L4.

---

## Level 2 Findings

### baseline_fairness

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 9 | **critical** | ## Independent Variables | The IV table declares 5 variants (baseline, thread_local, partial_rank, avx2_kernel, combined) as the treatment space, but Phase 4's `run_criterion_clean.sh` invokes only 3 bench binaries (tw_baseline_bench, tw_partial_rank_bench, tw_combined_bench). thread_local and avx2_kernel receive zero Criterion measurement at n=100K. |
| 10 | **critical** | ## H-100K analysis | The statistical plan specifies Holm-Bonferroni for m=4 pairwise comparisons (baseline vs each of thread_local, partial_rank, avx2_kernel, combined), but Phase 4 collects data for only 2 of those 4 pairs (partial_rank, combined). Running m=4 Holm correction with data for m_actual < 4 inflates the correction without empirical backing. |
| 11 | warning | ## Phase 4 script | thread_local and avx2_kernel variants receive no warm_up_time(30s), no sample_size=100 measurement cycle, and no controlled measurement. Any cross-variant comparison that includes them is unsupported. |
| 12 | info | ## Independent Variables | Resolution options: (a) add tw_thread_local_bench and tw_avx2_bench to Phase 4 and keep m=4, or (b) remove those variants from IVs and reduce to m=2. One choice must be made before execution. |

### unit_interference

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 13 | **critical** | ## Phase 4 script | Phase 4 runs all three bench binaries in a single `cargo criterion` invocation (`--bench tw_baseline_bench --bench tw_partial_rank_bench --bench tw_combined_bench`) with no pauses. This directly contradicts the stated mitigation (separate invocations with 1-minute pauses) and invalidates the isolation that separate bench binaries are intended to provide. |
| 14 | **critical** | ## Phase 5 (profiling) | Phase 5 runs all 5 variants sequentially in a single bash loop with no thermal or cache recovery interval. CPU frequency scaling, branch predictor state, and cache occupancy from earlier variants contaminate later measurements. |
| 15 | warning | ## Infrastructure Changes | CPU frequency scaling (Turbo Boost, Precision Boost) is not controlled for the benchmark session. The plan does not require setting a `performance` CPU governor, leaving per-variant speedups subject to transient frequency state. |
| 16 | warning | ## Threats to Internal Validity | The 1-minute pause mitigation is documented in Threats but absent from every execution script. A documented mitigation that is not implemented provides false assurance. |
| 17 | info | ## Infrastructure Changes | AVX2 variant has elevated power draw characteristics; if run before scalar variants, thermal throttling may suppress subsequent measurements. Consider placing AVX2 last in run order. |

---

## Level 3 Findings

### error_budget

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 18 | **critical** | ## Power analysis | The n_samples formula uses α/m = 0.0125 as the effective α for all four comparisons, but under Holm-Bonferroni the first test is the most conservative (α/m), while subsequent tests use α/(m−1), α/(m−2), and α. Sizing all comparisons at α/m over-corrects and produces a single n that applies only to the first (hardest) position. The plan should state which hypothesis the n=141 sizes and whether the others are powered at their respective Holm thresholds. |
| 19 | **critical** | ## H-partial-MERFISH | Holm-Bonferroni correction for H-partial-MERFISH is stated conditionally ("if this test is part of the multi-comparison family"). Family membership must be pre-specified before data collection. If included in m=4, the power analysis must account for it; if excluded, a separate α and power statement is required. As written, the inclusion decision is deferred to analysis time. |
| 20 | warning | ## Power analysis | The plan adopts sample_size=100 which achieves 80% power at r≈11.6%, not the stated r=10% target (power≈69% at r=10%). β at the primary effect size of interest is never stated. Explicitly declare: "At sample_size=100 and r=10%, power≈69% (β≈0.31)." |
| 21 | warning | ## H5 Analysis | H5 is declared as requiring no statistical test, but no rationale is given for why a single measurement suffices. Document whether H5 is exempt from error budgeting by design (deterministic binary criterion) or whether measurement variance could produce false verdicts. |
| 22 | warning | ## H0/H1-clean Analysis | 5 profiling iterations with means reported and no statistical test. No Type I/II error acknowledgment. With n=5 there is no meaningful power; either justify why 5 iterations are sufficient for a deterministic system, or declare this sub-analysis informal/exploratory. |
| 23 | warning | ## Success Criteria (H5) | The "inconclusive" zone ("both conditions fail by < 10% margin") has no stated power to detect the 10% margin. Without this, inconclusiveness cannot be distinguished from underpoweredness. |
| 24 | info | ## Statistical α | A consolidated error budget table (α, β, effect size, n, power) per hypothesis is absent, making the overall error accounting non-auditable as a unit. |

### statistical_corrections

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 25 | **critical** | ## Statistical α | The correction family is never explicitly defined across all 5 DVs. The global α statement says "Holm-Bonferroni for m=4 comparisons" but does not specify which DVs are in the family vs. exempt, nor whether FWER is controlled across all DVs or only within H-100K pairwise comparisons. |
| 26 | **critical** | ## H-100K analysis | H-100K specifies m=4 pairwise comparisons but Phase 4 collects data for only 2–3 variants (per baseline_fairness finding #9). Applying m=4 Holm correction to fewer actual tests inflates conservatism and makes the pre-registration inconsistent with execution. |
| 27 | **critical** | ## H-partial-MERFISH analysis | The correction for `partial_rank_ci_half_width` is conditional ("if part of the multi-comparison family"), deferring the inclusion decision to analysis time. This is not a pre-specification; it creates researcher degrees of freedom to exclude a marginal result from the correction family post-hoc. |
| 28 | warning | ## H5 analysis | wall_clock_speedup and delta_tw (both H5) constitute an implicit two-metric joint family. If either can be used post-hoc to support H5, no correction is an under-specified choice that should be explicitly justified. |
| 29 | warning | ## Dependent Variables | No single document enumerates the complete correction family for all 5 DVs. A consolidated correction table (DV → in family? → adjusted α → test type) is absent. |
| 30 | info | ## H0/H1-clean analysis | Threshold-based per_step_fraction applies no correction, but if the threshold result gates execution of other statistical tests, that gating logic is an undeclared multiplicity source. |

### variance_protocol

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 31 | **critical** | ## Phase 5 (step timing) | 5 profiling iterations with means only and no dispersion metrics (SD, IQR, CI). At 5 samples, a single outlier shifts the mean fraction by several percentage points, potentially crossing the 50% H1 threshold. Increase to ≥30 iterations and require CI lower bound (not mean) > 50% for H1 confirmation. |
| 32 | **critical** | ## Phase 5 (step timing) | No random seed is specified for tw_profiler. If any internal RNG-dependent path is non-deterministic, results are not reproducible across runs or machines. |
| 33 | warning | ## Phase 3 (H5) | "Near threshold" criterion for invoking the 5-seed sensitivity check is undefined. This is a post-hoc decision rule: the analyst observes the single result before deciding whether to run more seeds. Pre-register a concrete trigger interval (e.g., result within ±10% of threshold). |
| 34 | warning | ## Phase 3 (H5) | When the 5-seed path triggers, the plan says "report the distribution" but does not specify the summary statistic for the confirmatory decision (mean, median, worst-case). Aggregation rule must be pre-registered before execution. |
| 35 | warning | ## H-partial-MERFISH | Criterion's bootstrap CI is deterministic only if its internal RNG is seeded. Without a fixed bootstrap seed, exact CI bounds cannot be reproduced across re-runs. Set Criterion's bootstrap seed via BenchmarkConfig or document accepted variability. |
| 36 | info | ## Phase 4 | Phase 4 Criterion invocations also lack a pinned bootstrap seed, lower severity because sample_size=100 provides more stability. |
| 37 | info | ## Power analysis | CV=15% is assumed for the power calculation but its empirical source is not cited. Document provenance (prior run, literature, or pilot data). |

---

## Level 4 Findings

### data_acquisition

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 38 | **critical** | ## Existing Gaussian data | Pre-generated Gaussian n=1K–50K data at `data/gaussian/` (used as H-partial-MERFISH Gaussian baseline) has no acquisition or generation step in any phase. In a fresh worktree these files are absent; no gen_synthetic.py invocation or copy step is specified for these sizes. |
| 39 | **critical** | ## MERFISH source files | The MERFISH 100K NPZ files (`temp/merfish_100k/*.npz`) are gitignored and local-only. No acquisition command (download URL, copy-from-archive, or regeneration procedure) is provided. Any fresh worktree run of prepare_merfish.py will fail silently. |
| 40 | **critical** | ## prepare_data.sh | prepare_data.sh invokes prepare_merfish.py whose upstream inputs (the NPZ files above) have no acquisition step. The script will fail before producing merfish_n10k_x.npy, blocking H5 and H-partial-MERFISH. |
| 41 | warning | ## Phase 5 data generation | gen_synthetic.py may lack `--d` flag support for non-default dimensionality. The plan acknowledges this but provides no resolution step. If unsupported, Gaussian n=100K d=50 data is never generated, blocking H0/H1-clean and H-100K. |
| 42 | warning | ## H-partial-MERFISH baseline | The Gaussian CI baseline at n=10K is described as coming from "the new isolated bench run OR the existing bench_output.txt." No authoritative-source selection rule is given. If the prior file is absent, the fallback is the new run, but there is no verification criterion confirming both sources are numerically equivalent. |
| 43 | info | ## Data sources | Dependency graph for data acquisition is implicit. Recommend explicit ordering: raw NPZ → prepare_merfish.py → merfish_n10k_*.npy → MERFISH bench; gen_synthetic.py → Gaussian data → profiling and Criterion. |

### reproducibility_spec

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 44 | **critical** | ## Data dependencies | MERFISH source data at `temp/merfish_100k/*.npz` is gitignored with no download script, checksum, or archival reference. An independent reproducer cannot obtain the exact inputs; benchmark results are unverifiable against the primary hypothesis inputs. |
| 45 | **critical** | ## Data dependencies | `data/gaussian/` (n=1K–50K) is described as pre-present with no generation script or fixed seed documented. If generated stochastically, the exact matrices cannot be reconstructed. |
| 46 | **critical** | ## Infrastructure changes required (Phase 1) | Phase 1 requires source code edits (new Cargo features, bench binaries, instrumentation in src/metrics.rs) described only in prose. An independent reproducer must reverse-engineer all edits without a patch file or script, introducing divergence risk. |
| 47 | warning | ## Environment | statsmodels is bounded `>=0.14` but not pinned to an exact version. Version drift allowed; pin to exact version or provide a lockfile. |
| 48 | warning | ## Environment | rust-toolchain.toml is created as part of Phase 1a but its committed location and contents should be explicit in the plan. Without it, reproducers must manually install the exact nightly channel. |
| 49 | warning | ## Deployment context | Hardware profile states AVX2/FMA (x86-64-v3) but OS version, CPU model, core count, RAM, and NUMA topology are not documented. Micro-benchmark results are sensitive to these and cannot be reproduced on different hardware without a baseline. |
| 50 | info | ## Deployment context | The research worktree branch HEAD commit is not tagged or archived. If the worktree is dropped before results are published, the exact code state is lost. |

### ecological_validity

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 51 | **critical** | ## Deployment context | All benchmarks run on nightly-2026-03-26 (rustc 1.96.0-nightly). Production deployments use stable Rust. Nightly can apply unstable MIR/LLVM optimizations absent in stable, making measured speedups non-transferable to the production binary. |
| 52 | warning | ## Deployment context | The production workload is stated as n=100K but is likely MERFISH-like data. H-100K and H0/H1-clean use synthetic Gaussian (d=50), whose sparsity, co-variance, and numeric range differ substantially from real biological data. Gaussian speedups may not represent MERFISH speedups. |
| 53 | warning | ## Infrastructure changes (Phase 1) | Profiling instrumentation (thread-local RefCell, step-timing snapshots) alters binary layout, inlining decisions, and branch prediction relative to uninstrumented production builds. Measurements should be validated against a clean production build to confirm overhead is negligible. |
| 54 | info | ## Environment | AVX2/FMA runtime dispatch matches the stated x86-64-v3 production hardware profile — a positive alignment. Confirm CI/CD and cloud deployment targets also guarantee x86-64-v3 to avoid silent scalar fallback. |

### measurement_alignment

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 55 | **critical** | ## Metrics | `wall_clock_speedup` uses wall time for the H5 single-trial measurement. Wall time conflates OS scheduling jitter, memory pressure, and thermal state with algorithmic speedup. A single-run ratio can plausibly vary by ±20% on a loaded system, making the 5× pass/fail threshold unreliable. Use CPU time (CLOCK_PROCESS_CPUTIME_ID) or run ≥5 trials and evaluate against the CI. |
| 56 | **critical** | ## Metrics | `criterion_speedup_100k` derives the speedup ratio as `baseline_mean_ns / variant_mean_ns` from two independent Criterion runs. Criterion reports within-run variance of each arm independently; it does not produce a ratio CI. The two arms are measured in different invocations with correlated OS noise, so a simple ratio of means has no valid inferential CI. The ratio CI must be derived from a bootstrap of the paired timing samples. |
| 57 | **critical** | ## Measurement concerns | `per_step_fraction` from 5 iterations with non-uniform thread-local RefCell overhead per step is doubly invalid: (a) 5 samples provides no statistical stability, and (b) the acknowledged non-uniform overhead biases fractions against steps with more instrumentation points, potentially misidentifying the dominant step. |
| 58 | warning | ## Metrics | `delta_tw = t_approx − t_exact` is a point estimate with no CI. Trustworthiness at n=10K has estimator variance; a delta of 0.0009 may be within the noise floor of the statistic itself. Report a CI on delta_tw (bootstrap over subsampled rows or repeated seeds) to make the \|delta\| < 0.001 threshold meaningful. |

### benchmark_representativeness

| # | Severity | Section | Message |
|---|----------|---------|---------|
| 59 | **critical** | ## Generalizability | H-100K speedup results are tied to nightly-2026-03-26 and AVX2/FMA (x86-64-v3). SIMD speedups depend on vectorization width and are not portable to ARM (NEON/SVE), non-AVX2 x86, or future microarchitectures. Scope the shipped claim explicitly to this build target. |
| 60 | **critical** | ## Generalizability | H0/H1-clean uses isotropic Gaussian (d=50) profiling to identify the optimization target for a MERFISH production workload. Gaussian step costs (especially distance computation in uniform metric space) may not characterize MERFISH k-NN graph step costs (high dynamic range, gene dropout sparsity). Using Gaussian profiling to select a MERFISH optimization target is an invalid inference chain. |
| 61 | warning | ## Generalizability | H-partial-MERFISH uses one real dataset (MERFISH) vs one synthetic dataset. With n=2 conditions the experiment cannot distinguish distribution-dependent effects from MERFISH-specific artifacts (cluster imbalance, dropout). A second real dataset would be needed for a general claim. |
| 62 | warning | ## Generalizability | All benchmarks use nightly-2026-03-26. Nightly compiler changes can shift speedup ratios across releases; production use of stable Rust is unverified. Scope the representative claim to this build configuration or add a stable-channel comparison. |
| 63 | info | ## Generalizability | H5 quality claim (|delta| < 0.001 at n=10K, m=5000) does not bound delta at production scale (n=100K). Document that the quality guarantee is valid only at the tested (n, m) combination. |

---

## Adversarial Findings (Red-Team)

All red-team findings have `requires_decision: true`. Severity capped at **warning** for benchmark type (two intrinsically critical findings downgraded).

| # | Finding | Severity | Decision Required |
|---|---------|----------|------------------|
| RT-1 | **Survivorship / Optional stopping — H5:** "Near threshold" trigger for 5-seed check is undefined. Because the single result is observed before deciding to run more seeds, this is a post-hoc optional stopping rule. If seed=42 narrowly fails (e.g., speedup=4.8×), the analyst can reclassify as "near threshold," run additional seeds, aggregate to the mean, and potentially cross the gate. Pre-register the exact numeric trigger window and aggregation rule before any measurement. *[Downgraded from critical.]* | warning | yes |
| RT-2 | **Goodhart exploitation — H0/H1-clean:** The H1 threshold (tw_x_dist > 50%) was anchored on a prior contaminated observation of ~62%. Any cleanup of the ~6.25× overhead that leaves tw_x_dist above 50% confirms H1 without establishing that the step is genuinely dominant under normal conditions. The threshold needs a data-independent principled justification or the hypothesis should be reframed as an estimation task. *[Downgraded from critical.]* | warning | yes |
| RT-3 | **Measurement precision — step fractions:** Step fractions from 5 iterations with no variance reporting. A single outlier iteration can shift the mean by several percentage points across the 50% threshold. The CI lower bound should be required to exceed 50% for H1 confirmation, not the mean. | warning | yes |
| RT-4 | **Post-hoc family membership — H-partial-MERFISH correction:** The correction is applied "if this test is part of the multi-comparison family," deferring the decision to after results are observed. The family membership must be declared before data collection; conditional inclusion allows retroactive exclusion of marginal results from correction. | warning | yes |
| RT-5 | **Evaluation collision — Criterion ratio CI:** Criterion reports within-run variance for each arm independently. The speedup ratio baseline_mean / variant_mean has no valid inferential CI because the two arms are measured in separate invocations (correlated OS noise). If both arms share Criterion harness state, the CI may be artificially narrowed by correlated jitter. Derive the ratio CI from bootstrap of paired samples. | warning | yes |
| RT-6 | **Survivorship bias — MERFISH single dataset:** H5 uses MERFISH with seed=42 and makes a quality claim. If MERFISH's cluster structure causes m=5000 subsample points to land near cluster centroids (well-approximating the full topology), the result is a property of this dataset's geometry, not of the approximation method. No sensitivity analysis over datasets or subsample draws is included. Scope the quality claim to "MERFISH-like structured biological data." | warning | yes |

---

## Cannot Assess

The following items could not be evaluated due to absent plan content:

1. **Criterion bootstrap seed configurability:** Whether Criterion's BenchmarkConfig exposes a bootstrap RNG seed in the version required by the plan is unverifiable without checking the `cargo-criterion` release notes. Cannot assess whether CI half-widths are bit-reproducible across re-runs vs. floating within a known tolerance.

2. **Thermal state at benchmark session start:** The plan does not describe the initial hardware state (idle vs. loaded, ambient temperature, frequency-scaling governor) at the time benchmarks begin. Cannot assess whether the 30s warm-up period is sufficient to reach steady-state frequency for Phase 4 measurements.

3. **gen_synthetic.py `--d` flag availability:** The plan acknowledges this may need updating but provides no version history or check. Cannot assess whether the Phase 5 data generation step will succeed without inspecting the existing script.

4. **tw_profiler binary interface:** The `--seed` and `--iters` flags referenced in Phase 5 are assumed to exist in the current tw_profiler implementation. Cannot verify the CLI interface matches the plan without reading the binary source.

---

## Mechanizable Check Log

The following checks could be automated in a future CI gate:

| Check | Automatable? | Notes |
|-------|-------------|-------|
| Phase 4 script variant count matches IV table variant count | Yes | Parse `--bench` flags from run_criterion_clean.sh and count |
| Holm family size m matches bench binary count | Yes | Count `--bench` flags and compare to stated m |
| `sleep 60` present between bench invocations in criterion scripts | Yes | Grep for sleep between cargo criterion calls |
| `--iters` value ≥ 30 in profiling scripts | Yes | Parse tw_profiler invocation flags |
| statsmodels exact version pinned in environment.yml | Yes | YAML parse, check for `==` not `>=` |
| temp/merfish_100k/ paths exist before prepare_data.sh | Yes | Bash precondition check |
| rust-toolchain.toml present in experiment directory | Yes | File existence check |
| SD and CI fields present in step_timing JSON output | Yes | JSON schema validation on profiling output |
| delta_tw CI present in h5_result.json | Yes | JSON schema validation |
| Criterion bootstrap seed set in BenchmarkConfig | Partial | Code inspection of bench files |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 20
warning_count: 26
red_team_count: 6
```
