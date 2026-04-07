# Experiment Design Review: Trustworthiness Performance Re-run (Clean Infrastructure)

**Plan:** `experiment_plan_tw_perf_rerun_clean_2026-04-05_130500.md`
**Review date:** 2026-04-05
**Reviewer:** review-design skill (automated multi-level analysis)

---

## VERDICT BANNER

```
╔══════════════════════════════════════════════════════════════╗
║                        REVISE                                ║
║  9 critical findings across 4 dimensions.                    ║
║  Fix measurement infrastructure gaps before executing.       ║
╚══════════════════════════════════════════════════════════════╝
```

**Experiment type:** benchmark
**Secondary modifiers:** `+high_cost` (6–7 h compute), `+multi_metric` (10 DVs)

The plan is detailed, well-motivated, and a clear improvement over its predecessor. However, it contains measurement infrastructure design flaws and data acquisition gaps severe enough to invalidate multiple hypotheses if unaddressed. The most acute issues are: (1) the step-fraction profiling measures CPU-ns summed across rayon threads, not wall-clock fractions — the stated metric; (2) H0/H1-clean is framed at d=50 but all profiling data is Gaussian d=10; (3) H-partial-MERFISH has no data acquisition path for the required MERFISH n=50K fixture; (4) the new worktree's bench binaries hard-code a path to the old worktree's Gaussian data, which will cause runtime failures.

---

## Classification Summary

| Field | Value |
|-------|-------|
| Experiment type | benchmark |
| Active modifiers | +high_cost, +multi_metric |
| Triage rule | Rule 1: IVs are algorithm variants, DVs are performance metrics, 5 comparators |
| L1 gate | PASS (no critical L1 findings) |
| Overall verdict | **REVISE** |

---

## Dimension Scorecard

| Dimension | Weight | Findings | Severity Summary |
|-----------|--------|----------|-----------------|
| estimand_clarity (L1) | H | 3 findings | 3× warning |
| hypothesis_falsifiability (L1) | H | 1 finding | 1× warning |
| baseline_fairness (L2) | M | 4 findings | ~~1× critical (retracted)~~, 2× warning, 1× info |
| unit_interference (L2) | M | 4 findings | 2× critical, 2× warning |
| error_budget (L3) | M | 4 findings | 2× warning, 2× info |
| statistical_corrections (L3) | **H** (+multi_metric) | 4 findings | 2× warning, 2× info |
| variance_protocol (L3) | **H** | 7 findings | ~~2× critical (downgraded)~~, 2× critical, 5× warning |
| benchmark_representativeness | — | not spawned | foothold: insufficient dedicated content |
| ecological_validity (L4) | M | 5 findings | 1× critical, 3× warning, 1× info |
| measurement_alignment (L4) | M | 5 findings | 3× critical, 2× warning |
| reproducibility_spec (L4) | — | not spawned | merged into variance_protocol |
| data_acquisition (L4) | M | 5 findings | 2× critical, 2× warning, 1× info |
| red_team | — | 5 findings | 5× warning (capped per benchmark rule) |

**Critical count (final, after false-positive retraction and RT cap):** 9
**Warning count:** ~22
**Red-team count:** 5 (all capped at warning)

---

## Level 1 Findings

### estimand_clarity (L1 · warning)

1. **H-100K baseline unnamed** — The hypothesis references "combined vs baseline" but does not name which of the 5 variants is "baseline" at the point of the hypothesis statement. Minor; Independent Variables table clarifies, but formal estimand should be self-contained.

2. **H0/H1-clean dimensionality discrepancy** — The hypothesis is framed as an estimation task at "n=100K, d=50 (MERFISH PCA-50 regime)" but Phase 7 profiling uses Gaussian data (`gaussian_n100000_x.npy`, d=10). The DV context and the hypothesis context disagree. This is escalated as a critical measurement_alignment finding (see below).

3. **H-partial-MERFISH two-population contrast underspecified** — The comparison is MERFISH vs Gaussian CI half-width, but the hypothesis is stated as a single-threshold test (`< 0.26`). Recommend restating as an explicit cross-dataset contrast.

### hypothesis_falsifiability (L1 · warning)

4. **H0/H1-clean unfalsifiable by design** — "No threshold-based pass/fail verdict is assigned." The success criteria table provides CI-ordering criteria (which could serve as a formal falsification criterion) but the hypothesis section explicitly disavows them. Either remove from the hypothesis table or adopt the success-table criteria as the formal H0.

---

## Level 2 Findings

### baseline_fairness

> **Retracted finding:** The agent flagged "H5 protocol asymmetry across all 5 variants" as critical. This is a misreading of the experiment structure: `tw_approx_runner` (H5) is a separate approximation binary and hypothesis, not one of the 5 Criterion benchmark variants. The H5 hypothesis is specifically about `trustworthiness_approx` vs exact computation. This finding is retracted.

5. **Criterion settings asymmetric between n scales** (warning) — n=100K has explicitly pinned parameters (sample_size=63, SamplingMode::Flat, warm_up=30s, measurement_time=1500s). n=1K–50K uses "standard Criterion settings (not explicitly pinned)." This creates an implicit asymmetry: if Criterion's auto-tuning for smaller sizes produces different iteration counts per variant, comparisons across the scale sweep are not symmetric.

6. **Fixed run order without randomization** (warning) — Sequential order (baseline → combined) means the combined variant runs last and benefits from any system warm-up. No counterbalancing or randomized order is specified.

### unit_interference

7. **Relaxed atomic ordering for reset/accumulate** (critical) — `step_timing::reset()` uses `Ordering::Relaxed` for the store; worker threads use `fetch_add(Relaxed)`. The Relaxed model provides no synchronization guarantee: a reset on the main thread is not guaranteed to be visible to rayon worker threads before they execute their fetch_add. This can cause inter-iteration contamination in the accumulated totals. At minimum, reset should use `Ordering::Release` and reads should use `Ordering::Acquire`.

8. **reset() called after iteration, not before** (critical) — If the reset is called after each iteration (as described in the plan), the first measured iteration inherits accumulated state from the 5 warm-up iterations. State should be reset immediately before each measured iteration begins, not after it completes.

9. **Page cache effects from sequential runs** (warning) — Data fixtures loaded by earlier variants remain in the OS page cache for later variants. Combined (running last) sees effectively cached data. The 60s cool-down addresses thermal effects only.

10. **Profiling feature contamination in Criterion builds** (warning) — If the `profiling` feature flag is accidentally enabled during the Criterion benchmark phase, atomic fetch_add calls in the hot path add shared-cache-line contention across rayon threads. The Phase 6 build command should explicitly verify `--no-default-features` excludes profiling.

---

## Level 3 Findings

### error_budget

11. **CV=15% sensitivity analysis absent** (warning) — Power at n=63 assumes CV=15%, but this estimate comes from prior contaminated data. No sensitivity table shows how power degrades if true CV is 20–25%. If CV=20%, power drops to approximately 60–65%, below the 80% floor.

12. **n=10 seeds for H5 CI upper bound** (warning) — With n=10, the 95% CI upper bound is heavily influenced by the single worst seed. No bootstrap-on-seeds analysis or distributional assumption is provided to justify stability of the upper-bound estimate near the 0.001 threshold.

13. **z=1.96 instead of t(df=9) for n=10 H5 CI** (warning) — The analysis script uses `1.96 * std / sqrt(n)` for n=10. The correct critical value is `t(0.975, df=9) ≈ 2.262`, giving ~15% wider intervals. Under-coverage at a primary hypothesis gate is a validity threat. Fix: `scipy.stats.t.ppf(0.975, df=len(deltas)-1)`.

### statistical_corrections

14. **Secondary H5 sweep status unspecified** (warning) — The m-sweep (m ∈ {500, 1K, 2K, 5K, 10K}) generates multiple data points but is not explicitly committed to "descriptive only" vs "formally tested." If any inferential comparison is drawn from sweep results, it falls outside the Holm family.

15. **H0/H1-clean: 5 implicit simultaneous CI comparisons** (warning) — The success criterion "tw_x_dist CI lower bound exceeds all other steps' CI upper bounds" constitutes 5 simultaneous comparisons with no multiplicity adjustment. The plan should explicitly state this is a descriptive check with no family-wise error control.

### variance_protocol

> **Two variance_protocol critical findings are downgraded from the agent's assessment:**
> - "No Criterion seed" → warning: Criterion's internal RNG is not exposed as a user-configurable seed in the standard API; this is a Criterion limitation, not a plan omission.
> - "No repeated independent runs" → warning: single-run benchmarks are standard practice and the plan explicitly mitigates with n=63 samples at the primary scale.

16. **Criterion settings for n=1K–50K unspecified** (critical) — Sample size, warm_up_time, and measurement_time are not stated for n<100K scales. Variance budget for these scales cannot be audited or reproduced; results may not be comparable across variants if Criterion auto-tunes differently per binary.

17. **Rayon thread count uncontrolled** (critical) — The rayon global thread pool defaults to the host CPU count. Thread count is a controlled variable (it directly affects which variants benefit from parallelism), but it is not listed as a controlled variable and no pinning is specified. Results are not reproducible on machines with different core counts.

18. **Cache warm-state check is optional** (warning) — "Residual bias assessed by comparing first-variant results across repeated runs if time permits." This is a validity check, not a post-hoc curiosity; making it contingent on available time means the ordering bias may go undetected.

19. **CV=15% provisional** (warning) — Pre-experiment power calculation uses a CV estimate from contaminated prior data. This is documented as provisional and should be re-estimated and reported, but the sample size decision (n=63) was made on the prior CV.

---

## Level 4 Findings

### data_acquisition

20. **H-partial-MERFISH missing data source** (critical) — The hypothesis tests partial_rank CI half-width at n=50K on MERFISH data, but the plan defines no MERFISH n=50K fixture: no acquisition command, no generation script, no output path. The MERFISH data section covers n=10K only. This hypothesis has no data.

21. **Gaussian data path inaccessible from new worktree** (critical) — The bench templates hard-code `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/2026-04-04-tw-perf-scaling/data/gaussian")`. This path exists inside the *old* worktree, not the new one. No copy, symlink, or re-path step is specified. All Criterion benchmarks (H-100K, H-partial-MERFISH, H0/H1-clean) will fail at runtime with missing-file errors.

22. **temp/merfish_100k/ acquisition underspecified** (warning) — The gitignored source is "confirmed present as of 2025-04-04" with no hash, size range, or row/column assertions to verify a freshly generated copy. If this path is absent, the fallback ("run generate_merfish_subset.py") lacks command arguments, expected runtime, and output verification.

### measurement_alignment

23. **H0/H1-clean: d=10 data vs d=50 hypothesis** (critical) — Phase 7 profiling runs on Gaussian n=100K d=10, but the hypothesis claims step fractions at "n=100K, d=50 (MERFISH PCA-50 regime)" and the first-principles prediction ("tw_x_dist dominates because O(n·d) at d=50") is only valid at d=50. At d=10, O(n·d) per-row work is 5× smaller, and the predicted dominance pattern may not hold. The measured DV does not match the stated DV.

24. **Atomic counters measure CPU-ns, not wall-clock fractions** (critical) — `step_fractions` are defined as "mean fraction of total wall time," but atomic accumulators sum CPU-nanoseconds across all rayon threads. If steps differ in degree of parallelism, CPU-ns fraction and wall-clock fraction diverge. For a data-parallel workload using rayon, steps with higher rayon fan-out will appear inflated relative to their wall-clock share.

25. **tw_approx_runner timing underspecified** (critical) — `wall_exact_s` and `wall_approx_s` are parsed from the runner JSON but the plan does not specify (a) which "exact" algorithm is timed (which of the 5 variants?), (b) whether timing includes data loading/conversion, (c) whether both timings use the same warm-up protocol. If the timings differ in scope, the speedup ratio is not a clean algorithmic comparison.

26. **H-100K speedup claim valid only for Gaussian d=10** (warning) — The CI lower bound > 1.5× claim is validated on synthetic d=10 data. Production use cases are d=50–500 PCA components. Cache behavior and vectorization efficiency differ materially across this range. The claim should be explicitly scoped to Gaussian d=10 in the hypothesis statement.

27. **Criterion bootstrap may underestimate uncertainty** (warning) — Criterion's bootstrap resamples within-run timing observations, which may be autocorrelated (cache warming, CPU frequency ramp, NUMA affinity settling). An anti-conservative CI makes thresholds easier to pass.

### ecological_validity

28. **H-100K production relevance gap** (critical — merged with finding #26 above; keeping as warning per deduplication) — The primary deployment target is biological data at d=50–500, but speedup is validated only at d=10. The plan acknowledges "Gaussian d=10 may not represent all production use cases" but does not scope the H-100K claim accordingly in the hypothesis text. *(Deduplicated with measurement_alignment finding #26; kept here for dimension coverage.)*

---

## Adversarial Findings (Red-Team)

> **Severity cap for benchmark type: maximum = warning.** All red-team findings are at warning level per protocol.

All red-team findings have `requires_decision: true`.

| # | Challenge | Finding |
|---|-----------|---------|
| RT-1 | Goodhart exploitation | **Threshold sandbagging.** The H-100K threshold (1.5×) was set after observing the prior run's result (~1.95×–2.15×). The threshold is ~30% below the known expected outcome. A genuinely blinded threshold would be derived from a performance requirement (e.g., minimum acceptable speedup for production adoption), not calibrated to what the algorithm is already known to deliver. |
| RT-2 | Data leakage | **Prior data re-use.** Gaussian data is reused from the exact prior run that informed threshold selection and algorithm tuning. The data is not held-out; the experimental design decisions (thresholds, choice of d=10, k=15) were made while observing this data. The experiment cannot falsify the hypothesis in the way a truly independent dataset could. |
| RT-3 | Asymmetric tuning | **n=100K parameters specified; n<100K unspecified.** The n=100K settings (Flat sampling, 63 samples, 30s warm-up) are precisely calibrated for the primary claim. Smaller-n settings are left as Criterion defaults, creating asymmetric measurement rigor: the scale that matters for the claim is carefully configured; scales that could reveal scaling problems use unspecified auto-tuning. |
| RT-4 | Survivorship bias | **No re-run protocol.** Fixed run order (combined last) benefits from warm-up effects accumulated by prior variants. No protocol prevents selective re-running of the combined variant alone if initial results are unfavorable. No log of run attempts is mandated. |
| RT-5 | Evaluation collision | **Bootstrap fallback + H5 5× claim.** (a) The bootstrap fallback (`speedup_lower = baseline_ci_lower / combined_ci_upper`) is a ratio of summary statistics, not a CI over the speedup distribution — it systematically underestimates uncertainty. (b) The 5× H5 speedup claim from a 2× subsampling ratio (n/m = 10K/5K = 2×) lacks a mechanistic explanation. Without explaining the source of the extra 2.5×, the threshold cannot be distinguished from one reverse-engineered from an observed result. |

---

## Cannot Assess

These dimensions could not be evaluated due to absent plan content:

1. **Randomization mechanism** — The plan does not describe how, if at all, run order will be randomized across any replication. Cannot assess unit interference risk from run-order effects beyond noting their existence.

2. **Resource proportionality vs. alternatives** — No alternative approaches (e.g., simpler profiling via perf or flamegraph, reduced n, fewer seeds) are scoped. Cannot assess whether 6–7 hours of compute is the minimum needed to resolve the hypotheses or whether a cheaper design would suffice.

3. **tw_approx_runner implementation** — The plan references `tw_approx_runner` as a binary that outputs `wall_exact_s`, `wall_approx_s`, and `delta` but does not describe its implementation. Cannot assess whether its timing isolation and comparison logic are correctly specified.

4. **H5 5× speedup mechanism** — The theoretical speedup from subsampling m=5000 at n=10K is 2× (n/m). The claimed 5× speedup is not mechanistically explained. Cannot assess whether the threshold is achievable or was derived from preliminary data.

---

## Mechanizable Check Log

These binary checks could be automated in CI pre-flight:

- [ ] File exists: `research/2026-04-04-tw-perf-scaling/data/gaussian/gaussian_n100000_x.npy`
- [ ] File exists: `temp/merfish_100k/merfish_100k_expression.npz`
- [ ] New worktree contains Gaussian data path or copy step is present
- [ ] `cargo build --release --features cli,profiling --no-default-features` compiles clean (no `testing` feature in profiling build)
- [ ] Criterion bench template includes explicit `sample_size()` and `measurement_time()` for all n values, not just n=100K
- [ ] `reset()` call precedes each measured iteration (not follows)
- [ ] Rayon thread count is pinned via `build_global()` in bench or profiler binary
- [ ] H5 CI formula uses `t.ppf(0.975, df=n-1)` not hardcoded `1.96`
- [ ] MERFISH n=50K acquisition step exists (for H-partial-MERFISH)

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 9
warning_count: 22
red_team_count: 5
```
