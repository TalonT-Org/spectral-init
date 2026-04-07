# Review-Design Evaluation Dashboard

**Plan:** Trustworthiness Performance Re-run (Clean Infrastructure)
**Plan file:** `.autoskillit/temp/plan-experiment/experiment_plan_tw_perf_rerun_clean_2026-04-05_132911.md`
**Review date:** 2026-04-05 13:56:40

---

## Verdict

```
██████████████████████████████████████
  VERDICT: REVISE
  Experiment type: benchmark
  Critical findings: 0
  Warning findings: 32
  Red-team findings: 7 (all capped at warning)
██████████████████████████████████████
```

The plan is substantively sound and far more rigorous than the prior experiment. The core measurement
infrastructure redesign (isolated bench binaries, Holm correction, n=63 Criterion samples, separated
`profiling` feature) addresses the identified gaps correctly. However, **32 warning-level findings**
require resolution before execution — primarily: (1) statistically incorrect t-CI applied to a median,
(2) adaptive near-threshold rule with two test statistics on the same data, (3) data acquisition
blocking risk for fresh worktrees, and (4) multiple pre-specification gaps flagged by red-team.

---

## Classification Summary

| Field | Value | Source |
|---|---|---|
| experiment_type | benchmark | triage / rule 1 (IVs = algorithm variant names, DVs = performance metrics, multiple comparators) |
| secondary_modifiers | +multi_metric, +deployment | ≥3 DVs (11 total); motivation references "field-deployable" |
| +multi_metric effect | statistical_corrections M→H | 11 DVs present |
| +deployment effect | ecological_validity floor=M | already M; no change |

---

## Dimension Scorecard

| Dimension | Weight | Findings | Severity Summary |
|---|---|---|---|
| estimand_clarity | H (L1) | 1 | 1 info |
| hypothesis_falsifiability | H (L1) | 4 | 4 info |
| baseline_fairness | L2 | 2 | 2 info |
| unit_interference | L2 | 6 | 2 warning, 4 info |
| red_team | — | 7 | 7 warning (capped; type=benchmark) |
| error_budget | L3 | 7 | 2 warning, 5 info |
| statistical_corrections | H (L3) | 5 | 1 warning, 4 info |
| variance_protocol | H (L3) | 5 | 2 warning, 3 info |
| benchmark_representativeness | M (L4) | 6 | 4 warning, 2 info |
| ecological_validity | M (L4) | 7 | 6 warning, 1 info |
| measurement_alignment | M (L4) | 5 | 4 warning, 1 info |
| reproducibility_spec | M (L4) | 4 | 2 warning, 2 info |
| data_acquisition | M (L4) | 4 | 2 warning, 2 info |
| causal_structure | S | — | SILENT (not spawned) |
| resource_proportionality | L | 0 | No issues identified |

---

## Level 1 Findings (Fail-Fast Gate: PASSED)

### estimand_clarity — 1 info

**[INFO]** `## Hypotheses / H0/H1-clean` — H0/H1-clean is an estimation task with no formal A-vs-B
contrast (absent by design). The treatment and comparator are absent; only Y (per-step CPU fraction)
and Z (n=100K, d=10) are specified. Acceptable for estimation goal; cannot be expressed as a contrast.

### hypothesis_falsifiability — 4 info

**[INFO]** `## Hypotheses / H5` — The near-threshold rule (majority ≥6/10 trials ≥5× when median
∈ [4.5×, 5.5×]) is a secondary criterion that partially overrides the primary median criterion.
A result with median just below 5× but 6/10 trials above 5× yields H1 under the near-threshold rule
but H0 under the primary criterion. Minor goalposts risk near the threshold boundary.

**[INFO]** `## Hypotheses / H-100K` — "Combined variant speedup" is not formally defined as a
single aggregated statistic vs. minimum/mean across variants. If the aggregation rule is determined
post-hoc, goalposts risk exists.

**[INFO]** `## Hypotheses / H0/H1-clean` — Explicitly declared estimation task with no H0/H1.
Acceptable. No falsification criterion by design; downstream consumers should note that any set of
fractions constitutes a "result."

**[INFO]** `## Hypotheses / H-partial-MERFISH` — Dual specification: absolute threshold 0.26 vs.
relative comparison to Gaussian half-width from same experiment. Precedence unspecified if they
conflict (e.g., MERFISH half-width = 0.24 < 0.26 but also < Gaussian). In practice they should
align, but the precedence rule is absent.

---

## Level 2 Findings

### baseline_fairness — 2 info

**[INFO]** `## Controlled Variables` — N_THREADS is a compile-time constant (value 8 in template)
requiring manual update. `apply_phase1_changes.sh` does not verify that bench-file N_THREADS matches
tw_profiler.rs's N_THREADS or the actual physical core count. Risk: inconsistency between profiler
and Criterion bench binaries produces step-fraction data (H0/H1-clean) at a different thread count
than speedup ratios (H-100K).

**[INFO]** `## Execution Protocol` — `tw_partial_rank_merfish_bench` runs after the 5 Gaussian
variant benches with no pre-run 60s cool-down. Since this bench tests a separate hypothesis
(H-partial-MERFISH) and runs both partial_rank and baseline within the same binary (internally
symmetric), this is not a material fairness issue for primary comparisons.

### unit_interference — 2 warning, 4 info

**[WARNING]** `## Phase 2: Bench File Creation` — The plan mandates
`rayon::ThreadPoolBuilder::new().num_threads(N_THREADS).build_global().unwrap()` in each bench
binary, but the current codebase's bench file only calls `let _ = rayon::current_num_threads()` (a
no-op query). The thread-count pinning described in the plan is not yet implemented. When the new
bench files are created from the template, the Phase 1 changes must ensure this pinning is actually
present and verified.

**[WARNING]** `## Phase 1 Source Changes` — The `step_timing` module (Change 4) and `tw_profiler`
reset call-site fix (Change 5) are not yet implemented in the codebase. The H0/H1-clean measurement
depends on these changes being applied correctly before any profiling run. The `apply_phase1_changes.sh`
verification script must actively check for these changes, not just the Cargo.toml feature flag.

**[INFO]** Separate `[[bench]]` binaries provide genuine process isolation between variants. Each
`cargo criterion` invocation is a separate process; Rayon global pool, allocator state, and static
initializers cannot leak across variant binaries.

**[INFO]** Thread-local buffers in `metrics.rs` are cleared unconditionally before each row
computation. Stale data cannot leak between iterations.

**[INFO]** 60s cool-down between variants mitigates thermal throttling. The mandatory cache
warm-state check (W4) in reversed order provides the correct control for residual cache state.

**[INFO]** Within a single bench binary, Criterion runs groups sequentially but group state is
isolated per-`Criterion` instance; cross-group contamination of timing measurements does not occur.

---

## Adversarial Findings (Red-Team)

*All findings have `requires_decision: true`. For benchmark type, severity cap = warning (no red-team
finding can trigger STOP).*

**[WARNING — requires_decision]** `## H5 — Near-threshold rule` — The near-threshold rule
(majority ≥6/10 when median ∈ [4.5×, 5.5×]) is an adaptive test: two different test statistics
(median vs. count-based majority) are applied to the same data depending on a preliminary check.
This inflates type-I error in the near-threshold interval. The 6/10 majority threshold has no
power calculation: P(≥6 | n=10, p=0.5) ≈ 0.377, so the rule is not conservative near null. The
plan gives no rationale for abandoning the median criterion precisely where researcher degrees of
freedom matter most.

**[WARNING — requires_decision]** `## H-100K — 1.5× threshold selection` — The 1.5× CI lower-bound
threshold is derived by applying a 30% haircut to the prior experiment's point estimate (~1.95×) —
the same experiment acknowledged as having blocked/inconclusive hypotheses and measurement gaps.
The prior result is simultaneously distrusted (justifying a re-run) and trusted (anchoring the
success criterion). If the true speedup falls between 1.5× and 1.95×, the experiment declares H1
confirmed while the prior infrastructure failure remains unresolved.

**[WARNING — requires_decision]** `## Execution Order — warm-state adjudication` — The reversed-order
cache warm-state check (combined→baseline) is mandated but the protocol for adjudicating disagreement
between forward and reversed runs is unspecified. If they disagree (>5% difference), which result
takes precedence? The experiment has two execution orders with a single pre-specified analysis path,
creating an undeclared researcher degree of freedom.

**[WARNING — requires_decision]** `## H-partial-MERFISH — comparison baseline selection` — The plan
states the Gaussian half-width comparison uses the "NEW isolated Criterion run from this experiment"
but also pre-registers a threshold of 0.26 from contaminated prior data. It is not stated which
criterion takes precedence if they yield different verdicts. A result where MERFISH < fresh Gaussian
but MERFISH ≥ 0.26 (or vice versa) has no pre-specified resolution.

**[WARNING — requires_decision]** `## Asymmetric effort — variant vs baseline optimization` — No
evidence is provided that the baseline implementation has received equivalent engineering effort to
the optimized variants. The baseline is the reference against which speedups are measured, but if it
has not been subjected to the same profiling and optimization passes, any performance advantage of
the optimized variants may partly reflect development asymmetry rather than algorithmic improvement.
The baseline version/commit should be frozen and documented.

**[WARNING — requires_decision]** `## Goodhart exploitation — instruction cache footprint` — Identical
Criterion configuration does not guarantee identical measurement conditions if variants have different
instruction-cache footprints. The avx2_kernel variant likely has a substantially different I-cache
profile than baseline. Variants may be measured at different points on their warm-up curves despite
identical wall-clock configuration. No perf-counter (cache-miss, IPC) instrumentation is specified.

**[WARNING — requires_decision]** `## Survivorship bias — seed selection` — Seeds 42–51 form a
human-selected contiguous block starting at the conventional default seed 42. If any preliminary
runs with these seeds were executed before this plan was finalized (e.g., during development of
tw_approx), the seed selection may be de facto cherry-picked. With n=10 and the majority rule,
removing one seed can swing the decision. The plan provides no declaration that these seeds were
not previously observed.

---

## Level 3 Findings

### error_budget — 2 warning, 5 info

**[WARNING]** `## Power Analysis Details` — CV=15% is stated as "estimated from prior bench output"
but the prior experiment is acknowledged as having measurement infrastructure gaps. The specific run
and dataset providing this estimate are not cited, making the CV assumption unverifiable. If actual
CV exceeds 15%, power falls below 80% by design.

**[WARNING]** `## Hypotheses Summary / H5` — The plan uses a 95% t-CI (df=9) to assess |delta_tw|,
and separately uses median as the primary speedup metric. The t-CI formula (`mean ± t × SE`) is
appropriate for a mean, not a median. For median inference at n=10, a sign-based CI or bootstrap CI
is statistically appropriate; t-CI on median is only valid under symmetry assumptions not stated
in the plan. Type II error is unacknowledged for this gate.

**[INFO]** Holm correction family size (m=4) is adequate but could be more explicit (which 4 variants).

**[INFO]** H5 dual-gate conjunctive structure (speedup AND quality) is conservative; joint type I
error is no greater than either individual rate. Acceptable.

**[INFO]** H-partial-MERFISH has no power analysis (pure descriptive). Acceptable for benchmark calibration.

**[INFO]** W1 contingency (report as limitation if CV>20%) is adequate but lacks quantitative
guidance on how many additional samples would restore 80% power at CV=20%.

**[INFO]** Estimation tasks (H0/H1-clean) have no inferential error rates; omission is appropriate.

### statistical_corrections — 1 warning, 4 info

**[WARNING]** `## Statistical Plan / H5` — The m-sweep (W2) is labeled descriptive-only, but
H5's primary verdict uses a threshold criterion applied to the same infrastructure. If any post-hoc
selection of the reported speedup metric or m-value occurs based on W2 sweep results, the effective
number of comparisons is inflated. The plan should pre-specify that m=5000 is the confirmatory gate
and that the m-sweep output is never used to select or adjust the primary H5 metric.

**[INFO]** H-100K Holm m=4 correctly excludes baseline from the correction family (standard practice).

**[INFO]** 6 CPU step fractions are explicitly descriptive; reporting multiple CIs without family-wise
correction is acceptable.

**[INFO]** H5 conjunctive dual criterion does not require alpha correction (conjunctive = conservative).

**[INFO]** H-partial-MERFISH CI comparison is descriptive; no correction required.

### variance_protocol — 2 warning, 3 info

**[WARNING]** `## Sample Sizes — H5` — H5 uses n=10 seeds and reports median and range, but
provides no variance analysis for the median estimator itself. With n=10 and no distribution
assumption, the median's sampling variance is poorly characterized. The plan should either add
a bootstrap CI for the median over the 10 trials, or increase n, or explicitly bound the median's
standard error.

**[WARNING]** `## Success Criteria — W1` — The contingency plan for CV>20% states "report as
limitation and recommend additional samples" but does not specify a decision rule: how many
additional samples? At what CV level is the result declared inconclusive vs. needing more runs?
Without a concrete remediation protocol, W1 degrades into indefinite deferral.

**[INFO]** Criterion RNG non-determinism adequately acknowledged (W6); Criterion exposes no seed
API so this is an unavoidable tooling constraint.

**[INFO]** 30 profiling iterations with 5 warm-up is adequate for step-fraction CIs. Mean ± t-CI
(df=29) is appropriate.

**[INFO]** The tw_profiler CLI invocation in `run_profiling_clean.sh` must pass `--n-iters 30
--n-warmup 5` explicitly; the plan shows this correctly. If profiler defaults differ, results
would be incorrect — but the script as written is correct.

---

## Level 4 Findings

### benchmark_representativeness — 4 warning, 2 info

**[WARNING]** `## Scope Limitations` — The "field-deployable on MERFISH" claim is supported by a
single Zhuang-ABCA-1 dataset. No second biological dataset with different tissue, gene panel, or
spatial structure has been benchmarked. One dataset limits representativeness of the quality/speedup
claim for "biological data" as a category.

**[WARNING]** `## Benchmark Regime` — Gaussian Criterion benchmarks use d=10 only. AVX2 kernel
speedup is sensitive to vector length (d); results at d=10 do not bound behavior at d=50 (MERFISH
PCA). The "combined speedup" claim lacks coverage of the dimensionality axis.

**[WARNING]** `## Dataset Scope` — MERFISH trustworthiness values are known to saturate near 0.99
on this dataset. A near-ceiling testbed is insensitive to quality differences. Any quality
conclusion (H5 |delta| < 0.001) will have limited discriminative power; failing variants and passing
variants may produce near-identical trustworthiness scores.

**[WARNING]** `## Scope Limitations / H5` — H5 explicitly acknowledges n=10K scope as limited;
however, the plan lists "field-deployable on structured biological data" as a decision outcome.
The gap between n=10K (half-subsampling ratio m/n=0.5) and production n=100K (m/n=0.05) is
acknowledged but not bridged by additional measurement.

**[INFO]** W7 hardware specificity is documented (x86-64-v3, AVX2/FMA). The 3D V-Cache characteristic
of the benchmark machine is noted and materially affects cache-resident speedup results.

**[INFO]** Rust nightly specificity and need for stable-Rust follow-up are documented.

### ecological_validity — 6 warning, 1 info

**[WARNING]** `## Test Conditions` — No benchmark in the plan directly tests n=100K with MERFISH
biological data. H-100K uses Gaussian synthetic data; H5 uses n=10K MERFISH. The stated decision
"whether combined speedup survives cache-regime boundary at production scale on biological data"
is not directly answerable from this test matrix.

**[WARNING]** `## Test Conditions` — Gaussian d=10 (H-100K) is a structurally poor proxy for
high-dimensional biological embeddings (MERFISH d=50, manifold geometry). Speedup and approximation
quality are sensitive to intrinsic data geometry; Gaussian i.i.d. results may not generalize to
the biological deployment target.

**[WARNING]** `## Test Conditions` — At production n=100K with m=5000, the neighbor fraction
m/n drops to 0.05 vs. 0.5 at n=10K. This changes the sparsity regime, working set size, and
dominant code paths significantly. No benchmark characterizes performance or quality in this
low m/n regime. The field-deployability conclusion is underspecified for the production scale.

**[WARNING]** `## Test Conditions` — Dedicated benchmark machine with no competing workloads is
not representative of shared server or cloud environments where biological data pipelines typically
run. Rayon thread pinning may behave differently on NUMA nodes or containerized environments.
Reported throughput figures are likely optimistic relative to real deployment conditions.

**[WARNING]** `## Test Conditions` — All results are on Rust nightly-2026-03-26. No stable Rust
validation is included in this experiment. Production pipelines would target stable Rust; nightly-
specific SIMD lowering and autovectorization may not be available on stable.

**[WARNING]** `## RT-1 — 1.5× threshold` — The 1.5× threshold is not derived from a deployment
latency requirement. Without a target wall-clock budget from real pipeline constraints, even a
"passing" result at 1.5× may not satisfy field-deployable latency for the actual production workload.

**[INFO]** AVX2/FMA restriction is explicitly documented (W7). Broadly available on modern x86
server hardware; ARM limitation is acknowledged.

### measurement_alignment — 4 warning, 1 info

**[WARNING]** `## Research Questions / RQ2` — `wall_clock_speedup` is collected at n=10K, m=5000
(m/n=0.5), but RQ2 claims "field-deployable speedup on MERFISH." Production MERFISH may be 250K+
cells at m/n=0.05. The metric answers the n=10K question, not the field-deployable question. The
plan explicitly acknowledges this (n=10K scope only) but lists field-deployability as a decision
outcome — the decision cannot be conclusively answered by this metric.

**[WARNING]** `## Research Questions / RQ1` — `criterion_speedup_n100k` uses Gaussian synthetic
data at d=10, but RQ1 asks whether "the combined speedup survives the cache-regime boundary at
production scale" for a tool targeting MERFISH d=50. Gaussian d=10 and MERFISH d=50 differ in
both dimensionality (affecting distance computation cost) and distribution (affecting branch
prediction and cache behavior). The metric answers a narrower question than claimed.

**[WARNING]** `## Research Questions / RQ3` — CPU-ns step fractions at d=10 are used to identify
"which algorithmic step is the next optimization target." The plan explicitly notes "d=50 regime
requires a separate MERFISH profiling run not in scope." Step cost ratios are d-dependent (distance
computation scales with d; heap ops do not); the d=10 profile may rank steps differently than the
d=50 production profile. The "optimization target" conclusion may be incorrect for the actual
production workload.

**[WARNING]** `## Research Questions / RQ2 quality` — `delta_tw < 0.001` as operationalization of
"acceptable approximation quality for field deployment" lacks a domain-grounded threshold. No
downstream task accuracy or user-study basis is provided. The threshold appears inherited from prior
experiments without independent justification; a result that passes this gate may or may not be
distinguishable in practice from one that fails.

**[INFO]** CI half-width comparison (H-partial-MERFISH) addresses RQ4 structurally, though a
positive result is consistent with confounds beyond data distribution (different n, d, intrinsic
dimensionality). This limitation is a known bounded scope issue, not a design flaw.

### reproducibility_spec — 2 warning, 2 info

**[WARNING]** `## Environment Specification` — `cargo-criterion` is installed via `cargo install
cargo-criterion` with no version pin. The JSON output format or `--message-format` behavior could
change across cargo-criterion versions, silently breaking `analyze_clean.py`. The version actually
used will not be recorded unless explicitly captured (e.g., `cargo criterion --version >> hardware_profile.txt`).

**[WARNING]** `## Data Provenance / MERFISH source` — `data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad`
(2.0 GB) is the upstream source for `temp/merfish_100k/`. The plan does not document the exact
download URL, file checksum, or dataset version. An independent replicator attempting H5 needs this
provenance to reconstruct the `temp/merfish_100k/` NPZ files if absent.

**[INFO]** Hardware dependencies (x86-64-v3, AMD Ryzen, 3D V-Cache) are documented via W7 and the
hardware_profile.txt recording step. Known hardware dependency; partial reproducibility expected.

**[INFO]** Analysis formulas are documented with sufficient precision for independent re-implementation
(bootstrap CI, Holm correction via statsmodels `multipletests`, t-CI formula, step-fraction mean/CI).

### data_acquisition — 2 warning, 2 info

**[WARNING]** `## Inputs and Data — Source Data` — `temp/merfish_100k/` is gitignored (per
`/temp/` rule in project `.gitignore`). The plan notes the 5 NPZ artifacts are "confirmed present"
on the current machine, but provides no acquisition command or re-generation step for a fresh
worktree or CI environment where this directory will be empty. Steps 3.2 and 3.3 will fail
immediately on a fresh clone without this pre-populated directory. The plan must document either:
(a) how to re-acquire/re-generate `temp/merfish_100k/` from the raw H5AD source, or (b) an
explicit pre-flight check that gates on this directory existing.

**[WARNING]** `## Inputs and Data — Item 2 (MERFISH n=50K)` — Step 3.3 calls
`prepare_merfish_50k.py` to "slice expression[:50000] from prepare_merfish.py PCA output," but
Step 3.2 (`prepare_merfish.py`) only produces `merfish_n10k_x.npy` and `merfish_n10k_y.npy`.
The intermediate PCA-50 output used by Step 3.3 is never named or verified in Step 3.2. If
`prepare_merfish.py` does not write a persistent PCA-50 matrix for ≥50K rows, Step 3.3 has no
valid input. The dependency chain and intermediate artifact path must be made explicit.

**[INFO]** H-partial-MERFISH data coverage: Gaussian n=50K is generated by Step 3.1 (sizes include
50000). MERFISH n=50K is generated by Step 3.3. Both data sources are documented; the H-partial-
MERFISH data table could cross-reference both explicitly.

**[INFO]** Step 3.1 uses a relative `--output-dir data/gaussian` path. The experiment directory
context should be explicit to avoid misrouting output.

---

## Cannot Assess

The following dimensions could not be evaluated due to absent plan content:

1. **Seed pre-registration timeline** — The plan does not state when seeds 42–51 were selected or
   whether any runs with these seeds were executed prior to plan finalization. Cannot assess whether
   the seed block is truly pre-registered or was observed during development.

2. **Production latency budget** — No wall-clock latency requirement from a real deployment pipeline
   is stated. Cannot assess whether any speedup threshold (1.5×, 5×, or otherwise) meets actual
   deployment needs; the ecological validity of the success criteria is not assessable against
   real-world requirements.

3. **MERFISH pipeline uniqueness** — The plan does not describe the full preprocessing pipeline
   from raw H5AD to `temp/merfish_100k/` NPZ files. Cannot assess whether a fresh replication
   on a different machine would yield numerically identical intermediate data.

4. **Stable Rust codegen equivalence** — No stable Rust bench run is included or referenced. Cannot
   assess whether speedup results are stable or nightly-specific; the production claim requires
   stable Rust validation that is explicitly deferred.

---

## Mechanizable Check Log

The following binary checks could be automated in CI for this and future experiment plans:

- `[ ]` Verify `N_THREADS` constant is identical across all 6 bench binaries and `tw_profiler.rs`
- `[ ]` Verify `cargo-criterion` version is pinned in `Cargo.toml` dev-dependencies or captured at runtime
- `[ ]` Verify `temp/merfish_100k/` directory is non-empty before running `prepare_data.sh`
- `[ ]` Verify `prepare_merfish.py` produces an intermediate PCA-50 artifact that `prepare_merfish_50k.py` can read
- `[ ]` Verify `apply_phase1_changes.sh` tests `step_timing::reset()` call site position (before `variant_fn`, not after)
- `[ ]` Verify `build_global().unwrap()` is present in each bench `main()` (grep for `build_global`)
- `[ ]` Verify `run_profiling_clean.sh` passes `--n-iters 30 --n-warmup 5` (not defaults)

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: REVISE
experiment_type: benchmark
critical_count: 0
warning_count: 32
red_team_count: 7
```
