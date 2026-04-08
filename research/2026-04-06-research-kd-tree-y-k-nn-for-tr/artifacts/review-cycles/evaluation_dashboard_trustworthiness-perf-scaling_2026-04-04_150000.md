# Review-Design Evaluation Dashboard

**Plan:** `experiment_plan_trustworthiness_perf_scaling_2026-04-04_123526.md`
**Reviewed:** 2026-04-04
**Reviewer:** review-design skill (automated, parallel subagents)

---

## ⛔ VERDICT: STOP

**Classification:** `benchmark` (+deployment, +multi_metric modifiers active)

Three adversarial-critical red-team findings trigger the STOP path. The plan contains
a vacuous parity gate (trivially satisfied on Gaussian data), an unanchored
`sample_fraction` that admits post-hoc calibration on the test data, and a cold-start
allocation confound at the measurement tier most critical to the GO/NO-GO decision
(n=100K). Additionally, five non-red-team critical findings were identified across
variance, statistical corrections, ecological validity, and benchmark representativeness
dimensions.

The plan must be revised before implementation proceeds.

---

## Dimension Scorecard

| Dimension | Weight | Critical | Warning | Info | Status |
|-----------|--------|----------|---------|------|--------|
| estimand_clarity | L1-H | 0 | 0 | 1 | ✅ Clean |
| hypothesis_falsifiability | L1-H | 0 | 0 | 1 | ✅ Clean |
| baseline_fairness | L2-H | 1 | 2 | 0 | ❌ Critical |
| unit_interference | L2-M | 0 | 1 | 2 | ⚠️ Warning |
| red_team | adversarial | 3 | 2 | 0 | ⛔ Stop triggers |
| error_budget | L3-M | 0 | 2 | 1 | ⚠️ Warning |
| statistical_corrections | L3-H | 1 | 1 | 1 | ❌ Critical |
| variance_protocol | L3-H | 1 | 1 | 1 | ❌ Critical |
| benchmark_representativeness | L4-M | 1 | 2 | 0 | ❌ Critical |
| ecological_validity | L4-M | 1 | 2 | 0 | ❌ Critical |
| measurement_alignment | L4-M | 0 | 2 | 1 | ⚠️ Warning |
| reproducibility_spec | L4-L | 0 | 2 | 1 | ⚠️ Warning |
| causal_structure | SILENT | — | — | — | Not spawned |

**Totals:** 8 critical · 17 warning · 9 info
**Stop triggers (red-team critical):** 3

---

## L1 Analysis — Estimand & Falsifiability

Both L1 dimensions are clean. No fail-fast gate trigger.

**estimand_clarity — INFO**
A clear informal contrast is present: each optimization strategy vs. baseline on
wall-clock time at n=1K–100K, with quantitative thresholds (≥2x speedup at n=10K;
≥8x at n=50K with |Δ|<0.01). No formal estimand notation needed for a benchmark.

**hypothesis_falsifiability — INFO**
H0 is falsified if all exact variants show <1.5x speedup AND subsampling 10% has
max(|Δ|) ≥ 0.01. Both threshold and quality gate are explicit and measurable.

---

## L2 Analysis — Baseline Fairness & Unit Interference

### baseline_fairness — CRITICAL

**⚠️ [WARNING] Measurement method asymmetry between small-n and large-n tiers**
Exact variants are measured with Criterion (10 samples, outlier rejection, statistical
harness) at n=1K/5K/10K, but at n=25K/50K/100K all variants switch to the manual
`tw_large_scale` binary with 1–2 iterations and no outlier rejection. Speedup ratios
spanning both tiers conflate high-confidence Criterion estimates with potentially
5–20% CV wall-clock measurements. The plan does not flag this precision gap.

**⚠️ [WARNING] Criterion sample depth asymmetry between exact and sampled variants**
`sample_size(10)` with `measurement_time(120s)` gives exact variants (30–60s/call at
n=10K) barely 2–4 statistical samples, while sampled variants (~1s/call) accumulate
100+ samples in the same window. Confidence intervals for sampled variants will be
much tighter, making them appear more stable in comparison — a presentation fairness
issue that could mislead prioritization.

**❌ [CRITICAL] Iteration asymmetry at large-n makes speedup ratios statistically
unfounded**
The protocol uses `--iterations 2` at large-n but `1 for n=100K`. The rule is applied
by variant runtime, not consistently: sampled variants completing in seconds could
receive 2–3 iterations while exact variants receive 1. The speedup ratio
`t_baseline / t_sampled` at n=100K is computed from single-point measurements with
no variance estimate. The plan should explicitly fix iterations per (variant, n) cell
and collect ≥3 iterations for any variant completing in under 60s.

### unit_interference — WARNING

**⚠️ [WARNING] Thread-local buffers persist across Criterion benchmark groups**
When `trustworthiness_thread_local` is implemented, its `thread_local!` buffers
(DIST_X_BUF, RANK_X_BUF) will retain allocated capacity across all benchmark groups
in the same Criterion binary invocation, since Criterion does not reset thread-local
state between groups. A buffer grown during warmup at n=10K will be reused at full
capacity for later groups, artificially reducing allocation overhead for `thread_local`
relative to a cold-start baseline. Criterion benchmark groups should be run in
separate binaries or the buffers explicitly cleared between groups.

**[INFO]** Baseline implementation allocates fresh Vecs per-row per-call — no
cross-benchmark spillover at buffer level. Thread-local design will introduce
structural asymmetry: baseline pays allocation on every iteration, thread_local only
on first use per thread.

**[INFO]** No global shared mutable state between variants. Rayon thread pool state
(allocator arenas) is OS/allocator-dependent and not controllable at the Rust
application layer — minor residual risk, not a protocol defect.

---

## ⛔ Adversarial Findings (Red-Team) — ALL require_decision: true

Three critical findings trigger the STOP verdict. All five findings marked
`requires_decision: true`.

### 🛑 RED-TEAM CRITICAL #1 — Cold-start allocation confound at n=100K

**Section:** Phase 6 Large-Scale Timing

With `--iterations 1` at n=100K, the baseline pays full Vec allocation + OS page-fault
cost (X array ≈ 8 MB, rank_x ≈ 800 KB) on the single measured iteration. The
`thread_local` variant eliminates per-call allocation after the warmup call. If the
warmup call does not also incur the same page-fault profile (e.g., if OS recycles
physical pages differently), the measured speedup conflates allocation overhead with
algorithmic improvement. At steady state with `--iterations ≥ 3`, the page-fault
effect averages out. At `--iterations 1`, it is a one-time event that dominates the
measurement. The plan's warmup mechanism may partially mitigate this but does not
guarantee steady-state measurement.

**Fix:** Require `--iterations ≥ 3` at all n values; drop the first iteration as
extended warmup. Accept longer runtime at n=100K.

### 🛑 RED-TEAM CRITICAL #2 — sample_fraction not fixed before measurement

**Section:** Phase 4 / Phase 7 Parity Verification

The plan tests sample_fractions {0.05, 0.10, 0.20, 0.50} for parity and reports
"minimum safe fraction = smallest fraction where max(delta) < 0.01". This selection
procedure is applied to the same seed-42 synthetic data used for speedup timing.
`sample_fraction` is therefore chosen by observing which value passes the parity gate
on the test data — the gate is calibrated on the eval set. H1 nominally claims 10%,
but the plan's own protocol scans fractions and selects the passing one, leaving the
door open for a post-hoc 10% claim validated on the data that already informed the
selection. There is no held-out data to confirm generalization.

**Fix:** Pre-register `sample_fraction = 0.10` (as stated in H1) before any measurement
runs. Do not select the fraction based on parity results. The fraction scan is
permissible as an exploratory secondary analysis labeled as such, not as the GO
criterion.

### 🛑 RED-TEAM CRITICAL #3 — Parity gate is vacuous on Gaussian data

**Section:** Phase 7 Parity Verification

The parity gate `|T_approx − T_exact| < 0.01` is tested exclusively on
`StandardNormal(seed=42)` data. At n≥10K with d_x=10, Gaussian vectors are subject to
concentration of measure: pairwise distances cluster tightly around their mean, making
the trustworthiness penalty distribution near-degenerate (low variance, highly
predictable). In this regime, any fixed subsample of rows will produce a T estimate
close to the full-n result regardless of sampling quality, and the 0.01 gate will pass
trivially. This provides no evidence that the approximation error stays below 0.01 on
structured data (biological clusters, manifolds) where the gate actually matters.

The 100K MERFISH pipeline reports T ≈ 0.9887 on real data — a highly structured regime
where subsampling errors are qualitatively different from Gaussian. The GO decision for
`sampled_10pct` will be driven by a vacuous parity check that cannot transfer to
production.

**Fix:** Use a structured synthetic dataset (e.g., `make_blobs` with k=8 clusters) for
the parity gate. Optionally, use the existing MERFISH n=200 fixture for a small-scale
structural check. Label Gaussian results as "throughput characterization only."

### ⚠️ RED-TEAM WARNING #4 — Combined variant receives implicit hyperparameter composition

**Section:** Phase 4 combined variant

The `combined` variant (partial_select + thread_local + simd_avx2) is implemented and
tuned during development, then measured. The baseline receives no equivalent tuning
pass. `partial_select` cutoff, `thread_local` buffer pre-sizing, and `simd_avx2`
kernel width selection are all chosen during implementation and can be adjusted against
informal observations before the benchmark runs. The plan does not lock variant
parameters before measurement begins.

**Fix:** Document all variant-specific parameters in the plan before any implementation
begins. Prohibit post-measurement parameter adjustment.

### ⚠️ RED-TEAM WARNING #5 — Criterion harness and rayon thread pool share infrastructure

**Section:** Phase 5 Benchmarks

Criterion's iteration loop runs on the main thread and repeatedly invokes the rayon
global thread pool. The `thread_local` variant's measured benefit depends on how
frequently rayon reuses the same OS threads across Criterion iterations — a property of
Criterion's own measurement cadence. The measurement infrastructure is therefore a
parameter of the optimization under test. The step profiler (Phase 3) measures a single
row and misses rayon scheduler overhead entirely, further obscuring this entanglement.

**Fix:** Measure `thread_local` speedup with a standalone harness controlling rayon
thread lifetime explicitly.

---

## L3 Analysis — Error Budget, Statistical Corrections, Variance Protocol

### error_budget — WARNING

**⚠️ [WARNING] Inconclusive zone (1.5x–2.0x) has no named outcome branch**
A measured speedup of 1.7x with low Criterion variance satisfies neither the conclusive
positive (≥2x) nor the conclusive negative (<1.5x) condition, and the plan's
inconclusive criterion (CV>50%) does not cover this case. Add an explicit "partial
positive" branch: "speedup ∈ [1.5x, 2x) with CV<50% → real improvement below H1
threshold; report as insufficient for the 2x goal."

**⚠️ [WARNING] Criterion CI width not pre-specified**
Criterion produces confidence intervals automatically but the plan never specifies an
acceptable CI width, nor justifies `sample_size(10)` relative to expected measurement
variance. For high-variance O(n²) variants at large n, the plan should explicitly state
what CI width is acceptable and log Criterion's reported CI bounds in the results JSON.

**[INFO]** At `--iterations 1` for n=100K, `std_ms` is structurally zero. This is a
single-sample estimate; the plan should explicitly flag large-scale timing results as
"indicative, not conclusive."

### statistical_corrections — CRITICAL

**❌ [CRITICAL] No correction procedure pre-specified for 288+ potential comparisons**
The comparison space is 8 variants × 6 metrics × 6 n-values = 288 potential tests.
No inferential framework, no α adjustment, and no pre-registered decision rule
distinguishes confirmatory from exploratory comparisons. For a benchmark with the
`+multi_metric` modifier (H weight), correction pre-specification is required. The
plan contains only descriptive analyses (ranking tables, parity gates, polyfit) with
no statistical inference layer.

**⚠️ [WARNING] OR-framed H1 inflates FWER to ~14%**
Testing "at least one of [3 variants] delivers ≥2x speedup" across 3 independent
comparisons at α=0.05 yields a FWER of ~14% under the global null. No Bonferroni,
Holm, or equivalent pre-specification is present.

**[INFO]** The large-effect-size threshold (≥2x) partially mitigates this for the
primary speedup DV. But secondary metrics (per-step fraction, scaling exponent,
parity score) have no analogous mitigation and remain unprotected.

### variance_protocol — CRITICAL

**❌ [CRITICAL] n=100K timing uses --iterations 1, making std_ms structurally zero**
The plan specifies `--iterations 1` for n=100K. The output schema collects `std_ms`
but with a single sample, std_ms is mathematically undefined. This creates a false
impression of statistical rigor — the n=100K speedup result will appear in the ranking
table as a numeric value with no indication it is a single-sample estimate subject to
unquantified noise, thermal transients, and OS scheduler jitter.

**⚠️ [WARNING] Criterion may yield <10 samples at n=10K**
With `measurement_time(120s)` and an O(n²) function that may take 30–60s per call
at n=10K, Criterion will realistically collect 2–4 actual samples after warm-up.
The plan does not require verifying that Criterion's reported sample count met
`sample_size(10)`. Under-sampled Criterion runs should be flagged as a data quality
gate.

**[INFO]** Machine-state controls (CPU frequency scaling, turbo boost, thermal
throttling) are absent from the plan. The codebase has turbostat tooling from prior
experiments (`research/2026-03-27-avx512-gather-zen5-spmv/scripts/run_with_turbostat.sh`)
that is not referenced here.

---

## L4 Analysis — Representativeness, Ecological Validity, Measurement, Reproducibility

### benchmark_representativeness — CRITICAL

**❌ [CRITICAL] n=250K extrapolation is structurally unvalidated**
The primary motivation includes the n=250K case (the Rust process was killed after
>7 minutes). The protocol caps large-scale timing at n=100K. The log-log scaling
exponent fit over n=5K–100K will not capture L3 cache exhaustion at n=250K: at
n=100K, rank_x is ~800KB (L2/L3 range), but at n=250K, rank_x grows to ~2MB
(primarily main memory), materially changing the rank-scatter and penalty-step
cache behavior. The GO/NO-GO recommendation for 250K will be extrapolated from a
regime that does not include the cache transition.

**Fix:** Add a single n=150K or n=200K timing point for the top-2 variants
(`combined` and `sampled_10pct`) to bracket the L3 cache transition without
prohibitive runtime cost.

**⚠️ [WARNING] d_x=10 specificity is acknowledged but entirely unmitigated**
The plan contains zero protocol steps for any d_x other than 10. The optimization
rankings (SIMD vs partial-select vs thread-local) are known to be qualitatively
sensitive to d_x. The GO/NO-GO table will be labeled as if it applies to the
"MERFISH pipeline" but is valid only for the d_x=10 configuration. A cheap d_x
sensitivity sweep at n=10K for d_x ∈ {2, 10, 50} would cost ~30 min and bound the
claim scope.

**⚠️ [WARNING] Gaussian-only parity gate insufficient for structured-data GO decision**
The MERFISH 100K report shows T ≈ 0.9887 on real data — a highly structured,
near-perfect-trustworthiness regime. Subsampling error behavior in this regime is
qualitatively different from Gaussian. The MERFISH n=200 fixture at
`tests/fixtures/tw_parity/` already exists and could be used for a structural parity
spot-check without generating new data.

### ecological_validity — CRITICAL

**❌ [CRITICAL] MERFISH fixture mitigation is unscheduled and absent from the protocol**
The plan acknowledges the synthetic-data gap and proposes mitigation: "also timing
against the MERFISH NPZ fixture if temp/lobpcg_bench/merfish_10k_laplacian.npz can
be extended to include (X, Y) arrays." However, this is a conditional clause with
no resolution date, no fallback, no deliverable, and no entry in the Execution
Protocol. Meanwhile, `temp/merfish_100k/` already contains
`merfish_100k_expression.npz` and `merfish_100k_spatial.npz` from the existing 100K
pipeline run — meaning X and Y arrays are available right now. The mitigation is not
technically blocked; it is simply unscheduled. Promote to a required protocol step
or explicitly bound the decision scope to "synthetic-data throughput characterization
only."

**⚠️ [WARNING] Gaussian-vs-MERFISH gap is real but second-order for throughput**
The hot-path (O(n²) distance compute, sort, rank scatter) is data-value-agnostic at
the throughput level — memory access patterns don't depend on clustering structure.
However, two second-order effects are missed: (1) penalty accumulation distribution
differs (affecting branch prediction in the inner loop); (2) Y-embedding spatial
locality affects heap behavior. These are unlikely to reverse a GO/NO-GO conclusion
but should be noted in the results discussion.

**⚠️ [WARNING] RAYON_NUM_THREADS=8 not validated against production environment**
The production deployment thread count is undocumented. The Rust implementation uses
`into_par_iter()` — parallelism is the primary performance lever vs. sklearn's
single-threaded implementation. Speedup vs. Python sklearn is highly sensitive to
thread count. The plan should document the production core count and justify why
8 threads is representative, or sweep at {4, 8, 16, 32}.

### measurement_alignment — WARNING

**⚠️ [WARNING] #[inline(never)] step profiler measures a de-optimized code shape**
In release builds, the compiler inlines and fuses inner distance loops across step
boundaries, enabling auto-vectorization across the full row computation. Forcing
`#[inline(never)]` prevents this fusion. The per-step fraction metric reflects an
artificially de-optimized binary that does not represent any production variant. The
practical consequence: "step X is 40% of total time" in the profiler binary may be
meaningless or inverted in the actual release build. Treat step fractions as
directional indicators, not precise attribution.

**⚠️ [WARNING] 5 sampled rows insufficient for representative step attribution**
The 5-row sample (0, n/4, n/2, 3n/4, n-1) does not represent the bulk of rows where
Rayon work-stealing partition effects occur, nor does it capture the distribution of
per-row distance profiles critical for `partial_select` evaluation. Expand to ≥20
randomly sampled rows.

**[INFO]** The 6 metrics are largely non-redundant. Speedup is a deterministic function
of two wall-clock measurements — a standard and acceptable derived ratio.

### reproducibility_spec — WARNING

**⚠️ [WARNING] Rust toolchain version unspecified; nightly builds change daily**
No `rust-toolchain.toml` exists. The repo is on `nightly-x86_64-unknown-linux-gnu
(rustc 1.96.0-nightly 23903d01c 2026-03-26)`. With `-C target-cpu=native`, codegen
is sensitive to exact compiler version. An independent reproducer cannot verify
results without knowing the toolchain.

**Fix:** Add `rust-toolchain.toml` pinning channel and date to the experiment directory.

**⚠️ [WARNING] CPU feature requirements not documented alongside target-cpu=native**
An independent party on a CPU lacking AVX2/FMA will get SIGILL or silently different
results. The plan should document minimum required CPU features and state that timing
results are only comparable on identical microarchitectures.

**[INFO]** `environment.yml` created conditionally — if the existing conda env satisfies
requirements, the file may never be committed. An independent reproducer has no pinned
Python dependency manifest. Generate unconditionally as a first-class artifact.

---

## Cannot Assess

The following dimensions could not be evaluated due to absent or unresolvable plan content:

1. **Randomization mechanism for row subsampling at n≥25K** — The plan specifies
   Knuth shuffle or reservoir sampling for `trustworthiness_sampled` but does not
   specify which algorithm will be used in the timing binary vs. the parity script.
   Cannot assess whether the same subsampling algorithm is used in both places or
   whether the choice affects the speedup/parity comparison.

2. **Production MERFISH pipeline thread topology** — The plan references "production
   optimization work" but no documentation specifies the production deployment
   environment (cloud vs. on-prem, vCPU count, memory bandwidth class). Cannot assess
   whether RAYON_NUM_THREADS=8 is representative of production.

3. **AVX2 auto-vectorization baseline quantification** — The plan checks
   `grep -c ymm` to detect whether LLVM already vectorizes the distance loop, but does
   not specify how the result changes the experimental design. If LLVM already
   auto-vectorizes, the `simd_avx2` variant measures marginal gain over
   auto-vectorization — a different claim from "raw SIMD vs scalar." Cannot assess
   whether the plan has a separate protocol branch for the auto-vectorized baseline
   case.

4. **Statistical inference method for scaling exponent comparison** — The plan fits
   `log(t) = α·log(n) + β` per variant and predicts expected slopes (α≈2.0 for exact,
   α≈1.0 for subsampling), but specifies no test for whether two variants' scaling
   exponents are statistically distinguishable. Cannot assess whether the scaling
   exponent table will support confident variant ranking or merely descriptive reporting.

---

## Mechanizable Check Log

Binary checks that could be automated in future review passes:

- [ ] `rust-toolchain.toml` present in repo or experiment directory
- [ ] `--iterations` ≥ 3 at all n values in large-scale timing binary
- [ ] `environment.yml` committed unconditionally (not conditional on env existence)
- [ ] `sample_fraction` pre-specified as a constant, not derived from parity scan results
- [ ] n=100K included in timing range (or explicit cap with extrapolation warning)
- [ ] At least one structured-data (non-Gaussian) parity check present in protocol
- [ ] CPU feature requirements documented in controlled variables table
- [ ] Multiple comparison correction procedure named in statistical plan section

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: STOP
experiment_type: benchmark
critical_count: 8
warning_count: 17
red_team_count: 5
stop_trigger_count: 3
stop_trigger_dimensions:
  - red_team: cold-start allocation confound at n=100K
  - red_team: sample_fraction not pre-registered (post-hoc calibration risk)
  - red_team: parity gate vacuous on Gaussian data (concentration of measure)
```
