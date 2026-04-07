# Evaluation Dashboard — tw-perf-scaling

**Plan file:** `experiment_plan_tw_perf_scaling_2026-04-04_170000.md`
**Review timestamp:** 2026-04-04 17:20:00
**Reviewer:** review-design skill (parallel subagents, sonnet)

---

## Verdict

```
╔══════════════════════════════════════════════════╗
║                   VERDICT: STOP                  ║
║                                                  ║
║  Primary trigger: Red-team GOODHART — GO         ║
║  threshold validated at n≤50K while stated       ║
║  production problem is n=100K–250K.              ║
╚══════════════════════════════════════════════════╝
```

**Experiment type:** `benchmark`
**Secondary modifiers active:** `+deployment`, `+multi_metric`

---

## Stop Triggers

### STOP-1 (Red-team — Goodhart exploitation) `requires_decision: true`

> **Section:** Criterion benchmarks + Motivation
>
> The stated purpose is "a ranked GO/NO-GO recommendation grounded in measured
> wall-clock speedups at the practical target scale of 100K–250K cells." The
> primary GO criterion for individual variants is ≥1.5× Criterion throughput
> ratio — but Criterion benchmarks are capped at n=50K. Only H6 (combined
> variant) has an explicit n=100K check via `tw_profiler`. Individual variants
> H2 (thread_local), H3 (avx2_kernel), and H4 (avx-512) can pass the shipping
> gate entirely from n≤50K Gaussian data without any n=100K validation.
>
> A variant that achieves 1.6× at n=50K via cache-reuse effects but provides no
> meaningful improvement at n=100K (where L3 cache pressure and NUMA effects
> differ) would receive a GO recommendation that contradicts the motivating
> failure. The benchmark does not ground its primary decision in the regime it
> claims to characterize.
>
> **Fix required:** Extend Criterion benchmarks to n=100K (even with reduced
> sample count), or require tw_profiler n=100K measurements for every variant
> (not just H6), or rewrite the success criteria to reference only n=100K
> measurements.

---

### STOP-2 (L4 Measurement Alignment — H3 method mismatch) `critical`

> **Section:** ## Hypothesis + ## Analysis Plan
>
> H3 is defined quantitatively: "Manual AVX2 < 2× over the current
> auto-vectorized baseline at d=10 f64 (confirmed by `cargo asm` + benchmark)."
> The Analysis Plan resolves H3 via binary presence inspection: "If AVX2
> instructions present: mark H3 NO-GO — auto-vectorization confirmed, no manual
> SIMD needed."
>
> These are two different claims. Auto-vectorization being confirmed does not
> measure whether manual intrinsics would achieve < 2× improvement — it
> sidesteps the measurement entirely. A plan that measures X to conclude Y about
> Z produces uninterpretable results for that sub-hypothesis. The H3 go/no-go
> decision is structurally untethered from the stated threshold.
>
> **Fix required:** Either redefine H3 as a binary detection check ("is the
> current implementation auto-vectorized?") and remove the ≤2× claim, or add a
> Criterion benchmark that compares a manual AVX2 implementation against the
> baseline to actually measure the ratio.

---

### STOP-3 (L4 Reproducibility — MERFISH provenance) `critical`

> **Section:** ## Inputs and Data
>
> The H5 confirmatory gate — the only pre-registered gate using structured
> biological data (addressing prior STOP-3) — depends on
> `temp/merfish_100k/merfish_100k_expression.npz`. Per CLAUDE.md, the `temp/`
> directory is gitignored. No public URL, DOI, version hash, or download script
> is specified for this file. An independent party cannot obtain it, and the
> experiment's primary structural result (H5 PASS/FAIL) cannot be reproduced.
>
> **Fix required:** Commit the derived 10K PCA artifact
> (`merfish_n10k_x.npy`, `merfish_n10k_y.npy`) to `tests/fixtures/` or provide a
> fetch script with a stable URL/DOI in `scripts/prepare_merfish.py`.

---

### STOP-4 (L3 Variance Protocol — missing seed on confirmatory gate) `critical`

> **Section:** ## Phase 3d: H5 Confirmatory Gate
>
> The `run_h5_confirmatory.sh` script invokes `subsampling_sweep.py` with
> `--n-trials 5` but no `--seed` argument. The H5 GO/NO-GO verdict is sealed
> from this output and committed to git before the m-sweep. Without a fixed
> seed, the stochastic row sampling produces a different 5-trial result on each
> run. The sealed artifact is not reproducible: re-running
> `run_h5_confirmatory.sh` will produce a different `delta_max`, making the
> pre-registered GO gate non-deterministic.
>
> **Fix required:** Add `--seed 42` (or any pre-registered value) to the
> `run_h5_confirmatory.sh` invocation and the controlled variables table.

---

## Classification Summary

| Field | Value | Source |
|-------|-------|--------|
| experiment_type | benchmark | frontmatter (extracted) |
| hypothesis_h0 | "No single optimization achieves >2× speedup at n=100K warm AND m=5000 subsampling deviation >0.001 on MERFISH" | ## Hypothesis |
| hypothesis_h1 | "At least one exact optimization ≥2× at n=100K warm AND m=5000 deviation ≤0.001 on MERFISH" | ## Hypothesis |
| estimand | variant vs baseline on speedup ratio and T_approx deviation; population n=100K cells | extracted |
| metrics | 7 DVs including tw_speedup_ratio (≥1.5×), tw_approx_deviation (<0.001), step_fraction | ## Dependent Variables |
| baselines | current unmodified production trustworthiness() as-shipped | ## Controlled Variables |
| statistical_plan | Criterion 95% CI bootstrap; H5 pre-registered at m=5000; no multi-comparison correction | ## Analysis Plan |
| success_criteria | Three: conclusive positive, conclusive negative, inconclusive | ## Success Criteria |

---

## Dimension Scorecard

| Dimension | Weight | Findings | Severity Summary |
|-----------|--------|----------|-----------------|
| estimand_clarity | L1 (always) | 6 | 3 warning, 2 info, 1 warning |
| hypothesis_falsifiability | L1 (always) | 6 | 4 warning (1 calibrated from critical), 2 info |
| baseline_fairness | L2 | 5 | 3 warning (1 calibrated from critical), 1 info |
| unit_interference | L2 | 3 | 2 warning (1 calibrated from critical) |
| causal_structure | S | — | SILENT — not spawned |
| variance_protocol | H | 4 | **1 critical**, 2 warning, 1 info |
| error_budget | M | 5 | 4 warning, 1 info |
| statistical_corrections | H | 3 | 2 warning, 1 info |
| benchmark_representativeness | H | 7 | 2 warning (2 calibrated from critical), 3 warning, 1 info |
| ecological_validity | M | 6 | 2 warning (2 calibrated from critical), 2 warning, 1 info |
| measurement_alignment | M | 5 | **2 critical**, 3 warning |
| reproducibility_spec | M | 9 | **3 critical**, 4 warning, 1 info |
| red_team | — | 5 | **1 critical (STOP)**, 3 warning, 1 info |

---

## Adversarial Findings (Red-Team)

All red-team findings carry `requires_decision: true`.

### RT-1 (STOP trigger): Goodhart Exploitation — benchmark scale mismatch

The GO threshold (≥1.5×) is measured via Criterion capped at n=50K. The motivating problem is n=100K–250K. Variants H2, H3, H4 can clear the shipping gate without validating the production regime. **Critical — see STOP-1 above.**

### RT-2: Data leakage — implementation gating from benchmark workload

Phase 3 gating criteria (partial_rank only if X-sort ≥40%, avx2_kernel only if no AVX2 in assembly) use profiling measurements collected on the same workload instances that appear in subsequent benchmarks. Variants are selected to address observed weaknesses in the evaluation workload, meaning the variants are custom-fitted to the eval rather than to a held-out profiling set.

*Severity: warning. Gating criteria are mechanical thresholds (not calibrated against final speedup outcomes), and profiling on representative workloads is standard practice. Scope limitation should be documented.*

### RT-3: Asymmetric tuning — baseline receives zero optimization effort

The baseline is unmodified production code while variants are iteratively designed against profiling evidence from the same hardware and workload. This is acknowledged and intentionally disclosed via RT-3. The speedup ratios are explicitly documented as upper bounds.

*Severity: info. Intentional, disclosed, and appropriate for a benchmark measuring improvement over current production code.*

### RT-4: Survivorship bias — no pre-specified primary n for GO decision

The success criteria say "at least one exact optimization achieves ≥1.5× Criterion speedup with non-overlapping CI (H2 or H6 supported)." No specific n value is pre-registered as the primary endpoint for this disjunction. If thread_local achieves 1.6× only at n=5K but <1.2× at n=50K, the current plan's wording would still allow claiming H2 support. Pre-specify the primary n (e.g., n=50K or n=100K) for the GO criterion.

*Severity: warning.*

### RT-5: Evaluation collision — assembly identity check unreliable

The `run_criterion.sh` assembly check greps for AVX2 instruction counts across `target/release/deps/*.s` for both builds. The glob is non-deterministic when multiple `.s` files coexist, and the check counts instructions globally rather than for the specific `trustworthiness` function. A count match does not confirm the hot path is identical. The check may silently pass when the benchmarked binary inadvertently includes instrumentation.

*Severity: warning. The RT-4 fix (no `required-features = ["testing"]` on the bench target) provides the primary guard; the ASM check is a secondary verification that needs a more rigorous implementation.*

---

## Cannot Assess

The following dimensions could not be evaluated due to absent plan content:

1. **Randomization mechanism for subsampling seeds across variants** — the plan specifies RNG seed 42 for dataset generation and 5 trials for H5, but no RNG policy is defined for individual Criterion benchmark iterations (Criterion manages its own scheduling internally). Cannot assess whether Criterion's internal ordering introduces ordering effects between variants.

2. **Incremental implementation hygiene** — Phase 3 instructs the implementer to add new public functions (`trustworthiness_thread_local`, `trustworthiness_approx`) and modify `src/metrics.rs`. Whether these modifications preserve the existing `#[cfg(feature="testing")]` interface contract cannot be assessed from the plan alone; the current `MetricResult`/`AssessmentReport` structs (used for eigensolver metrics, not trustworthiness) might conflict with new timing annotations at the type level.

3. **Cross-platform correctness of `partial_rank` tie-breaking** — the plan specifies that equal X-distances must be broken consistently with `sort_unstable_by(total_cmp)` behavior, and mandates a unit test for ties. Whether the described O(n) linear scan (Phase 3b) can exactly replicate `sort_unstable_by` tie semantics cannot be verified from prose alone; this is only verifiable at code review time.

4. **Error propagation from `extract_criterion_summary.py`** — the plan lists this script in the directory layout and references it in `run_criterion.sh`, but no specification for its format or error handling is provided. If Criterion changes its HTML/JSON output format, the extraction step silently fails with no defined fallback, and the recommendation table will be unpopulated.

---

## Mechanizable Check Log

Binary checks that could be automated in a future CI gate:

| Check | Status | Notes |
|-------|--------|-------|
| `trustworthiness_bench` Cargo.toml has no `required-features` | Not verified (code not read) | Can be grepped from Cargo.toml after Phase 0 |
| RNG seed present in all shell script invocations of subsampling_sweep.py | **FAIL** | `run_h5_confirmatory.sh` lacks `--seed` |
| `rust-toolchain.toml` exists and pins exact version | **FAIL** | Plan specifies "stable (current)" with no pin |
| `research/2026-04-04-tw-perf-scaling/environment.yml` exists | **FAIL** | Plan explicitly says "no environment.yml will be created" |
| MERFISH source data has documented provenance | **FAIL** | temp/ gitignored, no URL/DOI |
| All hypothesis thresholds consistent (H0 vs success criteria) | **WARN** | H0 uses 2×; success criteria use 1.5× |
| Assembly identity check uses function-specific ASM | **FAIL** | Current check uses global instruction count |
| H5 confirmatory gate runs before m-sweep | PASS (enforced by script guard) | Script exits if h5_confirmatory_result.json absent |

---

## Additional Notable Findings (Non-STOP)

### L1: H0 threshold vs success criteria inconsistency
H0 says "no optimization achieves >2×" but the "conclusive positive" success criterion triggers at "≥1.5× Criterion speedup." A result of 1.8× would declare "conclusive positive" without falsifying H0. These thresholds should be unified.

### L2: Thread-local Vec state persists across Criterion groups
If `bench_thread_local` runs before `bench_baseline`, the thread-local allocations remain in TLS for all subsequent groups. The baseline group benefits from pre-allocated TLS memory it would not have in production cold-start. Group execution order must be fixed and documented, or each variant binary run as a separate `cargo bench` invocation.

### L4: Criterion CI not available at n=100K
The recommendation table format includes a "CI 95%" column for "Speedup (n=100K)." Criterion benchmarks run only to n=50K. The n=100K row will have no CI — only a `tw_profiler` point estimate from 5 iterations. The CI column should be omitted or labeled "N/A (tw_profiler)" for n=100K rows.

### L4: Reproducibility — "stable (current)" Rust toolchain
`stable (current)` resolves to different `rustc` versions at different times. SIMD codegen and Criterion measurements differ across compiler versions. A `rust-toolchain.toml` pinning the exact version (e.g., `1.87.0`) must be committed.

### L3: H2 confidence interval boundary not specified
H2 passes if speedup ∈ [1.05, 1.15]. A speedup of 1.20 registers as FAIL under this rule, even though it exceeds the expected range. No decision rule is specified for speedup > 1.15 (better than predicted but outside the band). This ambiguity should be resolved.

### L4: k=15 only — speedup rankings may invert at higher k
`trustworthiness()` cost is O(n·k) in neighborhood operations. The ranking of variants may differ at k=5 or k=50. The GO/NO-GO recommendation should be explicitly scoped to k≈15, or a secondary k sweep added.

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: STOP
experiment_type: benchmark
critical_count: 7
warning_count: 22
red_team_count: 5
stop_trigger_count: 4
stop_triggers:
  - "red_team: Goodhart — GO threshold at n<=50K while problem is n=100K+"
  - "measurement_alignment: H3 asm inspection does not measure <=2x ratio"
  - "reproducibility_spec: MERFISH source gitignored with no provenance"
  - "variance_protocol: H5 confirmatory subsampling_sweep.py missing --seed"
```
