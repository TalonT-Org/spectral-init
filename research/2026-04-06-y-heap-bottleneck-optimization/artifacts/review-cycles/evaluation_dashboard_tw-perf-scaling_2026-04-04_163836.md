# Review-Design Evaluation Dashboard

**Plan:** Trustworthiness Performance — Scaling Analysis and Optimization Evaluation  
**Plan file:** `.autoskillit/temp/plan-experiment/experiment_plan_tw_perf_scaling_2026-04-04_161626.md`  
**Review date:** 2026-04-04  
**Reviewer:** review-design skill (automated)

---

## ⛔ VERDICT: STOP

**Reason:** 4 adversarial critical findings from red-team analysis identify fundamental structural defects in the benchmark design that undermine the validity of the GO/NO-GO decision. All four must be resolved before the experiment can produce actionable evidence.

---

## Classification Summary

| Field | Value |
|---|---|
| Experiment type | **benchmark** |
| Triage rule | Rule 1 (IVs = system/method names, DVs = perf metrics, 6 comparators) |
| Secondary modifiers | `+multi_metric` (7 DVs), `+high_cost` (~2–3hr compute) |
| L1 gate | PASSED (no critical estimand/falsifiability findings) |
| Red-team gate | FAILED (4 critical red-team findings) |

---

## Dimension Scorecard

| Dimension | Weight | Findings | Severity Summary |
|---|---|---|---|
| estimand_clarity | H (L1) | 0 | Clean |
| hypothesis_falsifiability | H (L1) | 2 | 2 warnings, 1 info |
| baseline_fairness | H (L2) | 7 | **2 critical**, 3 warnings, 1 info |
| unit_interference | H (L2) | 7 | **3 critical**, 3 warnings, 1 info |
| red_team | — | 7 | **4 critical**, 3 warnings |
| error_budget | H (L3) | 6 | 5 warnings, 1 info |
| statistical_corrections | H (L3) | 6 | **1 critical**, 3 warnings, 2 info |
| variance_protocol | H (L3) | 7 | **2 critical**, 5 warnings |
| benchmark_representativeness | H (L4) | 4 | **2 critical**, 2 warnings |
| ecological_validity | M (L4) | 3 | **1 critical**, 2 warnings |
| measurement_alignment | M (L4) | 5 | **3 critical**, 2 warnings |
| reproducibility_spec | M (L4) | 6 | **2 critical**, 3 warnings, 1 info |
| causal_structure | **S** | — | *Not spawned (SILENT for benchmark)* |
| resource_proportionality | M (after +high_cost) | — | *Not assessed — see Cannot Assess* |

---

## Adversarial Findings (Red-Team) — All `requires_decision: true`

### CRITICAL-RT-1: Goodhart Exploitation
**Section:** Goodhart Exploitation  
**Severity:** critical | **requires_decision:** true

The GO threshold (≥1.5× speedup as `baseline_mean / variant_mean`) is vulnerable to two exploitation paths: (1) artificially inflating the baseline by sizing synthetic data to cause cache thrashing, making the baseline slow without any algorithmic improvement; (2) the `combined` variant is post-hoc composed of whichever individual variants pass — it is impossible to fail by definition since only winners are composed. H6 (combined ≥3× at n=100K) is not independently falsifiable under this design. The combined variant composition must be pre-specified unconditionally before any individual results are examined.

### CRITICAL-RT-2: Data Leakage — Confirmatory m=5000 Contaminated by Exploratory Sweep
**Section:** Data Leakage  
**Severity:** critical | **requires_decision:** true

The subsampling parameter m=5000 is declared pre-registered for the confirmatory H5 GO/NO-GO gate. However, the plan also runs an exploratory m-sweep over {500, 1000, 2000, 5000, 10000} with no structural blinding or ordering constraint. If the sweep reveals that m=5000 narrowly fails (e.g., delta=0.0012), nothing prevents post-hoc re-registration of m=10000. The fix (pre-register before examining sweep results) was declared in the plan but no enforcement mechanism is specified. The confirmatory H5 result must be computed before the sweep, or delegated to a blinded reviewer who evaluates m=5000 without access to sweep results.

### CRITICAL-RT-3: Asymmetric Tuning
**Section:** Asymmetric Tuning  
**Severity:** critical | **requires_decision:** true

The baseline is the unmodified `trustworthiness()` receiving zero optimization effort. All six variants are constructed with knowledge of profiling results from Phase 0 instrumentation — each variant targets the empirically measured hot path. The experiment measures "profiling-informed optimized variant vs. uninstrumented baseline," not "best achievable variant vs. best achievable baseline." The 1.5× GO threshold implicitly assumes the baseline represents the floor of achievable performance, but compiler PGO, trivial parallelism tuning, or other easily applicable optimizations on the baseline are never attempted. The plan must either (a) specify what optimization effort the baseline has received and justify why it is the appropriate comparator, or (b) acknowledge that reported speedups are an upper bound on real-world improvement.

### CRITICAL-RT-4: Evaluation Collision — Instrumentation Contaminates What Is Measured
**Section:** Evaluation Collision  
**Severity:** critical | **requires_decision:** true

Phase 0 adds `#[cfg(feature="testing")]` timing guards inside `trustworthiness()`. Optimization variants are then benchmarked in the same `src/metrics.rs` modified by instrumentation. Two collision risks: (1) if instrumentation guards are not excised before benchmarking (or if building without `--features testing` fails to fully remove them), the timing overhead is included in the baseline but may be structurally altered by Phase 3 loop restructuring, making the speedup ratio partially an artifact of instrumentation removal rather than algorithmic improvement; (2) adding timing code changes function body size and may alter compiler inlining decisions, meaning the benchmarked "baseline" is not the same machine code that motivated the experiment. The plan must require verification that Phase 3 benchmarks are run on a clean build with `testing` feature disabled, and that binary identity (via `cargo-show-asm` or `perf annotate`) matches the pre-instrumentation baseline before any speedup ratios are computed.

### Warning-RT-5: Survivorship Bias in Combined Variant
**Section:** Survivorship Bias  
**Severity:** warning | **requires_decision:** true

The `combined` variant is constructed from survivors of individual evaluations (only variants that passed ≥1.5× individually are included). Negative interactions between optimizations (cache line conflicts between `thread_local` buffers and `partial_rank` access patterns) will not manifest in individual evaluations and are invisible until `combined` runs. Single seed=42 throughout provides no mechanism to detect whether Gaussian data at seed=42 produces an unusually cache-friendly access pattern. Add at minimum a secondary seed for combined variant evaluation.

### Warning-RT-6: SIMD Pre-Registration as NO-GO Creates Asymmetric Implementation Effort
**Section:** Benchmark-Specific — Asymmetric SIMD Effort  
**Severity:** warning | **requires_decision:** true

H3 and H4 are framed as anticipated NO-GO outcomes (manual AVX2 <2× over auto-vectorized; AVX-512 <20% over AVX2). This reduces implementation motivation for `avx2_kernel`. A poorly implemented SIMD variant that confirms a pre-anticipated NO-GO is indistinguishable from a well-implemented one that genuinely underperforms. Given `target-cpu=native` already uses AVX2+FMA in the auto-vectorized baseline, the comparison requires expert-quality intrinsic code. Scope H3/H4 explicitly as "effort-bounded feasibility assessment" rather than a definitive architectural ruling, or require code review before treating them as definitive NO-GO.

### Warning-RT-7: Parity Gate Scope Ambiguity
**Section:** Parity Gate Proxy Validity  
**Severity:** warning | **requires_decision:** true

The parity gate (`|T_rust − T_sklearn| < 1e-6`) is undefined in the GO/NO-GO decision for approximate variants. `partial_rank` changes computation order (floating-point reordering may exceed 1e-6 non-catastrophically), and `approx_m5000` by construction introduces approximation error. The 1e-6 threshold conflates numerical precision (appropriate for exact reimplementations) with approximation error. The plan must specify which variants are subject to the 1e-6 gate vs. a looser tolerance, and what happens to GO status when parity fails.

---

## All Critical Findings (Summary)

| # | Dimension | Severity | Summary |
|---|---|---|---|
| 1 | red_team | critical | Goodhart exploitation — baseline manipulation + unfalsifiable combined variant |
| 2 | red_team | critical | Data leakage — m-sweep contaminates pre-registered m=5000 confirmatory gate |
| 3 | red_team | critical | Asymmetric tuning — profiling-informed variants vs. uninstrumented baseline |
| 4 | red_team | critical | Evaluation collision — instrumentation contaminates benchmarked baseline |
| 5 | baseline_fairness | critical | Conditional variant gating creates asymmetric comparison set |
| 6 | baseline_fairness | critical | approx_m5000 (approximate) mixed with exact variants under same GO threshold |
| 7 | unit_interference | critical | TLS buffers persist across Criterion benchmark groups — no process isolation |
| 8 | unit_interference | critical | All variants in same process; earlier runs contaminate later ones |
| 9 | unit_interference | critical | Baseline contaminated if run after thread_local has initialized TLS |
| 10 | statistical_corrections | critical | Per-n evaluation inflation — no confirmatory n anchor for H1–H5 |
| 11 | variance_protocol | critical | "5 independent seeds per m" not enumerated before execution |
| 12 | variance_protocol | critical | Rayon thread count not pinned — direct threat to H2 speedup ratio |
| 13 | benchmark_representativeness | critical | Gaussian d_x=10 unrepresentative of production (d_x=50/1122) |
| 14 | benchmark_representativeness | critical | n=100K absent from Criterion — policy-critical scale has no CI |
| 15 | ecological_validity | critical | Cold-start vs warm-start mismatch for interactive deployment contexts |
| 16 | measurement_alignment | critical | GO speedup threshold measured on Gaussian d_x=10, not deployment data |
| 17 | measurement_alignment | critical | Parity metric ambiguous about float precision (f32 vs f64 in Rust) |
| 18 | measurement_alignment | critical | Subsampling deviation uncomputable at n=100K (not in Criterion groups) |
| 19 | reproducibility_spec | critical | No environment.yml → sklearn_reference.py baseline not reproducible |
| 20 | reproducibility_spec | critical | MERFISH source data provenance not specified — H5 gate not reproducible |

---

## Warning Findings (Selected High-Priority)

| Dimension | Summary |
|---|---|
| hypothesis_falsifiability | H0/H1_combined AND-logic asymmetry: partial failure (one arm fails, one passes) not covered in success criteria |
| hypothesis_falsifiability | Inconclusive region doesn't address partial satisfaction of H1_combined |
| baseline_fairness | `combined` variant post-hoc composition — only winning components included (selection bias) |
| baseline_fairness | n=100K absent from Criterion benchmark groups (confirmed by L4) |
| baseline_fairness | Rayon thread count not pinned |
| unit_interference | CPU cache warming from earlier groups benefits later groups |
| unit_interference | bench_combined last — accumulates all spill-over effects |
| unit_interference | AVX2 frequency downclocking affects subsequent non-SIMD benchmarks |
| error_budget | No power analysis for any hypothesis |
| error_budget | Type I / Type II error rates not acknowledged |
| error_budget | 5 trials insufficient to estimate delta_max reliably |
| statistical_corrections | Total comparison count (60 cells) not enumerated; FWER not bounded |
| variance_protocol | OS jitter controls absent (WSL2 adds Hyper-V non-determinism) |
| variance_protocol | No bimodality detection before interpreting bootstrap CI |
| benchmark_representativeness | Single hardware platform — fallback behavior on non-AVX2 uncharacterized |
| ecological_validity | MERFISH validated at n=10K; bottleneck starts at n=100K |
| measurement_alignment | Speedup not stratified per data type (Gaussian vs MERFISH) |
| reproducibility_spec | Script contents not pre-registered; Phase 3 decision log not required |

---

## Cannot Assess

1. **Resource proportionality (M weight):** No explicit resource budget stated for individual phases. Total compute time is estimated at "~2–3 hours" but no per-step wall-clock budget is allocated, and no gate exists for aborting if a phase overruns. Cannot assess whether resource allocation is proportional to hypothesis priority.

2. **Randomization mechanism:** The plan does not describe any randomization of benchmark execution order (e.g., randomizing variant order within a Criterion session, or randomizing profiling n values). Cannot assess whether order effects are controlled via randomization or only via warm-up iterations.

3. **MERFISH PCA eigenspace stability:** The MERFISH d=50 PCA projection is computed with `PCA(50, random_state=42)`. Whether the first 50 components explain sufficient variance to preserve the cluster structure claimed to make STOP-3 relevant is not documented. Cannot assess whether the MERFISH PCA preprocessing preserves the biological properties being tested.

4. **Criterion version-specific behavior:** The exact Criterion 0.5.x patch version is not pinned. Bootstrap CI behavior, outlier filtering defaults, and throughput calculation changed between patch versions. Cannot assess reproducibility risk without knowing the exact version in use.

---

## Mechanizable Check Log

| Check | Automatable | Status |
|---|---|---|
| Subsampling seed enumeration (5 seeds explicitly listed in plan) | Yes — scan plan for seed list | FAIL: not found |
| RAYON_NUM_THREADS pin in benchmark harness | Yes — grep Cargo.toml / bench harness | NOT VERIFIED |
| n=100K present in Criterion BenchmarkId list | Yes — scan bench harness source | FAIL: n≤50K per spec |
| environment.yml presence in research directory | Yes — file existence check | FAIL: explicitly excluded |
| MERFISH data provenance URL or DOI in plan | Yes — scan plan for URL/DOI | FAIL: not found |
| `testing` feature disabled in benchmark build command | Yes — scan run_criterion.sh | NOT VERIFIED (script not pre-registered) |
| Criterion measurement_time = 10s uniformly set | Yes — scan bench harness | NOT VERIFIED |

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: STOP
experiment_type: benchmark
critical_count: 20
warning_count: 18
red_team_count: 7
stop_triggers:
  - dimension: red_team
    message: Goodhart exploitation — unfalsifiable combined variant + baseline manipulation
  - dimension: red_team
    message: Data leakage — m-sweep contaminates pre-registered m=5000 confirmatory gate
  - dimension: red_team
    message: Asymmetric tuning — profiling-informed variants vs uninstrumented baseline
  - dimension: red_team
    message: Evaluation collision — instrumentation contaminates benchmarked baseline
```
