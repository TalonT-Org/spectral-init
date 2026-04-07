# Review-Design Evaluation Dashboard
## Trustworthiness Performance — Scaling Analysis and Optimization Evaluation
**Plan:** `experiment_plan_tw_perf_scaling_2026-04-04_174500.md` (Revision 3)
**Verdict:** 🛑 **STOP**
**Experiment Type:** benchmark
**Review Date:** 2026-04-04 18:15

---

## Verdict Banner

> **STOP** — Four red-team adversarial findings at critical severity block execution. The plan's GO/NO-GO decision structure is fundamentally disconnected from the production problem it claims to solve (n=250K), the primary pre-registration mechanism is unenforceable, and the baseline measurement instrument introduces asymmetric overhead that invalidates speedup ratios before any data is collected. These are structural defects, not revisions at the margin. This is the plan's third revision; the STOP findings below are distinct from all prior STOP triggers.

---

## Dimension Scorecard

| Dimension | Level | Weight | Critical | Warning | Info | Summary |
|-----------|-------|--------|----------|---------|------|---------|
| estimand_clarity | L1 | H | 0 | 7 | 3 | All warnings — H0/H2/H3/H6 estimands partially underspecified |
| hypothesis_falsifiability | L1 | H | 0 | 4 | 5 | All warnings — H4 conditionally unfalsifiable, H1_combined inconclusive path unbounded |
| baseline_fairness | L2 | H | **2** | 3 | 2 | testing feature contamination + per-step eprintln! asymmetry |
| unit_interference | L2 | H | **2** | 3 | 1 | Criterion groups not isolated; thread_local state bleeds cross-group |
| error_budget | L3 | M | **3** | 4 | 1 | No power analysis; FWER uncontrolled at 26%; Type II entirely absent |
| statistical_corrections | L3 | H | **3** | 2 | 0 | No corrections pre-specified; existential H1_combined inflates FP risk |
| variance_protocol | L3 | H | **5** | 4 | 0 | 5-iter sample, no re-run protocol, WSL2 Hyper-V unaddressed |
| ecological_validity | L4 | M | **3** | 5 | 1 | d=10 benchmarks; H3/H4 SIMD conclusions don't generalize to d=50 |
| measurement_alignment | L4 | M | **2** | 4 | 0 | Criterion n≤50K cap; non-stationary distribution in tw_profiler |
| reproducibility_spec | L4 | L | **4** | 5 | 2 | No rust-toolchain.toml; MERFISH fixture uncommitted |
| red_team | — | — | **4** | 4 | 0 | **4 critical findings → STOP triggers** |

**Totals:** 28 critical · 45 warning · 14 info

---

## STOP Triggers (Red-Team Adversarial Findings)

All red-team findings carry `requires_decision: true`.

### RT-1 [CRITICAL] — Goodhart Exploitation: n=250K Production Gap
**Section:** Goodhart Exploitation — Benchmark n Mismatch

The cheapest path to a GO verdict is to optimize for n=50K (Criterion) and n=100K (tw_profiler) without benefit at the actual production ceiling n=250K. Trustworthiness is O(n²): from n=100K to n=250K the work grows 6.25×. An optimization that reduces a constant factor at cache-resident scales (n≤100K) may be memory-bandwidth-bound at n=250K with no relative benefit. **SUCCESS CRITERIA are defined only at ≤100K; there is no confirmatory gate at n=250K.** A variant could score GO at n=100K while providing zero benefit or a regression at n=250K, and this would never be detected by the experiment.

`requires_decision: true`

---

### RT-2 [CRITICAL] — Data Leakage: H5 Pre-Registration Integrity Unenforceable
**Section:** Data Leakage — H5 Pre-Registration Integrity

H5's confirmatory gate relies on a human-performed git commit step with no independent witness. Git commit timestamps are trivially forgeable with `--date`; even without forgery, a commit arriving in the same PR branch after the sweep provides no pre-registration guarantee because the ordering within a single-author branch is not independently witnessed. The pre-registration has epistemic value only if the confirmatory result is committed to a public remote **before** any measurement and can be independently verified — the plan provides no such mechanism. Additionally, the MERFISH fixture was used in two prior experiments (2026-04-03 param-sweep-robustness and 2026-04-03 MERFISH eval pipeline), making the experimenter familiar with its structure. If m=5000 was not chosen blindly but informed by prior knowledge of MERFISH spatial concentration, the sealed m value provides false assurance.

`requires_decision: true`

---

### RT-3 [CRITICAL] — Asymmetric Tuning: Partial-Rank Tie-Breaking Variance Asymmetry
**Section:** Asymmetric Tuning — Baseline vs Variant Configuration

`select_nth_unstable_by` (used in `partial_rank`) uses pdqselect's pattern-defeating quickselect. On input with many near-equal distances — plausible in MERFISH PCA-50 data where spatially proximate cells share many expression features — pdqselect can degrade toward O(n) degeneration. The baseline's `sort_unstable_by` is timing-stable across such inputs. This asymmetry makes the `partial_rank` variant's Criterion CI artificially narrow on Gaussian data (no ties) while the MERFISH fixture (used for the parity gate) has systematically different distance distributions. The experiment has no protocol for checking distance-distribution skew or tie density before running benchmarks on MERFISH data, meaning the favorable Gaussian-data speedup is not a valid predictor of partial_rank's MERFISH production behavior.

`requires_decision: true`

---

### RT-4 [CRITICAL] — Type-Specific: GO Threshold Structurally Insufficient for Production Problem
**Section:** Type-Specific: Asymmetric Effort at Production n — 250K Gap

The plan cites "~651s for 100K in Python UMAP, Rust expected ~10–60s" — a 6× range. If the actual Rust baseline at n=100K is 60s, a 1.5× GO means 40s. At n=250K (O(n²) scaling), this becomes 250s — still unacceptably slow for any real pipeline. The ≥1.5× threshold was calibrated against an unconstrained expectation that may not reflect reality. The experiment produces a binary GO/NO-GO **without measuring whether the resulting absolute performance is actually acceptable at the target scale**. GO on the speedup ratio does not imply "the pipeline can now process n=250K in reasonable time." The research question ("enable MERFISH evaluation at 250K+") is not answered by a relative speedup threshold tested only at n≤100K.

`requires_decision: true`

---

## Additional Red-Team Warnings (`requires_decision: true`)

### RT-5 [WARNING] — Survivorship Bias: 5 Iterations Insufficient on WSL2
With 5 measured iterations at n=100K on WSL2 (Windows Hyper-V scheduler), CPU thermal throttling and Hyper-V scheduling jitter can cause ±15% variance between iterations. Five samples from this distribution cannot reliably distinguish a 1.5× real speedup from a favorable thermal event. With 5 variants × 5 iterations = 25 chances to observe a favorable run, multi-comparison survivorship is compounded.

`requires_decision: true`

### RT-6 [WARNING] — Evaluation Collision: eprintln! Overhead in Timing Loop
The plan routes per-step timing via `eprintln!` inside the `par_iter()` closure. At n=100K this emits ~700K `eprintln!` calls per tw_profiler run, each acquiring the stderr Mutex. This serializes rayon workers asymmetrically — the baseline (which emits per-step output) pays this cost; optimized variants that eliminate steps reduce the eprintln! call count. The measurement instrument is coupled to the treatment. The contamination direction: baseline is measured slower than it would be without instrumentation.

`requires_decision: true`

### RT-7 [WARNING] — Combined Variant Laundering
The combined variant (thread_local + partial_rank + avx2_kernel) can score GO on H6 even if its components receive individual NO-GO verdicts. If H1 (partial_rank) is NO-GO due to MERFISH tie-density regression but thread_local compensates on Gaussian data, H6 GO does not mean all components are safe to ship. The plan provides no disambiguation protocol for this scenario.

`requires_decision: true`

### RT-8 [WARNING] — H5 MERFISH Fixture: Not Truly Blind
The MERFISH fixture structure (spatial stratification, PCA dimensionality, distance distribution at 10K rows) was extensively characterized in two prior 2026-04-03 experiments. An experimenter who has run those experiments knows which rows seed=42 selects and whether those rows exhibit favorable or unfavorable approximation behavior. The "pre-registered" m=5000 may reflect tacit knowledge from prior analysis rather than a truly blind choice.

`requires_decision: true`

---

## Critical Non-Red-Team Findings

### Baseline Fairness (L2, H-weight)

**[CRITICAL] Testing Feature Contaminates Baseline Compilation**
`tw_profiler` requires `--features testing cli`. The `testing` feature activates hard `assert!()` guards (replacing `debug_assert!()`), changes `dense_n_threshold()` to read an env var, and alters visibility of internal functions. The baseline variant runs under materially different compiled code than production, while optimized variants are not explicitly stated to share this penalty. Speedup ratios from tw_profiler reflect testing-vs-testing comparisons at best, but the plan describes tw_profiler as measuring "wall-clock per iteration" without clarifying that all variants compile with `--features testing`. If optimized variants omit per-step instrumentation but share the testing build, the critical concern is reduced — but this is never made explicit.

**[CRITICAL] Per-Step eprintln! Applied to Baseline Only**
The plan states per-step diagnostic output via `#[cfg(feature="testing")]` applies to the `baseline` variant only in tw_profiler. Any conditional I/O path in the baseline's hot loop that is absent in optimized variants introduces asymmetric work. The baseline pays branch and `eprintln!` cost that optimized variants do not, biasing measured latency against the baseline. All speedup ratios are therefore inflated relative to an uncontaminated comparison.

---

### Unit Interference (L2, H-weight)

**[CRITICAL] thread_local Buffer Persists Across Criterion Groups**
`thread_local! { static DIST_X: RefCell<Vec<(f64, usize)>> }` is bound to OS thread lifetime, not benchmark group scope. After `bench_tw_thread_local` completes, each rayon worker thread retains a fully-allocated Vec with n=50K capacity. Subsequent groups (`bench_tw_partial_rank`, `bench_tw_avx2`, `bench_tw_combined`) issue allocations against an allocator with pre-warmed pages, receiving systematically lower allocation latency than they would in isolation. Groups measured after `bench_tw_thread_local` appear faster due to contamination, not algorithmic improvement.

**[CRITICAL] All 5 Criterion Groups in a Single Binary Invocation**
`cargo criterion --bench trustworthiness_bench` executes all groups sequentially in one process. Criterion's per-group warm-up does not flush OS page caches, CPU LLC contents, or allocator free-lists accumulated by prior groups. `bench_tw_baseline` warms the instruction cache for all subsequent groups. `bench_tw_combined`, measured last, runs in the most favorable memory-subsystem state. There is no isolation between groups. Correct isolation requires either separate binary invocations or explicit allocator state resets between groups.

---

### Statistical (L3, H-weight)

**[CRITICAL] No Power Analysis for tw_profiler 5-Iteration Design**
A 20% wall-clock CV at n=100K (common for memory-bandwidth-bound workloads) requires ~14 iterations at 80% power to detect a 1.5× ratio at α=0.05. Five iterations provides unknown power. The plan asserts "sufficient for mean ± std" without calculation.

**[CRITICAL] FWER Uncontrolled at ~26%**
Six variants at implicit α=0.05 → FWER = 1−(0.95)^6 ≈ 26%. No Bonferroni, Holm, or BH correction pre-specified. A false GO on any single variant ships an unvalidated optimization.

**[CRITICAL] Type II Error Rate Absent**
False negative rate (missing a real speedup) has zero acknowledgment. No MDE stated, no β budget. Since shipping a fast-enough optimization is the goal, false negatives carry direct product cost.

**[CRITICAL] H1_combined Existential Framing Compounds False-Positive Risk**
"At least one exact optimization achieves ≥1.5×" is disjunctive over 4 variants. If each has 5% false-positive probability, joint FP probability exceeds 18%. No inflation adjustment or named primary variant declared.

**[CRITICAL] No Multiple-Comparisons Correction Pre-Specified**
~25 variant × n comparisons across tw_profiler with no correction. Uncontrolled family-wise false positive rate for the composite GO/NO-GO recommendation.

---

### Variance Protocol (L3, H-weight)

**[CRITICAL] 5 Iterations at n=100K Statistically Inadequate**
At df=4, t-critical = 2.78 for 95% CI, producing wide intervals. The chi-squared 95% CI on variance at df=4 spans [0.41σ², 5.85σ²] — reported std could understate true std by 1.5× or overstate it 2.4×. The mean ± std of 5 samples is not a defensible variance estimate.

**[CRITICAL] No High-Std Re-Run Protocol**
No CV threshold triggers a re-run. No outlier rejection policy. No minimum acceptable signal-to-noise ratio defined. Experimenters have no decision rule when 5-sample distribution is noisy.

**[CRITICAL] OS Performance Mode and CPU Isolation Not Specified**
No CPU frequency governor, CPU isolation (taskset), or process priority specification. On WSL2, Windows power management can cause significant inter-sample jitter.

**[CRITICAL] WSL2/Hyper-V Virtualization Overhead Unaddressed**
Hyper-V scheduling introduces bursty latency absent on bare-metal Linux. Two warmup iterations are insufficient to stabilize under Hyper-V VM scheduling preemptions. No mitigation specified.

**[CRITICAL] Statistical Adequacy of 5-Iteration std**
Reporting mean ± std from n=5 samples as a variance estimate is too weak for GO/NO-GO decisions at 1.5× margins. Minimum 10–20 samples or bootstrap CI required.

---

### Ecological Validity (L4, M-weight)

**[CRITICAL] d=10 Benchmarks vs Production d=50 — SIMD Conclusions Do Not Generalize**
All throughput benchmarks (H1–H4, H6) use d=10 Gaussian data; production MERFISH is PCA-50 (d=50). Distance computation is O(d): at d=50 with AVX2 there are ~12.5 SIMD ops per distance vs ~2.5 at d=10. Register utilization, loop unrolling, and auto-vectorization efficiency differ materially. **H3 and H4 GO/NO-GO decisions are valid only at d=10 and cannot be presented as production characterizations.**

**[CRITICAL] H3 AVX2 Detection at d=10 Doesn't Predict d=50 Behavior**
At d=10: 2 full 4-wide AVX2 passes + 2-element scalar tail — poor unroll opportunity. At d=50: 12 full passes — different register pressure and loop scheduling. A confirmed auto-vectorization at d=10 does not imply equivalent codegen quality at d=50.

**[CRITICAL] H4 AVX-512 Claim Inverted at d=50**
H4's core argument is 62.5% register utilization at d=10 (10/16 lanes). At d=50: 50/8 = 6.25 full passes with 2-element tail, giving ~93.75% register fill — the utilization argument is reversed. H4's NO-GO prediction at d=10 cannot be applied to production d=50.

---

### Measurement Alignment (L4, M-weight)

**[CRITICAL] Criterion n≤50K Doesn't Predict Cache-Limited n=100K Behavior**
`dist_x` at n=100K is 1.6 MB per row. This exceeds typical L2 cache, making the sort memory-bandwidth-bound above ~n=70–80K. A variant showing 1.6× speedup at n=50K via cache-residency effects may regress at n=100K. The plan requires both Criterion AND tw_profiler gates — but the two scales are not in the same memory-access regime. Criterion CI at n=50K is not a valid predictor of n=100K behavior.

**[CRITICAL] tw_profiler Mean from Non-Stationary Distribution**
The 300K Vec alloc/dealloc round-trips (baseline) grow the allocator's free-list state across iterations 1–5. Iterations 4–5 are systematically slower due to allocator fragmentation. Mean ± std of 5 samples from a non-stationary distribution does not represent steady-state throughput.

---

### Reproducibility Spec (L4, L-weight)

**[CRITICAL] No rust-toolchain.toml — Nightly Not Pinned**
Active toolchain is nightly (rustc 1.96.0-nightly, 2026-03-26) but not pinned. Independent reproducers get whatever nightly is current, affecting auto-vectorization decisions, LLVM IR codegen, and benchmark timing. A `rust-toolchain.toml` pinning the exact nightly channel and date is required.

**[CRITICAL] LLVM Version Not Pinned**
H3's cargo asm output is a direct function of LLVM codegen; different LLVM versions (shipped with different nightly builds) produce different instruction sequences. H3 is not independently reproducible without a pinned toolchain.

**[CRITICAL] MERFISH Fixture Not Yet Committed**
`tests/fixtures/merfish/` does not exist in the repository. The source MERFISH ABCA-1 data requires institutional access. If the fixture is not committed before the experiment runs, an independent reproducer cannot generate it without privileged data access.

**[CRITICAL] H3 cargo asm CPU Hardware-Specific**
`target-cpu=native` in `.cargo/config.toml` produces hardware-specific assembly. H3's binary detection result is specific to the exact CPU microarchitecture used and cannot be reproduced on different CPUs, even those supporting the same instruction set extensions.

---

## L1 Warning Summary

**estimand_clarity warnings (7):**
- H0 lacks formal contrast structure (threshold supplied retroactively by H1)
- H1 conflates profiling fraction and speedup-ratio estimands in one hypothesis
- H2 Criterion CI gate threshold not numerically stated in hypothesis definition
- H3 compound claim: binary asm check + "negligible benefit" lacks numeric threshold
- H6 range prediction 3–6× non-falsifiable; upper bound has no NO-GO rule
- DV-to-hypothesis mapping implicit — instrument selection risk post-hoc
- H0/H1/H3 missing pre-specified GO/NO-GO rules

**hypothesis_falsifiability warnings (4):**
- H0 rejection criterion only in Analysis Plan section, not Hypothesis definition
- H4 conditionally unfalsifiable when hardware lacks AVX-512 (outcome = "N/A")
- H1_combined compound AND gate creates indefinite inconclusive scope (n=75K/150K unplanned)
- H5 commit gate: file-presence check does not enforce temporal ordering vs. sweep

---

## Cannot Assess

1. **Allocator identity under WSL2** — The plan targets allocation overhead (H2: thread_local) as a primary optimization target. The actual allocator in use (system malloc via glibc, jemalloc, mimalloc) under WSL2 is not specified and not controllable from Rust without explicit Cargo feature flags. Whether thread_local buffer reuse provides the projected benefit depends critically on the allocator's free-list behavior at n=100K (1.6 MB per Vec × 2 Vecs per row × 300K allocator round-trips). Cannot assess H2's GO/NO-GO threshold validity without knowing the allocator identity.

2. **Rayon thread-pool configuration** — The plan relies on rayon's parallel_iter for penalty accumulation. Rayon's global thread pool size defaults to the number of logical CPUs, but WSL2's reported CPU count may differ from physical CPUs available due to Windows cgroup constraints. The expected parallelism for n=100K penalty accumulation — and therefore the measured wall-clock — depends on thread count, which is not specified or pinned anywhere in the plan.

3. **MERFISH fixture statistical representativeness** — The 10K subset of the 250K MERFISH dataset is derived using the first 10K rows after spatial stratification. Whether this 10K subset's distance distribution is representative of the full 250K distribution (relevant for H5's approximation quality generalization) cannot be assessed without running the actual analysis. The plan cites "MERFISH is more concentrated than Gaussian, so error may be tighter than O(1/sqrt(m)) Gaussian bound" as a justification for the 0.001 threshold, but this is stated as an expectation, not a verified property.

4. **Criterion internal sample count at n=50K** — Criterion 0.5 automatically determines iteration count based on a target time budget. At n=50K (each call takes ~5–30s), Criterion may collect only 5–15 samples before hitting its time budget. The width of the 95% CI depends on this auto-determined sample count, which is not specified in the plan and cannot be assessed without running the benchmark.

---

## Mechanizable Check Log

These are binary checks that could be automated in a CI pre-flight:

| Check | Automatable? | Status |
|-------|-------------|--------|
| `rust-toolchain.toml` exists | Yes — `test -f rust-toolchain.toml` | ABSENT |
| `tests/fixtures/merfish/` directory exists | Yes — `test -d tests/fixtures/merfish` | ABSENT |
| `tw_profiler` binary defined in `Cargo.toml` | Yes — `grep 'name = "tw_profiler"' Cargo.toml` | ABSENT |
| `h5_confirmatory_result.json` committed before sweep via git log | Yes — git log timestamp check | NOT ENFORCEABLE by script |
| Criterion bench has `required-features` conflict | Yes — parse Cargo.toml | PRESENT (no required-features) |
| `eprintln!` calls inside `par_iter` closure | Yes — grep for eprintln in metrics.rs inner closure | CHECK NEEDED |
| Per-step instrumentation scoped to baseline variant only | Semi — code inspection required | NOT VERIFIED |

---

## Machine-Readable YAML Summary

```yaml
# --- review-design machine summary ---
verdict: STOP
experiment_type: benchmark
critical_count: 28
warning_count: 45
red_team_count: 8
stop_trigger_count: 4
stop_trigger_dimensions:
  - red_team (critical × 4)
revision3_new_stops:
  - RT-1: n=250K production ceiling not measured
  - RT-2: H5 pre-registration not independently witnessable
  - RT-3: partial_rank asymmetric timing on MERFISH distance ties
  - RT-4: GO threshold disconnected from production absolute-performance requirement
```
