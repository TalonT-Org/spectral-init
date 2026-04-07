# Resolve-Design-Review — Triage Analysis

**Plan:** Trustworthiness Performance — Scaling Analysis and Optimization Evaluation
**Dashboard:** `evaluation_dashboard_tw-perf-scaling_2026-04-04_163836.md`
**Triage date:** 2026-04-04
**Timestamp:** 2026-04-04_164114

---

```
Triage complete (BEFORE any guidance written)
ADDRESSABLE: 4 | STRUCTURAL: 0 | DISCUSS: 0
```

---

## Stop Triggers Analyzed

4 critical red-team findings extracted from dashboard YAML block.

---

### RT-1: Goodhart Exploitation

**Verdict:** ADDRESSABLE

**Evidence:**
> H6 states "Combined exact optimizations (thread-local + partial-rank + distance kernel, **if applicable**)" — the phrase "if applicable" explicitly makes composition conditional on individual results. The plan lists "combined" as an independent variable without defining its composition rule. The GO threshold is defined as `baseline_mean_ms / variant_mean_ms` with no constraint on how baseline data sizing is controlled.

**Fix sketch:**
1. Add a pre-specification sentence before any runs: "The combined variant is unconditionally defined as the simultaneous application of thread_local + partial_rank + avx2_kernel regardless of individual variant outcomes." Remove the "if applicable" qualifier from H6.
2. Fix baseline integrity by adding a controlled variable specifying synthetic data size as a fixed multiple of L3 cache size (e.g., 2× to ensure consistent cache pressure), and require that the baseline binary is compiled and frozen before any variant is measured.

---

### RT-2: Data Leakage — m-sweep Contaminates m=5000 Confirmatory Gate

**Verdict:** ADDRESSABLE

**Evidence:**
> The plan states "The fraction scan over m={500, 1000, 2000, 5000, 10000} is conducted as an exploratory secondary analysis only" and "The GO/NO-GO criterion for H5 is evaluated at m=5000 exclusively," but the phase structure shows no ordering constraint between the H5 confirmatory test and the m-sweep, and there is no separate script for a confirmatory-only m=5000 run prior to the sweep.

**Fix sketch:**
Add an explicit phase-ordering constraint: create a dedicated script (`run_h5_confirmatory.sh`) that runs only the m=5000 confirmatory test and records its GO/NO-GO result before `run_subsampling_sweep.sh` is executed. Enforce this sequencing in the pipeline (e.g., make the sweep script depend on a sealed, timestamped output artifact from the confirmatory script). Pre-register the confirmatory result file with a hash or timestamp to a version-controlled log before any sweep results are examined.

---

### RT-3: Asymmetric Tuning — Profiling-Informed Variants vs. Uninstrumented Baseline

**Verdict:** ADDRESSABLE

**Evidence:**
> The plan states the purpose is to "characterize the per-step scaling profile, evaluate six specific algorithmic and SIMD alternatives, and produce a ranked GO/NO-GO recommendation" and compares "optimized variants against the current production baseline." The Controlled Variables section has no mention of baseline optimization effort, PGO, or justification for baseline as floor.

**Fix sketch:**
Add a disclosure statement to the Controlled Variables section explicitly stating: (1) the baseline is the current unmodified production code with no additional optimization effort applied; (2) reported speedups are therefore an upper bound on real-world improvement relative to a maximally-optimized baseline; (3) the GO threshold of ≥1.5× is calibrated against production-as-shipped, not a theoretical performance floor. This framing accurately represents what the experiment measures without requiring changes to the experimental design itself.

---

### RT-4: Evaluation Collision — Instrumentation Contaminates Benchmarked Baseline

**Verdict:** ADDRESSABLE

**Evidence:**
> The benchmark declaration `[[bench]] name = "trustworthiness_bench" harness = false required-features = ["testing"]` forces the `testing` feature on for all Criterion runs. The Phase 2 profiling script also uses `--features testing,cli`. No verification step exists to confirm binary identity between instrumented and clean builds. The `#[cfg(feature="testing")]` guards inside `trustworthiness()` are present in every benchmarked variant including the baseline.

**Fix sketch:**
Remove `required-features = ["testing"]` from the `[[bench]]` Cargo.toml entry so Criterion benchmarks compile without the testing feature. Extract timing instrumentation out of the hot inner loop (e.g., into a wrapper or separate profiling binary) so the benchmarked code path is identical to production. Add a mandatory pre-benchmark verification step that runs `cargo-show-asm` or `perf annotate` on both the clean build and the instrumented build to confirm the inner loop assembly matches before recording any speedup ratios.

---

## Resolution

All 4 stop-trigger findings are ADDRESSABLE → resolution = **revised**
