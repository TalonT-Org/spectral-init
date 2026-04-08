# Revision Guidance — Trustworthiness Performance Scaling Experiment

**Source dashboard:** `evaluation_dashboard_tw-perf-scaling_2026-04-04_163836.md`
**Plan:** `experiment_plan_tw_perf_scaling_2026-04-04_161626.md`
**Generated:** 2026-04-04_164114

All 4 stop-trigger findings are ADDRESSABLE. Apply the following fixes to the experiment plan before re-submitting for review.

---

## 1. Required Fixes

### Fix RT-1: Pre-Specify Combined Variant Composition Unconditionally

**Finding:** H6 uses "if applicable" to condition the `combined` variant on individual results, making it unfalsifiable by construction. The GO threshold is also vulnerable to baseline manipulation via synthetic data cache thrashing.

**Required changes to experiment plan:**

1. **Remove "if applicable" from H6.** Replace:
   > "Combined exact optimizations (thread-local + partial-rank + distance kernel, **if applicable**) achieve 3–6x end-to-end speedup at n=100K (warm)."

   With:
   > "Combined exact optimizations — unconditionally defined as the simultaneous application of `thread_local` + `partial_rank` + `avx2_kernel` — achieve 3–6x end-to-end speedup at n=100K (warm). This composition is fixed before any individual variant results are examined."

2. **Add a pre-specification constraint** in the Inputs and Data section or a new "Pre-specification" section:
   > "The `combined` variant composition is unconditionally fixed as thread_local + partial_rank + avx2_kernel regardless of individual variant outcomes. This is registered here before any Phase 3 measurements are run."

3. **Add baseline integrity control** to the Controlled Variables table:
   > | Synthetic data sizing | Fixed at n values sized ≥2× L3 cache to ensure consistent cache pressure across all runs; baseline binary compiled and frozen before any variant is measured |

---

### Fix RT-2: Enforce Ordering of H5 Confirmatory Test Before m-Sweep

**Finding:** The plan declares m=5000 pre-registered but provides no structural enforcement. The sweep can be examined before the confirmatory result is sealed, allowing post-hoc re-registration.

**Required changes to experiment plan:**

1. **Add a dedicated `run_h5_confirmatory.sh` script** to the scripts directory listing and describe it explicitly:
   > `run_h5_confirmatory.sh` — Runs ONLY the m=5000 subsampling test on MERFISH data and writes a sealed GO/NO-GO result to `results/subsampling/h5_confirmatory_result.json` before any sweep is executed. This script must complete and its output committed to git before `run_subsampling_sweep.sh` is invoked.

2. **Add a phase-ordering constraint** in Phase 3 (or a new Phase 2.5):
   > "**H5 Confirmatory Gate (must run before m-sweep):** Execute `run_h5_confirmatory.sh` to evaluate m=5000 on MERFISH data. Record the result in `results/subsampling/h5_confirmatory_result.json`. This file must be written and its path committed to git as a sealed artifact before `run_subsampling_sweep.sh` is executed. The GO/NO-GO decision for H5 is taken from this file only — sweep results do not affect the H5 verdict."

3. **Make the m-sweep explicitly secondary:**
   > "The subsequent `run_subsampling_sweep.sh` over m={500,1000,2000,5000,10000} is an exploratory secondary analysis for reporting purposes only and has no bearing on the H5 GO/NO-GO gate."

---

### Fix RT-3: Disclose Baseline Comparator Framing

**Finding:** The plan does not acknowledge that reported speedups are relative to current production code, not a maximally-optimized baseline. This omission makes the ≥1.5× GO threshold appear to measure absolute headroom rather than improvement over status quo.

**Required changes to experiment plan:**

1. **Add disclosure to the Controlled Variables table:**

   | Variable | Fixed Value | Rationale |
   |----------|-------------|-----------|
   | Baseline optimization effort | None — current production `trustworthiness()` as-shipped; no PGO, parallelism tuning, or code changes applied | Speedups are measured relative to production code, not a theoretical performance floor; reported ratios are upper bounds on real-world improvement over any further-optimized baseline |

2. **Add one sentence to the Motivation section:**
   > "Reported speedup ratios (≥1.5× GO threshold) are measured relative to the current unmodified production `trustworthiness()` function. They represent an upper bound on real-world gains if the baseline itself received additional optimization."

---

### Fix RT-4: Decouple Criterion Benchmarks from the `testing` Feature

**Finding:** The `[[bench]]` entry requires `features = ["testing"]`, forcing all Criterion benchmarks to run with timing guards compiled in. This (a) adds `Instant::now()` overhead inside the benchmarked hot path, and (b) may alter compiler inlining decisions, meaning the benchmarked baseline is not the same machine code that motivated the experiment.

**Required changes to experiment plan:**

1. **Remove `required-features = ["testing"]` from the bench Cargo.toml entry:**

   Change:
   ```toml
   [[bench]]
   name = "trustworthiness_bench"
   harness = false
   required-features = ["testing"]
   ```

   To:
   ```toml
   [[bench]]
   name = "trustworthiness_bench"
   harness = false
   ```

   The `testing` feature is needed only by the `tw_profiler` binary (Phase 0/2), not by Criterion benchmarks.

2. **Ensure timing guards do not appear inside the Criterion-benchmarked hot path.** If `#[cfg(feature="testing")]` guards exist inside `trustworthiness()` at the inner loop level, they must be either: (a) positioned outside the loop being benchmarked, or (b) replaced by a separate instrumented function that wraps the clean production function.

3. **Add a mandatory pre-benchmark verification step** to `run_criterion.sh`:
   > "Before recording any speedup ratios: run `cargo rustc --release --bench trustworthiness_bench -- --emit=asm` on both the clean build and the build with `--features testing` to confirm that the inner loop assembly for `trustworthiness()` is identical. Document this verification result in `results/criterion/binary_identity_check.txt`."

---

## 2. Design Questions for Human Review

None — all stop-trigger findings were ADDRESSABLE with mechanical fixes.

---

## 3. Structural Findings (for context)

None — no STRUCTURAL findings among stop triggers.

---

## Summary

| Finding | Classification | Fix Type |
|---------|---------------|----------|
| RT-1: Goodhart Exploitation | ADDRESSABLE | Pre-specify combined variant composition; add cache-pressure control |
| RT-2: Data Leakage | ADDRESSABLE | Add `run_h5_confirmatory.sh` + phase ordering constraint |
| RT-3: Asymmetric Tuning | ADDRESSABLE | Add disclosure statement to Controlled Variables |
| RT-4: Evaluation Collision | ADDRESSABLE | Remove `required-features = ["testing"]` from bench; add assembly verification step |

After applying all four fixes, re-submit the revised experiment plan through `plan-experiment` or directly to `review-design`.
