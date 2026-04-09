# Revision Guidance: Trustworthiness Step-Timing on MERFISH Real Data

**Verdict: REVISE**
**Experiment Type:** robustness_audit
**Review timestamp:** 2026-04-07 17:00:12

---

## Required Revisions (Critical Findings)

These gaps must be addressed before execution produces interpretable results.

---

### 1. Measurement metric does not align with optimization ROI inference

**Affected sections:** Dependent Variables, Threats to Validity, Analysis Plan

**Gap:** The primary metric (x_space_pct = thread-aggregate ns fraction) conflates parallelism degree with compute cost. Thread-aggregate ns accumulates across all Rayon threads; a step that parallelises well will have inflated thread-ns totals relative to its wall-clock contribution. The binary verdict "x_space_pct ≥ 50%" is used to infer which step to target for wall-clock speedup (Amdahl's law), but thread-ns fraction is not equivalent to wall-clock share when steps have different parallelisation efficiency. The plan acknowledges "thread-aggregate ns ≠ wall-clock" but then asserts "ratios between steps are valid for bottleneck analysis" without any qualification that limits this claim to equal-parallelism conditions.

**Risk:** The bottleneck conclusion and the engineering investment decision (X-space ANN target) may be drawn from a metric that systematically over-counts well-parallelised steps. If x_dist and y_dist have different thread utilisation efficiency (likely given different SIMD kernel widths), the fraction ordering may not reflect wall-clock priority.

---

### 2. Baseline comparison is structurally asymmetric

**Affected sections:** Inputs and Data, Controlled Variables, Analysis Plan

**Gap:** The synthetic Gaussian baseline is a historical snapshot (`profiler_baseline_n10000.json` from `2026-04-06-y-heap-bottleneck-optimization`) collected in a prior session with no documented warmup count, iteration count, binary version, commit hash, or machine state. The MERFISH run will be collected fresh under the plan's controlled conditions. At least one related baseline JSON (`gaussian_n10000_combined.json`) shows all step counter values at zero, suggesting the profiling feature may not have been active during some historical collections. The controlled-variable table applies only to the MERFISH run, not to both sides of the comparison.

**Risk:** Any observed difference in x_space_pct between MERFISH and synthetic may be attributable to differences in collection conditions (profiler version, binary hash, warmup count, environmental state) rather than to data geometry. The comparison cannot distinguish data effect from measurement artifact.

---

### 3. Warmup timing lines included in step_timing JSON

**Affected sections:** Implementation — Phase 2, Analysis Plan

**Gap:** Under the profiling feature flag, each `trustworthiness()` call emits `[timing:...]` lines to stderr. `--stderr-capture` records all stderr from process start through all calls (2 warmup + 5 timed = 7 total). `parse_step_timing()` reads the full capture file without distinguishing warmup-call lines from timed-call lines. The step_timing JSON therefore accumulates 7 call samples rather than the 5 timed iterations described in the analysis plan. Per-commit #242 (`fix(profiling): reset AtomicU64 counters at top of trustworthiness()`), counters reset at the top of each call — meaning each call's values are emitted independently — but the capture file cannot be post-hoc partitioned into warmup vs. timed without external tagging.

**Risk:** The "mean step fractions across 5 timed iterations" described in the analysis plan is computed from 7 samples (including 2 warmup samples that may not have reached steady-state thread behaviour). The contamination may be small but is unquantified and unacknowledged.

---

### 4. No statistical justification for thresholds or sample size

**Affected sections:** Hypothesis, Analysis Plan, Success Criteria

**Gap:** The ±5pp null band and ≥50% H1 cutoff were set after observing the 56.1% synthetic baseline — they were not pre-registered. No power analysis establishes whether n=5 timed iterations can detect a ±5pp difference against the measurement noise on this hardware. Type I and Type II error rates are entirely unacknowledged. The historical baseline has no distributional characterization; its measurement uncertainty is unknown. The thresholds are heuristic but are used as if they were statistically grounded decision boundaries.

**Risk:** The binary verdict (H0 vs. H1) may be made with insufficient precision to detect a real difference (Type II error) or may declare significance on noise (Type I error), neither of which can be assessed without characterizing the measurement variance.

---

### 5. Single process invocation — iterations are not statistically independent

**Affected sections:** Execution Protocol, Variance Protocol

**Gap:** The plan uses a single `tw_profiler` invocation producing 5 within-run iterations. These iterations share process state (Rayon thread pool, allocator arenas, TLB, branch predictor, CPU thermal state). A single adversarial event (OS scheduler jitter, thermal throttle event) cannot be distinguished from a true shift in step fractions without replicated independent process invocations.

**Risk:** The mean of 5 correlated within-run iterations may be systematically biased by a single session's hardware state. Without independent replications, run-to-run variance is entirely uncharacterized, and the reported mean may not represent the distribution of outcomes.

---

### 6. n=10K acknowledged as understating X-dominance at production scale

**Affected sections:** Inputs and Data, Threats to Validity (External)

**Gap:** The plan itself notes that PR #238's 61% figure may be from n=100K, and that penalty cost grows relative to distance cost as n increases (O(n²·k) vs O(n²·d)). Testing at n=10K may structurally suppress X-space fractions relative to production scale, making a negative result at n=10K inconclusive rather than falsifying. The conclusion from this experiment (whether to proceed with X-space ANN) applies to production n, not n=10K.

**Risk:** A result showing x_space_pct < 50% at n=10K may be reported as "X-space ANN not the productive target" when the correct conclusion is "X-space fraction at n=10K is insufficient to conclude for production n." The experiment scope does not match the decision scope.

---

### 7. d_x ambiguity makes the experiment's operating point indeterminate

**Affected sections:** Inputs and Data, Analysis Plan

**Gap:** d_x for the MERFISH n=10K fixture is described as "TBD (~48 at f64 from file size, ~8-20 per MERFISH norms)" — a 3-6× uncertainty range. Since x_dist cost scales as O(n² × d_x), d_x directly controls the x_dist fraction. The plan's central claim (whether X-space dominates depends on data geometry vs. synthetic) cannot be evaluated without knowing whether the experiment is operating at d_x=8 or d_x=48.

**Risk:** The experiment may be executed and interpreted without knowing the single most important input to the x_dist/y_dist ratio. If d_x turns out to be 8-10 (similar to the synthetic baseline assumption), the MERFISH result cannot demonstrate anything new about X-space dominance.

---

### 8. Reproducibility specification insufficient for independent replication

**Affected sections:** Controlled Variables, Inputs and Data, Environment

**Gap:** The plan specifies "tw_profiler at HEAD" (no commit hash), "Rust toolchain at HEAD" (no version), "Local development machine" (no CPU model, core count, cache sizes, or SIMD level), and references input files by path without checksums. The Python environment is pinned only to minor versions. An independent party has no means to reproduce the experiment or verify that two runs are comparable.

**Risk:** Without version anchors, results are non-reproducible and non-comparable across time. Two runs at different HEAD states may produce materially different step fractions due to codegen changes, SIMD kernel changes, or profiler instrumentation changes.

---

## Recommended Revisions (Warning Findings)

These should be addressed to strengthen interpretability and design quality.

---

### 9. Hypothesis structure is logically inconsistent

**Affected sections:** Hypothesis, Success Criteria

**Gap:** H1 has a compound structure (x_space differs by >5pp AND x_space ≥ 50%). The success criteria collapse this to a single ≥50% threshold, which is both inconsistent with H0 as stated and creates a region (x_space between ~45% and 51.1%) where neither H0 nor H1 is clearly declared. The ±5pp criterion and the ≥50% criterion are not reconciled.

**Risk:** If the result falls in the ambiguous region, no pre-specified resolution rule exists and the conclusion will be judgment-dependent rather than criterion-dependent.

---

### 10. Causal mechanism claims are not disentangleable from this design

**Affected sections:** Analysis Plan — d_x Effect, Analysis Plan — Clustering Geometry Effect

**Gap:** The plan attributes step-fraction changes to two distinct mechanisms (d_x magnitude and clustering geometry) that change simultaneously between synthetic Gaussian and MERFISH data. The design cannot isolate either mechanism. The theoretical FLOP-count prediction (`d_x / (d_x + d_y + constant) × 100`) conflates FLOP count with ns throughput; memory bandwidth and cache effects are independent mediators not accounted for.

**Risk:** The mechanism-level claims in the analysis plan ("if d_x = 48, expect x_dist fraction ~50-60%") cannot be validated or falsified by this design. Reporting them as explanations for observed differences would overstate the design's causal reach.

---

### 11. n=50K stretch scope and triggering criterion unspecified

**Affected sections:** Inputs and Data, Success Criteria, Execution Protocol

**Gap:** The n=50K run is present in `run_profiler.sh` but labeled "comment out if time-constrained" without a criterion for when to proceed. The scale-stability success criterion depends on this run. The plan does not specify what would constitute "time-constrained" or who makes the call.

**Risk:** The scale-stability criterion may be silently dropped without acknowledgment, narrowing the conclusions without formal documentation of the scope change.

---

### 12. Generalization boundary is implicit

**Affected sections:** Motivation, Success Criteria

**Gap:** The plan claims results "inform whether to proceed with X-space ANN approximation" without stating the scope of that inference. The experimental coverage (1 MERFISH dataset, 1 tissue, 1 hardware, n=10K, k=15) does not support generalizing to "all biological datasets" without explicit scope qualification.

**Risk:** A positive result may be applied to engineering decisions that require broader coverage, leading to suboptimal optimization targets in production contexts the experiment did not test.

---

## Red-Team Decision Points

*Each item requires an explicit decision by the plan author before proceeding.*

**RT-1: Is x_space_pct a valid proxy for optimization ROI?**
The plan's success metric (x_space_pct ≥ 50%) measures thread-aggregate ns share. This metric is partially constructed by d_x and thread count, not by data geometry. A decision is needed: is thread-ns fraction the intended measure, or should the plan define a wall-clock-based alternative?

**RT-2: Were the ±5pp and ≥50% thresholds set after observing the synthetic baseline?**
If yes, these thresholds are post-hoc and the plan should acknowledge this explicitly rather than presenting them as pre-registered hypotheses. A decision is needed: report as exploratory, or pre-register new thresholds on a held-out data subset before seeing MERFISH results.

**RT-3: Does the synthetic baseline have comparable profiling conditions to the planned MERFISH run?**
At least one historical baseline JSON shows zero counter values (inoperative profiling). A decision is needed: confirm the target baseline JSON was collected with active profiling under equivalent conditions, or re-run the synthetic baseline fresh under the current plan's controlled conditions.

**RT-4: Was the MERFISH dataset selected independent of expected results?**
If the MERFISH fixture was chosen because preliminary runs suggested it would show X-space dominance, the dataset selection is not independent. A decision is needed: acknowledge the non-blind selection and scope the conclusion accordingly.

**RT-5: Does profiling instrumentation overhead affect step fractions asymmetrically?**
The `Instant::now()` + `fetch_add` overhead per loop iteration may represent different fractions of measured time for short-kernel steps (y_dist flat_simd) vs. long-kernel steps (x_dist at high d_x). A decision is needed: confirm that instrumentation overhead is negligible relative to step compute time, or quantify the asymmetry.

**RT-6: Is the d_x effect analysis falsifiable?**
The theoretical prediction for x_dist/y_dist ratio has no pre-specified tolerance. A decision is needed: define what quantitative match to the FLOP-count model would constitute confirmation vs. refutation.

**RT-7: Are warmup timing lines excluded from the step_timing analysis?**
Given `parse_step_timing()` reads all captured stderr lines and counters reset per call, the capture file contains 2 warmup + 5 timed sample sets. A decision is needed: confirm whether the profiler harness separates warmup from timed emissions, or acknowledge that the reported fractions include warmup samples.

**RT-8: Can d_x and geometry effects be disentangled?**
MERFISH differs from synthetic data on d_x, distribution, and clustering simultaneously. Any observed step-fraction difference conflates all three. A decision is needed: scope the conclusion to "MERFISH vs. Gaussian under these combined conditions" rather than attributing differences to any single mechanism.
