# Evaluation Dashboard — KD-tree Y k-NN Trustworthiness Benchmark

**Plan:** `experiment_plan_kdtree_ynn_trustworthiness_50k_2026-04-07_060000.md`  
**Reviewed:** 2026-04-06  
**Reviewer:** review-design skill v1

---

## ✅ VERDICT: GO

The plan is well-designed with extensive pre-specification rationale, explicit threat documentation (W1–W13, RT1–RT8), and a coherent conjunctive success structure. Warning count (51) is below the proportional threshold (65). No critical findings. Proceed to implementation.

---

## Classification Summary

| Field | Value | Source |
|-------|-------|--------|
| Experiment type | `benchmark` | Prose extraction |
| Active modifiers | `+high_cost`, `+deployment`, `+multi_metric` | Triage analysis |
| Active dimensions | 13 | Spawned subagents |
| Warning threshold | 65 (13 × 5) | Calculated |
| Critical findings | 0 | Synthesis |
| Warning findings | 51 | Synthesis |
| Info findings | ~29 | Synthesis |
| Red-team findings | 7 (6 warning, 1 info) | Red-team agent |

**Modifier effects applied:**
- `+high_cost` (8–16h compute > 4h): `resource_proportionality` L→M
- `+deployment` (ships to production `trustworthiness()`): `ecological_validity` floor = M
- `+multi_metric` (6 DVs ≥ 3): `statistical_corrections` M→H

---

## Dimension Scorecard

| Dimension | Weight | Level | Findings | Severity Summary |
|-----------|--------|-------|----------|-----------------|
| estimand_clarity | H | L1 | 0 | ✅ No issues identified |
| hypothesis_falsifiability | H | L1 | 4 | ⚠️ 2W 2I |
| baseline_fairness | M | L2 | 5 | ⚠️ 4W 1I |
| unit_interference | M | L2 | 6 | ⚠️ 4W 2I |
| causal_structure | **S** | — | — | SILENT (not spawned) |
| error_budget | M | L3 | 7 | ⚠️ 4W 3I |
| statistical_corrections | H | L3 | 5 | ⚠️ 3W 2I |
| variance_protocol | H | L3 | 7 | ⚠️ 5W 2I |
| benchmark_representativeness | M | L4 | 6 | ⚠️ 4W 2I |
| ecological_validity | M | L4 | 8 | ⚠️ 4W 4I |
| measurement_alignment | M | L4 | 6 | ⚠️ 6W 0I |
| reproducibility_spec | M | L4 | 9 | ⚠️ 6W 3I |
| data_acquisition | M | L4 | 4 | ⚠️ 3W 1I |
| red_team | — | L3 | 7 | ⚠️ 6W 1I |

**Total: 0 critical · 51 warnings · ~29 informational**

---

## Adversarial Findings (Red-Team)

All red-team findings are capped at `warning` for benchmark experiments. All set `requires_decision: true`.

### RT-1 · Goodhart Exploitation — Distribution Specificity

> **Section:** ## Independent Variables  
> **Severity:** warning · `requires_decision: true`  
>
> Both test distributions (uniform random, Gaussian clusters) are precisely the conditions where KD-trees perform best — low intrinsic dimensionality, well-separated structure, no adversarial point arrangements. A KD-tree tuned implicitly or explicitly for these distributions could show strong benchmark speedups that collapse on real UMAP output embeddings, which are nonlinear manifold projections with density gradients. The research question is 'should we ship adaptive dispatch,' but the evaluation distributions may not represent actual Y inputs.

### RT-2 · Asymmetric Tuning — Leaf Size Without Correction Protocol

> **Section:** ## Controlled Variables  
> **Severity:** warning · `requires_decision: true`  
>
> `flat_simd` is a mature, production-hardened implementation with SIMD specialization tuned for `d_y=2` and `k=15`. The KD-tree path gets no equivalent tuning pass — `leaf_size=32` is the kiddo default, not optimized against this workload (acknowledged as RT2). If H1 is rejected, the rejection may be an artifact of the tuning asymmetry rather than a genuine structural advantage of `flat_simd`. The plan has no protocol for distinguishing these two failure modes.

### RT-3 · Survivorship Bias — Re-run Policy Creates Cherry-Pick Pathway

> **Section:** ## Analysis Plan  
> **Severity:** warning · `requires_decision: true`  
>
> The CV > 10% policy triggers a legitimate re-run of noisy cells, but does not define what constitutes a valid vs. invalid rep, or whether the replacement rep must be included in the final median. This creates a pathway where the analyst could effectively select from a larger pool than 3 reps under the guise of variance correction, without any protocol violation. The mandated `run_log.json` records outcomes but does not prevent selective exclusion of disqualified reps.

### RT-4 · Evaluation Collision — Profiler Atomics Compiled into Benchmark Binary

> **Section:** ## Implementation Plan Summary  
> **Severity:** warning · `requires_decision: true`  
>
> Phase 5 adds profiler atomics into `src/metrics.rs` via the same runtime bool dispatch used in the benchmark. If profiler instrumentation is compiled into the same binary as the Criterion benchmark (even gated by a feature flag), the atomic store/load instructions are present in the instruction stream and may perturb branch prediction, instruction cache layout, and LLVM optimization differently for `kdtree` vs `flat_simd`. The plan notes separate process invocations for the profiler but does not address whether profiler instrumentation code is compiled out during Criterion runs.

### RT-5 · Data Leakage — 75K Held-Out Point Selection Implies Foreknowledge

> **Section:** ## Independent Variables  
> **Severity:** warning · `requires_decision: true`  
>
> The 75K held-out point is called "independent validation," but the choice of 75K implies foreknowledge that the crossover lies in the 50K–100K range. If any informal profiling or prior knowledge of kiddo's scaling behavior informed placing the held-out point at 75K rather than, say, 25K or 150K, the validation point is not truly independent of the design — it was selected to be near the expected crossover, which inflates confidence in the crossover estimate.

### RT-6 · Goodhart Exploitation — const-generic d_y=2 Scope

> **Section:** ## Controlled Variables  
> **Severity:** warning · `requires_decision: true`  
>
> The KD-tree being benchmarked is a dimensionality-specialized variant (`const-generic K=2`) that may not reflect the general-case implementation for `d_y ≠ 2`. The framing of "should we ship adaptive dispatch" implies a production decision. If the shipped implementation must handle `d_y=3` or higher (e.g., 3D UMAP embeddings), the benchmark result for `d_y=2` provides no validity evidence for those cases, and the dispatch threshold `T_cross` derived here may be wrong by an order of magnitude for `d > 2`.

### RT-7 · Asymmetric Engineering Effort — New Method Written Under Evaluation Conditions (info)

> **Section:** ## Implementation Plan Summary  
> **Severity:** info · `requires_decision: true`  
>
> The KD-tree code path is being written fresh for this experiment, meaning the author controls its implementation quality with full awareness of the evaluation conditions (`d_y=2`, `k=15`, `n=50K/100K`, uniform/Gaussian). This creates an implicit Goodhart loop: the implementation is authored with knowledge of the exact test conditions, unlike `flat_simd` which was authored independently. This is an accepted bias in challenger-vs-incumbent benchmarks, noted for transparency.

---

## Key Warning Clusters by Dimension

### Measurement Alignment (6W — highest single-dimension count)

These findings concern whether the reported metrics actually map to the research question:

1. **Mean vs. median-of-means ambiguity** *(requires_decision)*: Criterion's internal `point_estimate` is an arithmetic mean over all samples within an iteration; the plan's "median of 3 reps" takes the median of three such means. The statistical properties (sensitivity to OS scheduler interference, outlier robustness) differ between the two. The plan does not clarify which aggregation level constitutes a "rep" or which level is the primary estimate.

2. **Absolute latency not a co-criterion** *(requires_decision)*: The ≥5× speedup ratio is dimensionless. The plan does not include an absolute latency saving as a co-criterion, leaving the connection between the measured ratio and the deployment decision (interactive usability at n=50K) underspecified.

3. **Correctness check at n=50 not representative of n=50K** *(requires_decision)*: Tie-breaking divergence frequency in `|T_kdtree − T_brute_force|` may differ substantially between n=50 (zero ties likely) and n=50K. The relaxed threshold at n=1K (< 1e-8 vs. < 1e-12) is unexplained.

4. **Crossover linear interpolation on nonlinear curve** *(requires_decision)*: The theoretical speedup vs. n relationship is O(n/log n), not linear. Linear interpolation between adjacent measured n values will introduce systematic bias in the T_cross estimate. No interpolation error bound is stated.

5. **Build fraction profiler vs. Criterion inconsistency**: Build cost fraction uses profiler atomics from a separate process invocation; Criterion measures total wall-clock from a different execution environment. No mechanism verifies their additivity.

6. **Query-only vs. total measured by different environments**: `tw_kdtree_query_speedup` (profiler atomics) and `tw_kdtree_total_speedup` (Criterion) cannot be arithmetically combined to attribute total speedup to build vs. query cost.

### Variance Protocol (5W)

1. Criterion's internal RNG is not seeded — randomized iteration ordering introduces uncontrolled stochastic components between reps.
2. CV estimate from n=3 reps has ~50–80% standard error — the 10% threshold is applied to a noisy estimate of variance.
3. CPU frequency scaling (turbo boost, thermal throttling) introduces systematic trends across the three sequential overnight runs, inflating CV.
4. Rayon's work-stealing scheduler is unseeded and its per-rep state is uncontrolled.
5. The 2× crossover variance bound is applied to a T_cross estimate with high inherent uncertainty at n=3 reps.

### Reproducibility Spec (6W)

1. **`target-cpu=native`** *(requires_decision)*: Produces binaries tied to the build machine's microarchitecture. An independent reproducer on different hardware compiles different code. CPU model not in `run_metadata.json`.
2. Conda environment uses wildcard minor versions without a lock file.
3. No conda lock file stored in experiment artifacts.
4. CPU model/microarchitecture not recorded in `run_metadata.json`.
5. Criterion sample size and warm-up configuration storage location not specified.
6. Six-step execution protocol with no top-level orchestration script.

### Unit Interference (4W)

1. `ImmutableKdTree` node data may reside in CPU L2/L3 cache between benchmark groups within the same process (W8 acknowledges OS cache, not CPU cache).
2. Rayon global thread pool state is shared between `flat_simd` and `kdtree` groups without a reset boundary.
3. Allocator free-list state is shaped by `flat_simd`'s allocation pattern before `kdtree`'s timed iterations.
4. **Repetition ordering unspecified** *(requires_decision)*: Whether reps are grouped by variant or interleaved is not stated; either creates different carryover effects.

### Baseline Fairness (4W)

1. **Allocation asymmetry unquantified** *(requires_decision)*: `kiddo`'s per-query heap priority queue vs. `flat_simd`'s thread-local Vec reuse is acknowledged but not bounded. At n=50K under Rayon parallelism, this could non-linearly depress `kdtree` throughput.
2. **Leaf-size tuning direction uncertain** *(requires_decision)*: `flat_simd` is benchmarked at its optimal configuration; `kiddo` is benchmarked at documentation defaults. The plan claims conservative bias against `kdtree`, but the direction is unvalidated.
3. Cache warming bias direction may not be conservative as claimed — instruction/data cache effects could run opposite to the assumed OS file cache warming direction.
4. Profiling decomposition is asymmetric: `kdtree` gets `build` + `query` breakdown; `flat_simd` gets a single composite label.

### Data Acquisition (3W)

1. **n=75K data missing from manifest and gen_data.py scope**: The crossover analysis plots speedup vs. log(n) for `n ∈ {1K, 5K, 10K, 50K, 75K, 100K}`, but `uniform_n75000_{x,y}.npy` and `gauss_n75000_{x,y}.npy` are not listed in the data manifest table and do not appear in the `gen_data.py` generation scope ({1K, 5K, 10K, 50K, 100K} only).
2. Re-use of existing n ≤ 10K data from prior research directory has no mandatory acquisition path in a fresh worktree.
3. Criterion JSON outputs and profiler JSON outputs are not listed as data manifest entries with acquisition commands or verification criteria.

---

## Cannot Assess

These dimensions were impossible to evaluate from plan design content alone:

1. **Internal RNG of ImmutableKdTree**: Whether `kiddo`'s tree construction algorithm uses any internal stochastic element (randomized split selection, approximate-NN sampling) cannot be assessed from the plan. If present, it is an unseeded variance source not acknowledged in the controlled variables.

2. **Profiler instrumentation overhead**: Whether the `#[cfg(feature = "profiling")]` atomics (`Y_KDTREE_BUILD_NS`, `Y_KDTREE_QUERY_NS`) introduced in Phase 3 are compiled out of the Criterion binary cannot be assessed without inspecting the feature flag gating. The plan does not specify whether `profiling` feature is enabled during Criterion benchmarking.

3. **Criterion internal iteration count asymmetry**: Criterion automatically determines iteration count per sample based on measurement duration. At identical wall-clock windows, the faster variant (`kdtree` at large n) will accumulate more iterations, creating asymmetric effective sample counts. Whether this affects the CV comparison cannot be assessed without knowing the iteration policies.

4. **Resource proportionality adequacy**: At 8–12h Criterion + 2–4h profiler (total 10–16h), whether this is proportionate to the research question value cannot be assessed without project-level priority context. Flagged only because `+high_cost` modifier fired.

5. **`d_x` participation in hot path**: Whether `d_x=10` (input dimension) participates in the `trustworthiness()` code path being benchmarked cannot be confirmed from plan design alone. The plan states `trustworthiness()` primarily operates on Y, but does not confirm that X is unused in the hot loop.

---

## Mechanizable Check Log

Binary checks that could be automated in future:

| Check | Status | Note |
|-------|--------|------|
| `n=75K` entries in data manifest | ❌ FAIL | 75K present in analysis plan but absent from manifest table |
| All seeds explicitly stated for data generation | ✅ PASS | Seeds 42, 99 declared |
| Criterion sample size explicitly stated | ✅ PASS | 10 samples |
| Criterion warm_up_time explicitly stated | ✅ PASS | 10s per group |
| Rust toolchain pinned to specific date | ✅ PASS | `nightly-2026-03-26` |
| Primary DV declared before data collection | ✅ PASS | W3 section |
| Pre-specification notice present | ✅ PASS | "Research Design Rationale" section |
| All DVs have stated thresholds | ⚠️ PARTIAL | DVs 3 (build_ms) and 4 (query_speedup) lack thresholds |
| run_log.json mandated | ✅ PASS | RT4 explicitly called out |
| Conda lock file present | ❌ FAIL | No lock file specified |
| CPU model in run_metadata | ❌ FAIL | Records only RAYON_NUM_THREADS and toolchain |
| Held-out set independence justified | ⚠️ PARTIAL | 75K rationale implies crossover range foreknowledge |

---

## Machine-Readable Summary

```yaml
# --- review-design machine summary ---
verdict: GO
experiment_type: benchmark
critical_count: 0
warning_count: 51
red_team_count: 7
active_dimensions: 13
warning_threshold: 65
secondary_modifiers:
  - +high_cost
  - +deployment
  - +multi_metric
notes: >
  Plan is mature with extensive pre-specification rationale.
  51/65 warnings. Most concerns already acknowledged in W*/RT* annotations.
  Key requires_decision items: measurement alignment ambiguities (mean vs
  median-of-means, absolute latency co-criterion, crossover interpolation),
  reproducibility (target-cpu=native), and red-team RT-3 (re-run cherry-pick
  pathway). These do not block execution but warrant author attention.
```
