# Revision Guidance: Trustworthiness Performance Re-run (Clean Infrastructure)

**Verdict:** REVISE
**Plan:** `experiment_plan_tw_perf_rerun_clean_2026-04-05_130500.md`
**Generated:** 2026-04-05

Fix all **Required** items before execution. **Recommended** items improve rigor but are not blocking. **Red-team** items require an explicit decision (accept risk with documented rationale, or mitigate).

---

## Required Revisions (Critical Findings)

### R1 — Provide MERFISH n=50K data for H-partial-MERFISH

**Finding:** H-partial-MERFISH requires a Criterion speedup CI at n=50K on MERFISH data, but no MERFISH n=50K fixture is defined anywhere in the plan. There is no acquisition command, generation script, or output path for this data.

**Fix (Option A — add data):** Extend `prepare_data.sh` to generate a MERFISH n=50K fixture from `temp/merfish_100k/`:
```python
# In prepare_merfish.py or a new prepare_merfish_50k.py:
x_50k = X[:50000, :]   # first 50K rows of the 100K PCA-50 output
y_50k = Y[:50000, :]
np.save("research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy", x_50k)
np.save("research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy", y_50k)
```
Then add a `tw_partial_rank_merfish_bench.rs` that loads these fixtures for the n=50K Criterion run.

**Fix (Option B — remove hypothesis):** If MERFISH n=50K data is not available without additional work, drop H-partial-MERFISH from this experiment and document it as a future measurement.

---

### R2 — Make Gaussian data accessible from the new worktree

**Finding:** The new bench templates hard-code `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/2026-04-04-tw-perf-scaling/data/gaussian")`. This path exists only inside the *old* worktree. All 5 Criterion benches will panic at runtime with a missing-file error.

**Fix:** In `apply_phase1_changes.sh`, add a symlink step that runs inside the new worktree:
```bash
# In the new worktree root:
ln -s /home/talon/projects/worktrees/research-20260404-174030/research/2026-04-04-tw-perf-scaling \
      research/2026-04-04-tw-perf-scaling
```
Or copy the data directory:
```bash
mkdir -p research/2026-04-04-tw-perf-scaling/data
cp -r /path/to/old-worktree/research/2026-04-04-tw-perf-scaling/data/gaussian \
      research/2026-04-04-tw-perf-scaling/data/
```
Add a verification step to Phase 3 (`prepare_data.sh`) that asserts the path exists before any build.

---

### R3 — Align H0/H1-clean hypothesis with actual profiling data

**Finding:** The step-fraction hypothesis is stated at "n=100K, d=50 (MERFISH PCA-50 regime)" and the first-principles prediction ("tw_x_dist dominates because O(n·d) at d=50") is valid only at d=50. Phase 7 profiling uses Gaussian n=100K d=10. The measured DV does not match the stated DV.

**Fix (Option A — use d=50 data):** Generate a Gaussian or MERFISH n=100K d=50 fixture and profile on it. The first-principles prediction is then testable as stated.

**Fix (Option B — reframe hypothesis at d=10):** Update H0/H1-clean to read "n=100K, d=10 (Gaussian benchmark regime)" and revise the first-principles prediction accordingly:
- At d=10, O(n·d) = O(10⁶) FLOPs for tw_x_dist vs O(n·log(k)) ≈ O(4×10⁵) for tw_y_heap — a ~2.5× raw FLOP advantage (not ~12× as stated for d=50).
- The dominance prediction may still hold at d=10 but the margin is smaller.

---

### R4 — Fix step-fraction metric: CPU-ns ≠ wall-clock fraction

**Finding:** The step_fractions DV is defined as "mean fraction of total wall time" but atomic counters accumulate CPU-nanoseconds summed across all rayon threads. For a data-parallel workload, steps with higher parallelism appear larger in CPU-ns than in wall-clock time, and vice versa. The instrument does not match the claim.

**Fix (Option A — reframe metric as CPU-time fraction):** Update the DV description to "mean fraction of total CPU-time-in-rayon-closures" and update H0/H1-clean success criteria accordingly. This is physically meaningful and what the counters actually measure.

**Fix (Option B — measure wall-clock per step):** Replace atomic accumulators with `Instant::now()` + wall-clock elapsed per step *in the outer serial loop* that dispatches rayon work. This requires restructuring the timing points to occur outside the parallel closure.

---

### R5 — Specify what tw_approx_runner times for wall_exact_s and wall_approx_s

**Finding:** The speedup DV (`wall_exact_s / wall_approx_s`) is underspecified: which variant is "exact"? Does the timing include data loading? What warm-up protocol applies?

**Fix:** In the H5 section and Phase 5 description, specify:
- `wall_exact_s`: wall-clock time for `trustworthiness_baseline(x, y, k)` called once after one warm-up call (or specify exact warm-up protocol)
- `wall_approx_s`: wall-clock time for `trustworthiness_approx(x, y, k, m, seed)` under the same warm-up protocol
- Both timings exclude data loading (x, y already loaded into memory before timing begins)
Add a code snippet or pseudocode to `tw_approx_runner` showing the timing boundaries.

---

### R6 — Fix atomic memory ordering in profiling infrastructure

**Finding:** `step_timing::reset()` uses `Ordering::Relaxed` for the store. Rayon worker threads use `fetch_add(Relaxed)`. The Relaxed model does not guarantee that a reset on the main thread is visible to worker threads before they execute fetch_add, causing inter-iteration contamination.

**Fix (Change 4 in Phase 1):**
```rust
// In reset():
X_DIST_NS.store(0, Ordering::Release);  // or SeqCst
// ... same for all 6 counters

// In read():
X_DIST_NS.load(Ordering::Acquire);  // or SeqCst
```
Additionally, change the reset call site in `tw_profiler.rs` from after-iteration to before:
```rust
// BEFORE: after iteration
step_timing::reset();  // ← wrong position

// AFTER: before each measured iteration
step_timing::reset();  // reset first
let _ = variant_fn(x, y, k);  // then time
let step_readings = step_timing::read();
```

---

### R7 — Pin Criterion parameters for all n values

**Finding:** Only n=100K has explicitly pinned Criterion parameters. n=1K–50K uses "standard Criterion settings (not explicitly pinned)," meaning variance budget and sample counts are undefined and non-reproducible.

**Fix:** In each bench template, explicitly set parameters for all n values before the n=100K special case:
```rust
fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("tw_baseline");
    group.sample_size(100);                          // explicit for n<100K
    group.warm_up_time(Duration::from_secs(10));     // explicit
    group.measurement_time(Duration::from_secs(60)); // explicit
    for &n in &[1000usize, 5000, 10000, 25000, 50000] {
        // ...
    }
    // n=100K override:
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(63);
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(1500));
    // ...
}
```

---

### R8 — Control rayon thread count

**Finding:** Rayon's global thread pool defaults to host CPU count. Thread count directly affects benchmark results and is a controlled variable for a parallelism-focused study, but is not listed as a controlled variable and not pinned.

**Fix:** In each bench binary and in `tw_profiler.rs`, add at the start of `main()` (or before Criterion setup):
```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(N_THREADS)  // e.g., 8 or physical core count
    .build_global()
    .unwrap();
```
Add `RAYON_NUM_THREADS` to the Controlled Variables table with the specific value used, and document it in the hardware profile section.

---

### R9 — Fix H5 CI formula to use t-distribution

**Finding:** The analysis script uses `z=1.96` for the 95% CI upper bound of `|delta|` with n=10. The correct critical value is `t(0.975, df=9) ≈ 2.262`, giving ~15% wider intervals. Under-coverage at a primary hypothesis gate is a validity threat.

**Fix in `analyze_clean.py`:**
```python
from scipy.stats import t as t_dist

n = len(deltas)
t_crit = t_dist.ppf(0.975, df=n - 1)  # ≈ 2.262 for n=10
delta_ci_upper = np.mean(deltas) + t_crit * np.std(deltas, ddof=1) / np.sqrt(n)
```

---

## Recommended Revisions (Warning Findings)

### W1 — CV sensitivity analysis for H-100K power

State the power degradation if CV=20%: approximate formula `power ≈ Φ(z_α - (z_β + z_α)) * ...` or a table row: "at CV=20%, n=63 yields ~65% power." If CV is re-estimated post-run and found to be >20%, note this as a limitation.

### W2 — Pre-register secondary H5 sweep as descriptive-only

Add one sentence to Phase 5: "The m-sweep (m ∈ {500, 1K, 2K, 5K, 10K}) is reported descriptively. No inferential comparisons or threshold decisions will be derived from sweep results."

### W3 — Clarify H0/H1-clean CI-ordering as descriptive

Add to the success criteria for H0/H1-clean: "This CI-ordering check is descriptive with no family-wise error control. It does not generate a p-value."

### W4 — Declare cache warm-state check as mandatory

Change "if time permits" to "required before publishing results." Alternatively, counterbalance run order across two passes (baseline…combined, combined…baseline) and report both.

### W5 — Document bootstrap fallback limitation

In `analyze_clean.py` and the Analysis Plan: if the Criterion sample fallback is used (from aggregate CI bounds rather than raw samples), flag the output with "FALLBACK CI: conservative bounds used; uncertainty is understated relative to bootstrap on raw samples."

### W6 — Add Criterion reproducibility note

Note in the plan that Criterion's internal RNG is not user-seeded. Document that exact CI values may differ across runs; only mean point estimates and CI widths are expected to be stable within measurement variance.

### W7 — Scope H-100K claim explicitly

Add to H-100K hypothesis: "Scope: speedup at n=100K, Gaussian d=10, k=15, x86-64-v3 AVX2/FMA only."

### W8 — Verify profiling feature excluded from Criterion builds

Add to Phase 6 build command:
```bash
# Verify profiling feature is NOT active:
cargo criterion --bench tw_baseline_bench --no-run -- --features cli
# Check: 'profiling' must NOT appear in cargo's feature resolution output
```

---

## Red-Team Decisions Required

Each item requires an explicit accept/mitigate decision. Document the chosen response in the plan.

### RT-1 — Threshold sandbagging (H-100K threshold of 1.5×)

**Risk:** The 1.5× threshold was set 30% below the observed ~1.95× prior result. It is not derived from a deployment requirement.

**Mitigate:** Replace 1.5× with a threshold derived from a concrete requirement (e.g., "minimum speedup that reduces wall time for MERFISH 100K below user-acceptable latency") or acknowledge explicitly in the report that 1.5× is a conservative lower bound on the prior result rather than an independent performance requirement.

**Accept:** Document: "The 1.5× threshold is a conservative lower bound on the prior result; the experiment tests whether the speedup survives the cache-regime transition, not whether it meets a specific user requirement."

---

### RT-2 — Data re-use from prior experiment (Gaussian fixtures)

**Risk:** The same Gaussian data on which the prior run informed threshold selection is re-used. A held-out dataset cannot falsify the hypothesis.

**Mitigate:** Generate fresh Gaussian data with a new seed for the new experiment (add `gen_synthetic.py --seed 2026 ...` to Phase 3). The shape and distribution remain identical; only the specific realization changes.

**Accept:** Document: "Gaussian data is synthetic randn; the specific realization is not expected to affect speedup ratios, as ratios are data-distribution-invariant for this algorithm at these scales. Data re-use risk is low."

---

### RT-3 — Asymmetric measurement configuration (n<100K unspecified)

**Risk:** The n=100K scale is precisely configured (the primary claim scale); smaller scales use defaults. This creates structurally higher confidence at the one scale that matters.

**Mitigate:** Pin Criterion parameters for all scales (see R7 above). This addresses both the red-team concern and the critical finding.

**Accept if R7 is implemented.** R7 resolves this.

---

### RT-4 — No re-run protocol / survivorship bias

**Risk:** If initial results are unfavorable, the plan does not prevent selective re-running of only the combined variant.

**Mitigate:** Add a pre-registered re-run policy: "If any benchmark must be re-run due to a system event (thermal throttle, background process), ALL 5 variants must be re-run together. The first complete set of results is the primary dataset."

**Accept:** Document: "Single-machine, single-user environment with manual run supervision. Re-run risk is acknowledged and mitigated by publishing the full raw criterion_output.json."

---

### RT-5 — H5 5× speedup: unexplained mechanism

**Risk:** Theoretical subsampling speedup is n/m = 2×. The claimed 5× is unexplained. If derived from prior observations, the threshold was reverse-engineered.

**Mitigate:** Add a mechanistic explanation: where does the extra 2.5× come from? Candidate explanations: approximation avoids the rank-sort step (O(n log n) savings), better cache behavior at smaller working set, etc. If the mechanism is known, state it. If unknown, lower the threshold to the theoretically justified 2× + safety margin, or set it empirically from the MERFISH dry-run result.

**Accept:** Document: "The 5× threshold is empirically observed from prior data. The mechanism is not fully understood; the experiment will report the measured speedup and its components without claiming a mechanistic explanation."
