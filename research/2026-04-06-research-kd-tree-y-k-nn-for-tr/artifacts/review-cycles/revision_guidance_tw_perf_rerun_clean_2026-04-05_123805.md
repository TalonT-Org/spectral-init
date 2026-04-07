# Revision Guidance: `tw_perf_rerun_clean`

**Source plan:** `.autoskillit/temp/plan-experiment/experiment_plan_tw_perf_rerun_clean_2026-04-05_122948.md`
**Review timestamp:** 2026-04-05 12:38:05
**Verdict:** REVISE

This document provides actionable fixes for all critical and warning findings from the evaluation dashboard. Address all **Required Revisions** before re-submitting for review. **Recommended Revisions** improve plan quality but are not blocking.

---

## Required Revisions (Critical Findings — 20 items, 9 root causes)

### R1 — Reconcile Variant Scope (baseline_fairness #9/#10, statistical_corrections #26)

**Problem:** The IV table declares 5 variants (baseline, thread_local, partial_rank, avx2_kernel, combined) as the treatment space, but `run_criterion_clean.sh` runs only 3 bench binaries. The Holm-Bonferroni correction specifies m=4 comparisons for data that only covers 2 (baseline vs partial_rank, baseline vs combined).

**Fix — choose ONE:**

**Option A (Expand):** Add thread_local and avx2_kernel to Phase 4:
```bash
cargo criterion \
  --bench tw_baseline_bench \
  --bench tw_thread_local_bench \
  --bench tw_partial_rank_bench \
  --bench tw_avx2_bench \
  --bench tw_combined_bench \
  --message-format=json \
  > results/criterion/criterion_output.json
```
Keep m=4 in the Holm-Bonferroni family (baseline vs thread_local, partial_rank, avx2_kernel, combined). Update wall-clock estimate (+30–60 min per additional variant at n=100K).

**Option B (Restrict):** Remove thread_local and avx2_kernel from the IV table. Change the Holm family to m=2 (baseline vs partial_rank and baseline vs combined). Update the analysis plan accordingly. Mark those variants as "out of scope for this experiment."

Either option resolves the inconsistency. Option A is recommended for completeness; Option B is appropriate if compute budget is binding.

---

### R2 — Implement Benchmark Isolation (unit_interference #13/#14)

**Problem:** Phase 4 runs 3 bench binaries in a single `cargo criterion` invocation. Phase 5 runs 5 profiling variants in a single bash loop. Both contradict the documented 1-minute pause mitigation, leaving CPU cache and frequency-scaling state shared across variants.

**Fix for Phase 4 — Rewrite `run_criterion_clean.sh` as separate invocations:**
```bash
#!/usr/bin/env bash
set -euo pipefail

OUT=results/criterion

# Reset CPU governor to performance before benchmarking session
# sudo cpupower frequency-set -g performance   # uncomment if permissions allow

for BENCH in tw_baseline_bench tw_partial_rank_bench tw_combined_bench; do
  echo "Running $BENCH..."
  cargo criterion --bench "$BENCH" --message-format=json 2>/dev/null \
    >> "$OUT/criterion_output.json"
  echo "Cooling down (60s)..."
  sleep 60
done
```

**Fix for Phase 5 — Add cool-down to `run_profiling_clean.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail

for VARIANT in baseline thread_local partial_rank avx2_kernel combined; do
  echo "Profiling $VARIANT..."
  ./target/release/tw_profiler \
    --x research/.../data/gaussian_n100k_d50_x.npy \
    --y research/.../data/gaussian_n100k_d50_y.npy \
    --k 15 --iters 30 --warmup 5 \
    --variant "$VARIANT" \
    --output "results/step_timing/gaussian_n100000_${VARIANT}.json"
  echo "Cooling down (60s)..."
  sleep 60
done
```

Remove the Threats to Validity mitigation note (or update it to say "implemented in scripts").

---

### R3 — Define the Correction Family Explicitly (statistical_corrections #25/#27, error_budget #19)

**Problem:** The Holm-Bonferroni correction family is never enumerated across all 5 DVs. H-partial-MERFISH family membership is conditional and deferred to analysis time.

**Fix:** Add a correction table to the Analysis Plan section (before any data is collected):

| DV | In Holm family? | Adjusted α | Test |
|----|----------------|------------|------|
| `wall_clock_speedup` | No (deterministic gate, no test) | — | Threshold ≥5× |
| `delta_tw` | No (deterministic gate, no test) | — | Threshold <0.001 |
| `criterion_speedup_100k` (baseline vs partial_rank) | **Yes** (rank 1) | 0.05/m | Criterion CI lower bound |
| `criterion_speedup_100k` (baseline vs combined) | **Yes** (rank 2) | 0.05/(m−1) | Criterion CI lower bound |
| `partial_rank_ci_half_width` | **Yes** (rank 3, or No — decide now) | 0.05/(m−2) | One-sided comparison |

Fill in the `partial_rank_ci_half_width` row with a definitive decision before data collection. If included, set m=3 (or m=5 if thread_local and avx2_kernel are added). This table must be committed to the plan before any Phase 4/6 data is collected.

---

### R4 — Increase Profiling Iterations and Report Variance (variance_protocol #31, measurement_alignment #57)

**Problem:** 5 profiling iterations provide no statistical stability. Step fractions are means only. A single outlier can shift the mean across the 50% H1 threshold. Non-uniform RefCell overhead further biases fractions.

**Fix:**

1. Change `--iters 5` to `--iters 30` (or minimum 20) in `run_profiling_clean.sh`.
2. Update `tw_profiler.rs` to emit per-step statistics:
   ```json
   {
     "step_fractions": {
       "tw_x_dist": { "mean": 0.62, "std": 0.03, "ci_lower_95": 0.57, "ci_upper_95": 0.67 }
     }
   }
   ```
3. Change the H0/H1-clean verdict rule in `analyze.py`:
   ```python
   # H1 confirmed only if CI lower bound > 0.5, not just mean
   verdict = "H1 supported" if fracs["tw_x_dist"]["ci_lower_95"] > 0.5 else "H0 not rejected"
   ```
4. For overhead: time a no-op step (zero work) to measure per-step RefCell overhead; subtract if > 0.5% of total (not 1% — be conservative).

---

### R5 — Add Data Acquisition Steps for All Inputs (data_acquisition #38/#39/#40, reproducibility_spec #44/#45)

**Problem:** Three critical data sources have no acquisition steps:
- `temp/merfish_100k/*.npz` — gitignored, local-only, no download/copy script
- `data/gaussian/` (n=1K–50K) — assumed present in worktree, no generation step
- Phase 1 code edits — described in prose only, not scripted

**Fix for MERFISH source:**
Add a prerequisite check at the top of `prepare_data.sh`:
```bash
# Verify MERFISH 100K source exists
MERFISH_SRC="/home/talon/projects/spectral-init/temp/merfish_100k"
if [ ! -f "$MERFISH_SRC/merfish_100k_expression.npz" ]; then
  echo "ERROR: MERFISH 100K source not found at $MERFISH_SRC"
  echo "This data must be obtained separately (gitignored, not reproducible from code)."
  echo "Copy the two NPZ files to $MERFISH_SRC and re-run."
  exit 1
fi
```
Also document in plan: "MERFISH 100K NPZ files are proprietary/large data not tracked in git. They are available at [internal archive path / contact author]. SHA-256 checksums: [expression.npz: XXX] [spatial.npz: YYY]."

**Fix for pre-generated Gaussian:**
Add to Phase 2 or `prepare_data.sh`:
```bash
# Generate Gaussian reference data if not present
for N in 1000 5000 10000 25000 50000; do
  python -c "
import numpy as np
rng = np.random.default_rng($N)  # use n as seed for reproducibility
np.save('data/gaussian/gaussian_n${N}_d10_x.npy', rng.standard_normal(($N, 10)))
np.save('data/gaussian/gaussian_n${N}_d10_y.npy', rng.standard_normal(($N, 2)))
"
done
```
Adjust dimensionality and file naming to match existing conventions in `data/gaussian/`.

---

### R6 — Script Phase 1 Code Changes (reproducibility_spec #46)

**Problem:** Phase 1 describes 5 source code changes in prose. An independent reproducer must manually interpret and apply these changes, introducing divergence risk.

**Fix:** Convert Phase 1 to a shell script `scripts/apply_phase1_changes.sh` that applies the changes, or provide a `phase1.patch` file generated from `git diff` after making the changes in a test environment. At minimum, ensure the plan includes the exact diff for each of the 5 changes (Cargo.toml features, bench entries, metrics.rs instrumentation, tw_profiler.rs update, ndarray-npy dev-dep) so they can be applied with `patch -p1 < phase1.patch`.

---

### R7 — Fix H5 Speedup Measurement (measurement_alignment #55)

**Problem:** `wall_clock_speedup = wall_exact_s / wall_approx_s` uses a single wall-clock measurement. OS scheduling jitter at n=10K can shift wall time by ±20%, making a single-trial 5× threshold unreliable.

**Fix — Option A (CPU time):** Change `tw_approx_runner` to measure CPU time using `std::time::Instant` on a single-threaded Tokio runtime or via `getrusage(RUSAGE_SELF)` to isolate CPU from I/O wait.

**Fix — Option B (repeated trials):** Run `tw_approx_runner` 10 times within the script (or expose a `--trials N` flag) and report median speedup with IQR. Evaluate the H5 gate against the median. Update `run_h5.sh`:
```bash
for i in $(seq 1 10); do
  ./target/release/tw_approx_runner \
    --x .../merfish_n10k_x.npy --y .../merfish_n10k_y.npy \
    --k 15 --sample 5000 --seed $((42 + i)) \
    --output "results/h5/h5_trial_${i}.json"
done
python -c "
import json, glob, numpy as np
results = [json.load(open(f)) for f in glob.glob('results/h5/h5_trial_*.json')]
speedups = [r['wall_exact_s'] / r['wall_approx_s'] for r in results]
print(f'median speedup={np.median(speedups):.2f}x, IQR=[{np.percentile(speedups,25):.2f},{np.percentile(speedups,75):.2f}]')
"
```

Either option resolves the single-trial reliability concern. If using Option B, note that using multiple seeds changes the H5 from a sealed single-seed gate to a multi-seed estimate — update the hypothesis framing accordingly.

---

### R8 — Fix Speedup Ratio CI for H-100K (measurement_alignment #56)

**Problem:** `criterion_speedup_100k` is computed as `baseline_mean_ns / variant_mean_ns` from independent Criterion runs. The arms are not jointly sampled; a simple ratio of means has no valid CI.

**Fix:** Derive the ratio CI via bootstrap over paired Criterion timing samples. Criterion's JSON output contains individual timing samples. Collect these and bootstrap the ratio:
```python
import json, numpy as np
from scipy import stats

# Parse all timing samples from JSON-lines
baseline_samples = []  # collect from criterion JSON for baseline n=100K
combined_samples = []  # collect from criterion JSON for combined n=100K

# Bootstrap ratio CI
n_boot = 10000
ratios = []
rng = np.random.default_rng(42)
for _ in range(n_boot):
    b = rng.choice(baseline_samples, len(baseline_samples))
    c = rng.choice(combined_samples, len(combined_samples))
    ratios.append(np.mean(b) / np.mean(c))

ci_lower, ci_upper = np.percentile(ratios, [2.5, 97.5])
speedup_point = np.mean(baseline_samples) / np.mean(combined_samples)
```

If Criterion's JSON format does not expose individual samples, add a `--save-baseline` step or use Criterion's programmatic API to extract raw timing vectors. Alternatively, write a custom harness that captures timing samples independently for both variants in the same process run to ensure paired measurement.

---

### R9 — Scope Nightly-Only Results (ecological_validity #51)

**Problem:** All speedup results are measured on nightly-2026-03-26 but production uses stable Rust. Nightly-only results cannot support a production performance claim.

**Fix:** Add an explicit scope statement to the Motivation and Success Criteria sections:

> "All performance results in this experiment are valid for the `nightly-2026-03-26` (rustc 1.96.0-nightly, x86-64-v3 AVX2/FMA) build configuration only. Stable Rust results may differ due to codegen differences. A follow-up measurement on stable Rust is recommended before publishing production performance claims."

Optionally: run the Phase 4 Criterion benchmarks once on stable Rust (accept higher runtime variance) and record both results. If stable results are within 10% of nightly, the production claim is substantiated.

---

## Recommended Revisions (Warning Findings)

### W1 — Invert H-100K H0/H1 to Standard Convention
The null should be the skeptical claim: "combined CI at n=100K does not exceed 1.5× (extrapolation overestimated)." The alternative should be: "combined CI lower bound > 1.5×." This restores the conventional direction where H0 is what the author seeks to reject.

### W2 — Pre-register "Near Threshold" Trigger for H5
Add to Controlled Variables: "H5 near-threshold window: result within [4.5×, 5.5×] for speedup or [0.0008, 0.0012] for |delta| triggers the 5-seed sensitivity check. Summary statistic for multi-seed verdict: median speedup and median |delta|."

### W3 — Justify H0/H1-clean Threshold Independently
The >50% threshold for tw_x_dist dominance was anchored on the contaminated prior observation (~62%). Add a first-principles justification: e.g., "50% represents the majority-time criterion from information-theoretic profiling literature" or change H0/H1-clean to an estimation task ("report step fraction distribution without a binary verdict").

### W4 — State β Explicitly for H-100K
Add: "At sample_size=100 and r=10%, power≈69% (β≈0.31). The minimum detectable effect at 80% power with sample_size=100 is r≈11.6% at the Holm-corrected threshold for the first comparison."

### W5 — Pin statsmodels Exactly
Change `statsmodels>=0.14` to `statsmodels=0.14.4` (or current stable) in environment.yml.

### W6 — Document Hardware Profile
Add to Environment section: CPU model, core count, L1/L2/L3 cache sizes, RAM, NUMA topology, and whether the system runs dedicated or shared. This allows future reproducers to assess hardware comparability.

### W7 — Resolve H-partial-MERFISH Gaussian Baseline Source Ambiguity
Specify one authoritative source for the Gaussian CI at n=10K (either from the new isolated Criterion run or from the prior experiment, not "OR"). Add a verification checksum or shape assertion.

### W8 — Add delta_tw Confidence Interval
Change `tw_approx_runner` to run on k different subsamples of the MERFISH 10K data (or k different seeds) and report delta_tw as mean ± 95% CI. This makes the |delta| < 0.001 threshold evaluation statistically grounded.

### W9 — Scope H5 Quality Claim
Add a note to H5: "The |delta| < 0.001 quality guarantee is valid at n=10K, m=5000 only. Extrapolation to n=100K requires additional measurement since the approximation ratio m/n changes from 0.5 to 0.05."

### W10 — Document CV=15% Provenance
Cite where the 15% coefficient of variation estimate comes from (prior run, literature benchmark, pilot data). If estimated from the contaminated prior run, note that CV may differ with clean infrastructure.

---

## Red-Team Mitigation Decisions

Each red-team finding has `requires_decision: true`. A response is required for each before re-submission.

| ID | Finding | Required Decision |
|----|---------|------------------|
| RT-1 | "Near threshold" undefined for H5 5-seed trigger | Pre-register the exact numeric window (see W2). Commit to the decision before any Phase 3 measurement. |
| RT-2 | H0/H1-clean threshold anchored on contaminated prior observation | Either justify independently (see W3) or reframe as estimation. This cannot be left as-is since it creates a guaranteed-pass hypothesis. |
| RT-3 | Step fraction variance from 5 iterations | Addressed by R4. No additional decision needed after that fix is applied. |
| RT-4 | H-partial-MERFISH conditional correction | Addressed by R3. Declare family membership before any data collection. |
| RT-5 | Criterion ratio CI invalid | Addressed by R8. Document the bootstrap procedure in analyze.py. |
| RT-6 | MERFISH single-dataset survivorship | Scope the quality claim to "MERFISH-like structured data" in the hypothesis and conclusion sections. Do not claim generality to arbitrary biological data from a single dataset. |
