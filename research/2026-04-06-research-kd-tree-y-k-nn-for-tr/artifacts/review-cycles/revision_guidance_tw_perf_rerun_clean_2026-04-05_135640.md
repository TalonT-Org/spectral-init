# Revision Guidance: Trustworthiness Performance Re-run (Clean Infrastructure)

**Verdict:** REVISE
**Review date:** 2026-04-05 13:56:40

This document provides actionable revisions for all warning-level findings. Required revisions
must be resolved before execution. Recommended revisions improve rigor but are not blocking.

---

## Required Revisions (Warning-Level Findings)

### R-1 · H5: Replace t-CI with statistically appropriate CI for median

**Finding:** error_budget, L3
**Section:** `## Hypotheses / H5`

The 95% t-CI formula is applied to the median speedup, but the t-CI computes confidence intervals
for a *mean*. For median inference at n=10, the t-CI is only valid under symmetry assumptions not
stated in the plan.

**Fix:** Choose one:
- **Option A (preferred):** Replace the median-based speedup gate with a mean-based gate.
  Report mean speedup ± 95% t-CI (df=9). The threshold changes accordingly (mean ≥ 5×).
  This eliminates the mismatch between estimator and CI formula.
- **Option B:** Keep the median as the point estimate but compute a bootstrap CI for the median:
  `np.percentile(bootstrap_medians, [2.5, 97.5])` from B=10,000 bootstrap samples of the 10 trials.
  The quality gate remains: bootstrap CI upper bound of |delta| < 0.001.

### R-2 · H5: Pre-commit to a single test statistic; remove or bound the near-threshold rule

**Finding:** red_team (requires_decision), L1 hypothesis_falsifiability
**Section:** `## Hypotheses / H5 — Near-threshold triggers`

The near-threshold rule introduces a second test statistic (majority ≥6/10) applied to the same
data when the primary statistic (median) lands in [4.5×, 5.5×]. This is adaptive testing that
inflates type-I error precisely where researcher degrees of freedom matter most.

**Fix:** Choose one:
- **Option A (preferred):** Remove the near-threshold rule. The median (or mean, per R-1) is
  the sole decision statistic. If the result is in [4.5×, 5.5×], report it as borderline and
  flag it for follow-up, but apply no alternative decision criterion.
- **Option B:** Pre-commit to the majority rule as the *only* decision statistic (discard the
  median gate entirely). Derive the 6/10 threshold from a power calculation:
  Under H0 (p=0.5), P(X≥6 | n=10) = 0.377 — not conservative. Adjust to 7/10 (P≈0.172)
  or 8/10 (P≈0.055) to match a 5% type-I error rate.

### R-3 · H-100K: Ground the 1.5× threshold independently of the prior data

**Finding:** red_team (requires_decision)
**Section:** `## Hypotheses / H-100K — 1.5× threshold rationale`

The 1.5× threshold is derived from a 30% haircut to the prior experiment's point estimate, but
that same prior experiment is acknowledged as unreliable (motivation for this re-run). Using
distrusted data to anchor the success criterion is asymmetric.

**Fix:** Choose one:
- **Option A:** Justify 1.5× from a deployment requirement or theoretical lower bound *independent*
  of the prior data. E.g., if any speedup >1.0× is acceptable for production, set threshold=1.0×
  and note this explicitly. If a specific latency budget exists, derive the threshold from it.
- **Option B:** Remove the binary pass/fail threshold. Report the Holm-corrected CI with its
  measured lower bound. State the *descriptive* claim: "The measured lower bound is X×; the prior
  extrapolation of 1.95× was X% above/below the confirmed result." This is more honest than a
  pass/fail against a threshold derived from distrusted data.

### R-4 · H-partial-MERFISH: Pre-specify which comparison baseline takes precedence

**Finding:** red_team (requires_decision), L1 hypothesis_falsifiability
**Section:** `## Hypotheses / H-partial-MERFISH`

Two criteria exist: (1) absolute threshold 0.26 from contaminated prior data, (2) relative
comparison to fresh Gaussian half-width from this experiment. The plan does not state which takes
precedence when they conflict.

**Fix:** Unambiguously pre-specify: "The primary verdict criterion is the relative comparison
to the fresh Gaussian half-width from this experiment (not the 0.26 threshold from prior data).
The 0.26 value is retained as a historical reference point only and does not affect the H1 verdict."

### R-5 · Execution order: Pre-specify adjudication protocol for reversed-order cache check

**Finding:** red_team (requires_decision)
**Section:** `## Execution Protocol / Cache warm-state check (W4)`

The reversed-order check (combined→baseline) is mandatory, but no protocol exists for what to
do when forward and reversed point estimates diverge by >5%. This creates an undeclared researcher
degree of freedom in result selection.

**Fix:** Add the following pre-specified adjudication rule to `run_criterion_clean.sh` and the
Analysis Plan:
> "If forward-order and reversed-order point estimates diverge by >5% for any variant, both runs
> are published in full as the primary dataset. The main analysis reports the forward-order result
> as the primary dataset. The reversed-order run is treated as a qualitative robustness check only
> and is NOT combined with or substituted for the forward-order bootstrap CI. If the divergence
> exceeds 10%, the result is reported as 'potentially cache-state-confounded' with a caveat."

### R-6 · Seeds: Declare prior-run status or use randomized seed selection

**Finding:** red_team (requires_decision)
**Section:** `## Hypotheses / H5 — Random seed`

Seeds 42–51 form a human-selected contiguous block starting at the conventional default. No
declaration is made about whether any runs with these seeds were executed prior to plan finalization.

**Fix:** Add one of the following to the plan before execution:
- **Declaration (preferred):** "No calls to `tw_approx_runner` or any equivalent function with
  seeds 42–51 were executed in any prior experiment. This is confirmed and logged in `results/
  analysis/seed_provenance.txt`."
- **Randomized selection:** Draw 10 seeds from `np.random.default_rng(master_seed).integers(0, 2**32, 10)`
  where `master_seed` is pre-registered (e.g., 20260405), document the master seed, and use the
  resulting seeds instead of 42–51.

### R-7 · Data acquisition: Document temp/ recovery strategy for fresh worktrees

**Finding:** data_acquisition, L4
**Section:** `## Inputs and Data — Source Data`

`temp/merfish_100k/` is gitignored. In a fresh worktree or CI environment, Steps 3.2 and 3.3
will fail immediately with "directory not found."

**Fix:** Add to `prepare_data.sh` a pre-flight check block:
```bash
# Pre-flight: verify temp/merfish_100k/ exists
NPZ_DIR="/home/talon/projects/spectral-init/temp/merfish_100k"
if [ ! -d "$NPZ_DIR" ] || [ -z "$(ls -A $NPZ_DIR/*.npz 2>/dev/null)" ]; then
  echo "ERROR: temp/merfish_100k/ is empty or missing."
  echo "Re-generate from source:"
  echo "  python scripts/generate_merfish_subset.py \\"
  echo "    --input data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad \\"
  echo "    --output-dir temp/merfish_100k --n 100000"
  echo "Requires: anndata, polars (install separately if needed)"
  exit 1
fi
```
Additionally, document in the plan that the H5AD source file
(`data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad`) was downloaded from the Allen Brain Cell Atlas
at [URL] with checksum [SHA256].

### R-8 · Data acquisition: Define intermediate PCA artifact for prepare_merfish_50k.py

**Finding:** data_acquisition, L4
**Section:** `## Inputs and Data / Item 2 (MERFISH n=50K)`

`prepare_merfish_50k.py` is described as slicing from the `prepare_merfish.py` PCA output, but
Step 3.2 only produces the final n=10K arrays. The intermediate PCA-50 matrix for ≥50K rows is
never named.

**Fix:** Modify `prepare_merfish.py` to also write a persistent intermediate artifact, e.g.:
```
data/merfish/merfish_pca50_full.npy   # full 100K × 50 PCA matrix
```
Update Step 3.2 verification: `merfish_pca50_full.npy` shape `(100000, 50)` float64.
Update `prepare_merfish_50k.py` to read `--pca-source data/merfish/merfish_pca50_full.npy`
rather than re-running PCA. This makes the dependency chain explicit and avoids redundant PCA.

### R-9 · Variance: W1 needs a concrete remediation decision rule

**Finding:** variance_protocol, L3
**Section:** `## Success Criteria — W1 CV sensitivity`

"Report as limitation and recommend additional samples" is not a decision rule. Without a
concrete threshold, W1 becomes indefinite deferral.

**Fix:** Replace the current W1 text with:
> "**W1 — CV>20% remediation protocol:** If post-run sample CV at n=100K exceeds 20% for
> any variant: (a) flag the H-100K result as INCONCLUSIVE; (b) the CI lower bound is still
> reported but not used for a binary H1 verdict; (c) to achieve 80% power at CV=20%, n≥112
> samples are required — add 49 additional Criterion iterations per affected variant and
> re-run only the affected variant before re-running the full analysis."

*(n≥112 derived from standard power formula: n = (z_α/2 + z_β)² × CV² / r² where
z_α/2=2.33 (Holm α=0.0125), z_β=0.842 (80% power), CV=0.20, r=0.10)*

### R-10 · Error budget: Justify CV=15% assumption or lower power claim

**Finding:** error_budget, L3
**Section:** `## Power Analysis Details`

CV=15% is "estimated from prior bench" but the specific run is not cited. The prior experiment
is acknowledged as having measurement gaps.

**Fix:** Either (a) cite the specific prior bench output file and iteration count from which
CV=15% was estimated, or (b) use a more conservative CV=20% for the power analysis, yielding
~65% power at n=63. If (b), adjust the power claim statement to: "n=63 provides ~65% power
at CV=20%, r=10%, Holm-corrected α=0.0125. If observed CV≤15%, power is ~80%."

### R-11 · Statistical corrections: Pre-specify m-sweep exclusion from H5 verdict

**Finding:** statistical_corrections, L3
**Section:** `## Statistical Plan / H5`

The m-sweep (W2) runs on the same data as H5's confirmatory gate. Implicit selection of m-value
or metric from the sweep would inflate effective comparisons.

**Fix:** Add an explicit statement to the Analysis Plan:
> "The H5 confirmatory verdict is derived *exclusively* from the 10-seed run at m=5000. The
> m-sweep output (m ∈ {500, 1000, 2000, 10000}) is tabulated separately and is never used to
> select, adjust, or re-interpret the H5 verdict. The analysis script will report both but
> the H5 verdict section references only the m=5000 runs."

### R-12 · Unit interference: Extend apply_phase1_changes.sh to verify key changes

**Finding:** unit_interference, L2
**Section:** `## Phase 1 Source Changes`

The Phase 1 verification script checks Cargo.toml features but does not verify:
- `build_global().unwrap()` is present in each bench binary's `main()`
- `step_timing::reset()` is called *before* (not after) `variant_fn()`
- N_THREADS constant is consistent across all bench binaries and tw_profiler.rs

**Fix:** Add to `apply_phase1_changes.sh`:
```bash
# Verify build_global() is present in each bench binary
for BENCH in tw_baseline_bench tw_thread_local_bench tw_partial_rank_bench tw_avx2_bench tw_combined_bench; do
  grep -q "build_global" benches/${BENCH}.rs || { echo "ERROR: build_global missing in ${BENCH}.rs"; exit 1; }
done

# Verify step_timing::reset() precedes variant_fn in tw_profiler.rs
python3 - << 'EOF'
import re
content = open("src/bin/tw_profiler.rs").read()
reset_pos = content.find("step_timing::reset()")
variant_pos = content.find("variant_fn(")
if reset_pos == -1 or variant_pos == -1:
    print("ERROR: step_timing::reset() or variant_fn not found")
    exit(1)
if reset_pos > variant_pos:
    print("ERROR: step_timing::reset() appears AFTER variant_fn — contamination bug")
    exit(1)
print("OK: step_timing::reset() precedes variant_fn")
EOF

# Verify N_THREADS is consistent across all bench files and tw_profiler
NTHREADS_VALUES=$(grep -h "const N_THREADS" benches/*.rs src/bin/tw_profiler.rs | sort -u | wc -l)
[ "$NTHREADS_VALUES" -eq 1 ] || { echo "ERROR: inconsistent N_THREADS across bench files"; exit 1; }
echo "OK: N_THREADS is consistent across all bench binaries"
```

---

## Recommended Revisions (Smaller Risk — Not Blocking)

### REC-1 · Reproducibility: Pin cargo-criterion version

Add cargo-criterion to `Cargo.toml` dev-dependencies:
```toml
[dev-dependencies]
cargo-criterion = "=1.1.0"  # pin exact version; verify current stable before use
```
Alternatively, add `cargo criterion --version >> results/analysis/hardware_profile.txt`
to the run script to capture the version actually used.

### REC-2 · Reproducibility: Document MERFISH H5AD download provenance

Add to the plan's Data Provenance section:
- Allen Brain Cell Atlas portal URL
- File name and SHA256 checksum
- Whether registration/agreement is required
- Date downloaded / version of the dataset

### REC-3 · Ecological validity: Add scope qualifier to "field-deployable" decision outcome

The plan lists "whether tw_approx is field-deployable on structured biological data" as a decision
outcome, but the test only covers n=10K at m/n=0.5. Revise the decision statement to:
> "Whether tw_approx delivers acceptable speedup and quality at n=10K, m=5000 on MERFISH PCA-50,
> as a necessary (but not sufficient) precondition for field deployability at larger scale."

### REC-4 · Measurement alignment: Qualify the optimization targeting claim for d=10

Add to H0/H1-clean:
> "**Scope limitation:** Step-fraction results at d=10 characterize the Gaussian benchmark regime
> only. The ordering of dominant steps may differ at d=50 (MERFISH production workload), where
> distance computation's FLOP advantage is ~3.6× larger. The d=50 profile requires a separate
> MERFISH profiling run as follow-up."

### REC-5 · Benchmark representativeness: Note MERFISH saturation as testbed sensitivity

Add to Threats to Validity:
> "**MERFISH saturation (W-MERFISH):** Trustworthiness values near the MERFISH ceiling (≈0.99)
> reduce the discriminative power of the quality gate (|delta| < 0.001). A passing result on
> this dataset does not rule out quality failure on higher-variance datasets. Report the absolute
> trustworthiness values (not just delta) to allow readers to assess ceiling effects."

### REC-6 · H-partial-MERFISH: Add a minimum meaningful difference

The plan has no pre-specified threshold for what constitutes a meaningful difference in CI
half-widths. Add:
> "A reduction of ≥0.05 in half-width (i.e., MERFISH half-width ≤ Gaussian half-width − 0.05)
> is treated as a practically meaningful difference; smaller differences are reported but noted
> as measurement-variance-scale variation."

---

## Red-Team Findings Requiring Decisions

All 7 red-team findings are marked `requires_decision: true`. The revisions above (R-2, R-3,
R-4, R-5, R-6) directly address them. The two remaining red-team findings not covered above:

**RT-Asymmetric-Effort (R-baseline):** Document the baseline variant's version/commit and confirm
it has not received performance modifications since the optimized variants were developed. Add to
the Controlled Variables table:
> "| Baseline implementation | Frozen at commit [HASH] prior to any variant optimization | Ensures comparison is between optimization strategies, not development effort |"

**RT-Cache-Footprint (R-cache):** Either add a note acknowledging this as an uncontrolled confound
(with estimated effect bound), or add a perf-stat check to the execution protocol:
> "Run `perf stat -e cache-misses,instructions` for one representative n=100K iteration of baseline
> and avx2_kernel. Record cache-miss rate ratio. If avx2_kernel has >2× fewer L1-I misses than
> baseline, flag this as a cache-footprint confound in the report."
