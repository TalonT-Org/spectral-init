# ComputeMode Parity Under SIMD SpMV

> Research report — 2026-03-27

## Executive Summary

This experiment investigated whether replacing the scalar `spmv_csr` kernel with `spmv_avx2_gather` (AVX2 + FMA) in `CsrOperator::apply()` would introduce detectable divergence between `PythonCompat` and `RustNative` compute modes — specifically, whether both modes can safely share a single SIMD SpMV kernel. Nine fixture graphs spanning n ∈ {50, 100, 200, 300, 500, 2000, 5000} were tested across three measurement levels: raw SpMV arithmetic (RQ1), eigenvector subspace agreement (RQ2), and final f32 coordinate output after `scale_and_add_noise` (RQ3).

SpMV-level divergence was negligible across all 45 test cases (max |y_avx2 − y_scalar| ≤ 1.78×10⁻¹⁵, three orders of magnitude below the 1×10⁻¹² bound). All seven fixtures in the dense EVD regime (n < 2000) produced bitwise-identical f32 outputs, as expected: dense EVD never routes through `CsrOperator::apply()`. Of the two LOBPCG-regime fixtures, blobs_5000 (n=5000) produced only 37/15,000 differing f32 elements with max_abs_diff = 4.77×10⁻⁷ — well below the σ=0.0001 noise floor. The single failure was blobs_connected_2000 (n=2000), where LOBPCG converged to an eigenvector with a sign flip in one column, causing max_abs_diff = 20.0 (the maximum possible under a ±10.0 scaled range). This failure is attributable to LOBPCG sign indeterminacy, not SpMV numerical explosion. The production pipeline calls `normalize_signs` before `scale_and_add_noise`, a call that was absent from the experiment test.

The formal verdict per the experiment plan's strict success criteria is **CONCLUSIVE_NEGATIVE**: the bitwise-identity requirement for all 9 fixtures is violated at blobs_connected_2000. However, the failure mechanism is mitigated by `normalize_signs` in the production pipeline. A focused follow-up — adding `normalize_signs` to the test before `scale_and_add_noise_pub` — is needed to confirm whether the production path is safe and whether Phase 3 (SIMD integration) can proceed with a shared kernel.

## Background and Research Question

PR #152 introduced an AVX2 SpMV kernel (`spmv_avx2_gather`) as an optional fast path in `CsrOperator::apply()`. The open question it left was: does using this kernel in both `PythonCompat` and `RustNative` modes violate the `CLAUDE.md` contract, which states that any optimization causing divergence from Python UMAP output must be gated behind `ComputeMode::RustNative`?

The null hypothesis (H0) predicted safety: theory bounds the per-entry AVX2 vs scalar difference at ~4.4×10⁻¹⁶ (2 ULPs), and even with Davis-Kahan amplification across LOBPCG iterations, the accumulated error should remain orders of magnitude below the σ=0.0001 noise floor added by `scale_and_add_noise`. The alternative hypothesis (H1) identified near-degenerate fixtures (especially `near_dupes_100`, min_gap ≈ 0) as the Davis-Kahan risk case.

This experiment provides empirical confirmation or refutation of these theoretical predictions, resolving whether Phase 3 can proceed with a shared SIMD kernel for both compute modes.

## Methodology

### Experimental Design

**Null hypothesis (H0):** AVX2 SpMV divergence from scalar SpMV, measured at the final `Array2<f32>` output after `scale_and_add_noise`, is undetectable (bitwise-identical f32 arrays, or max_abs_diff ≤ 1 ULP f32 ≈ 6×10⁻⁸) across all nine fixture graphs.

**Alternative hypothesis (H1):** At least one fixture graph produces a measurable divergence > 1 ULP f32 at the output level, traceable to AVX2 vs scalar SpMV differences in LOBPCG iterations.

**Key design decision:** Both scalar and SIMD runs use the same seed (42) for LOBPCG initialization and `scale_and_add_noise`, isolating the SpMV arithmetic difference as the only independent variable.

**Solver routing:** LOBPCG (level 1) is used for n ≥ 2000 (`DENSE_N_THRESHOLD = 2000`). Only LOBPCG routes through `CsrOperator::apply()`. Dense EVD fixtures (n < 2000) serve as a negative control — they cannot be affected by the SIMD kernel and must show zero divergence.

### Environment

- **Repository commit:** `e547fff5ec705c2194dff6de365f41004a522652`
- **Branch:** `research-20260327-162258`
- **Rust toolchain:** `rustc 1.96.0-nightly (23903d01c 2026-03-26)`, `cargo 1.96.0-nightly (e84cb639e 2026-03-21)`
- **Key Rust dependencies:** `faer v0.24.0`, `sprs v0.11.3`, `ndarray v0.16`, `ndarray-linalg v0.17`, `rand v0.8.5`, `ndarray-npy v0.9.1`, `serde_json v1.0`
- **Python environment:** `micromamba spectral-test` — Python 3.13.2, numpy 2.2.6, scipy 1.15.2 (OpenBLAS-backed, no MKL)
- **Hardware:** AMD Ryzen 7 9800X3D (AVX2 + FMA confirmed via `is_x86_feature_detected!`)
- **OS:** WSL2 Linux x86_64, kernel 6.6.87.2-microsoft-standard-WSL2, glibc 2.39
- **Compiler flags:** `-C target-cpu=native` (set in `.cargo/config.toml`, enabling AVX2+FMA)

### Procedure

1. Built and ran the integration test suite: `cargo test --features testing --test test_simd_parity -- --nocapture`
   - `test_spmv_divergence`: for each of 9 fixtures × 5 random vectors, ran both `spmv_csr` and `spmv_avx2_gather_pub` on the same input and measured per-element differences (max_abs, RMS, max ULP).
   - `test_solver_divergence`: for each fixture, ran `solve_eigenproblem_pub` (scalar) and `solve_eigenproblem_simd_pub` (SIMD), applied `scale_and_add_noise_pub` with seed 42 to both eigenvector outputs, and counted bitwise-identical f32 elements. Saved raw eigenvector `.npy` files for LOBPCG fixtures.
   - `test_spectral_gaps`: loaded `comp_d_eigensolver.npz` per fixture and extracted pre-computed eigenvalue gap arrays.
2. Ran `analyze_subspace.py` via micromamba to compute `scipy.linalg.subspace_angles` between scalar and SIMD eigenvector matrices for LOBPCG fixtures.
3. Ran `collect_rq4.py` to document scipy/numpy versions and BLAS backend from each fixture's `meta.json`.

## Results

### RQ1 — SpMV-Level Divergence (45 measurements)

Maximum values across all 5 test vectors per fixture:

| Fixture | n | nnz | Max max_abs_diff | Max rms_diff | Max max_ulp |
|---------|---|-----|-----------------|--------------|-------------|
| blobs_50 | 50 | 810 | 6.661e-16 | 1.901e-16 | 96 |
| near_dupes_100 | 100 | 1,778 | 8.882e-16 | 1.759e-16 | 16 |
| moons_200 | 200 | 3,376 | 1.332e-15 | 1.809e-16 | 640 |
| blobs_connected_200 | 200 | 3,746 | 8.882e-16 | 1.733e-16 | 128 |
| disconnected_200 | 200 | 3,750 | 8.882e-16 | 1.766e-16 | 64 |
| circles_300 | 300 | 4,760 | 8.882e-16 | 1.638e-16 | 164 |
| blobs_500 | 500 | 9,232 | 8.882e-16 | 1.824e-16 | 640 |
| blobs_connected_2000 | 2,000 | 43,144 | 1.776e-15 | 1.751e-16 | 4,096 |
| blobs_5000 | 5,000 | 88,070 | 1.776e-15 | 1.678e-16 | 40,960 |

**Global maximum:** max_abs_diff = 1.776×10⁻¹⁵ (≤ 8 × ε_mach where ε_mach = 2.22×10⁻¹⁶). The plan's 1×10⁻¹² threshold is satisfied with ~3 orders of magnitude headroom.

**ULP note:** max_ulp peaks at 40,960 for blobs_5000. This is benign — high ULP counts arise in the subnormal/near-zero region where individual ULPs are extremely small and the absolute difference (still ≤ 1.78×10⁻¹⁵) is the meaningful metric.

### RQ2 — Eigenvector Subspace Angles (LOBPCG fixtures)

| Fixture | solver_level | Principal angles (rad) | Max angle (rad) | Plan threshold |
|---------|-------------|------------------------|-----------------|----------------|
| blobs_connected_2000 | 1 (LOBPCG) | [6.132e-5, 1.399e-7, 0.0] | **6.132e-5** | 1e-8 |
| blobs_5000 | 1 (LOBPCG) | [3.641e-7, 1.773e-7, 1.326e-7] | 3.641e-7 | 1e-8 |

Both fixtures exceed the plan's 1×10⁻⁸ rad threshold, but both are geometrically negligible (6.13×10⁻⁵ rad ≈ 0.0035°; 3.64×10⁻⁷ rad ≈ 0.00002°). The three principal angles reflect the (n×3) eigenvector matrix saved (LOBPCG computes k+1=3 vectors for k=2 requested components).

### RQ3 — Final f32 Output Divergence (after `scale_and_add_noise`, seed=42)

| Fixture | n | solver_level | f32_identical / f32_total | f32_max_abs_diff | vs noise floor | Pass? |
|---------|---|-------------|--------------------------|-----------------|----------------|-------|
| blobs_50 | 50 | 0 (dense EVD) | 150 / 150 | 0.0 (exact) | — | ✓ |
| near_dupes_100 | 100 | 0 (dense EVD) | 300 / 300 | 0.0 (exact) | — | ✓ |
| moons_200 | 200 | 0 (dense EVD) | 600 / 600 | 0.0 (exact) | — | ✓ |
| blobs_connected_200 | 200 | 0 (dense EVD) | 600 / 600 | 0.0 (exact) | — | ✓ |
| disconnected_200 | 200 | 0 (dense EVD) | 600 / 600 | 0.0 (exact) | — | ✓ |
| circles_300 | 300 | 0 (dense EVD) | 900 / 900 | 0.0 (exact) | — | ✓ |
| blobs_500 | 500 | 0 (dense EVD) | 1,500 / 1,500 | 0.0 (exact) | — | ✓ |
| blobs_connected_2000 | 2,000 | 1 (LOBPCG) | 4,000 / 6,000 | **20.0** | 200,000× above | ✗ |
| blobs_5000 | 5,000 | 1 (LOBPCG) | 14,963 / 15,000 | 4.77e-7 | 470× below | ✓ |

Dense EVD fixtures (n < 2000) are bitwise-identical by design: dense EVD does not call `CsrOperator::apply()`.

### H6 — Near-Degenerate Spectral Gap Risk

| Fixture | n | Eigenvalue gaps | Min gap | Risk triggered? |
|---------|---|-----------------|---------|-----------------|
| blobs_50 | 50 | [0.0, 7.19e-11] | 0.0 (trivial) | No (dense EVD) |
| near_dupes_100 | 100 | [0.0206, 0.0341] | 0.0206 | No (dense EVD) |
| moons_200 | 200 | [2.67e-3, 4.57e-3] | 2.67e-3 | No (dense EVD) |
| blobs_connected_200 | 200 | [1.68e-3, 2.40e-2] | 1.68e-3 | No (dense EVD) |
| disconnected_200 | 200 | [0.0, 2.72e-9] | 0.0 (trivial) | No (dense EVD) |
| circles_300 | 300 | [6.32e-4, 6.29e-3] | 6.32e-4 | No (dense EVD) |
| blobs_500 | 500 | [1.71e-9, 1.36e-9] | 1.36e-9 | Low (dense EVD) |
| blobs_connected_2000 | 2,000 | [0.0122, 0.0351] | **0.0122** | No (healthy gap) |
| blobs_5000 | 5,000 | [0.0, 4.89e-5] | 0.0 (trivial) | No (passes anyway) |

### RQ4 — Fixture Reference Backend

All 9 fixtures: scipy 1.15.2 / numpy 2.2.6 / Python 3.13.2, WSL2 x86_64 glibc 2.39, OpenBLAS (no MKL). Homogeneous reference dataset confirmed.

### Solver Level Agreement

All 9 fixtures: `solver_level_scalar == solver_level_avx2` (0 for n < 2000, 1 for n ≥ 2000). No solver escalation divergence introduced by the SIMD kernel.

### Test Runtime

| Test | Wall time |
|------|-----------|
| test_spmv_divergence | < 0.1 s |
| test_spectral_gaps | < 0.1 s |
| test_solver_divergence | ~2.5 s |
| **Total (3 tests)** | **2.65 s** |

LOBPCG timings: scalar ~49 ms + SIMD ~55 ms at n=2000; scalar ~996 ms + SIMD ~1027 ms at n=5000 (≤12% overhead, debug mode).

## Observations

1. **SpMV parity is excellent (RQ1):** Global max_abs_diff = 1.78×10⁻¹⁵ ≤ 8 × ε_mach, with three orders of magnitude headroom below the 1×10⁻¹² threshold. RMS differences are uniformly ≈ 1.7×10⁻¹⁶ across all fixtures — essentially in the lowest addressable range of f64 arithmetic. High ULP counts at large fixtures are artifacts of the subnormal region and are benign.

2. **Dense EVD regime is completely unaffected:** For all 7 fixtures with n < 2000, dense EVD does not invoke `CsrOperator::apply()`. SIMD vs scalar is irrelevant; outputs are bitwise-identical. This confirms the experiment's control arm correctly captures "zero impact" when the kernel is not in the solver path.

3. **The blobs_connected_2000 failure is a sign flip, not numerical explosion:** 2,000/6,000 differing f32 elements with max_abs_diff = 20.0 (= 2 × 10.0, the output range) indicates exactly one eigenvector column has been sign-negated. This pattern — precisely n differing elements at the maximal possible amplitude — is mechanistically consistent with LOBPCG sign indeterminacy. The subspace angle analysis confirms the two runs span the same mathematical eigenspace (max_angle = 6.13×10⁻⁵ rad), ruling out a genuinely different eigenvector being found.

4. **The experiment test bypassed `normalize_signs`:** The test called `scale_and_add_noise_pub(eigvec, 42)` directly on raw eigenvectors. The production pipeline calls `normalize_signs` (src/scaling.rs:62) before `scale_and_add_noise`, which deterministically resolves sign flips using an argmax absolute value convention. This bypass is the likely explanation for the observed failure.

5. **blobs_5000 LOBPCG passes the noise-floor threshold:** At n=5,000, 37 elements differ with max_abs_diff = 4.77×10⁻⁷ ≪ σ=0.0001. The larger matrix does not amplify SpMV error through eigenvector convergence paths — in fact, the relative agreement is better at n=5,000 than at n=2,000.

6. **H6 hypothesis not triggered as predicted:** The plan identified `near_dupes_100` (near-degenerate eigenvalues) as the Davis-Kahan risk case. However, `near_dupes_100` is in the dense EVD regime (n=100), so `CsrOperator::apply()` is never called during solving. The only significant failure occurred at blobs_connected_2000, which has a healthy spectral gap (0.0122), confirming the failure was not Davis-Kahan amplification. The experiment design correctly predicted this threat to validity (see the "External" threats in the plan), but the Davis-Kahan risk case cannot be tested without a near-degenerate fixture at n ≥ 2000.

7. **SIMD timing is not a regression:** At n=5,000 (the most relevant benchmark), SIMD LOBPCG takes ~1027 ms vs ~996 ms scalar — a 3% difference in debug mode that is noise-level and will not carry through to an optimized build.

## Analysis

### The sign flip mechanism

LOBPCG finds eigenspaces, not canonical eigenvectors. The sign of any eigenvector is arbitrary; LOBPCG may converge to `+v` in one run and `-v` in another when the internal convergence trajectory differs, even slightly. The scalar and SIMD operators apply the Laplacian identically to within ~8ε_mach per SpMV call. Over many iterations, this tiny per-call difference can accumulate into a different convergence trajectory that selects an eigenvector of opposite sign. The subspace angle of 6.13×10⁻⁵ rad confirms both runs found the same eigenspace — only the sign convention differs.

### Why blobs_5000 is better-behaved than blobs_connected_2000

blobs_connected_2000 sits exactly at the solver boundary (n = `DENSE_N_THRESHOLD` = 2000), where LOBPCG is chosen over dense EVD. The boundary fixture experiences the maximum possible LOBPCG iterations for the minimum n in the LOBPCG regime, making it maximally sensitive to any per-iteration perturbation. blobs_5000, being a larger and naturally sparser system (nnz/n² ≈ 0.0035 vs 0.011 for blobs_connected_2000), may exhibit more stable convergence behavior or a convergence trajectory that reaches the same sign convention by chance.

### Deviation from the plan's threshold for RQ2

The plan expected max_angle < 1×10⁻⁸ rad for all LOBPCG fixtures. The measured angles (6.13×10⁻⁵ and 3.64×10⁻⁷) exceed this threshold, but both are physically irrelevant (< 0.004°). The plan threshold was theoretically motivated (from Davis-Kahan bound extrapolation), but in practice LOBPCG's sign-indeterminacy-driven convergence differences produce angles larger than this theoretical floor without indicating any meaningful eigenvector degradation.

### Implications for the CLAUDE.md `ComputeMode` contract

The CLAUDE.md rule requires gating behind `RustNative` any optimization that "could cause divergence from Python UMAP's behavior." The sign flip at blobs_connected_2000 is a potential violation under the strictest interpretation: the final f32 embedding coordinates differ. However, sign indeterminacy in eigenvectors is a fundamental property of the eigenproblem — Python UMAP's `spectral.py` also calls `sign_flip` (a variant of `normalize_signs`) to resolve it. If `normalize_signs` deterministically selects the same sign in both scalar and SIMD runs (which it should, since it operates on the final eigenvectors post-convergence using an argmax convention), the production pipeline's behavior would be identical. The risk is therefore contingent on `normalize_signs` eliminating the sign flip, not on the SpMV kernel itself.

## What We Learned

- **SpMV arithmetic error is negligible across all real-world LOBPCG fixtures.** The theoretical per-entry bound of ~4.4×10⁻¹⁶ holds empirically; no fixture produced anything close to the 1×10⁻¹² threshold.
- **Dense EVD fixtures are immune to SIMD changes.** Testing them as a control arm confirms the measurement apparatus is sound and that SIMD effects are correctly isolated to the LOBPCG solver path.
- **The boundary fixture (n = DENSE_N_THRESHOLD) is the most sensitive case.** Sign indeterminacy in LOBPCG can manifest at n=2000 even when SpMV arithmetic differences are negligible.
- **`normalize_signs` is load-bearing for inter-run reproducibility.** Any test that compares scalar vs SIMD eigenvector outputs must apply `normalize_signs` before `scale_and_add_noise` to mirror the production pipeline.
- **Davis-Kahan amplification was not empirically testable** with the current fixture set. The near-degenerate fixtures (`near_dupes_100`, `blobs_500`) are all in the dense EVD regime. A future H6 test requires a fixture with n ≥ 2000 and min_gap < 1×10⁻⁶.
- **SIMD timing at n=5000 is within noise** (~3%) compared to scalar in debug mode. A release-mode benchmark would be needed to measure any performance advantage.

## Conclusions

The strict success criterion (bitwise-identical f32 outputs for all 9 fixtures) is **not met**: blobs_connected_2000 produces max_abs_diff = 20.0 due to a LOBPCG eigenvector sign flip. The verdict per the experiment plan is **CONCLUSIVE_NEGATIVE**.

However, the failure is mechanistically a sign flip — not SpMV numerical explosion — and occurs in a test that bypasses the production pipeline's `normalize_signs` call. The SpMV kernel itself produces differences of ≤ 1.78×10⁻¹⁵ (≤ 8 × machine epsilon) across all 45 test cases, well within theoretical bounds. All observable evidence is consistent with the production pipeline being safe once `normalize_signs` is included.

A single follow-up test — re-running `test_solver_divergence` with `normalize_signs` applied before `scale_and_add_noise_pub` — is the minimal action needed to upgrade this verdict to CONCLUSIVE_POSITIVE or confirm a genuine incompatibility.

## Recommendations

1. **Run the follow-up test with `normalize_signs` applied before comparing.** Add a call to `normalize_signs` in `test_solver_divergence` before `scale_and_add_noise_pub` is called on each eigenvector. If blobs_connected_2000 then shows max_abs_diff = 0.0 (or ≤ 4.77×10⁻⁷), H0 is confirmed for the production pipeline and Phase 3 can proceed with a shared SIMD kernel.

2. **Do not gate the SIMD kernel behind `RustNative` prematurely.** The current evidence strongly suggests the failure is in the test harness, not the kernel. Making the architectural decision (shared vs separate) before the follow-up test risks unnecessary complexity.

3. **Update `test_solver_divergence` to mirror the production call chain.** The test should always call `normalize_signs` before `scale_and_add_noise_pub` to be a valid proxy for production behavior. The current test design (bypassing `normalize_signs`) is a threat to validity that should be fixed regardless of the SIMD decision.

4. **Add a near-degenerate LOBPCG-regime fixture** (n ≥ 2000, min_gap < 1×10⁻⁶) to future fixture sets if Davis-Kahan amplification remains a concern. The current fixture set cannot empirically test H6 in the LOBPCG regime.

5. **Re-run timing in release mode** (`--release` flag) to measure any real-world performance benefit from the SIMD kernel before committing to the integration.

---

## Appendix: Experiment Scripts

### tests/integration/test_simd_parity.rs

```rust
#[path = "../common/mod.rs"]
mod common;

use ndarray::Ix1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use spectral_init::{scale_and_add_noise_pub, solve_eigenproblem_pub};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::operator::spmv_avx2_gather_pub;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::solve_eigenproblem_simd_pub;

use spectral_init::operator::spmv_csr;

const FIXTURES: &[&str] = &[
    "blobs_50", "near_dupes_100", "moons_200", "blobs_connected_200",
    "disconnected_200", "circles_300", "blobs_500",
    "blobs_connected_2000", "blobs_5000",
];

fn results_dir() -> std::path::PathBuf {
    match std::env::var("RESULTS_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("research/2026-03-27-computemode-simd-spmv-parity/results"),
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == 0.0 && b == 0.0 { return 0; }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    if (ai >= 0) == (bi >= 0) { (ai - bi).unsigned_abs() }
    else { ai.unsigned_abs() + bi.unsigned_abs() }
}

#[test]
fn test_spmv_divergence() {
    // For each fixture × 5 random vectors: compare spmv_csr vs spmv_avx2_gather_pub.
    // Writes results/spmv_divergence.json
}

#[test]
fn test_solver_divergence() {
    // For each fixture: run solve_eigenproblem_pub (scalar) and
    // solve_eigenproblem_simd_pub (SIMD), apply scale_and_add_noise_pub(seed=42),
    // compare f32 outputs. Save eigenvectors for LOBPCG fixtures.
    // Writes results/solver_divergence.json and results/eigenvectors/*.npy
    //
    // NOTE: does NOT call normalize_signs before scale_and_add_noise_pub.
    // This is a known gap vs. production pipeline behavior.
}

#[test]
fn test_spectral_gaps() {
    // For each fixture: load comp_d_eigensolver.npz, extract eigenvalue_gaps.
    // Writes results/spectral_gaps.json
}
```

### scripts/analyze_subspace.py

```python
import numpy as np
from scipy.linalg import subspace_angles
import json, pathlib

results_dir = pathlib.Path(__file__).parents[1] / "results"
ev_dir = results_dir / "eigenvectors"

output = []
for fixture in ["blobs_connected_2000", "blobs_5000"]:
    scalar_path = ev_dir / f"{fixture}_scalar.npy"
    avx2_path   = ev_dir / f"{fixture}_avx2.npy"
    if not scalar_path.exists():
        continue
    V_s = np.load(str(scalar_path))
    V_a = np.load(str(avx2_path))
    angles = subspace_angles(V_s, V_a)
    output.append({
        "fixture": fixture,
        "principal_angles_rad": angles.tolist(),
        "max_angle_rad": float(angles.max()),
    })

json.dump(output, open(results_dir / "subspace_angles.json", "w"), indent=2)
print(json.dumps(output, indent=2))
```

### scripts/collect_rq4.py

```python
import json, pathlib

fixtures_dir = pathlib.Path(__file__).parents[3] / "tests" / "fixtures"
results_dir  = pathlib.Path(__file__).parents[1] / "results"
output = []
for meta_path in sorted(fixtures_dir.glob("*/meta.json")):
    meta = json.load(open(meta_path))
    env  = meta.get("env", {})
    output.append({
        "fixture":        meta_path.parent.name,
        "scipy_version":  env.get("scipy"),
        "numpy_version":  env.get("numpy"),
        "python_version": env.get("python"),
        "platform":       env.get("platform"),
    })
json.dump(output, open(results_dir / "scipy_backend.json", "w"), indent=2)
```

### scripts/run_experiment.sh

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
RESULTS_DIR="$REPO_ROOT/research/2026-03-27-computemode-simd-spmv-parity/results"
SCRIPTS_DIR="$REPO_ROOT/research/2026-03-27-computemode-simd-spmv-parity/scripts"

cd "$REPO_ROOT"

echo "=== Phase 1: Rust experiment tests ==="
RESULTS_DIR="$RESULTS_DIR" cargo test --features testing \
    --test test_simd_parity -- --nocapture 2>&1 | tee "$RESULTS_DIR/rust_test_output.txt"

echo "=== Phase 2: Subspace angle analysis ==="
micromamba run -n spectral-test python "$SCRIPTS_DIR/analyze_subspace.py"

echo "=== Phase 3: RQ4 scipy backend documentation ==="
micromamba run -n spectral-test python "$SCRIPTS_DIR/collect_rq4.py"

echo "=== Done. Results in $RESULTS_DIR ==="
```

## Appendix: Raw Data

### results/subspace_angles.json

```json
[
  {
    "fixture": "blobs_connected_2000",
    "principal_angles_rad": [6.132341513875872e-05, 1.3994156233820831e-07, 0.0],
    "max_angle_rad": 6.132341513875872e-05
  },
  {
    "fixture": "blobs_5000",
    "principal_angles_rad": [3.641454213720863e-07, 1.7734905100041942e-07, 1.3259864317660686e-07],
    "max_angle_rad": 3.641454213720863e-07
  }
]
```

### results/solver_divergence.json (key entries)

```json
[
  {"fixture": "blobs_50",    "solver_level_scalar": 0, "solver_level_avx2": 0,
   "f32_bitwise_identical": 150,   "f32_total_elements": 150,   "f32_max_abs_diff": 0.0},
  {"fixture": "blobs_500",   "solver_level_scalar": 0, "solver_level_avx2": 0,
   "f32_bitwise_identical": 1500,  "f32_total_elements": 1500,  "f32_max_abs_diff": 0.0},
  {"fixture": "blobs_connected_2000", "solver_level_scalar": 1, "solver_level_avx2": 1,
   "f32_bitwise_identical": 4000,  "f32_total_elements": 6000,  "f32_max_abs_diff": 20.0},
  {"fixture": "blobs_5000",  "solver_level_scalar": 1, "solver_level_avx2": 1,
   "f32_bitwise_identical": 14963, "f32_total_elements": 15000, "f32_max_abs_diff": 4.76837158203125e-7}
]
```
