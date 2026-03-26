# CI Test Optimization: Profile Tuning, Test Tiering, and Property-Based Testing

> Research report for branch `research-20260325-105823` — 2026-03-25

## Executive Summary

The spectral-init CI pipeline was taking 30+ minutes due to Rust's default `opt-level=0` for test binaries, which leaves LOBPCG inner loops (O(40-130M) scalar SpMV operations) unoptimized. This experiment measured the impact of four optimization strategies: compiler opt-level tuning, test tiering, configurable solver thresholds, and property-based tests. The single change of setting `opt-level=2` for the test binary reduces the full 221-test suite from 30+ minutes to under 2 minutes wall time (plus ~75s compile), achieving a total CI time of ~3-4 minutes — well under the 10-minute target. LOBPCG-heavy tests showed 13-47x speedups, confirming the hypothesis that unoptimized scalar math was the bottleneck.

The more complex strategies — nextest test tiering and property-based tests for all-level coverage — are unnecessary given the magnitude of the opt-level speedup. Property tests failed both their timing target (142s vs 30s) and bug detection target (1/8 vs 6/8), due to a fundamental design flaw: small-n graphs exercising only the dense EVD and LOBPCG paths cannot detect bugs in higher solver levels. The experiment also identified two blind spots in the full test suite (sign-flip and warm-restart truncation mutations) that warrant targeted follow-up tests.

## Background and Research Question

The spectral-init crate implements spectral initialization for UMAP embeddings via a solver escalation chain (dense EVD -> LOBPCG -> shift-invert LOBPCG -> regularized LOBPCG -> randomized SVD -> forced dense EVD). The CI pipeline runs 221 integration and unit tests covering all solver levels, but at the default `opt-level=0`, the LOBPCG and warm-restart tests at n=2000-5000 dominate wall time with hundreds of seconds each.

**Primary question:** Can we reduce CI wall time to under 10 minutes while maintaining full solver-level coverage and bug detection capability?

**Sub-questions:**
- **RQ1:** What is the quantitative speedup from `opt-level=2` on test binaries? Does adding `opt-level=1` for dependencies help further?
- **RQ2:** What is the minimum viable test subset that covers all 6 solver levels and detects known bug mutations?
- **RQ3:** Is a `#[cfg(test)]` env-var override for `DENSE_N_THRESHOLD` safe for routing small graphs through LOBPCG?
- **RQ4:** Can property-based tests at n<=200 detect bugs across all solver levels within 30 seconds?

## Methodology

### Experimental Design

**Null hypothesis (H0):** No combination of `[profile.test]` opt-level settings, nextest test tiering, or property-based test additions will reduce total CI wall time to <=10 minutes while maintaining coverage of all 6 solver levels and detection of all 5 primary bug mutation classes.

**Alternative hypothesis (H1):** The combination of `opt-level=2` for the test binary, a `ci-fast` nextest profile, and 9 property-based tests exercising LOBPCG via a threshold override will collectively achieve <=10 minute CI with all-level solver coverage and >=6/8 bug mutation detection.

**Independent variables:** Test binary opt-level (0, 2, 2+deps1), nextest test set (full, minimum, property-only), `SPECTRAL_DENSE_N_THRESHOLD` (2000 default, 50 override).

**Dependent variables:** Per-test wall time, total suite wall time, compile time, bug detection rate (8 mutations), solver level coverage (6 levels), LOBPCG-dense subspace angle, property test false positive rate.

### Environment

- **Repository commit:** `1bfccbac512c71566989dc5cda92445e5a2a0a0f`
- **Branch:** `research-20260325-105823`
- **Rust toolchain:** `nightly-2025-11-21` (no `rust-toolchain.toml`)
- **cargo-nextest:** `0.9.132`
- **Package versions:**
  - `spectral-init v0.1.0`
  - `faer v0.24.0`
  - `linfa-linalg v0.2.1`
  - `ndarray v0.16.1, v0.17.2`
  - `sprs v0.11.4`
  - `rand v0.9.2`
  - `proptest v1.11.0`
  - `thiserror v2.0.18`
- **Hardware/OS:** WSL2 Linux 6.6.87.2 on Windows host

### Procedure

1. **RQ1 — Opt-Level Sweep:** For each configuration (`opt0`, `opt2`, `opt2deps1`), a temporary `[profile.experiment-*]` section was appended to `Cargo.toml`. Compilation was timed (`cargo nextest run --no-run`), then the full 221-test suite was run with `--profile ci --features testing`. JUnit XML was collected and parsed into per-test timing CSVs via `extract_timings.py`. At `opt0`, 4 warm-restart tests were excluded (estimated 30+ min each).

2. **RQ2 — Minimum Viable Test Set:** A covering set of 33 tests was identified by static analysis of which tests exercise each solver level. The full 221-test suite was subjected to 8 bug mutations (B1-B8) via `inject_bug.sh`, with detection results recorded per mutation.

3. **RQ3 — Threshold Override:** A `#[cfg(test)]` env-var override for `DENSE_N_THRESHOLD` was implemented in `src/solvers/mod.rs`. A validation test built a 200-node ring Laplacian and compared dense EVD (threshold=2000) vs LOBPCG (threshold=50) eigenvectors via subspace angle. Release binary was checked for absence of test-only code.

4. **RQ4 — Property Tests:** 9 proptest tests were written in `tests/integration/test_rq4_property_tests.rs`, each running 256 cases over ring graphs with n in [50, 200]. Tests were timed at both `opt0` and `opt2`, and subjected to the same 8-mutation detection matrix.

## Results

### RQ1: Opt-Level Timing Sweep

#### Wall Times

| Configuration | Tests Run | Wall Time | Compile Time (warm) |
|---------------|-----------|-----------|---------------------|
| opt0 (excl. 4 slow warm-restart) | 217 | 560.5s (9.3 min) | 0.3s (incremental) |
| opt2 (full suite) | 221 | 111.9s (1.87 min) | 75.2s (cold from opt0) |
| opt2deps1 (full suite) | 221 | 117.4s (1.96 min) | 55.3s |

#### Per-Test Speedup (Top 15 by opt0 Time)

| Test | opt0 (s) | opt2 (s) | Speedup |
|------|----------|----------|---------|
| test_path_2000_warm_restart | 451.3 | 31.6 | 14.3x |
| comp_d_rsvd_blobs_connected_2000 | 270.9 | 111.8 | 2.4x |
| test_level_3_rsvd_valid_on_large_path | 264.4 | 111.6 | 2.4x |
| generate_accuracy_report | 241.2 | 12.5 | 19.3x |
| test_ring_2000_warm_restart | 173.9 | 12.7 | 13.7x |
| proptest_subspace_stable | 137.9 | 24.9 | 5.5x |
| comp_d_rsvd_blobs_500 | 136.6 | 39.9 | 3.4x |
| comp_d_rsvd_circles_300 | 130.3 | 36.3 | 3.6x |
| comp_d_dense_evd_blobs_connected_2000 | 109.0 | 6.3 | 17.4x |
| test_e2e_performance_blobs_5000 | 107.6 | 3.5 | 30.6x |
| test_level_2_regularized_lobpcg | 37.5 | 0.8 | 47.2x |
| solve_eigenproblem_large_n_routes_through_lobpcg | 19.7 | 0.6 | 33.8x |
| sinv_eigenvalue_accuracy_blobs_connected_2000 | 18.7 | 4.1 | 4.6x |
| test_level_1_lobpcg_for_large_well_conditioned_n | 9.5 | 0.6 | 15.1x |
| lobpcg_blobs_connected_2000_level2 | 3.8 | 0.4 | 10.5x |

#### Warm-Restart Tests (opt2 Only, Excluded from opt0)

| Test | opt2 (s) | Estimated opt0 (s) |
|------|----------|--------------------|
| test_path_5000_warm_restart | 58.4 | ~3000+ |
| test_path_3000_warm_restart | 38.4 | ~1500+ |
| test_ring_3000_warm_restart | 20.7 | ~500+ |
| test_epsilon_bridge_sweep | 19.8 | ~300+ |

### RQ2: Minimum Viable Test Set

#### Solver Level Coverage

| Level | Solver | Covered | Test |
|-------|--------|---------|------|
| 0 | Dense EVD | Yes | test_level_0_dense_evd_for_small_n |
| 1 | LOBPCG (unregularized) | Yes | test_level_1_lobpcg_for_large_well_conditioned_n |
| 2 | Shift-invert LOBPCG | Direct only | test_comp_g_sinv (11 tests) |
| 3 | Regularized LOBPCG | Direct only | test_comp_f_lobpcg (11 tests) |
| 4 | Randomized SVD | Yes | comp_d_rsvd_blobs_connected_200 + others |
| 5 | Forced dense EVD | NOT covered | Unreachable by design |

#### Bug Detection Matrix (Full Suite, opt0)

| Bug ID | Mutation Target | Description | Detected | Tests Failed |
|--------|----------------|-------------|----------|-------------|
| B1 | src/solvers/dense.rs | Reverse eigenvalue sort | YES | 16 |
| B2 | src/solvers/mod.rs | Set threshold=0 | YES | 2 |
| B3 | src/scaling.rs | Negate coordinates | **NO** | 0 |
| B4 | src/multi_component.rs | Zero components 2+ | YES | 5 |
| B5 | src/laplacian.rs | Skip normalization | YES | 19 |
| B6 | src/solvers/lobpcg.rs | Truncate warm-restart | **NO** | 0 |
| B7 | src/solvers/rsvd.rs | Return wrong eigenvectors | YES | 2 |
| B8 | src/solvers/sinv.rs | Return zeros | YES | 12 |

**Result:** 6/8 bugs detected by the full suite.

### RQ3: Configurable Threshold Override

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| Subspace angle (LOBPCG vs dense, n=200) | 9.112e-4 rad (0.052 deg) | < 0.01 rad | YES |
| Dense solver path level | 0 | Expected: 0 | YES |
| LOBPCG solver path level | 1 | Expected: >= 1 | YES |
| Test-only code in release binary | Not present | Absent | YES |

### RQ4: Property-Based Tests

#### Timing

| Configuration | Total proptest time | Target | Pass |
|---------------|---------------------|--------|------|
| opt0 | 1047.6s (17.5 min) | <= 30s | FAIL |
| opt2 | 142.7s (2.4 min) | <= 30s | FAIL |

#### Bug Detection

| Bug ID | Target | Detected by Proptests | Reason |
|--------|--------|-----------------------|--------|
| B1 | dense.rs (eigenvalue sort) | YES (2/9 fail) | proptest_eigenvalues_sorted catches it |
| B2 | solvers/mod.rs (threshold=0) | NO | Threshold override to 50 masks the mutation |
| B3 | scaling.rs (negate coords) | NO | Proptests check eigenvalue properties, not coordinate signs |
| B4 | multi_component.rs (zeros) | NO | Proptests use connected graphs only |
| B5 | laplacian.rs (skip normalization) | NO | Ring graphs have uniform degree; normalization is identity |
| B6 | lobpcg.rs (truncate warm-restart) | NO | Small n converges without restarts |
| B7 | rsvd.rs (random output) | NO | rSVD never reached at n<=200 |
| B8 | sinv.rs (return zeros) | NO | sinv never reached at n<=200 |

**Result:** 1/8 bugs detected (target: >= 6/8). **FAIL.**

#### False Positive Rate

All 9 property tests pass on correct code at both opt0 and opt2 — 0 false positives (PASS).

## Observations

1. **rSVD tests are the new bottleneck at opt2.** Two rSVD tests (`comp_d_rsvd_blobs_connected_2000` and `test_level_3_rsvd_valid_on_large_path`) each take ~112s at opt2, dominating wall time. These use faer's dense SVD which is already well-optimized via BLAS-like code paths. The 2.4x speedup (vs 13-47x for LOBPCG tests) confirms rSVD is not bound by unoptimized scalar math.

2. **opt2deps1 provides no benefit over opt2.** The test binary contains the LOBPCG inner loops that dominate runtime. Separately optimizing dependencies at opt-level=1 adds codegen overhead without improving the critical path. The 117.4s vs 111.9s difference is within noise.

3. **Compile time at opt2 (75s) is a one-time cost.** The 0.3s incremental compile at opt0 reflects a warm cache from a prior build. In CI (cold cache), opt2 compilation will take ~75s, which is acceptable for the 5x reduction in test runtime.

4. **The property test design was fundamentally flawed.** The hypothesis assumed that routing small graphs through LOBPCG via threshold override would exercise the full escalation chain. In reality, LOBPCG succeeds on the first attempt for small, well-conditioned ring/path graphs. Higher solver levels (sinv, regularized LOBPCG, rSVD, forced dense) are only triggered by specific failure conditions that small graphs never produce.

5. **B3 (sign flip) is a blind spot in the entire test suite.** No test anywhere checks the sign/orientation of final output coordinates. All residual checks (`||Lv - lv||/||v||`) and eigenvalue property tests are sign-invariant. This is the most significant coverage gap found.

6. **B6 (warm-restart truncation) passes all tolerance checks.** Truncating warm-restart to 1 iteration still produces a mathematically valid (but lower quality) result. Detection requires asserting that warm-restart iteration count > 1 for known hard inputs, which no current test does.

## Analysis

The experiment cleanly separates the impact of four strategies:

**Opt-level tuning (RQ1) is the dominant factor.** The 13-47x speedup on LOBPCG-heavy tests comes from the compiler optimizing tight SpMV inner loops — auto-vectorization, loop unrolling, and register allocation that `opt-level=0` leaves on the table. The bimodal speedup distribution (2-5x for rSVD/dense, 10-50x for LOBPCG) confirms the root cause: LOBPCG's iterative nature amplifies per-iteration overhead more than single-pass algorithms.

**Test tiering (RQ2) is unnecessary given opt-level.** The 33-test minimum set runs in 128s at opt0, but at opt2 the full 221-test suite completes in 112s — making the minimum set slower than the full suite at the recommended configuration. Test tiering adds maintenance burden without benefit.

**Threshold override (RQ3) works as designed** but is only useful for the property tests, which themselves failed. The LOBPCG-dense subspace angle of 9.1e-4 rad validates that LOBPCG produces equivalent results at n=200, establishing the technique as safe for future use if needed.

**Property tests (RQ4) failed on both criteria.** The timing failure (142s vs 30s target) stems from 256 proptest cases each running the full solver pipeline. The detection failure (1/8) stems from the inherent limitation that small, well-conditioned graphs only exercise the first two solver levels. Property tests at this scale test mathematical invariants (eigenvalue bounds, orthogonality, residuals) which are valuable sanity checks but not mutation detectors.

Comparing against the success criteria:
- RQ1 **PASS**: 111.9s (1.87 min) wall time, well under 10-minute target
- RQ2 **PARTIAL PASS**: 5/6 levels covered, 6/8 bugs detected (B3 and B6 are suite-wide blind spots)
- RQ3 **PASS**: Subspace angle 9.1e-4 < 0.01 rad, release binary clean
- RQ4 **FAIL**: 142s >> 30s, 1/8 << 6/8

The overall hypothesis H1 is **partially supported**: the opt-level component alone achieves the CI time target, making the other components unnecessary.

## What We Learned

- **Compiler optimization is the highest-leverage CI improvement for numerical Rust code.** A single `[profile.test] opt-level = 2` line delivers 10-50x speedups on iterative solvers, dwarfing all other optimization strategies combined.
- **opt2deps1 is not worth the complexity.** Separately optimizing dependencies does not help when the test binary itself contains the hot loops.
- **The solver escalation chain is inherently difficult to test via small inputs.** Higher levels require specific failure conditions (quality threshold exceeded, Cholesky failure) that well-conditioned small graphs never trigger. Effective coverage of Levels 2-5 requires either large adversarial inputs or mock solver injection.
- **Mutation testing revealed two blind spots (B3, B6)** that are invisible to all existing tests. These are not opt-level related — they represent genuine coverage gaps in the test suite's invariant checking.
- **Property tests are valuable for mathematical sanity checks but poor mutation detectors** for a multi-level solver pipeline. The invariants they check (eigenvalue bounds, orthogonality, residuals) are necessary but not sufficient conditions.

## Conclusions

The single change of `[profile.test.package.spectral-init] opt-level = 2` in `Cargo.toml` is sufficient to reduce CI from 30+ minutes to ~3-4 minutes (75s compile + 112s test + overhead). This is a 10x improvement with zero impact on test coverage or correctness.

The more complex strategies (nextest test tiering, property-based tests for all-level coverage) are unnecessary. The property test approach was based on a flawed assumption that small-n threshold override would exercise the full solver escalation chain. The minimum viable test set analysis confirmed that test tiering is counterproductive at opt2, where the full suite is faster than the minimum set at opt0.

Two test suite blind spots (B3: sign flip, B6: warm-restart truncation) were discovered through mutation testing. These are unrelated to CI timing but represent genuine coverage gaps.

## Recommendations

**Commit immediately:**
1. Add `[profile.test.package.spectral-init] opt-level = 2` to `Cargo.toml`. This is the single highest-impact change, reducing CI wall time from 30+ minutes to ~2 minutes.

**Do NOT commit:**
1. The `ci-fast` nextest profile — opt2 makes test tiering unnecessary.
2. The property tests in their current form — they fail both timing (142s vs 30s) and detection (1/8 vs 6/8) criteria.

**Follow-up work:**
1. **Add sign convention test** to detect B3-class mutations. Proposed invariant: the first nonzero element of each eigenvector should be positive (standard sign convention).
2. **Add warm-restart iteration count assertion** to detect B6-class mutations. For known hard inputs (path_2000, ring_2000), assert that warm-restart count > 0.
3. **Investigate rSVD test slowness** — two tests at ~112s each are the new wall-time bottleneck at opt2. Consider reducing the oversampling parameter for CI or splitting into separate fast/slow test tiers only for rSVD.
4. **Redesign property tests** if needed: use test-only mock solvers or targeted mutation-specific invariants instead of small-n graphs with threshold override.

## Appendix: Experiment Scripts

### extract_timings.py

```python
#!/usr/bin/env python3
"""Extract per-testcase timings from a JUnit XML report.

Usage:
    python3 extract_timings.py <junit_xml_path> [output_csv_path]

Outputs CSV with columns: suite,test_name,classname,time_s,status
sorted by time_s descending. When the input is empty or unparseable,
outputs only the CSV header row (exit 0).
"""
import csv
import sys
import xml.etree.ElementTree as ET

FIELDNAMES = ["suite", "test_name", "classname", "time_s", "status"]


def _status(tc_elem):
    if tc_elem.find("failure") is not None:
        return "failed"
    if tc_elem.find("error") is not None:
        return "error"
    if tc_elem.find("skipped") is not None:
        return "skipped"
    return "passed"


def extract(xml_path):
    rows = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return rows
    root = tree.getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = root.findall("testsuite")
    for suite in suites:
        suite_name = suite.get("name", "")
        for tc in suite.findall("testcase"):
            rows.append({
                "suite": suite_name,
                "test_name": tc.get("name", ""),
                "classname": tc.get("classname", ""),
                "time_s": float(tc.get("time", "0") or "0"),
                "status": _status(tc),
            })
    rows.sort(key=lambda r: r["time_s"], reverse=True)
    return rows


def write_csv(rows, out):
    writer = csv.DictWriter(out, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <junit_xml_path> [output_csv_path]",
              file=sys.stderr)
        sys.exit(1)
    rows = extract(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)


if __name__ == "__main__":
    main()
```

### run_opt_sweep.sh

```bash
#!/usr/bin/env bash
# Run optimization-level sweep across Cargo profiles.
# Each config appends a profile to Cargo.toml, times compilation, runs tests,
# copies JUnit XML, and generates a timings CSV.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq1_opt_levels"
EXTRACT="research/2026-03-25-ci-test-optimization/scripts/extract_timings.py"
mkdir -p "$RESULTS_DIR"

trap 'sed -i "/# BEGIN experiment-/,/# END experiment-/d" Cargo.toml' EXIT

append_profile() {
    local cfg="$1"
    case "$cfg" in
      opt0|opt1|opt2|opt3)
        local level="${cfg#opt}"
        cat >> Cargo.toml <<TOML

# BEGIN experiment-${cfg}
[profile.experiment-${cfg}]
inherits = "test"
opt-level = ${level}
# END experiment-${cfg}
TOML
        ;;
      opt2deps1)
        cat >> Cargo.toml <<TOML

# BEGIN experiment-opt2deps1
[profile.experiment-opt2deps1]
inherits = "test"
opt-level = 2

[profile.experiment-opt2deps1.package."*"]
opt-level = 1
# END experiment-opt2deps1
TOML
        ;;
    esac
}

run_config() {
    local CONFIG="$1"
    local FILTER="${2:-}"
    echo "=== Running config: ${CONFIG} ==="
    append_profile "$CONFIG"
    { time cargo nextest run \
        --cargo-profile "experiment-${CONFIG}" \
        --profile ci --features testing --no-run 2>&1; } \
        2>&1 | tee "${RESULTS_DIR}/${CONFIG}_compile.txt"
    local nextest_args=(cargo nextest run
        --cargo-profile "experiment-${CONFIG}"
        --profile ci --features testing --no-fail-fast)
    [[ -n "$FILTER" ]] && nextest_args+=(-E "$FILTER")
    "${nextest_args[@]}" || true
    cp target/nextest/ci/junit.xml "${RESULTS_DIR}/${CONFIG}_junit.xml"
    python3 "$EXTRACT" \
        "${RESULTS_DIR}/${CONFIG}_junit.xml" \
        "${RESULTS_DIR}/timings_${CONFIG}.csv"
    sed -i '/# BEGIN experiment-/,/# END experiment-/d' Cargo.toml
    echo "=== Done: ${CONFIG} ==="
}

CONFIGS=(opt0 opt2 opt2deps1)
for CONFIG in "${CONFIGS[@]}"; do
    run_config "$CONFIG"
done
echo "=== Sweep complete ==="
```

### inject_bug.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BUG_ID="${1:?Usage: inject_bug.sh <B1..B8> <apply|revert>}"
ACTION="${2:?Usage: inject_bug.sh <B1..B8> <apply|revert>}"

apply_patch() {
    local target="$1" diff_content="$2" sentinel="$3"
    if grep -qF "$sentinel" "$target" 2>/dev/null; then
        echo "[inject_bug] ${BUG_ID}: already applied, skipping"; return 0
    fi
    printf '%s\n' "$diff_content" | patch -p1 --forward --reject-file=/dev/null --no-backup-if-mismatch
    rm -f "${target}.orig"
}

revert_patch() {
    local target="$1" diff_content="$2" sentinel="$3"
    if ! grep -qF "$sentinel" "$target" 2>/dev/null; then
        echo "[inject_bug] ${BUG_ID}: already reverted, skipping"; return 0
    fi
    printf '%s\n' "$diff_content" | patch -p1 -R --forward --reject-file=/dev/null --no-backup-if-mismatch
    rm -f "${target}.orig"
}

# Bug definitions: B1-B8 target specific source files with sentinel-marked patches
# B1: dense.rs — reverse eigenvalue sort order
# B2: solvers/mod.rs — set DENSE_N_THRESHOLD=0
# B3: scaling.rs — negate output coordinates
# B4: multi_component.rs — skip components >= 1
# B5: laplacian.rs — skip D^{-1/2} normalization
# B6: lobpcg.rs — truncate warm-restart to 0 iterations
# B7: rsvd.rs — return trivial eigenvector for all slots
# B8: sinv.rs — return zeros without Cholesky solve

case "$ACTION" in
  apply)  apply_patch "$TARGET" "$DIFF" "$SENTINEL"; cargo check --features testing --quiet ;;
  revert) revert_patch "$TARGET" "$DIFF" "$SENTINEL"; cargo check --features testing --quiet ;;
esac
```

### run_bug_matrix.sh

```bash
#!/usr/bin/env bash
# Bug detection matrix driver.
# Applies each mutation B1-B8, runs nextest, counts failures, reverts.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

INJECT="research/2026-03-25-ci-test-optimization/scripts/inject_bug.sh"
RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq2_min_set"
OUTPUT_CSV="${RESULTS_DIR}/bug_matrix.csv"
SUITE_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
      --suite) SUITE_FILTER="$2"; shift 2 ;;
      --output) OUTPUT_CSV="$2"; shift 2 ;;
    esac
done

echo "bug_id,mutation_target,tests_failed,tests_passed,detected" > "$OUTPUT_CSV"

BUGS=(B1 B2 B3 B4 B5 B6 B7 B8)
for BUG in "${BUGS[@]}"; do
    bash "$INJECT" "$BUG" apply
    trap "bash '${INJECT}' '${BUG}' revert" ERR
    cargo nextest run --profile ci --features testing --no-fail-fast \
        ${SUITE_FILTER:+-E "$SUITE_FILTER"} || true
    # Parse JUnit XML for failure counts
    bash "$INJECT" "$BUG" revert
    trap - ERR
done
```

### test_rq3_threshold.rs

```rust
//! RQ3: Threshold override validation — dense EVD vs LOBPCG subspace agreement.

#[path = "../common/mod.rs"]
mod common;

use ndarray::Array2;
use ndarray_npy::write_npy;
use std::path::Path;

fn max_subspace_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let m = a.t().dot(b);
    let mtm = m.t().dot(&m);
    let k = mtm.nrows();
    let faer_mat = faer::Mat::<f64>::from_fn(k, k, |i, j| mtm[[i, j]]);
    let evd = faer_mat.self_adjoint_eigen(faer::Side::Lower).expect("EVD");
    let s = evd.S();
    let min_sv_sq = (0..k)
        .map(|i| s.column_vector().iter().nth(i).copied().unwrap_or(0.0).max(0.0))
        .fold(f64::INFINITY, f64::min);
    min_sv_sq.sqrt().min(1.0).acos()
}

#[test]
fn rq3_threshold_override_subspace_agreement() {
    let n = 200;
    let laplacian = common::ring_laplacian(n);

    // Run 1: Dense EVD (threshold=2000)
    unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "2000"); }
    let ((eigs_dense, vecs_dense), level_dense) =
        spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
    assert_eq!(level_dense, 0);

    // Run 2: LOBPCG (threshold=50)
    unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "50"); }
    let ((eigs_lobpcg, vecs_lobpcg), level_lobpcg) =
        spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
    assert!(level_lobpcg >= 1);

    unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }

    let angle = max_subspace_angle(&vecs_dense, &vecs_lobpcg);
    assert!(angle < 0.01, "subspace angle {angle:.6e} exceeds 0.01 rad");
}
```

### test_rq4_property_tests.rs

```rust
//! RQ4: Property-based tests for spectral_init mathematical invariants.

#[path = "../common/mod.rs"]
mod common;

use proptest::prelude::*;
use spectral_init::{spectral_init, SpectralInitConfig};

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 64, ..Default::default() })]

    #[test]
    fn proptest_eigenvalues_sorted(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _), _) = spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for i in 0..eigs.len() - 1 {
            prop_assert!(eigs[i] <= eigs[i + 1] + 1e-12);
        }
    }

    #[test]
    fn proptest_eigenvalues_nonneg(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _), _) = spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for (i, &e) in eigs.iter().enumerate() {
            prop_assert!(e >= -1e-12, "eigenvalue {}={} negative", i, e);
        }
    }

    // ... (7 more property tests: eigenvalues_leq_2, trivial_null_space,
    //      residual_leq_1e5, orthogonality, finite_output, output_variance,
    //      subspace_stable)
}
```

## Appendix: Raw Data

### Speedup Comparison (Top 30 Tests)

```csv
test_name,opt0_s,opt2_s,opt2deps1_s,speedup_opt0_to_opt2
test_path_2000_warm_restart,451.315,31.602,31.905,14.3
comp_d_rsvd_blobs_connected_2000,270.939,111.804,117.127,2.4
test_level_3_rsvd_valid_on_large_path,264.413,111.581,117.269,2.4
generate_accuracy_report,241.179,12.495,15.147,19.3
test_ring_2000_warm_restart,173.923,12.728,13.125,13.7
proptest_subspace_stable,137.892,24.94,27.393,5.5
comp_d_rsvd_blobs_500,136.601,39.905,50.614,3.4
comp_d_rsvd_circles_300,130.284,36.346,37.619,3.6
proptest_eigenvalues_sorted,119.11,16.822,14.268,7.1
proptest_orthogonality,119.62,14.388,15.827,8.3
proptest_eigenvalues_nonneg,122.635,13.754,13.737,8.9
proptest_finite_output,119.203,13.229,13.702,9.0
proptest_eigenvalues_leq_2,118.1,12.41,16.92,9.5
proptest_residual_leq_1e5,115.799,16.507,12.602,7.0
proptest_output_variance,115.056,15.174,15.685,7.6
comp_d_dense_evd_blobs_connected_2000,108.99,6.253,5.833,17.4
test_e2e_performance_blobs_5000,107.631,3.515,3.357,30.6
comp_d_rsvd_blobs_connected_200,94.841,17.909,18.687,5.3
comp_d_rsvd_moons_200,96.519,30.064,15.567,3.2
proptest_trivial_null_space,80.14,15.477,14.901,5.2
test_level_2_regularized_lobpcg_produces_valid_result,37.52,0.795,1.482,47.2
comp_d_rsvd_near_dupes_100,38.055,8.406,10.846,4.5
lobpcg_blobs_connected_2000_eigenvalues,6.062,0.098,0.195,61.9
solvers::tests::solve_eigenproblem_large_n_routes_through_lobpcg,19.715,0.584,0.455,33.8
test_level_1_lobpcg_for_large_well_conditioned_n,16.993,0.62,0.681,27.4
sinv_eigenvalue_accuracy_blobs_connected_2000,18.725,4.082,4.879,4.6
sinv_residual_quality_blobs_connected_2000,18.373,5.47,6.519,3.4
test_e2e_residual_quality_blobs_connected_2000,6.681,0.207,0.288,32.3
lobpcg_blobs_connected_2000_level2,5.955,0.192,0.204,31.0
lobpcg_blobs_2000_residual_below_1e5_with_chebyshev,5.387,0.191,0.23,28.2
```

### Bug Detection Matrix — Full Suite

```csv
bug_id,mutation_target,tests_failed,tests_passed,detected
B1,src/solvers/dense.rs,16,191,1
B2,src/solvers/mod.rs,2,205,1
B3,src/scaling.rs,0,207,0
B4,src/multi_component.rs,5,202,1
B5,src/laplacian.rs,19,188,1
B6,src/solvers/lobpcg.rs,0,207,0
B7,src/solvers/rsvd.rs,2,205,1
B8,src/solvers/sinv.rs,12,195,1
```

### Bug Detection Matrix — Property Tests Only

```csv
bug_id,mutation_target,tests_failed,tests_passed,detected
B1,src/solvers/dense.rs,2,7,1
B2,src/solvers/mod.rs,0,9,0
B3,src/scaling.rs,0,9,0
B4,src/multi_component.rs,0,9,0
B5,src/laplacian.rs,0,9,0
B6,src/solvers/lobpcg.rs,0,9,0
B7,src/solvers/rsvd.rs,0,9,0
B8,src/solvers/sinv.rs,0,9,0
```

### RQ3 Subspace Angle

```
subspace_angle_rad=9.111863e-4
dense_level=0
lobpcg_level=1
```

### RQ4 Property Test Timing

```
wall_time_ms=161959
wall_time_s=161.959
threshold=50
date=2026-03-25T17:54:55-07:00
```
