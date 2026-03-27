#!/usr/bin/env bash
# run_dry_run_verify.sh — Pre-flight verification for the rSVD oversampling experiment.
# Runs three checks; exits with code 1 if any fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
INJECT="$REPO_ROOT/research/2026-03-25-ci-test-optimization/scripts/inject_bug.sh"
OVERALL=0

# ---------------------------------------------------------------------------
# Check 1: Env-var seam — SPECTRAL_RSVD_OVERSAMPLING is read by production_sweep
# ---------------------------------------------------------------------------
echo "=== Check 1: Env-var seam ==="
if OUTPUT=$(SPECTRAL_RSVD_OVERSAMPLING=10 \
        cargo nextest run \
            --features testing \
            --test test_exp_rsvd_sweep \
            production_sweep \
            --test-threads=1 \
            --no-capture 2>&1); then
    if echo "$OUTPUT" | grep -q 'SWEEP_ROW'; then
        echo "[PASS] SWEEP_ROW lines present; test PASSED"
    else
        echo "[FAIL] Test passed but no SWEEP_ROW lines found in output"
        OVERALL=1
    fi
else
    echo "[FAIL] production_sweep test FAILED"
    echo "$OUTPUT" | grep -E 'SWEEP_ROW|PASSED|FAILED|error' || true
    OVERALL=1
fi

# ---------------------------------------------------------------------------
# Check 2: Single invariant test passes on clean source
# ---------------------------------------------------------------------------
echo ""
echo "=== Check 2: Single invariant test ==="
if cargo nextest run \
        --features testing \
        --test test_exp_solver_invariants \
        test_inv_eigenvalue_ascending_order 2>&1; then
    echo "[PASS] test_inv_eigenvalue_ascending_order passed on clean source"
else
    echo "[FAIL] test_inv_eigenvalue_ascending_order failed on clean source"
    OVERALL=1
fi

# ---------------------------------------------------------------------------
# Check 3: inject_bug cycle — B1 must be DETECTED, source reverted afterward
# ---------------------------------------------------------------------------
echo ""
echo "=== Check 3: inject_bug cycle (B1) ==="
bash "$INJECT" B1 apply

if cargo nextest run \
        --features testing \
        --test test_exp_solver_invariants \
        test_inv_eigenvalue_ascending_order 2>/dev/null; then
    DETECT_STATUS="NOT DETECTED"
else
    DETECT_STATUS="DETECTED"
fi

bash "$INJECT" B1 revert

if [ "$DETECT_STATUS" = "DETECTED" ]; then
    echo "[PASS] B1: DETECTED (test failed under mutation as expected)"
else
    echo "[FAIL] B1: NOT DETECTED (invariant test did not catch the mutation)"
    OVERALL=1
fi

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
echo ""
if [ "$OVERALL" -ne 0 ]; then
    echo "[ABORT] One or more checks failed. Fix before running full sweeps."
    exit 1
fi
echo "[ALL CHECKS PASSED] Ready to run full experiment sweeps."
