#!/usr/bin/env bash
# Phase 1 Change Verification Script
# Verifies all Phase 1 changes are correctly applied before running experiments.
# Reports PASS/FAIL for each check independently; exits 1 if any check fails.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" &>/dev/null; then
        echo "PASS: $label"
        ((PASS++)) || true
    else
        echo "FAIL: $label"
        ((FAIL++)) || true
    fi
}

echo "=== Phase 1 Change Verification ==="

# C1: rust-toolchain.toml exists
check "rust-toolchain.toml exists" test -f "$REPO_ROOT/rust-toolchain.toml"

# C2: rust-toolchain.toml contains nightly-2026-03-26
check "rust-toolchain.toml channel = nightly-2026-03-26" \
    grep -q "nightly-2026-03-26" "$REPO_ROOT/rust-toolchain.toml"

# C3: Cargo.toml contains profiling = []
check "Cargo.toml has profiling feature" \
    grep -q 'profiling = \[\]' "$REPO_ROOT/Cargo.toml"

# C4: tw_baseline_bench.rs exists
check "benches/tw_baseline_bench.rs exists" \
    test -f "$REPO_ROOT/benches/tw_baseline_bench.rs"

# C5: trustworthiness_bench.rs is absent
check "benches/trustworthiness_bench.rs absent" \
    bash -c "! test -f '$REPO_ROOT/benches/trustworthiness_bench.rs'"

# C6: cargo check --features cli,profiling
echo ""
echo "Running: cargo check --features cli,profiling"
if (cd "$REPO_ROOT" && cargo check --features cli,profiling 2>&1); then
    echo "PASS: cargo check --features cli,profiling"
    ((PASS++)) || true
else
    echo "FAIL: cargo check --features cli,profiling"
    ((FAIL++)) || true
fi

# C7: cargo check --features testing
echo ""
echo "Running: cargo check --features testing"
if (cd "$REPO_ROOT" && cargo check --features testing 2>&1); then
    echo "PASS: cargo check --features testing"
    ((PASS++)) || true
else
    echo "FAIL: cargo check --features testing"
    ((FAIL++)) || true
fi

# C8: cargo test --features testing --test test_trustworthiness
echo ""
echo "Running: cargo test --features testing --test test_trustworthiness"
if (cd "$REPO_ROOT" && cargo test --features testing --test test_trustworthiness 2>&1); then
    echo "PASS: cargo test --features testing --test test_trustworthiness"
    ((PASS++)) || true
else
    echo "FAIL: cargo test --features testing --test test_trustworthiness"
    ((FAIL++)) || true
fi

echo ""
echo "=== Results: $PASS PASS, $FAIL FAIL ==="
[[ $FAIL -eq 0 ]]
