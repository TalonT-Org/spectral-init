#!/usr/bin/env bash
# Run optimization-level sweep across 5 Cargo profiles.
# Each config appends a profile to Cargo.toml, times compilation, runs tests,
# copies JUnit XML, and generates a timings CSV.
#
# Usage:
#   ./run_opt_sweep.sh                      # all 5 configs
#   ./run_opt_sweep.sh --dry-run opt0       # single config, one fast test
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq1_opt_levels"
EXTRACT="research/2026-03-25-ci-test-optimization/scripts/extract_timings.py"
mkdir -p "$RESULTS_DIR"

# Safety: always remove any leftover experiment profile on exit
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
      *)
        echo "Unknown config: $cfg" >&2; exit 1 ;;
    esac
}

run_config() {
    local CONFIG="$1"
    local FILTER="${2:-}"   # optional nextest -E expression

    echo "=== Running config: ${CONFIG} ==="

    # 1. Append profile section
    append_profile "$CONFIG"

    # 2. Compile-only timing
    { time cargo nextest run \
        --cargo-profile "experiment-${CONFIG}" \
        --profile ci \
        --features testing \
        --no-run 2>&1; } 2>&1 | tee "${RESULTS_DIR}/${CONFIG}_compile.txt"

    # 3. Full (or filtered) test run; failures are non-fatal
    local nextest_args=(cargo nextest run
        --cargo-profile "experiment-${CONFIG}"
        --profile ci
        --features testing
        --no-fail-fast)
    if [[ -n "$FILTER" ]]; then
        nextest_args+=(-E "$FILTER")
    fi
    "${nextest_args[@]}" || true

    # 4. Copy JUnit XML
    cp target/nextest/ci/junit.xml "${RESULTS_DIR}/${CONFIG}_junit.xml"

    # 5. Extract timings CSV
    python3 "$EXTRACT" \
        "${RESULTS_DIR}/${CONFIG}_junit.xml" \
        "${RESULTS_DIR}/timings_${CONFIG}.csv"

    # 6. Remove profile section (trap also covers emergency cleanup)
    sed -i '/# BEGIN experiment-/,/# END experiment-/d' Cargo.toml

    echo "=== Done: ${CONFIG} ==="
}

# ── Argument parsing ────────────────────────────────────────────────────────────

DRY_RUN=false
DRY_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        DRY_CONFIG="${2:?--dry-run requires a config name (e.g. opt0)}"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Usage: $0 [--dry-run <config>]" >&2
        exit 1
        ;;
    esac
done

# ── Run ────────────────────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
    # Dry-run: one config, one fast unit test (validates pipeline without fixture cost)
    DRY_FILTER="test(dense_evd_trivial_2x2)"
    echo "=== Dry-run: config=${DRY_CONFIG}, filter=${DRY_FILTER} ==="
    run_config "$DRY_CONFIG" "$DRY_FILTER"
    echo "=== Dry-run complete ==="
else
    CONFIGS=(opt0 opt1 opt2 opt3 opt2deps1)
    for CONFIG in "${CONFIGS[@]}"; do
        run_config "$CONFIG"
    done
    echo "=== Sweep complete — results in ${RESULTS_DIR}/ ==="
fi
