#!/usr/bin/env bash
# Bug detection matrix driver (RQ2).
# Applies each mutation B1–B8 in sequence, runs nextest, counts failures, and reverts.
# Writes a CSV with one row per mutation to results/rq2_min_set/bug_matrix.csv.
#
# Usage:
#   ./run_bug_matrix.sh                        # full test suite
#   ./run_bug_matrix.sh --suite 'test(foo)'    # filter to nextest -E expression
#   ./run_bug_matrix.sh --output path/to.csv   # override output file
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

INJECT="research/2026-03-25-ci-test-optimization/scripts/inject_bug.sh"
RESULTS_DIR="research/2026-03-25-ci-test-optimization/results/rq2_min_set"
mkdir -p "$RESULTS_DIR"

SUITE_FILTER=""
OUTPUT_CSV="${RESULTS_DIR}/bug_matrix.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
      --suite) SUITE_FILTER="$2"; shift 2 ;;
      --output) OUTPUT_CSV="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

count_from_junit() {
    python3 - "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
tree = ET.parse(sys.argv[1])
root = tree.getroot()
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
total = sum(int(s.get("tests", 0)) for s in suites)
failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
print(f"{failed},{total - failed}")
PY
}

# CSV header
echo "bug_id,mutation_target,tests_failed,tests_passed,detected" > "$OUTPUT_CSV"

BUGS=(B1 B2 B3 B4 B5 B6 B7 B8)
TARGETS=(
    "src/solvers/dense.rs"
    "src/solvers/mod.rs"
    "src/scaling.rs"
    "src/multi_component.rs"
    "src/laplacian.rs"
    "src/solvers/lobpcg.rs"
    "src/solvers/rsvd.rs"
    "src/solvers/sinv.rs"
)

for i in "${!BUGS[@]}"; do
    BUG="${BUGS[$i]}"
    TARGET="${TARGETS[$i]}"

    echo "=== ${BUG}: applying mutation to ${TARGET} ==="
    bash "$INJECT" "$BUG" apply

    # Ensure revert happens even on error
    trap "bash '${INJECT}' '${BUG}' revert" ERR

    NEXTEST_ARGS=(cargo nextest run
        --profile ci
        --features testing
        --no-fail-fast)
    [[ -n "$SUITE_FILTER" ]] && NEXTEST_ARGS+=(-E "$SUITE_FILTER")
    "${NEXTEST_ARGS[@]}" || true

    cp target/nextest/ci/junit.xml "${RESULTS_DIR}/${BUG}_junit.xml"
    COUNTS=$(count_from_junit "${RESULTS_DIR}/${BUG}_junit.xml")
    FAILED=$(echo "$COUNTS" | cut -d, -f1)
    PASSED=$(echo "$COUNTS" | cut -d, -f2)
    DETECTED=$( [[ "$FAILED" -gt 0 ]] && echo 1 || echo 0 )

    echo "${BUG},${TARGET},${FAILED},${PASSED},${DETECTED}" >> "$OUTPUT_CSV"

    trap - ERR
    bash "$INJECT" "$BUG" revert
    echo "=== ${BUG}: reverted — failed=${FAILED} passed=${PASSED} detected=${DETECTED} ==="
    echo ""
done

echo "Bug matrix complete → ${OUTPUT_CSV}"
