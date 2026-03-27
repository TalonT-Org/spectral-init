#!/usr/bin/env bash
# run_rq2_mutations.sh — RQ2 mutation detection sweep.
# Applies each B1–B8 mutation, runs test_exp_solver_invariants, records detection.
# Output: research/2026-03-26-rsvd-oversampling-coverage/results/rq2_mutation_matrix.csv
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
INJECT="$REPO_ROOT/research/2026-03-25-ci-test-optimization/scripts/inject_bug.sh"
RESULTS="$REPO_ROOT/research/2026-03-26-rsvd-oversampling-coverage/results"

mkdir -p "$RESULTS"
printf 'bug_id,test_name,detected,suite_wall_time_s\n' > "$RESULTS/rq2_mutation_matrix.csv"

for BUG in B1 B2 B3 B4 B5 B6 B7 B8; do
    bash "$INJECT" "$BUG" apply

    START_NS=$(date +%s%N)
    if cargo nextest run \
        --manifest-path "$REPO_ROOT/Cargo.toml" \
        --features testing \
        --test test_exp_solver_invariants \
        2>/dev/null; then
        DETECTED=false
    else
        DETECTED=true
    fi
    END_NS=$(date +%s%N)
    WALL=$(echo "scale=3; ($END_NS - $START_NS) / 1000000000" | bc)

    printf '%s,test_exp_solver_invariants,%s,%s\n' \
        "$BUG" "$DETECTED" "$WALL" \
        >> "$RESULTS/rq2_mutation_matrix.csv"

    bash "$INJECT" "$BUG" revert
done

echo "Results written to $RESULTS/rq2_mutation_matrix.csv"
