#!/usr/bin/env bash
# Purity gate.
#
# The published library stays silent and independent of its environment, and Cargo
# features that belong to other workspace packages stay out of it.
#
# Usage: .github/scripts/check-purity.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

status=0

# forbid <description> <extended regex> <path>...
# Fails the gate when the pattern occurs in any file under the given paths whose name
# matches the glob in the variable `include` (default: Rust sources, '*.rs'). Files named
# in the space-separated variable `allowed` are exempt. Both variables are set per call.
forbid() {
    local description="$1" pattern="$2"
    shift 2
    local path hits file
    for path in "$@" ${allowed:-}; do
        if [ ! -e "$path" ]; then
            echo "error: ${path} does not exist" >&2
            exit 2
        fi
    done
    hits="$(grep -rnHE --include="${include:-*.rs}" -e "$pattern" "$@" || true)"
    for file in ${allowed:-}; do
        hits="$(grep -v "^${file}:" <<< "$hits" || true)"
    done
    if [ -n "$hits" ]; then
        echo "error: ${description}" >&2
        echo "$hits" >&2
        status=1
    fi
}

forbid "the library must not print or read the environment (src/)" \
    'println!|eprintln!|dbg!|env::var|SPECTRAL_' \
    src

forbid "random numbers come from the named generator ChaCha12Rng, not StdRng (src/)" \
    'StdRng' \
    src

allowed="src/solvers/nd16_bridge.rs" \
forbid "the library reaches ndarray 0.16 and linfa-linalg only through src/solvers/nd16_bridge.rs (src/)" \
    'ndarray16|linfa_linalg' \
    src

forbid "no package has a 'testing' feature" \
    'feature *= *"testing"' \
    src tests examples eval trustworthiness

forbid "the 'cli' and 'profiling' features belong to the trustworthiness package" \
    'feature *= *"(cli|profiling)"' \
    src tests examples

include='Cargo.toml' \
forbid "no package declares a 'testing' feature" \
    '^testing[[:space:]]*=' \
    Cargo.toml eval trustworthiness

include='Cargo.toml' \
forbid "only the trustworthiness package declares the 'cli' and 'profiling' features" \
    '^(cli|profiling)[[:space:]]*=' \
    Cargo.toml eval

if [ "$status" -eq 0 ]; then
    echo "purity gate passed"
fi
exit "$status"
