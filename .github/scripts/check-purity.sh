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
# Fails the gate when the pattern occurs in any Rust source under the given paths.
forbid() {
    local description="$1" pattern="$2"
    shift 2
    local path hits
    for path in "$@"; do
        if [ ! -e "$path" ]; then
            echo "error: ${path} does not exist" >&2
            exit 2
        fi
    done
    if hits="$(grep -rnE --include='*.rs' -e "$pattern" "$@")"; then
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

forbid "no package has a 'testing' feature" \
    'feature *= *"testing"' \
    src tests examples eval trustworthiness

forbid "the 'cli' and 'profiling' features belong to the trustworthiness package" \
    'feature *= *"(cli|profiling)"' \
    src tests examples

if [ "$status" -eq 0 ]; then
    echo "purity gate passed"
fi
exit "$status"
