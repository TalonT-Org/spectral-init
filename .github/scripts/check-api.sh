#!/usr/bin/env bash
# Public API gate for the published crate.
#
# The default-feature public API must equal the committed snapshot. Blanket
# implementations are left out of the listing: they follow from the dependency
# graph, not from this crate.
#
# Needs a nightly toolchain and cargo-public-api. The listing depends on both
# versions; CI pins them.
#
# Usage: .github/scripts/check-api.sh [--update]
#   --update  rewrite the snapshot instead of comparing against it
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="spectral-init"
snapshot=".github/api-snapshot.txt"

current="$(cargo public-api -p "$package" --simplified)"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$current" > "$snapshot"
    echo "wrote ${snapshot}"
    exit 0
fi

diff -u "$snapshot" <(printf '%s\n' "$current")
echo "public API gate passed"
