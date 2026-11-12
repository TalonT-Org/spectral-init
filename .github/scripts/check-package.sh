#!/usr/bin/env bash
# Package gate for the published crate.
#
# Checks that
#   1. `cargo package --list` equals the committed allow-list,
#   2. the .crate file is smaller than 1 MiB,
#   3. the unpacked .crate passes its own lib and doc tests and runs every example, and
#   4. `cargo publish --dry-run` succeeds.
#
# Usage: .github/scripts/check-package.sh [extra flags for cargo package/publish]
#   e.g. .github/scripts/check-package.sh --allow-dirty
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="spectral-init"
allowlist=".github/package-allowlist.txt"
max_bytes=$((1024 * 1024))

echo "==> package file list equals ${allowlist}"
diff -u \
    <(LC_ALL=C sort "$allowlist") \
    <(cargo package -p "$package" --locked --list "$@" | LC_ALL=C sort)

echo "==> build the .crate"
cargo package -p "$package" --locked --no-verify "$@"

pkgid="$(cargo pkgid -p "$package")"
version="${pkgid##*#}"
version="${version##*@}"
target_dir="$(cargo metadata --format-version 1 --no-deps --locked \
    | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
crate_file="${target_dir}/package/${package}-${version}.crate"

size="$(wc -c < "$crate_file")"
echo "    ${crate_file}: ${size} bytes"
if [ "$size" -ge "$max_bytes" ]; then
    echo "error: .crate is ${size} bytes; the limit is ${max_bytes}" >&2
    exit 1
fi

echo "==> the unpacked .crate passes its own tests and runs its examples"
# Unpack outside the repository: inside it, Cargo's workspace discovery would
# attach the unpacked crate to the root workspace.
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
tar -xzf "$crate_file" -C "$workdir"
(
    cd "${workdir}/${package}-${version}"
    cargo test --locked --lib
    cargo test --locked --doc
    # `cargo test --examples` only builds examples; running them executes their checks.
    for example in examples/*.rs; do
        cargo run --locked --example "$(basename "$example" .rs)"
    done
)

echo "==> publish dry run"
cargo publish -p "$package" --dry-run --locked "$@"

echo "package gate passed"
