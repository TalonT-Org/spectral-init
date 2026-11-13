#!/usr/bin/env bash
# Subresource-integrity gate for the KaTeX header that rustdoc embeds.
#
# Downloads every resource docs/katex-header.html loads with an `integrity`
# attribute, recomputes its sha384 digest and compares it with the attribute.
# A wrong digest makes the browser refuse the resource, and the formulas on
# docs.rs stay unrendered. A resource loaded without an `integrity` attribute, or
# not over https, fails the gate too: rustdoc puts the header into every docs.rs page.
#
# Usage: .github/scripts/check-sri.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

header="docs/katex-header.html"
status=0
checked=0

while IFS= read -r line; do
    if ! [[ "$line" =~ integrity=\"([^\"]+)\" ]]; then
        if [[ "$line" =~ (href|src)=\" ]]; then
            echo "error: resource without an integrity attribute: ${line}" >&2
            status=1
        fi
        continue
    fi
    expected="${BASH_REMATCH[1]}"
    if ! [[ "$line" =~ (href|src)=\"([^\"]+)\" ]]; then
        echo "error: integrity attribute without a URL: ${line}" >&2
        exit 2
    fi
    url="${BASH_REMATCH[2]}"
    if [[ "$url" != https://* ]]; then
        echo "error: resource not loaded over https: ${url}" >&2
        status=1
        continue
    fi

    digest="$(curl --fail --silent --show-error --location "$url" \
        | openssl dgst -sha384 -binary | openssl base64 -A)"
    actual="sha384-${digest}"
    checked=$((checked + 1))
    if [ "$actual" != "$expected" ]; then
        echo "error: integrity mismatch for ${url}" >&2
        echo "    in ${header}: ${expected}" >&2
        echo "    computed:     ${actual}" >&2
        status=1
    fi
done < "$header"

if [ "$checked" -eq 0 ]; then
    echo "error: no resource in ${header} was verified against an integrity attribute" >&2
    exit 2
fi
if [ "$status" -eq 0 ]; then
    echo "SRI gate passed (${checked} resources)"
fi
exit "$status"
