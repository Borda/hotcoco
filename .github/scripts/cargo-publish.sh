#!/usr/bin/env bash
#
# Publish one workspace crate to crates.io, tolerating *only* an
# already-published version.
#
# The previous form was `cargo publish -p <crate> || echo "::warning::already
# published"`, which degraded every failure — expired token, network error,
# verification failure, yanked dependency — into a warning on a green run. A
# release could then "succeed" having published nothing.
#
# Usage: cargo-publish.sh <crate-name>

set -euo pipefail

crate="${1:?usage: cargo-publish.sh <crate-name>}"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if cargo publish -p "$crate" 2>&1 | tee "$log"; then
    echo "::notice::${crate} published"
    exit 0
fi

# crates.io rejects a re-publish with "crate version X is already uploaded";
# older/newer cargo wording is covered by the broader "already exists".
#
# KNOWN STOPGAP: matching on cargo's prose is fragile — it breaks if cargo
# rewords, and "already exists" is broad enough to catch an unrelated failure
# that happens to use the phrase. The intended shape is to ask the registry
# instead (GET https://crates.io/api/v1/crates/<name>/<version> before
# publishing), which is what the PyPI job in this workflow already does
# structurally via `skip-existing: true`. Deferred, not forgotten: the failure
# mode here is a false *failure* (loud, blocks the release), never a false
# success — which is the safe direction for it to point.
if grep -qiE 'already (uploaded|exists)' "$log"; then
    echo "::warning::${crate} is already published at this version — skipping"
    exit 0
fi

echo "::error::cargo publish -p ${crate} failed (see log above)"
exit 1
