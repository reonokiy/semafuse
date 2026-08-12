#!/usr/bin/env bash
set -euo pipefail

version="$(cargo metadata --locked --no-deps --format-version 1 \
  | jq -er '[.packages[] | select(.name == "semafuse")] |
    if length == 1 then .[0].version
    else error("expected exactly one Semafuse package") end')"
tag="v$version"

for required_file in \
  .github/release-targets.json Cargo.lock CHANGELOG.md LICENSE \
  THIRD_PARTY_NOTICES.html semafuse-daemon.example.toml
do
  if [[ ! -s "$required_file" ]]; then
    echo "Required release file is missing or empty: $required_file" >&2
    exit 1
  fi
done

if ! grep -Fqx "## $version" CHANGELOG.md; then
  echo "CHANGELOG.md has no section for version $version." >&2
  exit 1
fi

expected_matrix='{
  "include": [
    {"runner":"ubuntu-24.04","target":"x86_64-unknown-linux-gnu","archive":"tar.gz"},
    {"runner":"ubuntu-24.04-arm","target":"aarch64-unknown-linux-gnu","archive":"tar.gz"},
    {"runner":"macos-15-intel","target":"x86_64-apple-darwin","archive":"tar.gz"},
    {"runner":"macos-15","target":"aarch64-apple-darwin","archive":"tar.gz"},
    {"runner":"windows-2025","target":"x86_64-pc-windows-msvc","archive":"zip"}
  ]
}'
matrix="$(jq -cer --argjson expected "$expected_matrix" '
  if . == $expected then .
  else error("release target policy differs from the audited allowlist")
  end
' .github/release-targets.json)"
assets="$(jq -cn --arg tag "$tag" --argjson matrix "$matrix" '
  $matrix.include | map("semafuse-\($tag)-\(.target).\(.archive)")
')"

jq -cn \
  --arg version "$version" \
  --arg tag "$tag" \
  --argjson assets "$assets" \
  '{version: $version, tag: $tag, assets: $assets}'
