#!/usr/bin/env bash
# Build the Darwin-only SystemLanguageModel helper.
#
# Usage: scripts/build-apple-lm-sidecar.sh [arch]
#   arch defaults to host arch (arm64 / x86_64).
#
# FoundationModels.framework ships with the macOS 26+ SDK. Build against the
# selected Xcode so local verification matches the release runner contract.
set -euo pipefail

ARCH="${1:-$(uname -m)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/apple-lm-sidecar/main.swift"
OUT="$ROOT/bin/steno-apple-lm"
MODULE_CACHE="$ROOT/build/apple-lm-module-cache"

if [[ ! -f "$SRC" ]]; then
    echo "missing sidecar source: $SRC" >&2
    exit 1
fi

SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
SDK_MAJOR="${SDK_VERSION%%.*}"
if [[ ! "$SDK_MAJOR" =~ ^[0-9]+$ ]] || (( SDK_MAJOR < 26 )); then
    echo "Apple LM sidecar requires the macOS 26 SDK or newer; selected SDK is ${SDK_VERSION}" >&2
    exit 1
fi

mkdir -p "$ROOT/bin" "$MODULE_CACHE"

TMP_OUT="${OUT}.tmp.$$"
trap 'rm -f "$TMP_OUT"' EXIT

xcrun --sdk macosx swiftc \
    -O \
    -parse-as-library \
    -module-cache-path "$MODULE_CACHE" \
    -target "${ARCH}-apple-macos26.0" \
    -framework FoundationModels \
    "$SRC" \
    -o "$TMP_OUT"

test -x "$TMP_OUT"
codesign --sign - "$TMP_OUT" 2>/dev/null || true
mv -f "$TMP_OUT" "$OUT"
trap - EXIT
file "$OUT"
