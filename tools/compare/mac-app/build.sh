#!/bin/bash
# Builds "Kera vs Ollama.app" (an Apple-silicon Mac app, macOS 14 or later) with the kera program inside it, signed
# for this Mac only (ad hoc: no developer certificate).
#
#   KERA=build/bin/kera [OUT=folder] [VERSION=dev] tools/compare/mac-app/build.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
out=${OUT:-$here/build}
kera=${KERA:?set KERA to the kera program to put inside the app}
app="$out/Kera vs Ollama.app"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
xcrun --sdk macosx swiftc -O -swift-version 5 -parse-as-library -target arm64-apple-macos14.0 \
    -o "$app/Contents/MacOS/KeraVsOllama" "$here"/Sources/*.swift
sed "s/@VERSION@/${VERSION:-dev}/" "$here/Info.plist" > "$app/Contents/Info.plist"
cp "$kera" "$app/Contents/Resources/kera"
cp "$root/assets/brand/icons/kera.icns" "$app/Contents/Resources/kera.icns"
codesign --force --sign - "$app/Contents/Resources/kera"
codesign --force --sign - "$app"
echo "built $app"
