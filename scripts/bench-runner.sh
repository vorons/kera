#!/usr/bin/env bash
# Runs the section 7 bench on a self-hosted Mac or Linux runner, using a finished Kera download
# (nothing is built on the owner's machines, docs/DECISIONS.md D-002).
#
#   bench-runner.sh OUT_DIR [TAG] [LABEL]
#
# The models to bench: $KERA_BENCH_MODELS (paths separated by ";"), or else the lines of
# bench_models.txt in the Kera folder. Needs $GH_TOKEN to download the release from the private repo.
set -euo pipefail
out=$1
tag=${2:-}
label=${3:-nightly}
repo=$GITHUB_REPOSITORY
api=(-fsSL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")

if [[ -n "$tag" ]]; then
    rel=$(curl "${api[@]}" "https://api.github.com/repos/$repo/releases/tags/$tag")
else
    rel=$(curl "${api[@]}" "https://api.github.com/repos/$repo/releases?per_page=1" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)[0]))')
fi
case "$(uname -s)" in
    Darwin) pattern=macos-arm64.zip ;;
    *) pattern=linux-x64.tar.gz ;;
esac
read -r rel_tag url < <(python3 -c 'import json,sys; r=json.load(sys.stdin); a=[x for x in r["assets"] if x["name"].endswith(sys.argv[1])]; print(r["tag_name"], a[0]["url"] if a else "")' "$pattern" <<<"$rel")
[[ -n "$url" ]] || { echo "Release $rel_tag has no $pattern download"; exit 1; }

work="$RUNNER_TEMP/kera-bench"
rm -rf "$work" && mkdir -p "$work"
curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" -o "$work/kera.pkg" "$url"
if [[ "$pattern" == *.zip ]]; then
    ditto -x -k "$work/kera.pkg" "$work"
    kera=$(find "$work" -path '*Kera.app/Contents/MacOS/kera' | head -1)
else
    tar -xzf "$work/kera.pkg" -C "$work"
    kera=$(find "$work" -name kera -type f | head -1)
fi
echo "Kera $rel_tag: $kera"

if [[ -n "${KERA_BENCH_MODELS:-}" ]]; then
    models=$(tr ';' '\n' <<<"$KERA_BENCH_MODELS")
else
    home=${KERA_HOME:-$([[ "$(uname -s)" == Darwin ]] && echo "$HOME/Library/Application Support/Kera" || echo "${XDG_DATA_HOME:-$HOME/.local/share}/kera")}
    models=$(cat "$home/bench_models.txt")
fi

mkdir -p "$out"
"$kera" setup --quick
while IFS= read -r m; do
    m=$(echo "$m" | xargs)
    [[ -z "$m" || "$m" == \#* ]] && continue
    name=$(basename "$m" .gguf)
    "$kera" bench --model "$m" --label "$label $rel_tag" --out "$out/$name.json"
done <<<"$models"
