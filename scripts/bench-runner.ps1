# Runs the section 7 bench on a self-hosted Windows runner (the owner's Minisforum), using a finished
# Kera download, as the owner asked: nothing is built on that machine (docs/DECISIONS.md D-002).
#
#   bench-runner.ps1 -Out DIR [-Tag v0.1.0] [-Label nightly]
#
# The models to bench: $env:KERA_BENCH_MODELS (paths separated by ";"), or else the lines of
# bench_models.txt in the Kera folder. Needs $env:GH_TOKEN to download the release from the private repo.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Tag = "",
    [string]$Label = "nightly"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repo = $env:GITHUB_REPOSITORY
$api = @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/vnd.github+json" }
if ($Tag) {
    $rel = Invoke-RestMethod -Headers $api "https://api.github.com/repos/$repo/releases/tags/$Tag"
} else {
    $rel = (Invoke-RestMethod -Headers $api "https://api.github.com/repos/$repo/releases?per_page=1")[0]
}
$asset = $rel.assets | Where-Object { $_.name -like "*windows-x64.zip" } | Select-Object -First 1
if (-not $asset) { throw "Release $($rel.tag_name) has no Windows download" }

$work = Join-Path $env:RUNNER_TEMP "kera-bench"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory $work | Out-Null
Invoke-WebRequest -Headers @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/octet-stream" } `
    -Uri $asset.url -OutFile "$work\kera.zip"
Expand-Archive "$work\kera.zip" $work
$kera = (Get-ChildItem -Recurse $work -Filter kera.exe | Select-Object -First 1).FullName
Write-Host "Kera $($rel.tag_name): $kera"

if ($env:KERA_BENCH_MODELS) {
    $models = $env:KERA_BENCH_MODELS -split ";"
} else {
    $home_dir = if ($env:KERA_HOME) { $env:KERA_HOME } else { Join-Path $env:LOCALAPPDATA "Kera" }
    $models = Get-Content (Join-Path $home_dir "bench_models.txt")
}
$models = $models | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") }
if (-not $models) { throw "No bench models listed" }

New-Item -ItemType Directory -Force $Out | Out-Null
& $kera setup --quick | Out-Host
foreach ($m in $models) {
    $name = [IO.Path]::GetFileNameWithoutExtension($m)
    & $kera bench --model $m --label "$Label $($rel.tag_name)" --out (Join-Path $Out "$name.json") | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "The bench failed on $m" }
}
