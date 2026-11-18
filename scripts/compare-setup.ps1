# Gets a Windows machine ready for tools/compare/compare.py (stock llama.cpp against Kera on the same model), with
# everything in the Kera folder's "compare" folder: llama.cpp's own Vulkan build of the version Kera is built on, a
# finished Kera download, Python if the machine has none, and compare.py with the prompts it sends. Nothing outside
# that folder is changed.
#
#   compare-setup.ps1 [-KeraZip kera-windows-x64.zip] [-FilesZip compare-files.zip] [-LlamaTag b11619]
#
# -KeraZip is a release download (the repo is private) and -FilesZip holds tools/compare and bench/prompts, laid out
# as in the repo. By default they are tmp\kera.zip and tmp\compare-files.zip in the
# folder, where the agent copies them.
param(
    [string]$KeraZip = "",
    [string]$FilesZip = "",
    [string]$LlamaTag = "b11619"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$home_dir = if ($env:KERA_HOME) { $env:KERA_HOME } else { Join-Path $env:LOCALAPPDATA "Kera" }
$dir = Join-Path $home_dir "compare"
New-Item -ItemType Directory -Force "$dir\tmp" | Out-Null
if (-not $KeraZip -and (Test-Path "$dir\tmp\kera.zip")) { $KeraZip = "$dir\tmp\kera.zip" }
if (-not $FilesZip -and (Test-Path "$dir\tmp\compare-files.zip")) { $FilesZip = "$dir\tmp\compare-files.zip" }

# llama.cpp's own Windows Vulkan build, of the version pinned in docs/UPSTREAM_PATCHES.md
$llama = Join-Path $dir "llama-$LlamaTag"
if (-not (Test-Path "$llama\llama-server.exe")) {
    $rel = Invoke-RestMethod -Headers @{ "User-Agent" = "kera" } `
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LlamaTag"
    $asset = $rel.assets | Where-Object { $_.name -like "*-bin-win-vulkan-x64.zip" } | Select-Object -First 1
    if (-not $asset) { throw "llama.cpp $LlamaTag has no Windows Vulkan download" }
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile "$dir\tmp\llama.zip"
    Expand-Archive -Force "$dir\tmp\llama.zip" $llama
}

if ($KeraZip) {
    if (Test-Path "$dir\kera") { Remove-Item -Recurse -Force "$dir\kera" }
    Expand-Archive -Force $KeraZip "$dir\kera"
}
if ($FilesZip) {
    if (Test-Path "$dir\repo") { Remove-Item -Recurse -Force "$dir\repo" }
    Expand-Archive -Force $FilesZip "$dir\repo"
}

# the machine's own Python if it has one (not the Microsoft Store stand-in), else the embeddable one from python.org
$py = $null
foreach ($c in "py", "python") {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*\WindowsApps\*") { $py = $cmd.Source; break }
}
if (-not $py) {
    $py = "$dir\python\python.exe"
    if (-not (Test-Path $py)) {
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" `
            -OutFile "$dir\tmp\python.zip"
        Expand-Archive -Force "$dir\tmp\python.zip" "$dir\python"
    }
}
Set-Content -Encoding ascii "$dir\python.txt" $py
Remove-Item -Recurse -Force "$dir\tmp\*"

$server = Get-ChildItem -Recurse $llama -Filter llama-server.exe | Select-Object -First 1
$kera = Get-ChildItem -Recurse "$dir\kera" -Filter kera.exe -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Output "llama.cpp $LlamaTag : $($server.FullName)"
Write-Output "Kera: $(if ($kera) { $kera.FullName } else { 'missing (pass -KeraZip)' })"
Write-Output "Python: $py ($(& $py --version 2>&1))"
Write-Output "compare.py: $(Test-Path "$dir\repo\tools\compare\compare.py")"
