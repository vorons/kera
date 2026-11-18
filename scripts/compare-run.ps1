# Runs tools/compare/compare.py (stock llama.cpp against Kera on the same model) with what compare-setup.ps1 put in
# the Kera folder's "compare" folder, keeping its temporary files there too. Results go to its "results" folder:
# compare-TAG.log (progress and the table), compare-TAG.json, compare-TAG.err and compare-TAG.status.
#
#   compare-run.ps1 -Model MODEL.gguf [-Tag nemotron] [-Ngl 99] [-Ctx 16384] [-Runs 3] [-Extra "-fa on"]
#                   [-More "--engines ollama,kera --ollama nemotron-3-nano:30b"] [-KeraDir kera-next]
#                   [-SetEnv "KERA_IO_TRACE=1;HIP_LAUNCH_BLOCKING=1"] [-Tool multi_user|mixed_day] [-Quick] [-Detach]
#
# -Tool multi_user runs tools/compare/multi_user.py (several people at once) instead of compare.py, and -Tool mixed_day
# tools/compare/mixed_day.py (a day of chats and a long document on Vulkan, ROCm and Kera choosing per message); -Runs
# doesn't apply to them.
#
# -Extra goes to both servers, -More to compare.py itself, -SetEnv to the servers' environment. -KeraDir names the
# folder (in the compare folder) holding the Kera build to test, "kera" (from compare-setup.ps1) by default. -Detach
# starts the run outside the current session and returns at once, so it keeps going if an SSH connection drops.
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Tag = "run",
    [string]$Ngl = "99",
    [int]$Ctx = 16384,
    [int]$Runs = 3,
    [string]$Extra = "",
    [string]$More = "",
    [string]$KeraDir = "kera",
    [string]$SetEnv = "",
    [string]$Tool = "compare",
    [switch]$Quick,
    [switch]$Detach,
    [string]$Dir = ""
)
$ErrorActionPreference = "Stop"

if (-not $Dir) {
    $home_dir = if ($env:KERA_HOME) { $env:KERA_HOME } else { Join-Path $env:LOCALAPPDATA "Kera" }
    $Dir = Join-Path $home_dir "compare"
}
$out = Join-Path $Dir "results\compare-$Tag"
New-Item -ItemType Directory -Force "$Dir\results", "$Dir\tmp" | Out-Null

if ($Detach) {
    # a process made through WMI isn't part of the SSH session, so closing the connection doesn't stop it. The
    # command goes encoded, its values in single quotes: on a plain command line a lone value such as
    # "--no-auto-guess" is taken for a parameter's name, and the run never starts.
    $q = { param($s) "'" + ($s -replace "'", "''") + "'" }
    $c = "& $(& $q $PSCommandPath) -Model $(& $q $Model) -Tag $(& $q $Tag) -Ngl $(& $q $Ngl) -Ctx $Ctx -Runs $Runs" +
         " -Dir $(& $q $Dir) -KeraDir $(& $q $KeraDir)"
    if ($Extra) { $c += " -Extra $(& $q $Extra)" }
    if ($More) { $c += " -More $(& $q $More)" }
    if ($SetEnv) { $c += " -SetEnv $(& $q $SetEnv)" }
    if ($Tool -ne "compare") { $c += " -Tool $(& $q $Tool)" }
    if ($Quick) { $c += " -Quick" }
    # a run that fails says why in its status file (a detached process has nowhere else to say it)
    $c = "try { $c; exit `$LASTEXITCODE } catch { ""failed: `$_"" | Add-Content -Encoding utf8 $(& $q "$out.status"); exit 1 }"
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($c))
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $enc"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Dir }
    if ($r.ReturnValue -ne 0) { throw "Couldn't start the comparison (WMI error $($r.ReturnValue))" }
    Write-Output "Started (process $($r.ProcessId)); progress in $out.log"
    exit 0
}

$env:TMPDIR = "$Dir\tmp"; $env:TEMP = "$Dir\tmp"; $env:TMP = "$Dir\tmp"; $env:PYTHONUTF8 = "1"
foreach ($kv in ($SetEnv -split ";" | Where-Object { $_ -match "=" })) {
    $name, $value = $kv -split "=", 2
    Set-Item "env:$($name.Trim())" $value
}
$py = (Get-Content "$Dir\python.txt" | Select-Object -First 1).Trim()
$llama = (Get-ChildItem -Directory $Dir -Filter "llama-*" | Get-ChildItem -Recurse -Filter llama-server.exe |
    Select-Object -First 1).FullName
$kera = (Get-ChildItem -Recurse (Join-Path $Dir $KeraDir) -Filter kera.exe | Select-Object -First 1).FullName
$a = @("-u", "$Dir\repo\tools\compare\$Tool.py", "--model", $Model, "--llama-server", $llama, "--kera", $kera,
       "--ngl", $Ngl, "--ctx", "$Ctx", "--out", "$out.json")
if ($Tool -eq "compare") { $a += @("--runs", "$Runs") }
# one piece, so a lone option such as "--no-auto-guess" isn't read as compare.py's own
if ($Extra) { $a += "--extra=$Extra" }
if ($More) { $a += $More -split "\s+" | Where-Object { $_ } }
if ($Quick) { $a += "--quick" }
# Start-Process joins the arguments with spaces, so each one is quoted (paths can have spaces)
$q = $a | ForEach-Object { '"' + $_ + '"' }
"started $(Get-Date -Format s)" | Set-Content -Encoding utf8 "$out.status"
$p = Start-Process -FilePath $py -ArgumentList $q -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" `
    -NoNewWindow -Wait -PassThru
"finished $(Get-Date -Format s), exit code $($p.ExitCode)" | Add-Content -Encoding utf8 "$out.status"
exit $p.ExitCode
