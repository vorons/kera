# Starts or stops a private Ollama for tools/compare: the machine's own Ollama program and models, on a port of its
# own and only for this machine (127.0.0.1). A test then never waits behind, or pushes out, the models the machine's
# own Ollama is serving to other apps. Its log goes to the Kera folder's compare\results folder.
#
#   ollama-private.ps1 [-Port 11500] [-Parallel N]   start it (-Parallel: chats at once, OLLAMA_NUM_PARALLEL);
#                                                    then: compare.py --ollama-url http://127.0.0.1:11500
#   ollama-private.ps1 -Stop                         stop it and the model runners it started
param([int]$Port = 11500, [int]$Parallel = 0, [switch]$Stop)
$ErrorActionPreference = "Stop"

$home_dir = if ($env:KERA_HOME) { $env:KERA_HOME } else { Join-Path $env:LOCALAPPDATA "Kera" }
$dir = Join-Path $home_dir "compare"
$pid_file = "$dir\ollama-private.pid"

function Stop-Tree([int]$id) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" | ForEach-Object { Stop-Tree $_.ProcessId }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

if ($Stop) {
    if (Test-Path $pid_file) {
        $id = [int](Get-Content $pid_file)
        Stop-Tree $id
        Remove-Item $pid_file
        Write-Output "Stopped the private Ollama (process $id)"
    }
    exit 0
}

$exe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $exe) { $exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
if (-not (Test-Path $exe)) { throw "Ollama isn't installed here" }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is in use" }
$models = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "Machine")
if (-not $models) { $models = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User") }
if (-not $models) { $models = Join-Path $env:USERPROFILE ".ollama\models" }
New-Item -ItemType Directory -Force "$dir\results" | Out-Null

# made through WMI so it outlives the SSH session; cmd.exe sets its address and models folder
$parallel = if ($Parallel -gt 0) { "set OLLAMA_NUM_PARALLEL=$Parallel&& " } else { "" }
$cmd = "cmd.exe /c set OLLAMA_HOST=127.0.0.1:$Port&& set OLLAMA_MODELS=$models&& set OLLAMA_KEEP_ALIVE=30m&& $parallel" +
       "`"$exe`" serve > `"$dir\results\ollama-private.log`" 2>&1"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $dir }
if ($r.ReturnValue -ne 0) { throw "Couldn't start Ollama (WMI error $($r.ReturnValue))" }
Set-Content -Encoding ascii $pid_file $r.ProcessId
for ($i = 0; $i -lt 60; $i++) {
    try {
        $v = Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -TimeoutSec 2
        Write-Output "Private Ollama $($v.version) on http://127.0.0.1:$Port (process $($r.ProcessId))"
        exit 0
    } catch { Start-Sleep -Milliseconds 500 }
}
Stop-Tree $r.ProcessId
throw "The private Ollama didn't start; see $dir\results\ollama-private.log"
