# Looks around a Windows machine before tools/compare runs on it, changing nothing: model files (GGUF over 1 GB, and
# Ollama's models), free space, memory, the graphics chip, Python, and AI apps already running.
#
#   compare-survey.ps1 [-Name nemotron] [-Wsl]
#
# -Name keeps only models whose path matches it. -Wsl also looks for models inside WSL's Linux (read-only).
# Plain output, not Write-Host, so it comes back as plain text over SSH.
param([string]$Name = "", [switch]$Wsl)
$ErrorActionPreference = "SilentlyContinue"
$ProgressPreference = "SilentlyContinue"

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
"Windows: $($os.Caption) $($os.Version); memory $([math]::Round($cs.TotalPhysicalMemory / 1GB)) GB, " +
    "$([math]::Round($os.FreePhysicalMemory / 1MB)) GB free"
Write-Output "Processor: $((Get-CimInstance Win32_Processor).Name)"
Get-CimInstance Win32_VideoController | ForEach-Object { Write-Output "Graphics: $($_.Name), driver $($_.DriverVersion)" }
$drives = (Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3")
$drives | ForEach-Object {
    Write-Output "Drive $($_.DeviceID) $([math]::Round($_.FreeSpace / 1GB)) GB free of $([math]::Round($_.Size / 1GB)) GB"
}
foreach ($c in "py", "python") {
    $cmd = Get-Command $c
    if ($cmd -and $cmd.Source -notlike "*\WindowsApps\*") { Write-Output "Python: $($cmd.Source) $(& $c --version 2>&1)" }
}
Get-Process | Where-Object { $_.ProcessName -match "ollama|lm ?studio|^lms$|llama|kobold|^jan$|msty|gpt4all|anythingllm|lemonade|kera" } |
    ForEach-Object { Write-Output "Running: $($_.ProcessName) (process $($_.Id), $([math]::Round($_.WorkingSet64 / 1GB, 1)) GB) $($_.Path)" }

# dir /s is much faster than Get-ChildItem over whole drives
foreach ($d in $drives.DeviceID) {
    cmd /c "dir /s /b /a-d `"$d\*.gguf`" 2>nul" | ForEach-Object {
        $f = Get-Item -LiteralPath $_
        if ($f -and $f.Length -gt 1GB -and $f.FullName -match $Name) {
            Write-Output "Model: $([math]::Round($f.Length / 1GB, 1)) GB $($f.FullName)"
        }
    }
}
# Ollama keeps each model as a blob named after its digest; the manifests say which is which
$ollama = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
Get-ChildItem -Recurse -File "$ollama\manifests" | Where-Object { $_.FullName -match $Name } | ForEach-Object {
    $layers = (Get-Content -Raw $_.FullName | ConvertFrom-Json).layers | Where-Object { $_.mediaType -like "*.model" }
    foreach ($l in $layers) {
        "Ollama model: $($_.FullName) is $ollama\blobs\$($l.digest -replace ':', '-') " +
            "($([math]::Round($l.size / 1GB, 1)) GB)"
    }
}

if ($Wsl) {
    # passed as base64 so no shell on the way changes the script
    $sh = @'
find / -xdev \( -path /proc -o -path /sys -o -path /mnt \) -prune -o -type f -size +1G -iname '*.gguf' -print 2>/dev/null |
    grep -i -- '__FILTER__' | while read -r f; do echo "WSL model: $(du -h "$f" | cut -f1) $f"; done
find / -xdev \( -path /proc -o -path /sys -o -path /mnt \) -prune -o -type f -path '*models/manifests/*' -print 2>/dev/null |
    grep -i -- '__FILTER__' | while read -r m; do
        echo "WSL Ollama model: $m is $(grep -o '"application/vnd.ollama.image.model","digest":"[^"]*"' "$m" | cut -d'"' -f6)"
    done
'@
    $sh = $sh.Replace("__FILTER__", $Name)  # String.Replace: exact, so -iname stays as it is
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($sh.Replace("`r", "")))
    wsl.exe -e bash -c "echo $b64 | base64 -d | bash" 
}
