$ErrorActionPreference = 'Stop'

param(
    [string]$Model = 'mamba-370m',
    [int]$MaxSamples = 20,
    [int]$MaxLength = 4096,
    [string]$GpuIds = '2 3',
    [string]$RemoteRoot = '/home1/mabo1215/COREY_Transformer',
    [string]$RemoteOutputBase = 'src/outputs/mgpu_longbench_remote',
    [switch]$SkipCodeSync,
    [switch]$SkipDataSync,
    [switch]$SkipPullBack
)

$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$envPath = Join-Path $workspaceRoot '.env'
if (-not (Test-Path $envPath)) {
    throw ".env not found at $envPath"
}

$envLines = Get-Content -Path $envPath
if ($envLines.Count -lt 2) {
    throw ".env must contain at least two lines: remote host on line 1 and password on line 2."
}

$remote = $envLines[0].Trim()
$password = $envLines[1].Trim()
$hostKey = 'ssh-ed25519 255 SHA256:Jj7AizwqBqF1buL3ZBUiE5P37N9XXvel+rxwrYIPty0'

$plink = 'C:\Program Files\PuTTY\plink.exe'
$pscp = 'C:\Program Files\PuTTY\pscp.exe'
if (-not (Test-Path $plink)) {
    throw "plink.exe not found at $plink"
}
if (-not (Test-Path $pscp)) {
    throw "pscp.exe not found at $pscp"
}

function Invoke-RemoteCommand {
    param([string]$Command)

    & $plink -ssh -batch -hostkey $hostKey -pw $password $remote $Command
}

function Ensure-RemoteDir {
    param([string]$Path)

    Invoke-RemoteCommand "mkdir -p $Path" | Out-Null
}

function Copy-FileToRemote {
    param(
        [string]$LocalPath,
        [string]$RemotePath
    )

    $destination = "${remote}:$RemotePath"
    & $pscp -batch -hostkey $hostKey -pw $password $LocalPath $destination | Out-Null
}

function Copy-DirectoryToRemote {
    param(
        [string]$LocalPath,
        [string]$RemoteParent
    )

    $destination = "${remote}:$RemoteParent"
    & $pscp -r -batch -hostkey $hostKey -pw $password $LocalPath $destination | Out-Null
}

Write-Host "[remote-mgpu] remote=$remote"
Invoke-RemoteCommand 'hostname; pwd'

if (-not $SkipCodeSync) {
    Write-Host '[remote-mgpu] syncing code...'
    Ensure-RemoteDir "$RemoteRoot/src"
    Ensure-RemoteDir "$RemoteRoot/src/scripts"

    Copy-DirectoryToRemote (Join-Path $workspaceRoot 'src\algorithms') "$RemoteRoot/src"
    Copy-DirectoryToRemote (Join-Path $workspaceRoot 'src\experiments') "$RemoteRoot/src"
    Copy-FileToRemote (Join-Path $workspaceRoot 'src\scripts\wsl_run_multigpu_longbench.sh') "$RemoteRoot/src/scripts/wsl_run_multigpu_longbench.sh"
    Copy-FileToRemote (Join-Path $workspaceRoot 'src\scripts\remote_run_multigpu.sh') "$RemoteRoot/src/scripts/remote_run_multigpu.sh"
}

if (-not $SkipDataSync) {
    Write-Host '[remote-mgpu] syncing LongBench subset data...'
    Ensure-RemoteDir "$RemoteRoot/src/data"
    Copy-DirectoryToRemote (Join-Path $workspaceRoot 'src\data\longbench_subset') "$RemoteRoot/src/data"
}

$remoteCommand = @"
set -euo pipefail
cd $RemoteRoot
export REMOTE_ROOT=$RemoteRoot
export MODEL=$Model
export MAX_SAMPLES=$MaxSamples
export MAX_LENGTH=$MaxLength
export GPU_IDS='$GpuIds'
export OUTPUT_BASE=$RemoteOutputBase
bash src/scripts/remote_run_multigpu.sh
"@

Write-Host '[remote-mgpu] launching remote multigpu run...'
Invoke-RemoteCommand $remoteCommand

if (-not $SkipPullBack) {
    Write-Host '[remote-mgpu] pulling merged results back...'
    $localOutputs = Join-Path $workspaceRoot 'src\outputs'
    if (-not (Test-Path $localOutputs)) {
        New-Item -ItemType Directory -Path $localOutputs | Out-Null
    }
    $remoteMerged = "${remote}:$RemoteRoot/${RemoteOutputBase}_merged"
    & $pscp -r -batch -hostkey $hostKey -pw $password $remoteMerged $localOutputs | Out-Null
}

Write-Host '[remote-mgpu] done.'