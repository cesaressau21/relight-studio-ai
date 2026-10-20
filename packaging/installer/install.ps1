<#
Sets up (or repairs) the Python environment of an installed Relight Studio AI.

Run by the installer after copying the application, and by the "Repair" shortcut:

    powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [-Force]

Layout under the installation folder (the parent of this script's folder):
    app\      application source, presets, assets (copied by the installer)
    tools\    uv, downloaded from its GitHub release and checked against a pinned SHA-256
    python\   the managed Python interpreter
    env\      the virtual environment the shortcut runs
    install.log

The PyTorch build follows the GPU: cu130 for NVIDIA GPUs of compute capability 7.5 or
newer with a CUDA 13 driver, cu126 for older NVIDIA GPUs (not RTX 50, which needs CUDA
12.8+), CPU otherwise. Override with -Variant cu130|cu126|cpu or RELIGHT_TORCH_VARIANT.
RELIGHT_UV_CACHE reuses an existing uv cache instead of a temporary one.

-DetectOnly prints the variant and exits; -GpuInfo "<compute_cap>,<driver>" replaces the
nvidia-smi query (tests).
#>
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot),
    [string]$Variant = $env:RELIGHT_TORCH_VARIANT,
    [switch]$Force,
    [switch]$DetectOnly,
    [string]$GpuInfo
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is very slow while drawing progress

$UvVersion = '0.13.0'
$UvSha256 = '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'
$PythonVersion = '3.12'
$TorchVersion = '2.14.1'
$Cuda13Driver = 580.0    # minimum NVIDIA driver for CUDA 13 builds
$Cuda12Driver = 528.33   # minimum NVIDIA driver for CUDA 12.x builds (minor version compatibility)

function Get-TorchVariant([string]$Info) {
    if (-not $Info) {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $smi) {
            $candidate = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
            if (Test-Path $candidate) { $smi = $candidate }
        }
        if (-not $smi) { return 'cpu' }
        $lines = & $smi --query-gpu=compute_cap,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $lines) { return 'cpu' }
        $Info = @($lines)[0]
    }
    $parts = $Info.Split(',') | ForEach-Object { $_.Trim() }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $cap = [double]::Parse($parts[0], $culture)
    $driver = [double]::Parse(($parts[1].Split('.')[0..1] -join '.'), $culture)
    if ($cap -ge 7.5 -and $driver -ge $Cuda13Driver) { return 'cu130' }
    if ($cap -ge 5.0 -and $cap -lt 12.0 -and $driver -ge $Cuda12Driver) { return 'cu126' }
    return 'cpu'
}

if (-not $Variant) { $Variant = Get-TorchVariant $GpuInfo }
if ($Variant -notin @('cu130', 'cu126', 'cpu')) { throw "Unknown PyTorch variant '$Variant'" }
if ($DetectOnly) { Write-Output $Variant; exit 0 }

$Log = Join-Path $AppDir 'install.log'
function Write-Log([string]$Message) {
    $line = '{0:yyyy-MM-dd HH:mm:ss}  {1}' -f (Get-Date), $Message
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Invoke-Native([string]$What, [string]$Exe, [string[]]$Arguments) {
    Write-Log $What
    # Progress goes to the console and the log; tools write it to stderr, which is not an error.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $Log -Value "    $line" -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "$What failed (exit code $code)" }
}

$Tools = Join-Path $AppDir 'tools'
$EnvDir = Join-Path $AppDir 'env'
$Lock = Join-Path $PSScriptRoot 'requirements-lock.txt'
$Python = Join-Path $EnvDir 'Scripts\python.exe'
$StampFile = Join-Path $EnvDir 'relight-stamp.txt'
$Stamp = "python=$PythonVersion torch=$TorchVersion+$Variant lock=$((Get-FileHash $Lock -Algorithm SHA256).Hash)"

try {
    Write-Log "Relight Studio AI environment setup in $AppDir (PyTorch build: $Variant)"
    if (-not $Force -and (Test-Path $StampFile) -and (Get-Content $StampFile -Raw).Trim() -eq $Stamp -and (Test-Path $Python)) {
        Write-Log 'The environment is already up to date.'
        exit 0
    }

    # uv: a pinned release, verified before use.
    $Uv = Join-Path $Tools 'uv.exe'
    if (-not (Test-Path $Uv)) {
        New-Item -ItemType Directory -Force -Path $Tools | Out-Null
        $Zip = Join-Path $Tools 'uv.zip'
        $Url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
        Write-Log "Downloading uv $UvVersion"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
        $Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Hash -ne $UvSha256) { throw "uv download is corrupted or was tampered with (SHA-256 $Hash)" }
        Expand-Archive -Path $Zip -DestinationPath $Tools -Force
        Remove-Item $Zip
    }

    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppDir 'python'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_LINK_MODE = 'copy'
    $env:UV_NO_CONFIG = '1'
    $TempCache = -not $env:RELIGHT_UV_CACHE
    $env:UV_CACHE_DIR = if ($TempCache) { Join-Path $AppDir 'cache' } else { $env:RELIGHT_UV_CACHE }

    if (Test-Path $EnvDir) { Remove-Item -Recurse -Force $EnvDir }
    # Private to this installation: no python.exe on the user's PATH, no Windows registry entry.
    Invoke-Native "Installing Python $PythonVersion" $Uv @('python', 'install', $PythonVersion, '--no-bin', '--no-registry')
    Invoke-Native 'Creating the environment' $Uv @('venv', '--python', $PythonVersion, $EnvDir)
    Invoke-Native "Downloading PyTorch $TorchVersion ($Variant, the largest download)" $Uv @(
        'pip', 'install', '--python', $Python, "torch==$TorchVersion",
        '--index-url', "https://download.pytorch.org/whl/$Variant")
    Invoke-Native 'Downloading the other components' $Uv @('pip', 'install', '--python', $Python, '-r', $Lock)
    # No double quotes in arguments: Windows PowerShell 5.1 strips them when calling programs.
    Invoke-Native 'Checking the installation' $Python @('-c',
        "import torch, transformers, PySide6.QtWidgets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")

    Set-Content -Path $StampFile -Value $Stamp -Encoding ASCII
    if ($TempCache) { Remove-Item -Recurse -Force $env:UV_CACHE_DIR -ErrorAction SilentlyContinue }
    Write-Log 'Done.'
    exit 0
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Log "Check the internet connection, then run 'Repair Relight Studio AI' from the Start menu."
    exit 1
}
