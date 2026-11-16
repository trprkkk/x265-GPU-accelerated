# Configure the x265 fork with the Visual Studio generator and MSVC.
# The CUDA kernels are built with nvcc at configure time, so this script
# imports vcvars64 (cl.exe on PATH) before running CMake.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File configure-vs.ps1 [-BuildDir build-vs]
param(
    [string]$BuildDir = "build-vs",
    [string]$CudaPath = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if( -not $vs ) { throw "Visual Studio not found via vswhere" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if( -not (Test-Path $vcvars) ) { throw "vcvars64.bat not found: $vcvars" }

# import the vcvars environment into this session
$lines = cmd /c "`"$vcvars`" >nul && set"
foreach( $line in $lines )
{
    if( $line -match '^([^=]+)=(.*)$' ) { Set-Item "Env:$($Matches[1])" $Matches[2] }
}

if( -not $CudaPath )
{
    $roots = @($env:CUDA_PATH, "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
    foreach( $r in $roots ) { if( $r -and (Test-Path (Join-Path $r "bin\nvcc.exe")) ) { $CudaPath = $r; break } }
}
if( $CudaPath ) { $env:CUDA_PATH = $CudaPath }

$build = Join-Path $root $BuildDir
New-Item -ItemType Directory -Force -Path $build | Out-Null
Push-Location $build
try
{
    cmake ../x265/source -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_ASSEMBLY=OFF -DENABLE_GPU_ME=ON
    if( $LASTEXITCODE -ne 0 ) { exit 1 }
}
finally { Pop-Location }
Write-Host "configured: $build"
