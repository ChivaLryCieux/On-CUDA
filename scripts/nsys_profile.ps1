# Capture a system-level timeline with Nsight Systems - the pipeline microscope.
#
#   powershell -ExecutionPolicy Bypass -File scripts\nsys_profile.ps1 -Exe ..\build\src\conv2d\bench_conv.exe
#
# Nsight Systems answers a different question than Nsight Compute: it shows the
# whole timeline - CUDA API calls, kernel launches, memcpys, and whether the GPU
# ever sits idle waiting for the CPU. Use it to find gaps between kernels,
# missing stream overlap, and unexpected synchronization.

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Out = "timeline",
    [string]$Args = ""
)

$ErrorActionPreference = "Stop"

$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"
if (-not (Test-Path $nsys)) {
    $found = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter nsys.exe -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -like "*x64*" } | Select-Object -First 1
    if (-not $found) { Write-Error "nsys.exe not found"; exit 1 }
    $nsys = $found.FullName
}

$nsysArgs = @(
    "profile",
    "--trace", "cuda,nvtx,osrt",
    "--sample", "none",
    "--cpuctxsw", "none",
    "--stats", "true",
    "-o", $Out,
    "-f"
)

if ($Args -ne "") { $nsysArgs += $Args.Split(" ") }
$nsysArgs += $Exe

Write-Host "Running: $nsys $($nsysArgs -join ' ')"
& $nsys @nsysArgs

Write-Host ""
Write-Host "Timeline written to $Out.nsys-rep - open it in the Nsight Systems GUI."
Write-Host "In the timeline look for: white gaps on the CUDA row (GPU starved),"
Write-Host "and memcpy rows that do not overlap with kernel rows."
