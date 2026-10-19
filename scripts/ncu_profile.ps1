# Profile one kernel with Nsight Compute - the per-kernel microscope.
#
#   powershell -ExecutionPolicy Bypass -File scripts\ncu_profile.ps1 `
#       -Exe ..\bin\bench_matmul.exe -Kernel RegTile
#
# Nsight Compute answers: is this kernel compute-bound or memory-bound, what is
# the achieved occupancy, how much of peak memory bandwidth am I getting.
# The "Speed Of Light" and "Roofline" sections are the ones to read first.
#
# IMPORTANT: on Windows ncu must run ELEVATED (admin). Without admin it fails
# with "Launching the target application failed" because it cannot inject its
# profiling library - that error is about privileges, not your code.

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Kernel = "",
    [int]$LaunchSkip = 3,
    [int]$LaunchCount = 1,
    [string]$Out = "profile",
    [string]$Args = ""
)

$ErrorActionPreference = "Stop"

$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe"
if (-not (Test-Path $ncu)) {
    $found = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter ncu.exe -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if (-not $found) { Write-Error "ncu.exe not found"; exit 1 }
    $ncu = $found.FullName
}

# ncu warns (and can misbehave) when HOME is unset.
$env:HOME = if ($env:HOME) { $env:HOME } else { $env:USERPROFILE }

# --set full is slow but gives everything; use "basic" for a quick pass.
$setName = if ($Kernel -ne "") { "full" } else { "basic" }

$ncuArgs = @(
    "--target-processes", "all",
    "--set", $setName,
    "--launch-skip", "$LaunchSkip",
    "--launch-count", "$LaunchCount",
    "--import-source", "yes",
    "-o", $Out,
    "-f"
)

if ($Kernel -ne "") {
    $ncuArgs += @("--kernel-name", "regex:$Kernel")
}
if ($Args -ne "") {
    $ncuArgs += $Args.Split(" ")
}
$ncuArgs += $Exe

Write-Host "Running: $ncu $($ncuArgs -join ' ')"
& $ncu @ncuArgs

Write-Host ""
Write-Host "Report written to $Out.ncu-rep - open it in the Nsight Compute GUI."
